import os
import sys
import time
import random
from pathlib import Path

import hydra
import numpy as np
import torch

from dgl.dataloading import GraphDataLoader
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, open_dict
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ""))
if project_path not in sys.path:
    sys.path.append(project_path)

from python.create_dgl_dataset import TelemacDatasetWithSourceNodes, collate_source_sequences, save_json
from python.CustomMeshGraphNet import MeshGraphNetWithSourceNodes

from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.launch.utils import load_checkpoint, save_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_resume(cfg: DictConfig, ckpt_path: Path) -> int:
    model_epochs = {
        int(path.name.split(".")[-2])
        for path in ckpt_path.glob("*.0.*.mdlus")
    }
    state_epochs = {
        int(path.name.split(".")[-2])
        for path in ckpt_path.glob("checkpoint.0.*.pt")
    }
    complete_epochs = model_epochs & state_epochs
    if not complete_epochs:
        raise FileNotFoundError(f"Aucun checkpoint complet trouvé dans {ckpt_path}")

    epoch = max(complete_epochs)
    with open_dict(cfg):
        cfg.train_from_scratch = False
        cfg.init_ckpt_path = str(ckpt_path)
        cfg.init_epoch = epoch
        cfg.resume_training = True
    return epoch


class SourceNodeTrainer:
    def __init__(self, cfg: DictConfig, r0: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp = bool(getattr(cfg, "amp", False)) and self.dist.device.type == "cuda"

        self.sequence_length = int(cfg.sequence_length)
        if self.sequence_length < 1:
            raise ValueError("sequence_length doit etre >= 1.")
        self.use_q_feature = bool(getattr(cfg, "use_q_feature", False))

        hydro_files = getattr(cfg, "hydro_dir", None)
        cli_file = getattr(cfg, "cli_file", None)
        inlet_node_lists = getattr(cfg, "inlet_node_lists", None)
        if hydro_files is None:
            raise ValueError("hydro_dir is required for source-node training.")
        if cli_file is None and inlet_node_lists is None:
            raise ValueError("Provide cli_file or inlet_node_lists in config.")

        output_ckpt = to_absolute_path(cfg.ckpt_path)
        resuming = not cfg.get("train_from_scratch", False) and cfg.get("resume_training", True)
        stats_ckpt = to_absolute_path(cfg.get("init_ckpt_path", output_ckpt)) if resuming else output_ckpt

        self.dataset = TelemacDatasetWithSourceNodes(
            name="telemac_train_source_nodes",
            data_dir=to_absolute_path(cfg.data_dir),
            dynamic_data_files=[to_absolute_path(p) for p in cfg.dynamic_dir],
            hydro_data_files=[to_absolute_path(p) for p in hydro_files],
            cli_file=to_absolute_path(cli_file) if cli_file else None,
            inlet_node_lists=inlet_node_lists,
            use_q_feature=self.use_q_feature,
            # The split selects how statistics are obtained, independently of the event list.
            split="test" if resuming else "train",
            ckpt_path=stats_ckpt,
            normalize=True,
            sequence_length=self.sequence_length,
            overlap=int(getattr(cfg, "overlap", self.sequence_length)),
            dt_seconds=float(getattr(cfg, "dt_seconds", 1800.0)),
        )

        expected_phys_features = (
            self.dataset.base_graph.ndata["static"].shape[1]
            + self.dataset.physical_dynamic_dim
        )
        if int(cfg.num_input_features) != int(expected_phys_features):
            raise ValueError(
                f"num_input_features={cfg.num_input_features} incompatible with source-node dataset "
                f"({expected_phys_features})."
            )

        expected_source_features = len(self.dataset.source_feature_names)
        cfg_source_dim = getattr(cfg, "num_input_features_source", None)
        if cfg_source_dim is not None and int(cfg_source_dim) != int(expected_source_features):
            raise ValueError(
                f"num_input_features_source={cfg_source_dim} incompatible with source-node dataset "
                f"({expected_source_features})."
            )

        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=self.dist.world_size > 1,
            num_workers=cfg.num_dataloader_workers,
            collate_fn=collate_source_sequences,
        )

        hidden_dim = int(getattr(cfg, "hidden_dim", 64))
        self.model = MeshGraphNetWithSourceNodes(
            input_dim_nodes_phys=expected_phys_features,
            input_dim_nodes_src=expected_source_features,
            input_dim_edges=cfg.num_edge_features,
            output_dim=cfg.num_output_features,
            processor_size=cfg.mp_layers,
            hidden_dim_processor=hidden_dim,
            hidden_dim_node_encoder=hidden_dim,
            hidden_dim_edge_encoder=hidden_dim,
            hidden_dim_node_decoder=hidden_dim,
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
        ).to(self.dist.device)

        if bool(cfg.jit):
            if not getattr(self.model, "meta", None) or not getattr(self.model.meta, "jit", False):
                raise ValueError("MeshGraphNetWithSourceNodes n'est pas JIT-compatible.")
            self.model = torch.jit.script(self.model).to(self.dist.device)

        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        self.model.train()
        self.criterion = torch.nn.MSELoss()

        self.optimizer = None
        try:
            if bool(cfg.use_apex):
                from apex.optimizers import FusedAdam

                self.optimizer = FusedAdam(self.model.parameters(), lr=cfg.lr)
        except ImportError:
            r0.warning("Apex non installe, utilisation d'Adam standard.")

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        if str(getattr(cfg, "lr_scheduler", "exponential")) == "warmup_cosine":
            warmup_epochs = int(cfg.lr_warmup_epochs)
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=float(cfg.lr_warmup_start) / float(cfg.lr),
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=int(cfg.epochs) - warmup_epochs,
                eta_min=float(cfg.lr_min),
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch,
            )
        self.scaler = GradScaler(enabled=self.amp)

        if self.dist.world_size > 1:
            torch.distributed.barrier()

        self.epoch_init = self._load_init_weights(cfg, r0)
        if resuming and self.dist.rank == 0 and Path(stats_ckpt).resolve() != Path(output_ckpt).resolve():
            save_json(self.dataset.node_stats, output_ckpt, "node_stats.json")
            save_json(self.dataset.edge_stats, output_ckpt, "edge_stats.json")
            save_json(self.dataset.source_stats, output_ckpt, "source_stats.json")

        self.pf_start_epoch = int(getattr(cfg, "pf_start_epoch", self.epoch_init))
        self.pf_warmup_epochs = int(getattr(cfg, "pf_warmup_epochs", 0))
        self.p_tf = (
            lambda epoch: 0.0
            if self.pf_warmup_epochs <= 0
            else max(0.0, 1.0 - max(0, epoch - self.pf_start_epoch) / self.pf_warmup_epochs)
        )

        node_stats = self.dataset.node_stats
        self.mx = torch.tensor(
            [node_stats["h"].item(), node_stats["u"].item(), node_stats["v"].item()],
            dtype=torch.float32,
            device=self.dist.device,
        )
        self.sx = torch.tensor(
            [node_stats["h_std"].item(), node_stats["u_std"].item(), node_stats["v_std"].item()],
            dtype=torch.float32,
            device=self.dist.device,
        )
        self.my = torch.tensor(
            [node_stats["delta_h"].item(), node_stats["delta_u"].item(), node_stats["delta_v"].item()],
            dtype=torch.float32,
            device=self.dist.device,
        )
        self.sy = torch.tensor(
            [
                node_stats["delta_h_std"].item(),
                node_stats["delta_u_std"].item(),
                node_stats["delta_v_std"].item(),
            ],
            dtype=torch.float32,
            device=self.dist.device,
        )

        self.static_dim = self.dataset.base_graph.ndata["static"].shape[1]
        self.physical_dynamic_dim = self.dataset.physical_dynamic_dim

    def _load_init_weights(self, cfg: DictConfig, r0: RankZeroLoggingWrapper) -> int:
        if bool(getattr(cfg, "train_from_scratch", False)):
            r0.info("train_from_scratch=True, skipping checkpoint loading.")
            return 1

        init_ckpt_path = to_absolute_path(getattr(cfg, "init_ckpt_path", cfg.ckpt_path))
        init_epoch = getattr(cfg, "init_epoch", None)
        resume_training = bool(getattr(cfg, "resume_training", True))
        if not Path(init_ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {init_ckpt_path}")

        load_kwargs = {
            "models": self.model,
            "device": self.dist.device,
        }
        if init_epoch is not None:
            init_epoch = int(init_epoch)
            model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
            model_path = Path(init_ckpt_path) / f"{model.meta.name}.0.{init_epoch}.mdlus"
            if not model_path.is_file():
                raise FileNotFoundError(f"Poids d'initialisation introuvables: {model_path}")
            load_kwargs["epoch"] = init_epoch
        if resume_training:
            load_kwargs.update(
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )

        try:
            loaded_epoch = load_checkpoint(init_ckpt_path, **load_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Could not load this checkpoint into MeshGraphNetWithSourceNodes. "
                "If the checkpoint comes from the old MeshGraphNet architecture, "
                "start with train_from_scratch=true. Use resume_training=false to "
                "load model weights without optimizer state."
            ) from exc

        if resume_training and init_epoch is not None and int(loaded_epoch or 0) != init_epoch:
            raise RuntimeError(f"État d'entraînement de l'époque {init_epoch} absent dans {init_ckpt_path}")
        checkpoint_epoch = int(init_epoch if init_epoch is not None else loaded_epoch or 0)
        if resume_training:
            start_epoch = checkpoint_epoch + 1
            r0.info(
                f"Resumed source-node training from {init_ckpt_path} "
                f"after epoch {checkpoint_epoch}"
            )
            return start_epoch

        r0.info(
            f"Initialized source-node weights from {init_ckpt_path} "
            f"at epoch {checkpoint_epoch}"
        )
        return 1

    @staticmethod
    def _denorm(xn, mean, std):
        return xn * std + mean

    @staticmethod
    def _renorm(x, mean, std):
        return (x - mean) / (std + 1e-12)

    def _apply_h_boundary_inplace(self, x_pred, x_gt, onehot):
        h_mask = (onehot == torch.tensor([0, 1, 0, 0], device=onehot.device)).all(dim=1)
        x_pred[h_mask, 0] = x_gt[h_mask, 0]
        return x_pred

    def train_batch(self, sequence, epoch):
        one_step = len(sequence) == 1
        k_steps = 1 if one_step else len(sequence) - 1

        current_x_phys = sequence[0]["x_phys"].to(self.dist.device)
        current_x_src = sequence[0]["x_src"].to(self.dist.device)
        total_loss = torch.zeros((), device=self.dist.device)

        self.optimizer.zero_grad()

        for t in range(k_steps):
            step = sequence[t]
            graph = step["graph"].to(self.dist.device)
            edge_features = graph.edata["x"].to(self.dist.device)

            with autocast(enabled=self.amp):
                y_pred_n = self.model(
                    graph=graph,
                    x_phys=current_x_phys,
                    x_src=current_x_src,
                    edge_features=edge_features,
                )
                y_gt_n = step["y_phys"].to(self.dist.device)
                loss_t = self.criterion(y_pred_n, y_gt_n)

            total_loss = total_loss + loss_t
            if one_step:
                break

            next_step = sequence[t + 1]
            onehot = current_x_phys[:, :4]
            xn_t = current_x_phys[:, self.static_dim:self.static_dim + 3]
            x_t = self._denorm(xn_t, self.mx.to(xn_t.device), self.sx.to(xn_t.device))

            y_pred = self._denorm(y_pred_n, self.my.to(xn_t.device), self.sy.to(xn_t.device))
            x_t1_pred = x_t + y_pred

            xn_t1_gt = next_step["x_phys"][:, self.static_dim:self.static_dim + 3].to(self.dist.device)
            x_t1_gt = self._denorm(xn_t1_gt, self.mx.to(xn_t.device), self.sx.to(xn_t.device))

            x_t1_pred = self._apply_h_boundary_inplace(x_t1_pred, x_t1_gt, onehot)

            use_tf = torch.rand(1, device=self.dist.device).item() < self.p_tf(epoch)
            x_t1_next = x_t1_gt if use_tf else x_t1_pred
            xn_t1_next = self._renorm(x_t1_next, self.mx.to(xn_t.device), self.sx.to(xn_t.device))

            static_features = current_x_phys[:, :self.static_dim]
            forcing_t1 = next_step["x_phys"][
                :,
                self.static_dim + 3:self.static_dim + self.physical_dynamic_dim,
            ].to(self.dist.device)
            current_x_phys = torch.cat(
                (static_features, xn_t1_next, forcing_t1),
                dim=1,
            ).detach()
            current_x_src = next_step["x_src"].to(self.dist.device)

        total_loss = total_loss / k_steps

        if self.amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        return total_loss.detach()


def main(cfg: DictConfig, resume: bool = False):
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("train")
    r0 = RankZeroLoggingWrapper(logger, dist)

    ckpt_path = Path(to_absolute_path(cfg.ckpt_path))
    if resume and not ckpt_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")
    if resume:
        resume_epoch = configure_resume(cfg, ckpt_path)
        r0.info(f"Latest complete checkpoint: epoch {resume_epoch}")
    if dist.rank == 0:
        if not resume:
            ckpt_path.mkdir(parents=True)
        r0.info(f"Checkpoint directory: {ckpt_path}")
    if dist.world_size > 1:
        torch.distributed.barrier()

    base_seed = int(getattr(cfg, "seed", 0))
    seed_everything(base_seed + int(dist.rank))
    r0.info(f"Seed: {base_seed} (rank-adjusted: {base_seed + int(dist.rank)})")

    trainer = SourceNodeTrainer(cfg, r0)

    start = time.time()
    r0.info(f"len(dataloader) = {len(trainer.dataloader)}")
    mode = "one-step" if trainer.sequence_length == 1 else "pushforward"
    r0.info(f"Training ({mode}, source nodes) started...")

    for epoch in range(trainer.epoch_init, cfg.epochs):
        epoch_lr = trainer.optimizer.param_groups[0]["lr"]
        epoch_loss = 0.0
        for sequence in trainer.dataloader:
            loss = trainer.train_batch(sequence, epoch)
            epoch_loss += loss.item()
        epoch_loss /= len(trainer.dataloader)

        trainer.scheduler.step()

        r0.info(
            f"epoch: {epoch}, p_tf={trainer.p_tf(epoch):.3f}, "
            f"lr: {epoch_lr:.3e}, "
            f"loss: {epoch_loss:10.3e}, time/epoch: {(time.time() - start):.2f}s"
        )
        start = time.time()

        if dist.world_size > 1:
            torch.distributed.barrier()
        if (dist.rank == 0) and (epoch % 10 == 0):
            save_checkpoint(
                str(ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            logger.info(f"Saved model on rank {dist.rank}")

    r0.info("Training completed!")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != "resume"):
        print("Usage: python train_push_source_nodes.py <config_name> [resume]")
        sys.exit(1)

    config_name = sys.argv[1]
    resume = len(sys.argv) == 3
    if config_name.endswith((".yaml", ".yml")):
        config_name = os.path.splitext(config_name)[0]
    conf_dir = Path(__file__).resolve().parent / "conf"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = hydra.compose(config_name=config_name)
        main(cfg, resume=resume)
