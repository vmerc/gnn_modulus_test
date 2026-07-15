import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import torch

from dgl.dataloading import GraphDataLoader
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ""))
if project_path not in sys.path:
    sys.path.append(project_path)

from python.create_dgl_dataset import TelemacDatasetWithSourceNodes, collate_source_sequences
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


class SourceNodeTrainer:
    def __init__(self, cfg: DictConfig, r0: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp = bool(getattr(cfg, "amp", False)) and self.dist.device.type == "cuda"

        self.sequence_length = int(cfg.sequence_length)
        if self.sequence_length < 2:
            raise ValueError("sequence_length doit etre >= 2 pour le pushforward.")
        self.use_q_feature = bool(getattr(cfg, "use_q_feature", False))

        hydro_files = getattr(cfg, "hydro_dir", None)
        cli_file = getattr(cfg, "cli_file", None)
        inlet_node_lists = getattr(cfg, "inlet_node_lists", None)
        if hydro_files is None:
            raise ValueError("hydro_dir is required for source-node training.")
        if cli_file is None and inlet_node_lists is None:
            raise ValueError("Provide cli_file or inlet_node_lists in config.")

        self.dataset = TelemacDatasetWithSourceNodes(
            name="telemac_train_source_nodes",
            data_dir=to_absolute_path(cfg.data_dir),
            dynamic_data_files=[to_absolute_path(p) for p in cfg.dynamic_dir],
            hydro_data_files=[to_absolute_path(p) for p in hydro_files],
            cli_file=to_absolute_path(cli_file) if cli_file else None,
            inlet_node_lists=inlet_node_lists,
            use_q_feature=self.use_q_feature,
            split="train",
            ckpt_path=to_absolute_path(cfg.ckpt_path),
            normalize=True,
            sequence_length=self.sequence_length,
            overlap=self.sequence_length,
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

        self.model = MeshGraphNetWithSourceNodes(
            input_dim_nodes_phys=expected_phys_features,
            input_dim_nodes_src=expected_source_features,
            input_dim_edges=cfg.num_edge_features,
            output_dim=cfg.num_output_features,
            processor_size=cfg.mp_layers,
            hidden_dim_processor=64,
            hidden_dim_node_encoder=64,
            hidden_dim_edge_encoder=64,
            hidden_dim_node_decoder=64,
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

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch,
        )
        self.scaler = GradScaler(enabled=self.amp)

        if self.dist.world_size > 1:
            torch.distributed.barrier()

        self.epoch_init = self._load_init_weights(cfg, r0)

        self.pf_start_epoch = self.epoch_init
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
            return 0

        init_ckpt_path = to_absolute_path(getattr(cfg, "init_ckpt_path", cfg.ckpt_path))
        init_epoch = getattr(cfg, "init_epoch", None)
        if not Path(init_ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint introuvable: {init_ckpt_path}")

        try:
            if init_epoch is None:
                loaded_epoch = load_checkpoint(
                    init_ckpt_path,
                    models=self.model,
                    device=self.dist.device,
                )
            else:
                loaded_epoch = load_checkpoint(
                    init_ckpt_path,
                    models=self.model,
                    device=self.dist.device,
                    epoch=int(init_epoch),
                )
        except Exception as exc:
            raise RuntimeError(
                "Could not load this checkpoint into MeshGraphNetWithSourceNodes. "
                "If the checkpoint comes from the old MeshGraphNet architecture, "
                "start with train_from_scratch=true or provide a checkpoint produced "
                "by train_push_source_nodes.py."
            ) from exc

        start_epoch = int(init_epoch if init_epoch is not None else loaded_epoch or 0)
        r0.info(f"Loaded source-node weights from {init_ckpt_path} at epoch {start_epoch}")
        return start_epoch

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

    def train_pushforward(self, sequence, epoch):
        k_steps = len(sequence) - 1
        if k_steps < 1:
            raise ValueError("La sequence doit contenir au moins 2 pas.")

        current_x_phys = sequence[0]["x_phys"].to(self.dist.device)
        current_x_src = sequence[0]["x_src"].to(self.dist.device)
        total_loss = torch.zeros((), device=self.dist.device)

        self.optimizer.zero_grad()

        for t in range(k_steps):
            step = sequence[t]
            next_step = sequence[t + 1]
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

            onehot = current_x_phys[:, :4]
            xn_t = current_x_phys[:, self.static_dim:self.static_dim + 3]
            x_t = self._denorm(xn_t, self.mx.to(xn_t.device), self.sx.to(xn_t.device))

            y_pred = self._denorm(y_pred_n, self.my.to(xn_t.device), self.sy.to(xn_t.device))
            x_t1_pred = x_t + y_pred

            xn_t1_gt = next_step["x_phys"][:, self.static_dim:self.static_dim + 3].to(self.dist.device)
            x_t1_gt = self._denorm(xn_t1_gt, self.mx.to(xn_t.device), self.sx.to(xn_t.device))

            x_t1_pred = self._apply_h_boundary_inplace(x_t1_pred, x_t1_gt, onehot)
            total_loss = total_loss + loss_t

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

        total_loss = total_loss / max(1, k_steps)

        if self.amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        return total_loss.detach()


def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("train")
    r0 = RankZeroLoggingWrapper(logger, dist)
    r0.file_logging()

    base_seed = int(getattr(cfg, "seed", 0))
    seed_everything(base_seed + int(dist.rank))
    r0.info(f"Seed: {base_seed} (rank-adjusted: {base_seed + int(dist.rank)})")

    trainer = SourceNodeTrainer(cfg, r0)

    start = time.time()
    r0.info(f"len(dataloader) = {len(trainer.dataloader)}")
    r0.info("Training (pushforward source nodes) started...")

    for epoch in range(trainer.epoch_init, cfg.epochs):
        epoch_loss = 0.0
        for sequence in trainer.dataloader:
            loss = trainer.train_pushforward(sequence, epoch)
            epoch_loss += loss.item()
        epoch_loss /= len(trainer.dataloader)

        trainer.scheduler.step()

        r0.info(
            f"epoch: {epoch}, p_tf={trainer.p_tf(epoch):.3f}, "
            f"lr: {trainer.optimizer.param_groups[0]['lr']:.3e}, "
            f"loss: {epoch_loss:10.3e}, time/epoch: {(time.time() - start):.2f}s"
        )
        start = time.time()

        if dist.world_size > 1:
            torch.distributed.barrier()
        if (dist.rank == 0) and (epoch % 10 == 0):
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            logger.info(f"Saved model on rank {dist.rank}")

    r0.info("Training completed!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_push_source_nodes.py <config_name>")
        sys.exit(1)

    config_name = sys.argv.pop(1)
    script_dir = Path(__file__).resolve().parent
    conf_path = script_dir / "conf" / config_name
    if conf_path.suffix not in {".yaml", ".yml"}:
        conf_path = conf_path.with_suffix(".yaml")
    if not conf_path.exists():
        raise FileNotFoundError(f"Config file not found: {conf_path}")

    cfg = OmegaConf.load(conf_path)
    main(cfg)
