import os
import sys
import time
import random
from pathlib import Path

import dgl
import hydra
import numpy as np
import torch

from dgl.dataloading import GraphDataLoader
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ""))
if project_path not in sys.path:
    sys.path.append(project_path)

from python.create_dgl_dataset import TelemacDataset, TelemacDatasetWithQ
from python.CustomMeshGraphNet import MeshGraphNet

from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.launch.utils import load_checkpoint, save_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_sequence(batch):
    seq_len = len(batch[0])
    batched_graphs = []
    for t in range(seq_len):
        gt_list = [seq[t] for seq in batch]
        batched_graphs.append(dgl.batch(gt_list))
    return batched_graphs


class MGNTrainer:
    def __init__(self, cfg: DictConfig, r0: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp = bool(getattr(cfg, "amp", False)) and self.dist.device.type == "cuda"

        self.sequence_length = int(cfg.sequence_length)
        if self.sequence_length < 2:
            raise ValueError("sequence_length doit etre >= 2 pour le pushforward.")

        self.use_q_feature = bool(getattr(cfg, "use_q_feature", False))
        if self.use_q_feature:
            hydro_files = getattr(cfg, "hydro_dir", None)
            if hydro_files is None:
                raise ValueError("use_q_feature=True requires hydro_dir in config.")
            dataset = TelemacDatasetWithQ(
                name="telemac_train_q",
                data_dir=to_absolute_path(cfg.data_dir),
                dynamic_data_files=[to_absolute_path(p) for p in cfg.dynamic_dir],
                hydro_data_files=[to_absolute_path(p) for p in hydro_files],
                split="train",
                ckpt_path=to_absolute_path(cfg.ckpt_path),
                normalize=True,
                sequence_length=self.sequence_length,
                overlap=self.sequence_length,
                dt_seconds=float(getattr(cfg, "dt_seconds", 1800.0)),
            )
        else:
            dataset = TelemacDataset(
                name="telemac_train",
                data_dir=to_absolute_path(cfg.data_dir),
                dynamic_data_files=[to_absolute_path(p) for p in cfg.dynamic_dir],
                split="train",
                ckpt_path=to_absolute_path(cfg.ckpt_path),
                normalize=True,
                sequence_length=self.sequence_length,
                overlap=self.sequence_length,
            )

        expected_input_features = dataset.base_graph.ndata["static"].shape[1] + (4 if self.use_q_feature else 3)
        if int(cfg.num_input_features) != int(expected_input_features):
            raise ValueError(
                f"num_input_features={cfg.num_input_features} incompatible with dataset ({expected_input_features})."
            )

        self.dataloader = GraphDataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=self.dist.world_size > 1,
            num_workers=cfg.num_dataloader_workers,
            collate_fn=collate_sequence,
        )

        self.model = MeshGraphNet(
            cfg.num_input_features,
            cfg.num_edge_features,
            cfg.num_output_features,
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
                raise ValueError("MeshGraphNet n'est pas JIT-compatible.")
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

        r0.info(f"Optimizer: {self.optimizer.__class__.__name__}")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda e: cfg.lr_decay_rate**e
        )
        self.scaler = GradScaler(enabled=self.amp)

        if self.dist.world_size > 1:
            torch.distributed.barrier()

        self.epoch_init = self._load_finetune_weights(cfg, r0)

        self.pf_start_epoch = self.epoch_init
        self.pf_warmup_epochs = int(getattr(cfg, "pf_warmup_epochs", 0))
        self.p_tf = (
            lambda ep: 0.0
            if self.pf_warmup_epochs <= 0
            else max(0.0, 1.0 - max(0, ep - self.pf_start_epoch) / self.pf_warmup_epochs)
        )

        ns = dataset.node_stats
        self.mx = torch.tensor(
            [ns["h"].item(), ns["u"].item(), ns["v"].item()],
            dtype=torch.float32,
            device=self.dist.device,
        )
        self.sx = torch.tensor(
            [ns["h_std"].item(), ns["u_std"].item(), ns["v_std"].item()],
            dtype=torch.float32,
            device=self.dist.device,
        )
        self.my = torch.tensor(
            [ns["delta_h"].item(), ns["delta_u"].item(), ns["delta_v"].item()],
            dtype=torch.float32,
            device=self.dist.device,
        )
        self.sy = torch.tensor(
            [ns["delta_h_std"].item(), ns["delta_u_std"].item(), ns["delta_v_std"].item()],
            dtype=torch.float32,
            device=self.dist.device,
        )

        self.DYN_START = dataset.base_graph.ndata["static"].shape[1]
        self.DYN_LEN = 4 if self.use_q_feature else 3

    def _load_finetune_weights(self, cfg: DictConfig, r0: RankZeroLoggingWrapper) -> int:
        if bool(getattr(cfg, "train_from_scratch", False)):
            r0.info("train_from_scratch=True, skipping checkpoint loading.")
            return 0

        init_ckpt_path = to_absolute_path(getattr(cfg, "init_ckpt_path", cfg.ckpt_path))
        init_epoch = getattr(cfg, "init_epoch", None)

        if not Path(init_ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint introuvable pour le fine-tuning: {init_ckpt_path}")

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

        start_epoch = int(init_epoch if init_epoch is not None else loaded_epoch or 0)
        r0.info(f"Loaded fine-tuning weights from {init_ckpt_path} at epoch {start_epoch}")
        return start_epoch

    @staticmethod
    def _denorm(xn, mean, std):
        return xn * std + mean

    @staticmethod
    def _renorm(x, mean, std):
        return (x - mean) / (std + 1e-12)

    def _apply_bc_inplace(self, x_pred, x_gt, onehot):
        q_mask = (onehot == torch.tensor([0, 0, 1, 0], device=onehot.device)).all(dim=1)
        h_mask = (onehot == torch.tensor([0, 1, 0, 0], device=onehot.device)).all(dim=1)
        x_pred[q_mask] = x_gt[q_mask]
        x_pred[h_mask, 0] = x_gt[h_mask, 0]
        return x_pred

    def train_pushforward(self, graphs, epoch):
        k_steps = len(graphs) - 1
        if k_steps < 1:
            raise ValueError("La sequence doit contenir au moins 2 pas.")

        g = graphs[0].to(self.dist.device)
        onehot = g.ndata["x"][:, :4]

        total_loss = torch.zeros((), device=self.dist.device)
        self.optimizer.zero_grad()

        for t in range(k_steps):
            with autocast(enabled=self.amp):
                y_pred_n = self.model(g.ndata["x"], g.edata["x"], g)
                y_gt_n = graphs[t].ndata["y"].to(self.dist.device)
                loss_t = self.criterion(y_pred_n, y_gt_n)

            xn_t_full = g.ndata["x"][:, self.DYN_START:self.DYN_START + self.DYN_LEN]
            xn_t = xn_t_full[:, :3]
            x_t = self._denorm(xn_t, self.mx.to(xn_t.device), self.sx.to(xn_t.device))

            y_pred = self._denorm(y_pred_n, self.my.to(xn_t.device), self.sy.to(xn_t.device))
            x_t1_pred = x_t + y_pred

            xn_t1_gt_full = graphs[t + 1].ndata["x"][:, self.DYN_START:self.DYN_START + self.DYN_LEN].to(xn_t.device)
            xn_t1_gt = xn_t1_gt_full[:, :3]
            x_t1_gt = self._denorm(xn_t1_gt, self.mx.to(xn_t.device), self.sx.to(xn_t.device))

            x_t1_pred = self._apply_bc_inplace(x_t1_pred, x_t1_gt, onehot)
            total_loss = total_loss + loss_t

            use_tf = torch.rand(1, device=xn_t.device).item() < self.p_tf(epoch)
            x_t1_next = x_t1_gt if use_tf else x_t1_pred

            xn_t1_next = self._renorm(x_t1_next, self.mx.to(xn_t.device), self.sx.to(xn_t.device))
            if self.use_q_feature:
                q_t1_n = xn_t1_gt_full[:, 3:4]
                xn_t1_next_full = torch.cat([xn_t1_next, q_t1_n], dim=1)
            else:
                xn_t1_next_full = xn_t1_next

            x_next_full = torch.cat([g.ndata["x"][:, :self.DYN_START], xn_t1_next_full], dim=1).detach()

            g = dgl.graph(g.edges(), num_nodes=g.num_nodes(), device=g.device)
            g.ndata["x"] = x_next_full
            g.edata["x"] = graphs[0].edata["x"].to(g.device)

        total_loss = total_loss / max(1, k_steps)

        if self.amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        return total_loss.detach()


@hydra.main(version_base="1.3", config_path="conf", config_name=None)
def main(cfg: DictConfig):
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("train")
    r0 = RankZeroLoggingWrapper(logger, dist)
    r0.file_logging()

    base_seed = int(getattr(cfg, "seed", 0))
    seed_everything(base_seed + int(dist.rank))
    r0.info(f"Seed: {base_seed} (rank-adjusted: {base_seed + int(dist.rank)})")

    trainer = MGNTrainer(cfg, r0)

    start = time.time()
    r0.info(f"len(dataloader) = {len(trainer.dataloader)}")
    r0.info("Training (pushforward fine-tuning) started...")

    for epoch in range(trainer.epoch_init, cfg.epochs):
        epoch_loss = 0.0
        for graphs in trainer.dataloader:
            loss = trainer.train_pushforward(graphs, epoch)
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
        print("Usage: python train_push_fine.py <config_name>")
        sys.exit(1)

    config_name = sys.argv.pop(1)
    if config_name.endswith((".yaml", ".yml")):
        config_name = os.path.splitext(config_name)[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    conf_dir = os.path.join(script_dir, "conf")
    with hydra.initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = hydra.compose(config_name=config_name)
        main(cfg)
