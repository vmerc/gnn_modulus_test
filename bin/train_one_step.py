import os
import sys
import time
import random
import torch
import dgl
import hydra
import numpy as np

from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

# repo root (ajuste si besoin)
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
if project_path not in sys.path:
    sys.path.append(project_path)

from python.create_dgl_dataset import TelemacDataset
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


# --- collate: sequence_length=1 -> on renvoie un seul DGLGraph batched ---
def collate_graph(batch):
    # batch: liste d'items du dataset; chaque item est [graph] (len=1)
    graphs = [item[0] if isinstance(item, list) else item for item in batch]
    return dgl.batch(graphs)


class MGNTrainer:
    def __init__(self, cfg: DictConfig, r0: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp = bool(cfg.amp)

        # === Dataset ===
        dataset = TelemacDataset(
            name="telemac_train",
            data_dir=to_absolute_path(cfg.data_dir),
            dynamic_data_files=[to_absolute_path(p) for p in cfg.dynamic_dir],
            split="train",
            ckpt_path=to_absolute_path(cfg.ckpt_path),
            normalize=True,
            sequence_length=1,
            overlap=0,
        )

        # === DataLoader ===
        self.dataloader = GraphDataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=self.dist.world_size > 1,
            num_workers=cfg.num_dataloader_workers,
            collate_fn=collate_graph,
        )

        # === Model ===
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

        # DDP
        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        self.model.train()

        # === Optim & Loss & Scheduler ===
        self.criterion = torch.nn.MSELoss()
        self.optimizer = None
        try:
            if bool(cfg.use_apex):
                from apex.optimizers import FusedAdam
                self.optimizer = FusedAdam(self.model.parameters(), lr=cfg.lr)
        except ImportError:
            r0.warning("Apex non installé, utilisation d'Adam standard.")
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        r0.info(f"Optimizer: {self.optimizer.__class__.__name__}")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda e: cfg.lr_decay_rate**e
        )
        self.scaler = GradScaler(enabled=self.amp)

        # === Checkpoint load ===
        if self.dist.world_size > 1:
            torch.distributed.barrier()

        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )

    def train_one_step(self, g):
        g = g.to(self.dist.device)
        self.optimizer.zero_grad()

        # AMP (CUDA si dispo)
        with autocast(enabled=self.amp):
            pred = self.model(g.ndata["x"], g.edata["x"], g)
            loss = self.criterion(pred, g.ndata["y"])

        if self.amp:
            self.scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # optionnel
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # optionnel
            self.optimizer.step()

        return loss


@hydra.main(version_base="1.3", config_path="conf", config_name=None)
def main(cfg: DictConfig):
    # Init distrib
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
    r0.info("Training started...")

    for epoch in range(trainer.epoch_init, cfg.epochs):
        epoch_loss = 0.0
        for g in trainer.dataloader:
            loss = trainer.train_one_step(g)
            epoch_loss += loss.item()
        epoch_loss /= len(trainer.dataloader)

        # scheduler -> **par epoch**
        trainer.scheduler.step()

        r0.info(f"epoch: {epoch}, lr: {trainer.optimizer.param_groups[0]['lr']:.3e}, "
                f"loss: {epoch_loss:10.3e}, time/epoch: {(time.time()-start):.2f}s")
        start = time.time()

        # checkpoint
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
    # Ex.: python train_simplified.py config_name=config.yaml
    if len(sys.argv) < 2:
        print("Usage: python train_simplified.py <config_name>")
        sys.exit(1)

    # Laisse Hydra gérer `config_path="conf"` si tu utilises un répertoire de confs
    config_name = sys.argv.pop(1)
    with hydra.initialize_config_dir(config_dir=os.path.abspath("./bin/conf")):
        cfg = hydra.compose(config_name=config_name)
        main(cfg)
