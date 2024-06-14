import time
import hydra
from hydra.utils import to_absolute_path
import torch
import sys
import os


from dgl.dataloading import DataLoader,MultiLayerFullNeighborSampler

from omegaconf import DictConfig

from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
sys.path.append(project_path)

from python.create_dgl_dataset import TelemacDataset
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
    initialize_wandb,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
from python.CustomMeshGraphNetSample import MeshGraphNet

class MGNTrainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()

        self.amp = cfg.amp

        # instantiate dataset
        self.dataset = TelemacDataset(
            name="telemac_train",
            data_dir=to_absolute_path(cfg.data_dir),
            dynamic_data_file= to_absolute_path(cfg.dynamic_dir),
            split="train",
            num_samples=cfg.num_training_samples,
            num_steps=cfg.num_training_time_steps
        )


        # instantiate the model
        self.model = MeshGraphNet(
            cfg.num_input_features,
            cfg.num_edge_features,
            cfg.num_output_features,
            processor_size=5,
            hidden_dim_processor=128,
            hidden_dim_node_encoder=128,
            hidden_dim_edge_encoder=128,
            hidden_dim_node_decoder=128,
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
        )
        
        if cfg.jit:
            if not self.model.meta.jit:
                raise ValueError("MeshGraphNet is not yet JIT-compatible.")
            self.model = torch.jit.script(self.model).to(self.dist.device)
        else:
            self.model = self.model.to(self.dist.device)


        # distributed data parallel for multi-node training
        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.dist.local_rank],
                    output_device=self.dist.device,
                    broadcast_buffers=self.dist.broadcast_buffers,
                    find_unused_parameters=self.dist.find_unused_parameters,
                )

        # enable train mode
        self.model.train()

        # instantiate loss, optimizer, and scheduler
        self.criterion = torch.nn.MSELoss()

        self.optimizer = None
        try:
            if cfg.use_apex:
                from apex.optimizers import FusedAdam

                self.optimizer = FusedAdam(self.model.parameters(), lr=cfg.lr)
        except ImportError:
            rank_zero_logger.warning(
                    "NVIDIA Apex (https://github.com/nvidia/apex) is not installed, "
                    "FusedAdam optimizer will not be used."
            )
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        rank_zero_logger.info(f"Using {self.optimizer.__class__.__name__} optimizer")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch
            )
        self.scaler = GradScaler()

        # load checkpoint
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

    def train(self, input_nodes, seeds, blocks,common_edges):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.forward(input_nodes, seeds, blocks,common_edges)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def forward(self, input_nodes, seeds, blocks,common_edges):
        with autocast(enabled=self.amp):
            node_features=  blocks[0].srcdata['x'].to(self.dist.device)
            edge_features = blocks[0].edata['x'].to(self.dist.device)
            common_edges = [c.to(self.dist.device) for c in common_edges]
            blocks = [b.to(self.dist.device) for b in blocks]
            pred = self.model(node_features,edge_features,blocks,common_edges)
            loss = self.criterion(pred, blocks[-1].dstdata['y'].to(self.dist.device))
            return loss
        
    def backward(self, loss):
        # backward pass
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

from typing import Callable, List, Tuple, Union
from dgl import DGLGraph
from dgl.heterograph import DGLBlock

class CustomSampler(MultiLayerFullNeighborSampler):
    def __init__(self, fanouts: List[int]):
        super().__init__(fanouts)

    def sample(self, g: DGLGraph, seed_nodes: torch.Tensor, exclude_eids: torch.Tensor = None) -> Tuple[List[DGLBlock], List[torch.Tensor]]:
        input_nodes, output_nodes, blocks = super().sample_blocks(g, seed_nodes, exclude_eids=exclude_eids)
        common_edges = self.find_common_edges(blocks)
        return input_nodes, output_nodes,blocks,common_edges

    def find_common_edges(self, blocks: List[DGLBlock]) -> List[torch.Tensor]:
        common_edges = []
        for i in range(len(blocks) - 1):
            current_block = blocks[i]
            next_block = blocks[i + 1]
            
            src_current, dst_current = current_block.edges()
            src_next, dst_next = next_block.edges()
            
            edges_current = {(src_current[j].item(), dst_current[j].item()): j for j in range(len(src_current))}
            common_indices = [edges_current.get((src_next[k].item(), dst_next[k].item()), -1) for k in range(len(src_next))]
            
            valid_indices = [idx for idx in common_indices if idx != -1]
            common_edges.append(torch.tensor(valid_indices, dtype=torch.long))
        
        return common_edges
    

@hydra.main(version_base="1.3", config_path="conf", config_name="config_Tet_sample")
def main(cfg: DictConfig) -> None:
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    rank_zero_logger.file_logging()

    trainer = MGNTrainer(cfg, rank_zero_logger)
    start = time.time()
    rank_zero_logger.info("Training started...")
    for epoch in range(trainer.epoch_init, cfg.epochs)
        loss = 0 
        for graph in trainer.dataset:
            acc = 0
            graph_time = time.time()
            sampler = CustomSampler(5)
            train_nids = torch.arange(graph.num_nodes())
            dataloader = DataLoader(
                        graph, 
                        train_nids,
                        sampler,
                        batch_size=cfg.batch_size,
                        shuffle=True,
                        drop_last=False,
                        use_ddp=trainer.dist.world_size > 1,
                        num_workers=cfg.num_dataloader_workers,
                        )
            for input_nodes,seeds,blocks,common_edges in dataloader:
                loss = trainer.train(input_nodes,seeds,blocks,common_edges)

        if epoch%1==0 : 
            rank_zero_logger.info(
                f"epoch: {epoch}, loss: {loss:10.3e}, time per epoch: {(time.time()-start):10.3e}"
            )
        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if (dist.rank == 0) and  (epoch%1==0):
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            logger.info(f"Saved model on rank {dist.rank}")
        start = time.time()
    rank_zero_logger.info("Training completed!")

if __name__ == "__main__":
    main()
