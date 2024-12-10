import time
import hydra
from hydra.utils import to_absolute_path
import torch
import sys
import os
import torch.nn as nn
import dgl

from dgl.dataloading import GraphDataLoader
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

# Adjust the import paths as necessary
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
sys.path.append(project_path)

from python.create_dgl_dataset import TelemacDataset,TelemacDatasetOld  # Import the new dataset
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
from python.CustomMeshGraphNet import MeshGraphNet  # Import your model


def collate_fn(batch):
    # batch is a list of sequences
    # Each sequence is a list of graphs (of length sequence_length)
    # We want to batch the graphs at each time step across sequences

    sequence_length = len(batch[0])  # Assuming all sequences have the same length

    batched_graphs = []
    for t in range(sequence_length):
        graphs_at_t = [sequence[t] for sequence in batch]
        batched_graph = dgl.batch(graphs_at_t)
        batched_graphs.append(batched_graph)

    return batched_graphs

class MGNTrainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()

        self.amp = cfg.amp

        # Instantiate the dataset with sequence_length=1
        dataset = TelemacDataset(
            name="telemac_train",
            data_dir=to_absolute_path(cfg.data_dir),
            dynamic_data_files=[to_absolute_path(path) for path in cfg.dynamic_dir],  # Handle list of files
            split="train",
            ckpt_path=cfg.ckpt_path,
            normalize=True,
            sequence_length=2,  # Set sequence length to 1
        )
        
        
        # Instantiate the data loader
        self.dataloader = GraphDataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=self.dist.world_size > 1,
            num_workers=cfg.num_dataloader_workers,
            collate_fn=collate_fn,
        )

        # Instantiate the model with parameters from cfg
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
        )
        if cfg.jit:
            if not self.model.meta.jit:
                raise ValueError("MeshGraphNet is not yet JIT-compatible.")
            self.model = torch.jit.script(self.model).to(self.dist.device)
        else:
            self.model = self.model.to(self.dist.device)
        

        # Distributed data parallel for multi-node training
        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        # Enable train mode
        self.model.train()

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

        # Load checkpoint
        if self.dist.world_size > 1:
            torch.distributed.barrier()
            
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.loading_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )

    def train(self, batch):
        # batch is a list of sequences, each containing one graph
        self.optimizer.zero_grad()
        loss = self.forward(batch)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def forward(self, batch):
        graph_0 = batch[0].to(self.dist.device)
        graph_1 = batch[1].to(self.dist.device)
        
        # Step 1: Predict output on graph_0
        node_features_0 = graph_0.ndata["x"]
        edge_features_0 = graph_0.edata["x"]
        target_0 = graph_0.ndata["y"]

        with autocast(enabled=self.amp):
            pred_0 = self.model(node_features_0, edge_features_0, graph_0)

            # Compute loss_0
            loss_0 = self.compute_loss(pred_0, target_0, graph_0)

        # Step 2: Modify graph_0 to create predicted_modified_graph
        predicted_modified_graph = self.modify_graph(graph_0, pred_0, graph_1)
        
        # Step 3: Predict output on predicted_modified_graph
        node_features_mod = predicted_modified_graph.ndata["x"]
        edge_features_mod = predicted_modified_graph.edata["x"]
        target_1 = graph_1.ndata["y"]

        with autocast(enabled=self.amp):
            pred_1 = self.model(node_features_mod, edge_features_mod, predicted_modified_graph)

            # Compute loss_1
            loss_1 = self.compute_loss(pred_1, target_1, predicted_modified_graph)

        # Step 4: Combine losses
        total_loss = loss_0 + loss_1

        return total_loss
    
    
    def modify_graph(self, graph_0, pred_0, graph_1):
        # Clone graph_0 to avoid modifying the original
        predicted_modified_graph = graph_0.clone()

        # Detach pred_0 to prevent gradients from flowing back
        pred_0_detached = pred_0.detach()

        # Update node features in predicted_modified_graph
        node_features_mod = predicted_modified_graph.ndata['x'].clone()
        one_hot_vectors = node_features_mod[:, :4]

        # Nodes with one-hot [0, 0, 1, 0]: replace x values with those from graph_1
        mask_replace = (one_hot_vectors == torch.tensor([0, 0, 1, 0], device=self.dist.device)).all(dim=1)
        node_features_mod[mask_replace] = graph_1.ndata['x'][mask_replace]

        # Nodes with one-hot [0, 1, 0, 0]: replace 7th feature with that from graph_1
        mask_7th = (one_hot_vectors == torch.tensor([0, 1, 0, 0], device=self.dist.device)).all(dim=1)
        node_features_mod[mask_7th, 6] = graph_1.ndata['x'][mask_7th, 6]

        # For other nodes, update node features with pred_0_detached
        mask_other = ~(mask_replace | mask_7th)
        node_features_mod[mask_other, 6:] =  pred_0_detached[mask_other]

        # Update the node features in predicted_modified_graph
        predicted_modified_graph.ndata['x'] = node_features_mod

        return predicted_modified_graph

    def compute_loss(self, pred, target, graph):
        # Masks and loss computation as before
        one_hot_vectors = graph.ndata['x'][:, :4]
        mask_exclude = (one_hot_vectors == torch.tensor([0, 0, 1, 0], device=self.dist.device)).all(dim=1)
        mask_include = ~mask_exclude
        mask_specific = (one_hot_vectors == torch.tensor([0, 1, 0, 0], device=self.dist.device)).all(dim=1)
        mask_specific = mask_specific & mask_include
        mask_other = mask_include & ~mask_specific

        # Exclude nodes with [0, 0, 1, 0]
        pred_included = pred[mask_include]
        target_included = target[mask_include]

        # Nodes with one-hot [0, 1, 0, 0]: predict only y[:,1:3]
        pred_specific = pred[mask_specific][:, 1:3]
        target_specific = target[mask_specific][:, 1:3]

        # Other nodes: predict all features
        pred_other = pred[mask_other]
        target_other = target[mask_other]

        # Compute losses
        loss_specific = self.criterion(pred_specific, target_specific)
        loss_other = self.criterion(pred_other, target_other)

        # Combine losses
        total_nodes = mask_include.sum()
        weight_specific = mask_specific.sum().float() / total_nodes
        weight_other = mask_other.sum().float() / total_nodes

        loss = weight_specific * loss_specific + weight_other * loss_other
        return loss

    def backward(self, loss):
        # Backward pass
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

@hydra.main(version_base="1.3", config_path="conf", config_name=None)
def main(cfg: DictConfig) -> None:
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    rank_zero_logger.file_logging()

    trainer = MGNTrainer(cfg, rank_zero_logger)
    start = time.time()
    print("len dataloader : {}".format(len(trainer.dataloader)))
    rank_zero_logger.info("Training started...")
    for epoch in range(trainer.epoch_init, cfg.epochs):
        total_loss = 0 
        for batch in trainer.dataloader:
            loss = trainer.train(batch)
            total_loss += loss.item()
                
        total_loss = total_loss / len(trainer.dataloader)
                
        if epoch % 1 == 0:
            rank_zero_logger.info(
                f"epoch: {epoch}, loss: {total_loss:10.3e}, time per epoch: {(time.time()-start):10.3e}"
            )
        # Save checkpoint
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
        start = time.time()
    rank_zero_logger.info("Training completed!")

if __name__ == "__main__":

    # Ensure a configuration file name is provided
    if len(sys.argv) < 2:
        print("Usage: python parse_string_hydra.py <config_name>")
        sys.exit(1)

    # Get the config name from the command line arguments
    config_name = sys.argv.pop(1)
    config_dir = os.path.abspath("./bin/conf")
    # Initialize Hydra and compose the configuration
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name=config_name)
        main(cfg)
