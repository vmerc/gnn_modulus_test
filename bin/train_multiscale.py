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

from python.create_dgl_dataset import TelemacDataset  # Import the new dataset
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
from python.MsMGN import MultiScaleMeshGraphNet

from python.create_dgl_dataset import add_mesh_info
from python.MultiScaleUtils import create_transition_graph  
from python.python_code.data_manip.extraction.telemac_file import TelemacFile


def load_coarse_graph(cfg, device):
    # Get the absolute path from the configuration
    coarse_graph_path = to_absolute_path(cfg.coarse_graph_path)
    if not os.path.exists(coarse_graph_path):
        raise FileNotFoundError(f"Coarse graph file not found: {coarse_graph_path}")
    # Load the graph(s)
    graphs, _ = dgl.load_graphs(coarse_graph_path)
    # Use the first graph (or adapt if you have multiple graphs)
    graph_coarse = graphs[0]
    return graph_coarse

def normalize_graph_edges(graph, stats):
    """
    Normalise les données des arêtes (edata['x']) d'un graphe DGL in-place en utilisant un dictionnaire de statistiques.

    Args:
        graph (dgl.DGLGraph): Le graphe contenant les données des arêtes.
        stats (dict): Un dictionnaire contenant les moyennes et écarts-types des features des arêtes,
                      avec les clés suivantes :
                      - 'xrel', 'yrel', 'norm' (moyennes)
                      - 'xrel_std', 'yrel_std', 'norm_std' (écarts-types)

    Returns:
        None (modifie le graphe in-place)
    """
    if 'x' not in graph.edata:
        raise ValueError("Le graphe ne contient pas edata['x'].")

    # Extraire les valeurs des moyennes et écarts-types
    mean_values = torch.tensor([
        stats['xrel'].item(),
        stats['yrel'].item(),
        stats['norm'].item()
    ], dtype=torch.float32)

    std_values = torch.tensor([
        stats['xrel_std'].item(),
        stats['yrel_std'].item(),
        stats['norm_std'].item()
    ], dtype=torch.float32)

    # Éviter la division par zéro
    epsilon = 1e-8
    std_values = std_values + epsilon

    # Normalisation in-place
    graph.edata['x'] = (graph.edata['x'] - mean_values) / std_values

def normalize_graph_edges_static_nodes(graph, edge_stats, nodes_stats):
    """
    Normalise les données des arêtes (edata['x']) et les données statique des noeuds (ndata['static']) d'un graphe DGL in-place en utilisant deux dictionnaire de statistiques.

    Args:
        graph (dgl.DGLGraph): Le graphe contenant les données des arêtes.
        edge_stats (dict): Un dictionnaire contenant les moyennes et écarts-types des features des arêtes,
                      avec les clés suivantes :
                      - 'xrel', 'yrel', 'norm' (moyennes)
                      - 'xrel_std', 'yrel_std', 'norm_std' (écarts-types)
                      
        node_stats (dict): Un dictionnaire contenant les moyennes et écarts-types des features des arêtes,
                      avec les clés suivantes :
                      - 'strickler', 'z' (moyennes)
                      - 'strickler_std', 'z_std' (écarts-types)

    Returns:
        None (modifie le graphe in-place)
    """
    if 'x' not in graph.edata:
        raise ValueError("Le graphe ne contient pas edata['x'].")
        
    if 'static' not in graph.ndata:
        raise ValueError("Le graphe ne contient pas ndata['static'].")

    # Extraire les valeurs des moyennes et écarts-types
    mean_values_edge = torch.tensor([
        edge_stats['xrel'].item(),
        edge_stats['yrel'].item(),
        edge_stats['norm'].item()
    ], dtype=torch.float32)
    
    
    mean_values_node = torch.tensor([
        nodes_stats['strickler'].item(),
        nodes_stats['z'].item(),
    ], dtype=torch.float32)

    std_values_edge = torch.tensor([
        edge_stats['xrel_std'].item(),
        edge_stats['yrel_std'].item(),
        edge_stats['norm_std'].item()
    ], dtype=torch.float32)

    std_values_node = torch.tensor([
        nodes_stats['strickler_std'].item(),
        nodes_stats['z_std'].item(),
    ], dtype=torch.float32)
    
    # Éviter la division par zéro
    epsilon = 1e-8
    std_values_edge = std_values_edge + epsilon
    
    std_values_node = std_values_node + epsilon

    
    # Normalisation in-place
    graph.edata['x'] = (graph.edata['x'] - mean_values_edge) / std_values_edge
    graph.ndata['static'][:,-2:] = (graph.ndata['static'][:,-2:] - mean_values_node) / std_values_node

def normalize_edge_features(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize edge features of shape (nb_edges, 3).
    
    For each feature column, subtract the mean and divide by the standard deviation.
    If the standard deviation is zero, it is replaced with epsilon (1e-8) to avoid division by zero.
    
    Args:
        x (torch.Tensor): Tensor with shape (nb_edges, 3).
    
    Returns:
        torch.Tensor: Normalized tensor of the same shape.
    """
    epsilon = 1e-8
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    print("stats graph")
    print(mean)
    print(std)
    std = torch.where(std == 0, torch.tensor(epsilon, device=x.device), std)
    return (x - mean) / std

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
            sequence_length=1,  # Set sequence length to 1
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
        
        #----- Chargement des graphes statiques --------#
        # Mesh grossier
        mesh_coarse_path = to_absolute_path(cfg.mesh_coarse_path)
        res_mesh_coarse = TelemacFile(mesh_coarse_path)
        X_coarse, Triangles_coarse = add_mesh_info(res_mesh_coarse)
        # Mesh fin
        mesh_fine_path = to_absolute_path(cfg.mesh_fine_path)
        res_mesh_fin = TelemacFile(mesh_fine_path)
        X_fin, Triangles_fin = add_mesh_info(res_mesh_fin)
        
        self.graph_coarse = load_coarse_graph(cfg, self.dist.device)
        # Création du graphe de downscaling (passage de fin vers coarse)
        self.graph_down = create_transition_graph(X_fin, X_coarse, Triangles_fin)
        # Création du graphe d'upscaling (passage de coarse vers fin)
        self.graph_up = create_transition_graph(X_coarse, X_fin, Triangles_coarse)
        
        self.graph_coarse.edata['x'] = normalize_edge_features(self.graph_coarse.edata['x'])
        self.graph_down.edata['x'] = normalize_edge_features(self.graph_down.edata['x'])
        self.graph_up.edata['x'] = normalize_edge_features(self.graph_up.edata['x'])
        
        #Normalisation edge et noeuds grossier 
        if cfg.use_coarse_values :
            #normalize_graph_edges_static_nodes(self.graph_coarse,dataset.edge_stats,dataset.node_stats)
            None
        else : 
            self.graph_coarse.ndata['static'] = self.graph_coarse.ndata['static'][:,:4]

        # Déplacer les graphes sur le device
        self.graph_coarse = self.graph_coarse.to(self.dist.device)
        self.graph_down = self.graph_down.to(self.dist.device)
        self.graph_up   = self.graph_up.to(self.dist.device)
        

        # Instantiate the model with parameters from cfg
        self.model = MultiScaleMeshGraphNet(
            input_dim_nodes_fine=cfg.num_input_features,
            input_dim_nodes_coarse=cfg.num_input_features_coarse,
            input_dim_edges=cfg.num_edge_features,
            output_dim=cfg.num_output_features,
            processor_size_fine=cfg.mp_layers_fine,
            processor_size_coarse=cfg.mp_layers_coarse,
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
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )

    def train(self, batch,epoch,test):
        # batch is a list of sequences, each containing one graph
        self.optimizer.zero_grad()
        loss = self.forward(batch)
        self.backward(loss)
        
        if epoch%50 ==0 : 
            if test ==0:
                print(f"epoch {epoch}")
                #TESTS :
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
        self.scheduler.step()
        
        #TESTS :
        if epoch%50==0 : 
            if test ==0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"learning rate = {current_lr}")

        return loss

    def forward(self, batch):
        graph_fine = batch[0].to(self.dist.device)
        
        # Récupération des features pour le graphe fin
        node_features_fine = graph_fine.ndata["x"]
        edge_features_fine = graph_fine.edata["x"]

        # Utilisation des graphes statiques pour coarse, down et up
        node_features_coarse = self.graph_coarse.ndata["static"]
        edge_features_coarse = self.graph_coarse.edata["x"]
        
        # Extract one-hot vectors
        one_hot_vectors = graph_fine.ndata['x'][:, :4]
        
        # Masks
        mask_exclude = (one_hot_vectors == torch.tensor([0, 0, 1, 0], device=one_hot_vectors.device)).all(dim=1)
        mask_include = ~mask_exclude
        mask_specific = (one_hot_vectors == torch.tensor([0, 1, 0, 0], device=one_hot_vectors.device)).all(dim=1)
        mask_specific = mask_specific & mask_include
        mask_other = mask_include & ~mask_specific

        with autocast(enabled=self.amp):
            pred = self.model(
                node_features_fine=node_features_fine,
                node_features_coarse=node_features_coarse,
                edge_features_fine=edge_features_fine,
                edge_features_coarse=edge_features_coarse,
                graph_fine=graph_fine,
                graph_coarse=self.graph_coarse,
                graph_down=self.graph_down,
                graph_up=self.graph_up,
            )
            
            target = graph_fine.ndata['y']
            
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
        test = 0
        for batch in trainer.dataloader:
            loss = trainer.train(batch,epoch,test)
            total_loss += loss.item()
            test += 1
                
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
