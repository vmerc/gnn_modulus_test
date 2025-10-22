import time
import hydra
from hydra.utils import to_absolute_path
import torch
import sys
import os
import torch.nn as nn

import argparse

from dgl.dataloading import GraphDataLoader

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
from python.CustomMeshGraphNet import MeshGraphNet

class TestRollout:
    def __init__(self, cfg: DictConfig, logger: PythonLogger):
        self.num_test_time_steps = cfg.num_test_time_steps
        self.frame_skip = cfg.frame_skip

        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {self.device} device")
        
        # instantiate dataset
        self.dataset = TelemacDataset(
            name="telemac_test",
            data_dir=to_absolute_path(cfg.data_dir),
            dynamic_data_file= to_absolute_path(cfg.dynamic_dir),
            split="test",
            num_samples=cfg.num_training_samples,
            num_steps=cfg.num_training_time_steps,
            ckpt_path=cfg.ckpt_path,
            stride=cfg.timestep,
            starting_ts = cfg.starting_ts
        )

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=1,  
            shuffle=False,
            drop_last=False,
        )

        # instantiate the model
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
            self.model = torch.jit.script(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)

        # enable train mode
        self.model.eval()

        # load checkpoint
        load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            device=self.device,
        )

        self.var_identifier = {"h": 0, "u": 1, "v": 2}

    def predict(self):
        self.pred, self.exact, self.graphs = [], [], []
        node_stats =  self.dataset.node_stats
        for i, graph in enumerate(self.dataloader):
            graph = graph.to(self.device)
            
            pred = 
            #get stats 
            h_u_v_i_0_mean = torch.tensor([node_stats['h'].item(),node_stats['u'].item(),node_stats['v'].item()])
            h_u_v_i_0_std = torch.tensor([node_stats['h_std'].item(),node_stats['u_std'].item(),node_stats['v_std'].item()])

            delta_h_u_v_i_diff_mean = torch.tensor([node_stats['delta_h'].item(),node_stats['delta_u'].item(),node_stats['delta_v'].item()])
            delta_h_u_v_i_diff_std = torch.tensor([node_stats['delta_h_std'].item(),node_stats['delta_u_std'].item(),node_stats['delta_v_std'].item()])
            
            # denormalize data
            h_u_v_i_0 = _denormalize_data(graph.ndata['x'][:,6:9],h_u_v_i_0_mean,h_u_v_i_0_std)
            
            h_u_v_i_1 = _denormalize_data(graph.ndata['y'][:,0:3],delta_h_u_v_i_diff_mean,delta_h_u_v_i_diff_std) + h_u_v_i_0
            
            h_u_v_i_1_pred = _denormalize_data(pred,delta_h_u_v_i_diff_mean,delta_h_u_v_i_diff_std) + h_u_v_i_0
            # do not update the "wall_boundary" & "outflow" &"inflow" nodes
            

    def _denormalize_data(tensor,mean,std):
        assert(tensor.shape[1]==mean.shape[0])
        assert(tensor.shape[1]==std.shape[0])
        return tensor*std + mean 
    def get_raw_data(self, idx):
        self.pred_i = [var[:, idx] for var in self.pred]
        self.exact_i = [var[:, idx] for var in self.exact]

        return self.graphs, self.faces, self.pred_i, self.exact_i
