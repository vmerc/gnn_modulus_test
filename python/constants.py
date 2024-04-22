import os
from pathlib import Path
from pydantic import BaseModel
from typing import Tuple, Optional


class Constants(BaseModel):
    """vortex shedding constants"""

    # data configs
    data_dir: str = "./telemac_bdd_process/data/dgl_datasets/toy_one_traj.bin"

    # training configs
    batch_size: int = 10
    epochs: int = 50
    num_training_samples: int = 1
    num_training_time_steps: int = 100
    lr: float = 0.0005
    lr_decay_rate: float = 0.9999991
    num_input_features: int = 9
    num_output_features: int = 3
    num_edge_features: int = 3
    ckpt_path: str = "checkpointsv2"
    ckpt_name: str = "model.pt"

    # performance configs
    amp: bool = False
    jit: bool = False

    # test & visualization configs
    num_test_samples: int = 1
    num_test_time_steps: int = 60
    viz_vars: Tuple[str, ...] = ("h", "u", "v")
    frame_skip: int = 10
    frame_interval: int = 1

    # wb configs
    wandb_mode: str = "disabled"
    watch_model: bool = False
