import numpy as np 
import os 
import sys
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))
sys.path.append(project_path)

from python.create_dgl_dataset import TelemacDataset,create_dgl_dataset_chunked
from python.python_code.data_manip.extraction.telemac_file import TelemacFile

import torch
from tqdm import trange
import copy
import pandas as pd
import random
import dgl

res_list = ['/projets/aniti-daml/vmercier/simu_valentin/TetQ2500_intermediaire_bdd.res']
mesh_list = ['/projets/aniti-daml/vmercier/simu_valentin/maillage_3.slf']
cli_list = ['/projets/aniti-daml/vmercier/simu_valentin/cli']
dt_list = [1]

data_folder = './data/TetQ2500inter_1min_chunk/'
dataset_name = 'TetQ2500inter_1min'

create_dgl_dataset_chunked(mesh_list,res_list,cli_list,dt_list,data_folder,dataset_name, chunk_size=500)