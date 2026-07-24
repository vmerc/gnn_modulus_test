import numpy as np 
import enum
import os 
import sys
import torch
import dgl 
import pickle
from dgl.data import DGLDataset
os.chdir('../')
from python.python_code.data_manip.extraction.telemac_file import TelemacFile
from python.ghost_nodes import (
    add_ghost_source_nodes,
    extract_inlet_node_lists_from_conlim,
    normalize_inlet_node_lists,
)
from scipy.spatial import KDTree
from pathlib import Path, PurePath

class NodeType(enum.IntEnum):
    """
    Define the code for the one-hot vector representing the node types.
    """
    NORMAL = 0
    PRESCRIBED_H = 1
    PRESCRIBED_Q = 2
    WALL_BOUNDARY = 3
    SIZE = 4


def extract_node_type(tri,bnd_info):
    """Get the node type 
    @param tri (matplotlib.tri.Triangulation) triangular mesh
    @param bnd_info (tuple) : boundary conditions information (default None)

    Returns:
        np.array : array de taille [nb_pts,NodeType.size] : one hot vector encoding du type de chaque noeuds
    """
    x, y = tri.x, tri.y
    nbor, lihbor, liubor, livbor, _ = bnd_info
    nbnd_poin = len(nbor)
    nb_points = x.shape[0]
    
    bnd_types_dict = {'Closed boundaries/walls (2,2,2)': [2, 2, 2],
                      'Prescribed H (5,4,4)':            [5, 4, 4],
                      'Prescribed Q (4,5,5)':            [4, 5, 5],
                      'Prescribed Q and H (5,5,5)':      [5, 5, 5],
                      'Prescribed UV (4,6,6)':           [4, 6, 6],
                      'Prescribed UV and H (5,6,6)':     [5, 6, 6],
                      'Incident waves (1,1,1)':          [1, 1, 1],
                      'Custom (0,0,0)':                  [0, 0, 0],
                      'Free boundaries (4,4,4)':         [4, 4, 4]}
    
    bnd_one_hot_dict = {'[2, 2, 2]':np.array([0,0,0,1]),
                     '[5, 4, 4]':np.array([0,1,0,0]),
                      '[4, 5, 5]':np.array([0,0,1,0]),
                      }
    # on crée un one hot vectot de la taille de tous les pts ou tous les points sont normaux
    output = np.zeros((nb_points,NodeType.SIZE))
    output[:,0] = 1
    
    for i in range(nbnd_poin):
        bc_type = [lihbor[i], liubor[i], livbor[i]]
        item = bnd_one_hot_dict[str(bc_type)]
        output[nbor[i],:] = item
    return output 


def extract_h_u_v(res,timestep):
    """

    Args:
        res (res telemac): résultats télémac 
        
    Outputs : 
        output : np.array : nb_points,3 
    """
    hauteur = res.get_data_value("HAUTEUR D'EAU", timestep)
    u = res.get_data_value("VITESSE U", timestep)
    v = res.get_data_value("VITESSE V", timestep)
    
    #On corrige les erreurs Telemac 
    u[hauteur==0.0]=0.0
    v[hauteur==0.0]=0.0
    
    result = np.stack([hauteur,u,v],axis=1)
    return result

def add_mesh_info(res_mesh):
    """
    ajoute les info sur le mesh au 0 de la trajectoire

    Args:
        res_mesh (_type_): _description_
    """
    
    x,y = res_mesh.tri.x,res_mesh.tri.y
    pos = np.stack([x[:],y[:]],axis=1)
    return pos,res_mesh.tri.triangles


def extract_fond(res,timestep):
    """

    Args:
        res (res telemac): résultats télémac 
        
    Outputs : 
        output : np.array : nb_points,1 
    """
    fond = res.get_data_value("FOND", timestep)
    return np.expand_dims(fond,axis=1)

def extract_coeff(res,timestep):
    """

    Args:
        res (res telemac): résultats télémac 
        
    Outputs : 
        output : np.array : nb_points,1
    """
    coeff = res.get_data_value("FROTTEMENT", timestep)
    return np.expand_dims(coeff,axis=1)


#def get_node_features(res,res_mesh,timestep):
#    tri = res.tri
#    bnd_info = res.get_bnd_info()
#    node_type = extract_node_type(tri,bnd_info)
#    
#    huv = extract_h_u_v(res,timestep)
#    
#    cf = extract_coeff(res_mesh,0)
#    
#    z = extract_fond(res_mesh,0)
#    
#    return np.concatenate([node_type,huv,cf,z],axis=1).astype('float32')

def get_dynamic_node_features(res, timestep):
    huv = extract_h_u_v(res, timestep)
    return huv.astype('float32')

def get_static_node_features(res, res_mesh):
    tri = res.tri
    bnd_info = res.get_bnd_info()
    node_type = extract_node_type(tri, bnd_info)
    cf = extract_coeff(res_mesh, 0)
    z = extract_fond(res_mesh, 0)
    return np.concatenate([node_type, cf, z], axis=1).astype('float32')


def get_edge_index(tri):
    """Return unique directed edges from a triangular mesh."""
    return get_edge_index_from_triangles(tri.triangles)


def get_edge_index_from_triangles(triangles):
    """Return both directions of every non-degenerate triangle edge."""
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (n, 3)")

    edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ]
    )
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.concatenate([edges, edges[:, ::-1]])
    edges = np.unique(edges, axis=0)
    return edges.T

def get_edges_features(tri,coo,res):
    """_summary_

    Returns:
        _type_: _description_
    """    
    node_positions = np.column_stack((tri.x, tri.y))
    return get_edge_features_from_positions(node_positions, coo)


def get_edge_features_from_positions(node_positions, coo):
    """Return relative displacement and distance for directed edges."""
    node_positions = np.asarray(node_positions)
    first_endpoint_coordinates = node_positions[coo[0]]
    second_endpoint_coordinates = node_positions[coo[1]]
    u_ij = first_endpoint_coordinates - second_endpoint_coordinates
    norm = np.expand_dims(np.linalg.norm(u_ij,axis=1),axis=1)
    return np.concatenate([u_ij,norm],axis=1).astype('float32')


#def get_node_outputs(x,x_future,dt):
#    """
#    Contains the fluid acceleration between the current graph and the graph 
#    in the next time step. These are the features used for training: y=(v_t_next-v_t_curr)/dt [num_nodes x 2]
#
#    Returns:
#        _type_: _description_
#    """
#    result = (x_future[:,4:7]-x[:,4:7])/dt
#    
#    return result

def get_node_outputs(x, x_future):
    """
    Contains the fluid acceleration between the current graph and the graph 
    in the next time step. These are the features used for training: y=(v_t_next-v_t_curr) [num_nodes x 2]

    Returns:
        np.array: array with fluid acceleration features [num_nodes, 3]
    """
    # Indices for h, u, v in the dynamic features
    h_idx = 0
    u_idx = 1
    v_idx = 2
    
    result = (x_future[:, h_idx:v_idx+1] - x[:, h_idx:v_idx+1]) 
    
    return result


#def put_boundary_infos(x,x_future):
#    """
#    Args:
#        x (_type_): _description_
#        y (_type_): _description_
#    """
#    #Q imposé
#    x[(x[:,:4] == [0,0,1,0]).all(axis=1),4:7] = x_future[(x[:,:4] == [0,0,1,0]).all(axis=1),4:7]
#    #print((x[:,:4] == [0,0,1,0]).all(axis=1).sum())
#    # on impose hauteur à l'instant t+1 
#    x[(x[:,:4] == [0,1,0,0]).all(axis=1),4:5] = x_future[(x[:,:4] == [0,1,0,0]).all(axis=1),4:5]
#    #print((x[:,:4] == [0,1,0,0]).all(axis=1).sum())
#    return x

def put_boundary_infos(
    x,
    x_future,
    static_features,
    enforce_q_boundary=True,
    enforce_h_boundary=True,
):
    """
    Args:
        x (np.array): current timestep node features
        x_future (np.array): next timestep node features
        static_features (np.array): static node features including node type

    Returns:
        np.array: updated node features for the current timestep
    """
    # Indices for h, u, v in the dynamic features
    h_idx = 0
    u_idx = 1
    v_idx = 2
    
    # Indices for node type in the static features
    node_type_idx = 0
    node_type_length = 4
    
    # Apply boundary conditions
    q_mask = (static_features[:, node_type_idx:node_type_idx+node_type_length] == [0, 0, 1, 0]).all(axis=1)
    h_mask = (static_features[:, node_type_idx:node_type_idx+node_type_length] == [0, 1, 0, 0]).all(axis=1)
    
    if enforce_q_boundary:
        x[q_mask, h_idx:v_idx+1] = x_future[q_mask, h_idx:v_idx+1]
    if enforce_h_boundary:
        x[h_mask, h_idx:h_idx+1] = x_future[h_mask, h_idx:h_idx+1]
    
    return x

def put_boundary_infos_on_changes(
    y,
    static_features,
    zero_q_boundary_changes=True,
    zero_h_boundary_changes=True,
):
    """
    Args:
        x (np.array): current timestep node features
        x_future (np.array): next timestep node features
        static_features (np.array): static node features including node type

    Returns:
        np.array: updated node features for the current timestep
    """
    # Indices for h, u, v in the dynamic features
    h_idx = 0
    u_idx = 1
    v_idx = 2
    
    # Indices for node type in the static features
    node_type_idx = 0
    node_type_length = 4
    
    # Apply boundary conditions
    q_mask = (static_features[:, node_type_idx:node_type_idx+node_type_length] == [0, 0, 1, 0]).all(axis=1)
    h_mask = (static_features[:, node_type_idx:node_type_idx+node_type_length] == [0, 1, 0, 0]).all(axis=1)
    
    if zero_q_boundary_changes:
        y[q_mask, h_idx:v_idx+1] = 0.0
    if zero_h_boundary_changes:
        y[h_mask, h_idx:h_idx+1] = 0.0
    
    return y

#def get_dgl_graph(tri):
#    """
#    Create a DGL graph from the triangulation information.
#    """
#    coo_edges = get_edge_index(tri)  # get connectivity
#    g = dgl.graph((coo_edges[0], coo_edges[1]))  # Create a DGL graph
#    return g

def get_dgl_graph(tri):
    """
    Create a DGL graph from the triangulation information + edges features
    """
    coo_edges = get_edge_index(tri)
    g = dgl.graph((coo_edges[0], coo_edges[1]), num_nodes=len(tri.x))
    edge_features = get_edges_features(tri, coo_edges, None)  # Precompute edge features
    return g, edge_features

#def add_features_to_graph(g, node_features, edge_features):
#    """
#    Add node and edge features to the DGL graph.
#    """
#    g.ndata['x'] = torch.tensor(node_features, dtype=torch.float32)  # Add node features
#    g.edata['x'] = torch.tensor(edge_features, dtype=torch.float32)  # Add edge features

def create_dgl_dataset_chunked(
    mesh_list,
    res_list,
    cli_list,
    dt_list,
    data_folder,
    dataset_name,
    chunk_size=20,
    enforce_q_boundary=True,
    enforce_h_boundary=True,
):
    """
    mesh_list : list(string) : liste des fichiers .slf qui contiennent les maillages associées aux .res
    
    res_list  : list(string) : liste des fichiers .res qui contiennent les résultats associées aux .slf
    
    cli_list  : list(string) : liste des fichiers .cli qui contiennent les conditions aux limites des .slf
    
    dt_list   : list(string) : liste des pas de temps (pour l'instant tous égal à 1)
    
    data_folder : string : liste du folder qui contiendras les chunks
    
    dataset_name : string : noms du dataset produit
    
    """
    assert len(mesh_list) == len(res_list)
    assert len(dt_list) == len(res_list)
    assert len(cli_list) == len(res_list)
    number_trajectories = len(mesh_list)

    base_graph_list = []

    for traj in range(number_trajectories):
        mesh_path = mesh_list[traj]
        res_path = res_list[traj]
        cli_path = cli_list[traj]
        dt = dt_list[traj]
        res = TelemacFile(res_path, bnd_file=cli_path)
        res_mesh = TelemacFile(mesh_path)

        # Create DGL graph and precompute edge features
        g, edge_features = get_dgl_graph(res.tri)

        # Add edge features to the graph
        g.edata['x'] = torch.tensor(edge_features, dtype=torch.float32)

        # Add static node features to the graph
        static_node_features = get_static_node_features(res, res_mesh)
        g.ndata['static'] = torch.tensor(static_node_features, dtype=torch.float32)

        base_graph_list.append(g)

        number_ts = int(res.times.shape[0])

        for start_ts in range(0, number_ts - 1, chunk_size):
            end_ts = min(start_ts + chunk_size, number_ts - 1)
            dynamic_data_list = []
            for ts in range(start_ts, end_ts):

                # Get dynamic node features for current and next timesteps
                dynamic_node_features = get_dynamic_node_features(res, ts)
                dynamic_node_features_future = get_dynamic_node_features(res, ts + 1)
                dynamic_node_features = put_boundary_infos(
                    dynamic_node_features,
                    dynamic_node_features_future,
                    static_node_features,
                    enforce_q_boundary=enforce_q_boundary,
                    enforce_h_boundary=enforce_h_boundary,
                )
                
                # Get outputs for training
                y = get_node_outputs(dynamic_node_features, dynamic_node_features_future)
                
                #differences = np.abs((dynamic_node_features + y) - dynamic_node_features_future)
                #print("Max difference:", np.max(differences))
                #print("Mean difference:", np.mean(differences))
   
                #print(np.allclose(dynamic_node_features + y, dynamic_node_features_future,rtol=1e-4, atol=1e-7))
                
                y = put_boundary_infos_on_changes(
                    y,
                    static_node_features,
                    zero_q_boundary_changes=enforce_q_boundary,
                    zero_h_boundary_changes=enforce_h_boundary,
                )

                dynamic_data_list.append((dynamic_node_features, y, int(ts)))

            # Save dynamic data for this chunk
            with open(os.path.join(data_folder, f"{dataset_name}_{traj}_{start_ts}-{end_ts}.pkl"), 'wb') as f:
                pickle.dump(dynamic_data_list, f)

    # Save the base graphs separately
    dgl.save_graphs(os.path.join(data_folder, f"{dataset_name}_base.bin"), base_graph_list)
    return True

def create_multimesh(base_graph_path, fine_mesh_path, coarse_mesh_paths, output_path):
    """Add projected coarse-mesh edges to an existing physical base graph."""
    if not coarse_mesh_paths:
        raise ValueError("At least one coarse mesh is required.")

    base_graph_path = Path(base_graph_path)
    output_path = Path(output_path)
    if base_graph_path.resolve() == output_path.resolve():
        raise ValueError("The output path must differ from the base graph path.")

    base_graphs, graph_labels = dgl.load_graphs(str(base_graph_path))
    if len(base_graphs) != 1:
        raise ValueError("The base .bin must contain exactly one graph.")
    base_graph = base_graphs[0]

    fine_mesh = TelemacFile(str(fine_mesh_path))
    fine_xy, _ = add_mesh_info(fine_mesh)
    fine_mesh.close()
    if base_graph.num_nodes() != len(fine_xy):
        raise ValueError(
            "The base graph and fine mesh have different node counts: "
            f"{base_graph.num_nodes()} != {len(fine_xy)}"
        )

    src, dst = base_graph.edges()
    edge_blocks = [
        np.column_stack(
            [
                src.cpu().numpy().astype(np.int64),
                dst.cpu().numpy().astype(np.int64),
            ]
        )
    ]

    fine_tree = KDTree(fine_xy)
    for coarse_mesh_path in coarse_mesh_paths:
        coarse_mesh = TelemacFile(str(coarse_mesh_path))
        coarse_xy, coarse_triangles = add_mesh_info(coarse_mesh)
        coarse_mesh.close()
        distances, fine_indices = fine_tree.query(coarse_xy)
        mapped_triangles = fine_indices[np.asarray(coarse_triangles, dtype=np.int64)]
        collapsed_triangles = np.any(
            np.diff(np.sort(mapped_triangles, axis=1), axis=1) == 0,
            axis=1,
        )
        coarse_edges = get_edge_index_from_triangles(mapped_triangles).T
        edge_blocks.append(coarse_edges)
        print(
            f"{coarse_mesh_path}: nodes={len(coarse_xy)}, "
            f"edges={len(coarse_edges)}, "
            f"collapsed_triangles={int(collapsed_triangles.sum())}, "
            f"max_mapping_distance={float(np.max(distances)):.3f}"
        )

    edges = np.unique(np.concatenate(edge_blocks), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    edge_features = get_edge_features_from_positions(fine_xy, edges.T)
    if np.any(edge_features[:, 2] == 0.0):
        raise ValueError("The multimesh contains zero-length edges.")

    multimesh = dgl.graph(
        (edges[:, 0], edges[:, 1]),
        num_nodes=base_graph.num_nodes(),
    )
    multimesh.edata["x"] = torch.tensor(edge_features, dtype=torch.float32)
    for name, values in base_graph.ndata.items():
        multimesh.ndata[name] = values.clone()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dgl.save_graphs(str(output_path), [multimesh], graph_labels)

    base_edges = np.unique(edge_blocks[0], axis=0)
    base_edges = base_edges[base_edges[:, 0] != base_edges[:, 1]]
    print(
        f"base edges={len(base_edges)}, "
        f"multimesh edges={multimesh.num_edges()}, "
        f"added edges={multimesh.num_edges() - len(base_edges)}, "
        "self_loops=0"
    )
    return output_path
    
#################################
def somme_par_groupe(liste_tuples, k):
    resultat = []
    n = len(liste_tuples)
    for i in range(0, n, k):
        if i + k <= n:
            somme_y = sum(y for _, y in liste_tuples[i:i+k])
            x = liste_tuples[i][0]
            resultat.append((x, somme_y))
    return resultat

    
import json
from pathlib import Path


def save_json(var,path,file_name) :
    """
    Saves a dictionary of tensors to a JSON file.

    Parameters
    ----------
    var : Dict[str, torch.Tensor]
        Dictionary where each value is a PyTorch tensor.
    file : str
        Path to the output JSON file.
    """
    if not Path(path).is_dir():
        Path(path).mkdir(parents=True, exist_ok=True)
    # cast en float (float32) avant sauvegarde
    var_list = {k: v.to(torch.float32).cpu().numpy().tolist() for k, v in var.items()}
    with open(str(PurePath(path, file_name)), "w") as f:
        json.dump(var_list, f)


def load_json(file, dtype=torch.float32):
    """
    Loads a JSON file into a dictionary of PyTorch tensors.

    Parameters
    ----------
    file : str
        Path to the JSON file.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary where each value is a PyTorch tensor.
    """
    with open(file, "r") as f:
        var_list = json.load(f)
    # charge directement au dtype demandé (float32)
    var = {k: torch.tensor(v, dtype=dtype) for k, v in var_list.items()}
    return var


def unpack_dynamic_sample(sample):
    """
    Support legacy samples (x, y) and timestamped samples (x, y, ts).
    Also supports dict format {"x": ..., "y": ..., "ts": ...}.
    """
    if isinstance(sample, dict):
        if "x" not in sample or "y" not in sample:
            raise ValueError("Dynamic sample dict must contain keys 'x' and 'y'.")
        return sample["x"], sample["y"], sample.get("ts", None)

    if isinstance(sample, (tuple, list)):
        if len(sample) == 2:
            x, y = sample
            return x, y, None
        if len(sample) == 3:
            x, y, ts = sample
            return x, y, ts

    raise ValueError("Unsupported dynamic sample format. Expected (x,y), (x,y,ts) or dict.")


def collate_source_sequences(batch):
    seq_len = len(batch[0])
    output_sequence = []

    for t in range(seq_len):
        graphs = [item[t]["graph"] for item in batch]
        x_phys = torch.cat([item[t]["x_phys"] for item in batch], dim=0)
        x_src = torch.cat([item[t]["x_src"] for item in batch], dim=0)
        y_phys = torch.cat([item[t]["y_phys"] for item in batch], dim=0)

        output_sequence.append(
            {
                "graph": dgl.batch(graphs),
                "x_phys": x_phys,
                "x_src": x_src,
                "y_phys": y_phys,
            }
        )

    return output_sequence


class TelemacDataset(DGLDataset):
    """In-memory MeshGraphNet Dataset for stationary mesh
    Notes:
        - This dataset prepares and processes the data available in MeshGraphNet's repo:
            https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
        - A single adjacency matrix is used for each transient simulation.
          Do not use with adaptive mesh or remeshing

    Parameters
    ----------
    name : str, optional
        Name of the dataset, by default "dataset"
    data_dir : str, optional
        Directory that stores the raw data in .TFRecord format, by default None
    dynamic_data_files : list, optional
        List of paths to the pickle files containing dynamic node data, by default None
    split : str, optional
        Dataset split ["train", "eval", "test"], by default "train"
    ckpt_path : str, optional 
        Path where to find or save normalization values
    force_reload : bool, optional
        Force reload, by default False
    verbose : bool, optional
        Verbose, by default False
    normalize : bool, optional
        Whether to normalize the data, by default True
    sequence_length : int, optional
        Length of the sequences to provide, by default 1
    """

    def __init__(
        self,
        name="dataset",
        data_dir=None,
        dynamic_data_files=None,
        split="train",
        ckpt_path='.',
        force_reload=False,
        verbose=False,
        normalize=True,
        sequence_length=1,
        overlap=0,
    ):
        super().__init__(
            name=name,
            force_reload=force_reload,
            verbose=verbose,
        )
        self.data_dir = data_dir
        self.dynamic_data_files = dynamic_data_files
        self.split = split
        self.node_stats = None
        self.edge_stats = None
        self.sequence_length = sequence_length
        self.overlap=overlap

        # Load base graph (assuming a single graph)
        self.base_graph, _ = dgl.load_graphs(data_dir)
        self.base_graph = self.base_graph[0]

        # Load dynamic data from multiple pickle files and create sequences
        self.sequences = []
        for file_path in dynamic_data_files:
            with open(file_path, 'rb') as f:
                dynamic_data = pickle.load(f)
                step = max(1, sequence_length - overlap)
                for i in range(0, len(dynamic_data) - sequence_length + 1, step):
                    sequence = dynamic_data[i:i + sequence_length]
                    self.sequences.append(sequence)
                # Note: Remaining data at the end of the file is ignored to avoid overlapping outputs

        self.length = len(self.sequences)

        # Define dictionaries for nodes and edges
        self.node_var_info = {
            "h": {"source": "x", "index": 0},
            "u": {"source": "x", "index": 1},
            "v": {"source": "x", "index": 2},
            "strickler": {"source": "x", "index": 4},
            "z": {"source": "x", "index": 5},
            "delta_h": {"source": "y", "index": 0},
            "delta_u": {"source": "y", "index": 1},
            "delta_v": {"source": "y", "index": 2},
        }
        self.edge_var_info = {
            "xrel": {"source": "x", "index": 0},
            "yrel": {"source": "x", "index": 1},
            "norm": {"source": "x", "index": 2},
        }
        
        if normalize:
            if split == "train":
                print("Normalizing data...")
                self.node_stats = self._get_node_stats(self.node_var_info)
                self.edge_stats = self._get_edge_stats(self.edge_var_info)
                save_json(self.node_stats, ckpt_path, "node_stats.json")
                save_json(self.edge_stats, ckpt_path, "edge_stats.json")
            else:
                print("Loading normalization statistics...")
                self.node_stats = load_json(f"{ckpt_path}/node_stats.json", dtype=torch.float32)
                self.edge_stats = load_json(f"{ckpt_path}/edge_stats.json", dtype=torch.float32)

            self._normalize_data(self.node_stats, self.edge_stats, self.node_var_info, self.edge_var_info)



    def _normalize_data(self, node_stats, edge_stats, node_var_info, edge_var_info):
        # static node features
        for var_name, info in node_var_info.items():
            if var_name in ["strickler", "z"]:
                mean = node_stats[var_name].item()
                std  = node_stats[f"{var_name}_std"].item()
                if std != 0.0:
                    data_tensor = self.base_graph.ndata['static'][:, info['index']:info['index']+1]
                    self.base_graph.ndata['static'][:, info['index']:info['index']+1] = (data_tensor - mean) / std

        # edge features
        for var_name, info in edge_var_info.items():
            mean = edge_stats[var_name].item()
            std  = edge_stats[f"{var_name}_std"].item()
            if std != 0.0:
                data_tensor = self.base_graph.edata[info["source"]][:, info['index']:info['index']+1]
                self.base_graph.edata[info["source"]][:, info['index']:info['index']+1] = (data_tensor - mean) / std

        # dynamic node features and targets
        for seq_index, sequence in enumerate(self.sequences):
            normalized_sequence = []
            for sample in sequence:
                x, y, ts = unpack_dynamic_sample(sample)
                x = x.copy()
                y = y.copy()
                # dynamic h,u,v
                for var_name, info in node_var_info.items():
                    if var_name in ["h", "u", "v"]:
                        mean = node_stats[var_name].item()
                        std  = node_stats[f"{var_name}_std"].item()
                        if std != 0.0:
                            idx = info['index']  # indexes are *within dynamic block* (x)
                            x[:, idx:idx+1] = (x[:, idx:idx+1] - mean) / std
                # targets delta_h, delta_u, delta_v
                for var_name, info in node_var_info.items():
                    if var_name in ["delta_h", "delta_u", "delta_v"]:
                        mean = node_stats[var_name].item()
                        std  = node_stats[f"{var_name}_std"].item()
                        if std != 0.0:
                            idx = info['index']
                            y[:, idx:idx+1] = (y[:, idx:idx+1] - mean) / std

                if ts is None:
                    normalized_sequence.append((x, y))
                else:
                    normalized_sequence.append((x, y, int(ts)))
            self.sequences[seq_index] = normalized_sequence


    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        graphs = []
        for sample in sequence:
            x, y, _ = unpack_dynamic_sample(sample)
            # Combine static and dynamic features
            static_features = self.base_graph.ndata['static']
            dynamic_features = torch.tensor(x, dtype=torch.float32)
            combined_features = torch.cat((static_features, dynamic_features), dim=1)
            # Create a new graph for the current timestep
            g = self.base_graph.clone()
            g.ndata.pop('static')
            g.ndata['x'] = combined_features
            g.ndata['y'] = torch.tensor(y, dtype=torch.float32)
            graphs.append(g)
        return graphs  # Return a list of graphs representing the sequence
    
    def __len__(self):
        return self.length 
    
    def _get_node_stats(self, var_info):
        # Accumulateurs en float32 (cohérents avec I/O)
        stats = {key: torch.zeros(1, dtype=torch.float32) for key in var_info.keys()}
        meansqr_stats = {f"{key}_meansqr": torch.zeros(1, dtype=torch.float32) for key in var_info.keys()}

        # ----- statiques -----
        static = self.base_graph.ndata['static'].to(torch.float32)
        strickler_col, z_col = 4, 5
        if "strickler" in var_info:
            v = static[:, strickler_col:strickler_col+1]
            m = v.mean(); ms = (v**2).mean()
            stats["strickler"] = m
            meansqr_stats["strickler_meansqr"] = ms
            stats["strickler_std"] = torch.sqrt(torch.clamp(ms - m*m, min=0.0))
        if "z" in var_info:
            v = static[:, z_col:z_col+1]
            m = v.mean(); ms = (v**2).mean()
            stats["z"] = m
            meansqr_stats["z_meansqr"] = ms
            stats["z_std"] = torch.sqrt(torch.clamp(ms - m*m, min=0.0))

        # ----- dynamiques & deltas -----
        total_steps = 0
        for sequence in self.sequences:
            for sample in sequence:
                x, y, _ = unpack_dynamic_sample(sample)
                total_steps += 1
                x_t = torch.tensor(x, dtype=torch.float32)
                y_t = torch.tensor(y, dtype=torch.float32)

                if "h" in var_info:
                    v = x_t[:, 0:1]; stats["h"] += v.mean(); meansqr_stats["h_meansqr"] += (v*v).mean()
                if "u" in var_info:
                    v = x_t[:, 1:2]; stats["u"] += v.mean(); meansqr_stats["u_meansqr"] += (v*v).mean()
                if "v" in var_info:
                    v = x_t[:, 2:3]; stats["v"] += v.mean(); meansqr_stats["v_meansqr"] += (v*v).mean()

                if "delta_h" in var_info:
                    v = y_t[:, 0:1]; stats["delta_h"] += v.mean(); meansqr_stats["delta_h_meansqr"] += (v*v).mean()
                if "delta_u" in var_info:
                    v = y_t[:, 1:2]; stats["delta_u"] += v.mean(); meansqr_stats["delta_u_meansqr"] += (v*v).mean()
                if "delta_v" in var_info:
                    v = y_t[:, 2:3]; stats["delta_v"] += v.mean(); meansqr_stats["delta_v_meansqr"] += (v*v).mean()

        denom = max(total_steps, 1)
        for var_name in ["h", "u", "v", "delta_h", "delta_u", "delta_v"]:
            if var_name in var_info:
                stats[var_name] /= denom
                meansqr_stats[f"{var_name}_meansqr"] /= denom
                mean = stats[var_name]
                ms = meansqr_stats[f"{var_name}_meansqr"]
                var = torch.clamp(ms - mean*mean, min=0.0)
                stats[f"{var_name}_std"] = torch.sqrt(var)
                del meansqr_stats[f"{var_name}_meansqr"]

        if "strickler_meansqr" in meansqr_stats: del meansqr_stats["strickler_meansqr"]
        if "z_meansqr" in meansqr_stats: del meansqr_stats["z_meansqr"]

        return stats

    def _get_edge_stats(self, var_info):
        stats = {key: torch.zeros(1, dtype=torch.float32) for key in var_info.keys()}
        meansqr_stats = {f"{key}_meansqr": torch.zeros(1, dtype=torch.float32) for key in var_info.keys()}

        graph = self.__getitem__(0)[0]
        for var_name, info in var_info.items():
            value = graph.edata[info["source"]][:, info["index"]:info["index"]+1].to(torch.float32)
            m = value.mean()
            stats[var_name] = stats[var_name] + m
            meansqr_stats[f"{var_name}_meansqr"] = meansqr_stats[f"{var_name}_meansqr"] + (value*value).mean()

        for var_name in var_info.keys():
            mean = stats[var_name]
            ms = meansqr_stats[f"{var_name}_meansqr"]
            var = torch.clamp(ms - mean*mean, min=0.0)
            stats[f"{var_name}_std"] = torch.sqrt(var)
            del meansqr_stats[f"{var_name}_meansqr"]

        return stats


def _is_float_token(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


def load_liq_table(liq_path):
    header = None
    rows = []

    with open(liq_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not all(_is_float_token(token) for token in parts):
                if header is None:
                    header = parts
                continue

            rows.append([float(token) for token in parts])

    if not rows:
        raise ValueError(f"No numeric hydrograph data found in {liq_path}")

    data = np.asarray(rows, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Hydrograph {liq_path} must contain at least time and one value column.")

    if header is None or len(header) != data.shape[1]:
        header = ["T"] + [f"COL_{idx}" for idx in range(1, data.shape[1])]

    order = np.argsort(data[:, 0])
    return data[order], header


def load_liq_q_series(liq_path):
    data, header = load_liq_table(liq_path)
    time_seconds = data[:, 0]

    q_indices = [
        idx
        for idx, name in enumerate(header[1:], start=1)
        if name.upper().startswith("Q")
    ]
    if not q_indices:
        q_indices = [data.shape[1] - 1]

    q_values = data[:, q_indices].astype(np.float32)
    q_labels = [header[idx] for idx in q_indices]

    if q_values.ndim == 1:
        q_values = q_values[:, None]

    return time_seconds, q_values, q_labels


def load_liq_hydrograph(liq_path):
    """
    Load TELEMAC .liq hydrograph file and return (time_seconds, discharge_q).
    If several Q columns are present, the returned discharge is their sum.
    """
    time_seconds, q_values, _ = load_liq_q_series(liq_path)
    total_q = q_values.sum(axis=1, dtype=np.float32)
    return time_seconds, total_q


class TelemacDatasetWithQ(TelemacDataset):
    """
    Separate dataset class that augments dynamic node features with global Q(t),
    interpolated from associated .liq hydrographs using per-sample ts.
    """

    def __init__(
        self,
        name="dataset_q",
        data_dir=None,
        dynamic_data_files=None,
        hydro_data_files=None,
        split="train",
        ckpt_path='.',
        force_reload=False,
        verbose=False,
        normalize=True,
        sequence_length=1,
        overlap=0,
        dt_seconds=1800.0,
    ):
        if hydro_data_files is None:
            raise ValueError("hydro_data_files is required for TelemacDatasetWithQ.")
        if dynamic_data_files is None:
            raise ValueError("dynamic_data_files is required for TelemacDatasetWithQ.")
        if len(dynamic_data_files) != len(hydro_data_files):
            raise ValueError("dynamic_data_files and hydro_data_files must have same length.")

        self.hydro_data_files = [str(p) for p in hydro_data_files]
        self.dt_seconds = float(dt_seconds)
        self.sequence_meta = []
        self.hydrographs = {}

        super().__init__(
            name=name,
            data_dir=data_dir,
            dynamic_data_files=dynamic_data_files,
            split=split,
            ckpt_path=ckpt_path,
            force_reload=force_reload,
            verbose=verbose,
            normalize=normalize,
            sequence_length=sequence_length,
            overlap=overlap,
        )

        self._build_sequence_meta(dynamic_data_files)
        self._load_hydrographs()
        self.node_var_info["q"] = {"source": "x", "index": 3}

        if normalize:
            if split == "train":
                q_mean, q_std = self._get_q_stats()
                self.node_stats["q"] = q_mean
                self.node_stats["q_std"] = q_std
                save_json(self.node_stats, ckpt_path, "node_stats.json")
            else:
                if ("q" not in self.node_stats) or ("q_std" not in self.node_stats):
                    raise ValueError(
                        "Missing q/q_std in node_stats.json for eval/test with TelemacDatasetWithQ."
                    )

    def _build_sequence_meta(self, dynamic_data_files):
        step = max(1, self.sequence_length - self.overlap)
        for file_path, hydro_path in zip(dynamic_data_files, self.hydro_data_files):
            with open(file_path, 'rb') as f:
                dynamic_data = pickle.load(f)
            for _ in range(0, len(dynamic_data) - self.sequence_length + 1, step):
                self.sequence_meta.append({"hydro_path": hydro_path})

        if len(self.sequence_meta) != len(self.sequences):
            raise ValueError(
                f"Sequence metadata mismatch: {len(self.sequence_meta)} meta vs {len(self.sequences)} sequences."
            )

    def _load_hydrographs(self):
        for hydro_path in set(self.hydro_data_files):
            self.hydrographs[hydro_path] = load_liq_hydrograph(hydro_path)

    def _q_at_ts(self, hydro_path, ts):
        if ts is None:
            raise ValueError("Sample is missing ts. TelemacDatasetWithQ requires (x, y, ts).")
        t_sec = float(ts) * self.dt_seconds
        t_arr, q_arr = self.hydrographs[hydro_path]
        return float(np.interp(t_sec, t_arr, q_arr, left=q_arr[0], right=q_arr[-1]))

    def _get_q_stats(self):
        sum_q = 0.0
        sum_q2 = 0.0
        total = 0
        for seq_idx, sequence in enumerate(self.sequences):
            hydro_path = self.sequence_meta[seq_idx]["hydro_path"]
            for sample in sequence:
                _, _, ts = unpack_dynamic_sample(sample)
                q_val = self._q_at_ts(hydro_path, ts)
                sum_q += q_val
                sum_q2 += q_val * q_val
                total += 1

        denom = max(total, 1)
        mean_q = sum_q / denom
        var_q = max(sum_q2 / denom - mean_q * mean_q, 0.0)
        std_q = float(np.sqrt(var_q))
        return (
            torch.tensor([mean_q], dtype=torch.float32),
            torch.tensor([std_q], dtype=torch.float32),
        )

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        hydro_path = self.sequence_meta[idx]["hydro_path"]
        graphs = []
        for sample in sequence:
            x, y, ts = unpack_dynamic_sample(sample)
            static_features = self.base_graph.ndata['static']
            dynamic_features = torch.tensor(x, dtype=torch.float32)

            q_val = self._q_at_ts(hydro_path, ts)
            q_tensor = torch.full((dynamic_features.shape[0], 1), q_val, dtype=torch.float32)

            if self.node_stats is not None and "q" in self.node_stats and "q_std" in self.node_stats:
                q_mean = self.node_stats["q"].item()
                q_std = self.node_stats["q_std"].item()
                if q_std != 0.0:
                    q_tensor = (q_tensor - q_mean) / q_std

            dynamic_features = torch.cat((dynamic_features, q_tensor), dim=1)
            combined_features = torch.cat((static_features, dynamic_features), dim=1)

            g = self.base_graph.clone()
            g.ndata.pop('static')
            g.ndata['x'] = combined_features
            g.ndata['y'] = torch.tensor(y, dtype=torch.float32)
            graphs.append(g)
        return graphs

    def _get_edge_stats(self, var_info):
        # Keep independent from __getitem__ so super().__init__ remains safe.
        stats = {key: torch.zeros(1, dtype=torch.float32) for key in var_info.keys()}
        meansqr_stats = {f"{key}_meansqr": torch.zeros(1, dtype=torch.float32) for key in var_info.keys()}

        for var_name, info in var_info.items():
            value = self.base_graph.edata[info["source"]][:, info["index"]:info["index"]+1].to(torch.float32)
            m = value.mean()
            stats[var_name] = stats[var_name] + m
            meansqr_stats[f"{var_name}_meansqr"] = meansqr_stats[f"{var_name}_meansqr"] + (value*value).mean()

        for var_name in var_info.keys():
            mean = stats[var_name]
            ms = meansqr_stats[f"{var_name}_meansqr"]
            var = torch.clamp(ms - mean*mean, min=0.0)
            stats[f"{var_name}_std"] = torch.sqrt(var)
            del meansqr_stats[f"{var_name}_meansqr"]

        return stats


class TelemacDatasetWithSourceNodes(TelemacDataset):
    """
    Dataset variant that augments the physical mesh with ghost/source inlet nodes.
    Physical and source features stay separated:
      - x_phys: [N_phys, d_phys]
      - x_src:  [N_src, d_src]
      - y_phys: [N_phys, 3]
    """

    def __init__(
        self,
        name="dataset_source_nodes",
        data_dir=None,
        dynamic_data_files=None,
        hydro_data_files=None,
        cli_file=None,
        inlet_node_lists=None,
        use_q_feature=False,
        split="train",
        ckpt_path='.',
        force_reload=False,
        verbose=False,
        normalize=True,
        sequence_length=1,
        overlap=0,
        dt_seconds=1800.0,
    ):
        if hydro_data_files is None:
            raise ValueError("hydro_data_files is required for TelemacDatasetWithSourceNodes.")
        if dynamic_data_files is None:
            raise ValueError("dynamic_data_files is required for TelemacDatasetWithSourceNodes.")
        if len(dynamic_data_files) != len(hydro_data_files):
            raise ValueError("dynamic_data_files and hydro_data_files must have same length.")
        if inlet_node_lists is None and cli_file is None:
            raise ValueError("Provide either cli_file or inlet_node_lists.")

        self.hydro_data_files = [str(p) for p in hydro_data_files]
        self.dt_seconds = float(dt_seconds)
        self.cli_file = cli_file
        self.use_q_feature = bool(use_q_feature)
        self.sequence_meta = []
        self.hydrographs = {}
        self.source_stats = None
        self.source_feature_names = ["q"]

        super().__init__(
            name=name,
            data_dir=data_dir,
            dynamic_data_files=dynamic_data_files,
            split=split,
            ckpt_path=ckpt_path,
            force_reload=force_reload,
            verbose=verbose,
            normalize=normalize,
            sequence_length=sequence_length,
            overlap=overlap,
        )

        self._build_sequence_meta(dynamic_data_files)
        if inlet_node_lists is None:
            inlet_node_lists = extract_inlet_node_lists_from_conlim(cli_file)
        self.inlet_node_lists = normalize_inlet_node_lists(inlet_node_lists)

        self.num_physical_nodes = self.base_graph.num_nodes()
        self._validate_inlet_node_lists()

        self.num_source_nodes = len(self.inlet_node_lists)
        self.global_q_feature_names = [
            f"q_{source_id}"
            for source_id in range(self.num_source_nodes)
        ]
        self.physical_dynamic_dim = 3 + (
            self.num_source_nodes if self.use_q_feature else 0
        )
        self.physical_static = self.base_graph.ndata["static"].clone()
        self.base_graph_with_sources = add_ghost_source_nodes(
            self.base_graph,
            self.inlet_node_lists,
            edge_feature_dim=self.base_graph.edata["x"].shape[1],
        )

        self._load_source_hydrographs()

        if normalize:
            if split == "train":
                self.source_stats = self._get_source_stats()
                save_json(self.source_stats, ckpt_path, "source_stats.json")
                if self.use_q_feature:
                    self.node_stats.update(self._get_global_q_stats())
                    save_json(self.node_stats, ckpt_path, "node_stats.json")
            else:
                self.source_stats = load_json(
                    f"{ckpt_path}/source_stats.json",
                    dtype=torch.float32,
                )
                if self.use_q_feature:
                    self._validate_global_q_stats()

    def _build_sequence_meta(self, dynamic_data_files):
        step = max(1, self.sequence_length - self.overlap)
        for file_path, hydro_path in zip(dynamic_data_files, self.hydro_data_files):
            with open(file_path, 'rb') as f:
                dynamic_data = pickle.load(f)
            for _ in range(0, len(dynamic_data) - self.sequence_length + 1, step):
                self.sequence_meta.append({"hydro_path": hydro_path})

        if len(self.sequence_meta) != len(self.sequences):
            raise ValueError(
                f"Sequence metadata mismatch: {len(self.sequence_meta)} meta vs {len(self.sequences)} sequences."
            )

    def _validate_inlet_node_lists(self):
        for source_id, inlet_nodes in enumerate(self.inlet_node_lists):
            for node_id in inlet_nodes:
                if node_id < 0 or node_id >= self.num_physical_nodes:
                    raise ValueError(
                        f"Inlet node {node_id} from source {source_id} is outside the physical graph."
                    )

    def _load_source_hydrographs(self):
        for hydro_path in set(self.hydro_data_files):
            t_arr, q_values, q_labels = load_liq_q_series(hydro_path)
            if q_values.shape[1] != self.num_source_nodes:
                raise ValueError(
                    f"{hydro_path} provides {q_values.shape[1]} Q series but "
                    f"{self.num_source_nodes} source nodes are defined."
                )
            self.hydrographs[hydro_path] = {
                "time_seconds": t_arr,
                "q_values": q_values,
                "q_labels": q_labels,
            }

    def _source_q_at_ts(self, hydro_path, ts):
        if ts is None:
            raise ValueError(
                "Sample is missing ts. TelemacDatasetWithSourceNodes requires (x, y, ts)."
            )

        hydro = self.hydrographs[hydro_path]
        t_sec = float(ts) * self.dt_seconds
        t_arr = hydro["time_seconds"]
        q_values = hydro["q_values"]

        return np.asarray(
            [
                np.interp(
                    t_sec,
                    t_arr,
                    q_values[:, source_id],
                    left=q_values[0, source_id],
                    right=q_values[-1, source_id],
                )
                for source_id in range(self.num_source_nodes)
            ],
            dtype=np.float32,
        )

    def _build_source_features_raw(self, hydro_path, ts):
        q_values = self._source_q_at_ts(hydro_path, ts)
        return q_values[:, None].astype(np.float32)

    def _build_global_q_features_raw(self, hydro_path, ts):
        return self._source_q_at_ts(hydro_path, ts)

    def _normalize_source_features(self, source_features):
        if self.source_stats is None:
            return source_features

        source_features = source_features.clone()
        for feature_idx, feature_name in enumerate(self.source_feature_names):
            mean = self.source_stats[feature_name].item()
            std = self.source_stats[f"{feature_name}_std"].item()
            if std != 0.0:
                source_features[:, feature_idx:feature_idx+1] = (
                    source_features[:, feature_idx:feature_idx+1] - mean
                ) / std
        return source_features

    def _normalize_global_q_features(self, global_q_features):
        if (not self.use_q_feature) or self.node_stats is None:
            return global_q_features

        global_q_features = global_q_features.clone()
        for source_id, feature_name in enumerate(self.global_q_feature_names):
            mean = self.node_stats[feature_name].item()
            std = self.node_stats[f"{feature_name}_std"].item()
            if std != 0.0:
                global_q_features[:, source_id:source_id+1] = (
                    global_q_features[:, source_id:source_id+1] - mean
                ) / std
        return global_q_features

    def _validate_global_q_stats(self):
        missing_keys = []
        for feature_name in self.global_q_feature_names:
            if feature_name not in self.node_stats:
                missing_keys.append(feature_name)
            std_key = f"{feature_name}_std"
            if std_key not in self.node_stats:
                missing_keys.append(std_key)

        if missing_keys:
            raise ValueError(
                "Missing global Q statistics in node_stats.json for "
                f"TelemacDatasetWithSourceNodes(use_q_feature=True): {missing_keys}"
            )

    def _get_source_stats(self):
        sums = torch.zeros(len(self.source_feature_names), dtype=torch.float32)
        sumsqr = torch.zeros(len(self.source_feature_names), dtype=torch.float32)
        total = 0

        for seq_idx, sequence in enumerate(self.sequences):
            hydro_path = self.sequence_meta[seq_idx]["hydro_path"]
            for sample in sequence:
                _, _, ts = unpack_dynamic_sample(sample)
                source_features = torch.tensor(
                    self._build_source_features_raw(hydro_path, ts),
                    dtype=torch.float32,
                )
                sums += source_features.sum(dim=0)
                sumsqr += (source_features * source_features).sum(dim=0)
                total += source_features.shape[0]

        denom = max(total, 1)
        stats = {}
        for feature_idx, feature_name in enumerate(self.source_feature_names):
            mean = sums[feature_idx] / denom
            meansqr = sumsqr[feature_idx] / denom
            std = torch.sqrt(torch.clamp(meansqr - mean * mean, min=0.0))
            stats[feature_name] = mean.unsqueeze(0)
            stats[f"{feature_name}_std"] = std.unsqueeze(0)

        return stats

    def _get_global_q_stats(self):
        sums = torch.zeros(self.num_source_nodes, dtype=torch.float32)
        sumsqr = torch.zeros(self.num_source_nodes, dtype=torch.float32)
        total = 0

        for seq_idx, sequence in enumerate(self.sequences):
            hydro_path = self.sequence_meta[seq_idx]["hydro_path"]
            for sample in sequence:
                _, _, ts = unpack_dynamic_sample(sample)
                q_values = torch.tensor(
                    self._build_global_q_features_raw(hydro_path, ts),
                    dtype=torch.float32,
                )
                sums += q_values
                sumsqr += q_values * q_values
                total += 1

        denom = max(total, 1)
        stats = {}
        for source_id, feature_name in enumerate(self.global_q_feature_names):
            mean = sums[source_id] / denom
            meansqr = sumsqr[source_id] / denom
            std = torch.sqrt(torch.clamp(meansqr - mean * mean, min=0.0))
            stats[feature_name] = mean.unsqueeze(0)
            stats[f"{feature_name}_std"] = std.unsqueeze(0)

        return stats

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        hydro_path = self.sequence_meta[idx]["hydro_path"]
        output_sequence = []

        for sample in sequence:
            x, y, ts = unpack_dynamic_sample(sample)
            dynamic_features = torch.tensor(x, dtype=torch.float32)
            if self.use_q_feature:
                raw_global_q = self._build_global_q_features_raw(hydro_path, ts)
                global_q_features = torch.tensor(
                    raw_global_q,
                    dtype=torch.float32,
                ).unsqueeze(0).repeat(dynamic_features.shape[0], 1)
                global_q_features = self._normalize_global_q_features(global_q_features)
                dynamic_features = torch.cat(
                    (dynamic_features, global_q_features),
                    dim=1,
                )
            x_phys = torch.cat((self.physical_static, dynamic_features), dim=1)

            raw_source_features = self._build_source_features_raw(hydro_path, ts)
            x_src = torch.tensor(raw_source_features, dtype=torch.float32)
            x_src = self._normalize_source_features(x_src)

            output_sequence.append(
                {
                    "graph": self.base_graph_with_sources.clone(),
                    "x_phys": x_phys,
                    "x_src": x_src,
                    "y_phys": torch.tensor(y, dtype=torch.float32),
                }
            )

        return output_sequence

    def _get_edge_stats(self, var_info):
        # Keep independent from __getitem__ so super().__init__ remains safe.
        stats = {key: torch.zeros(1, dtype=torch.float32) for key in var_info.keys()}
        meansqr_stats = {f"{key}_meansqr": torch.zeros(1, dtype=torch.float32) for key in var_info.keys()}

        for var_name, info in var_info.items():
            value = self.base_graph.edata[info["source"]][:, info["index"]:info["index"]+1].to(torch.float32)
            m = value.mean()
            stats[var_name] = stats[var_name] + m
            meansqr_stats[f"{var_name}_meansqr"] = meansqr_stats[f"{var_name}_meansqr"] + (value * value).mean()

        for var_name in var_info.keys():
            mean = stats[var_name]
            ms = meansqr_stats[f"{var_name}_meansqr"]
            var = torch.clamp(ms - mean * mean, min=0.0)
            stats[f"{var_name}_std"] = torch.sqrt(var)
            del meansqr_stats[f"{var_name}_meansqr"]

        return stats
