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
from scipy.spatial import KDTree

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
    """ 
    Return the connectivity of the graph in COO format

    Returns:
        np.array : [2 x num_edges]
    """
    edges = set()
    for triangle in tri.triangles:
        for i in range(3):
            edge_1 = tuple(sorted([triangle[i], triangle[(i + 1) % 3]]))
            edge_2 = tuple(sorted([triangle[i], triangle[(i + 1) % 3]],reverse=True))
            edges.add(edge_1)
            edges.add(edge_2)

    coo = np.array(list(edges)).T
    return coo.astype('int64')

def get_edges_features(tri,coo,res):
    """_summary_

    Returns:
        _type_: _description_
    """    
    first_endpoint_indices = coo[0]
    second_endpoint_indices = coo[1]
    
    x,y = tri.x,tri.y
    
    first_endpoint_coordinates = np.column_stack((x[first_endpoint_indices], y[first_endpoint_indices]))
    second_endpoint_coordinates = np.column_stack((x[second_endpoint_indices], y[second_endpoint_indices]))
    
    u_ij = first_endpoint_coordinates-second_endpoint_coordinates
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

def put_boundary_infos(x, x_future, static_features):
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
    
    x[q_mask, h_idx:v_idx+1] = x_future[q_mask, h_idx:v_idx+1]
    x[h_mask, h_idx:h_idx+1] = x_future[h_mask, h_idx:h_idx+1]
    
    return x

def put_boundary_infos_on_changes(y, static_features):
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
    
    y[q_mask, h_idx:v_idx+1] = 0.0
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
    g = dgl.graph((coo_edges[0], coo_edges[1]))
    edge_features = get_edges_features(tri, coo_edges, None)  # Precompute edge features
    return g, edge_features

#def add_features_to_graph(g, node_features, edge_features):
#    """
#    Add node and edge features to the DGL graph.
#    """
#    g.ndata['x'] = torch.tensor(node_features, dtype=torch.float32)  # Add node features
#    g.edata['x'] = torch.tensor(edge_features, dtype=torch.float32)  # Add edge features

def create_dgl_dataset_chunked(mesh_list, res_list, cli_list, dt_list, data_folder, dataset_name, chunk_size=20):
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
                dynamic_node_features = put_boundary_infos(dynamic_node_features, dynamic_node_features_future, static_node_features)
                #put no modification on boundaries 
                
                # Get outputs for training
                y = get_node_outputs(dynamic_node_features, dynamic_node_features_future)
                
                #differences = np.abs((dynamic_node_features + y) - dynamic_node_features_future)
                #print("Max difference:", np.max(differences))
                #print("Mean difference:", np.mean(differences))
   
                #print(np.allclose(dynamic_node_features + y, dynamic_node_features_future,rtol=1e-4, atol=1e-7))
                
                y = put_boundary_infos_on_changes(y,static_node_features) #put 0 on changes  

                dynamic_data_list.append((dynamic_node_features, y))

            # Save dynamic data for this chunk
            with open(os.path.join(data_folder, f"{dataset_name}_{traj}_{start_ts}-{end_ts}.pkl"), 'wb') as f:
                pickle.dump(dynamic_data_list, f)

    # Save the base graphs separately
    dgl.save_graphs(os.path.join(data_folder, f"{dataset_name}_base.bin"), base_graph_list)
    return True

def replace_triangle_indices(tri, indices):
    """
    Remplace les indices des triangles par les indices du KD-tree.

    Parameters:
    tri (np.ndarray): Tableau de triangles (n x 3).
    indices (np.ndarray): Tableau d'indices du KD-tree (m,).

    Returns:
    np.ndarray: Nouveau tableau de triangles avec indices remplacés.
    """
    # Assurez-vous que les triangles et les indices sont des tableaux numpy
    tri = np.asarray(tri)
    indices = np.asarray(indices)
    
    # Remplacer les indices des triangles par les indices du KD-tree
    new_tri = indices[tri]
    
    return new_tri

def create_multimesh(fine_mesh,coarse_mesh_list,res_list,cli_list,data_folder,dataset_name):
    mesh_path = fine_mesh
    res_path = res_list[0]
    cli_path = cli_list[0]
    res = TelemacFile(res_path, bnd_file=cli_path)
    res_mesh = TelemacFile(mesh_path)
    
    X,triangles = add_mesh_info(res_mesh)
    fine_kd_tree = KDTree(X)
    print(triangles.shape)
    for coarse_mesh_path in coarse_mesh_list : 
        coarse_mesh = TelemacFile(coarse_mesh_path)
        X_coarse,triangles_coarse = add_mesh_info(coarse_mesh)
        distances, indices = fine_kd_tree.query(X_coarse)
        new_tri = replace_triangle_indices(triangles_coarse, indices)
        triangles = np.concatenate([triangles,new_tri])
    
    # Extract x and y coordinates
    x = X[:, 0]
    y = X[:, 1]
    # Create the triangulation object
    triangulation = tri.Triangulation(x, y, triangles)
    
    # Create DGL graph and precompute edge features
    g, edge_features = get_dgl_graph(res.tri)

    # Add edge features to the graph
    g.edata['x'] = torch.tensor(edge_features, dtype=torch.float32)

    # Add static node features to the graph
    static_node_features = get_static_node_features(res, res_mesh)
    g.ndata['static'] = torch.tensor(static_node_features, dtype=torch.float32)
    
    dgl.save_graphs(os.path.join(data_folder, f"{dataset_name}_multimesh_base.bin"), [g])
    
    return True
    
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

class TelemacDatasetOld(DGLDataset):
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
    data_dir : _type_, optional
        Specifying the directory that stores the raw data in .TFRecord format., by default None
    dynamic_data_file : str, optional
        Path to the pickle file containing dynamic node data, by default None
    split : str, optional
        Dataset split ["train", "eval", "test"], by default "train"
    num_samples : int, optional
        Number of samples, by default 1000
    num_steps : int, optional
        Number of time steps in each sample, by default 600
    ckpt_path : str, optional 
        Path where to find or save normalization values 
    force_reload : bool, optional
        force reload, by default False
    verbose : bool, optional
        verbose, by default False
    """

    def __init__(
        self,
        name="dataset",
        data_dir=None,
        dynamic_data_file=None,
        split="train",
        num_samples=1000,
        num_steps=600,
        ckpt_path='.',
        force_reload=False,
        verbose=False,
        normalize=True,
        stride=1,
        starting_ts=0
    ):
        super().__init__(
            name=name,
            force_reload=force_reload,
            verbose=verbose,
        )
        self.data_dir = data_dir
        self.dynamic_data_file = dynamic_data_file
        self.split = split
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.length = (num_samples * num_steps) // stride
        self.node_stats = None
        self.edge_stats = None
        # Load base graph (assuming a single graph)
        self.base_graph, _ = dgl.load_graphs(data_dir)
        self.base_graph = self.base_graph[0]

        # Load dynamic data
        with open(dynamic_data_file, 'rb') as f:
            all_dynamic_data = pickle.load(f)
            self.dynamic_data_list = all_dynamic_data[starting_ts:starting_ts + self.length]

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
                # Save normalization statistics
                #save_json(self.node_stats, ckpt_path, "node_stats.json")
                #save_json(self.edge_stats, ckpt_path, "edge_stats.json")
                # Normalize node and edge data
                self._normalize_data(self.node_stats, self.edge_stats, self.node_var_info, self.edge_var_info)
            else:
                print("Loading normalization statistics...")
                self.node_stats = load_json(f"{ckpt_path}/node_stats.json")
                self.edge_stats = load_json(f"{ckpt_path}/edge_stats.json")
                self._normalize_data(self.node_stats, self.edge_stats, self.node_var_info, self.edge_var_info)

    def _normalize_data(self, node_stats, edge_stats, node_var_info, edge_var_info):
        """Normalize node and edge data in all graphs based on computed statistics."""

        # Normalize static node features
        for var_name, info in node_var_info.items():
            if var_name in ["strickler", "z"]:
                mean = node_stats[var_name].item()
                std = node_stats[f"{var_name}_std"].item()
                if std != 0.0:
                    data_tensor = self.base_graph.ndata['static'][:, info['index']:info['index'] + 1]
                    self.base_graph.ndata['static'][:, info['index']:info['index'] + 1] = (data_tensor - mean) / std

        # Normalize edge features
        for var_name, info in edge_var_info.items():
            mean = edge_stats[var_name].item()
            std = edge_stats[f"{var_name}_std"].item()
            if std != 0.0:
                data_tensor = self.base_graph.edata[info["source"]][:, info['index']:info['index'] + 1]
                self.base_graph.edata[info["source"]][:, info['index']:info['index'] + 1] = (data_tensor - mean) / std

        # Normalize dynamic node features and targets
        for index, dynamic_data in enumerate(self.dynamic_data_list):
            x, y = dynamic_data

            # Normalize node features (h, u, v)
            for var_name, info in node_var_info.items():
                if var_name in ["h", "u", "v"]:
                    mean = node_stats[var_name].item()
                    std = node_stats[f"{var_name}_std"].item()
                    if std != 0.0:
                        x[:, info['index']:info['index'] + 1] = (x[:, info['index']:info['index'] + 1] - mean) / std

            # Normalize target variables (delta_h, delta_u, delta_v)
            for var_name, info in node_var_info.items():
                if var_name in ["delta_h", "delta_u", "delta_v"]:
                    mean = node_stats[var_name].item()
                    std = node_stats[f"{var_name}_std"].item()
                    if std != 0.0:
                        y[:, info['index']:info['index'] + 1] = (y[:, info['index']:info['index'] + 1] - mean) / std

            # Update the dynamic data list with normalized data
            self.dynamic_data_list[index] = (x, y)

    def __getitem__(self, idx):
        tidx = idx  # timestep index
        dynamic_node_features, y = self.dynamic_data_list[tidx]
        # Combine static and dynamic features
        static_features = self.base_graph.ndata['static']
        combined_features = torch.cat((static_features, torch.tensor(dynamic_node_features, dtype=torch.float32)), dim=1)
        # Create a new graph for the current timestep
        g = self.base_graph.clone()
        g.ndata.pop('static')
        g.ndata['x'] = combined_features
        g.ndata['y'] = torch.tensor(y, dtype=torch.float32)
        return g

    def __len__(self):
        return self.length

    def _get_node_stats(self, var_info):
        """
        Compute statistics (mean and std) for node variables.
        """

        # Initialize stats dictionary with float64 tensors
        stats = {key: torch.zeros(1, dtype=torch.float64) for key in var_info.keys()}
        meansqr_keys = [f"{key}_meansqr" for key in var_info.keys()]
        stats.update({key: torch.zeros(1, dtype=torch.float64) for key in meansqr_keys})

        total_steps = self.length

        for i in range(self.length):
            graph = self.__getitem__(i)
            for var_name, info in var_info.items():
                source = graph.ndata[info["source"]]
                value = source[:, info["index"]:info["index"] + 1].double()  # Convert to float64
                mean_value = value.mean()
                stats[var_name] += mean_value
                stats[f"{var_name}_meansqr"] += (value ** 2).mean()

        # Compute mean and std
        for var_name in var_info.keys():
            stats[var_name] /= total_steps
            stats[f"{var_name}_meansqr"] /= total_steps
            mean = stats[var_name]
            meansqr = stats[f"{var_name}_meansqr"]
            variance = meansqr - mean ** 2
            # Handle potential small negative variance due to numerical errors
            variance = torch.clamp(variance, min=0.0)
            stats[f"{var_name}_std"] = torch.sqrt(variance)

            # Remove intermediate meansqr stats
            del stats[f"{var_name}_meansqr"]

        return stats

    def _get_edge_stats(self, var_info):
        """
        Computes statistics for edge data based on the provided variable information.
        """
        # Initialize stats dictionary with float64 tensors
        stats = {key: torch.zeros(1, dtype=torch.float64) for key in var_info.keys()}
        meansqr_keys = [f"{key}_meansqr" for key in var_info.keys()]
        stats.update({key: torch.zeros(1, dtype=torch.float64) for key in meansqr_keys})

        # Use the first graph to compute edge stats (edges are static)
        graph = self.__getitem__(0)
        for var_name, info in var_info.items():
            source = graph.edata[info["source"]]
            value = source[:, info["index"]:info["index"] + 1].double()  # Convert to float64
            mean_value = value.mean()
            stats[var_name] += mean_value
            stats[f"{var_name}_meansqr"] += (value ** 2).mean()

        # Compute mean and std
        for var_name in var_info.keys():
            mean = stats[var_name]
            meansqr = stats[f"{var_name}_meansqr"]
            variance = meansqr - mean ** 2
            variance = torch.clamp(variance, min=0.0)
            stats[f"{var_name}_std"] = torch.sqrt(variance)

            # Remove intermediate meansqr stats
            del stats[f"{var_name}_meansqr"]

        return stats


    
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
    var_list = {k: v.numpy().tolist() for k, v in var.items()}
    with open(path+'/'+file_name, "w") as f:
        json.dump(var_list, f)


def load_json(file):
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
    var = {k: torch.tensor(v, dtype=torch.float) for k, v in var_list.items()}
    return var


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
                # Save normalization statistics
                save_json(self.node_stats, ckpt_path, "node_stats.json")
                save_json(self.edge_stats, ckpt_path, "edge_stats.json")
                # Normalize node and edge data
                self._normalize_data(self.node_stats, self.edge_stats, self.node_var_info, self.edge_var_info)
            else:
                print("Loading normalization statistics...")
                self.node_stats = load_json(f"{ckpt_path}/node_stats.json")
                self.edge_stats = load_json(f"{ckpt_path}/edge_stats.json")
                self._normalize_data(self.node_stats, self.edge_stats, self.node_var_info, self.edge_var_info)

    def _normalize_data(self, node_stats, edge_stats, node_var_info, edge_var_info):
        """Normalize node and edge data in all graphs based on computed statistics."""
        # Normalize static node features
        for var_name, info in node_var_info.items():
            if var_name in ["strickler", "z"]:
                mean = node_stats[var_name].item()
                std = node_stats[f"{var_name}_std"].item()
                if std != 0.0:
                    data_tensor = self.base_graph.ndata['static'][:, info['index']:info['index']+1]
                    self.base_graph.ndata['static'][:, info['index']:info['index']+1] = (data_tensor - mean) / std

        # Normalize edge features
        for var_name, info in edge_var_info.items():
            mean = edge_stats[var_name].item()
            std = edge_stats[f"{var_name}_std"].item()
            if std != 0.0:
                data_tensor = self.base_graph.edata[info["source"]][:, info['index']:info['index']+1]
                self.base_graph.edata[info["source"]][:, info['index']:info['index']+1] = (data_tensor - mean) / std

        # Normalize dynamic node features and targets
        for seq_index, sequence in enumerate(self.sequences):
            normalized_sequence = []
            for x, y in sequence:
                x = x.copy()
                y = y.copy()
                # Normalize node features (h, u, v)
                for var_name, info in node_var_info.items():
                    if var_name in ["h", "u", "v"]:
                        mean = node_stats[var_name].item()
                        std = node_stats[f"{var_name}_std"].item()
                        if std != 0.0:
                            dynamic_index = info['index'] 
                            x[:, dynamic_index:dynamic_index+1] = (x[:, dynamic_index:dynamic_index+1] - mean) / std

                # Normalize target variables (delta_h, delta_u, delta_v)
                for var_name, info in node_var_info.items():
                    if var_name in ["delta_h", "delta_u", "delta_v"]:
                        y_index = info['index']
                        mean = node_stats[var_name].item()
                        std = node_stats[f"{var_name}_std"].item()
                        if std != 0.0:
                            y[:, y_index:y_index+1] = (y[:, y_index:y_index+1] - mean) / std

                normalized_sequence.append((x, y))
            self.sequences[seq_index] = normalized_sequence

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        graphs = []
        for x, y in sequence:
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
        """
        Compute statistics (mean and std) for node variables.
        """
        # Initialize stats dictionary with float64 tensors
        stats = {key: torch.zeros(1, dtype=torch.float64) for key in var_info.keys()}
        meansqr_stats = {f"{key}_meansqr": torch.zeros(1, dtype=torch.float64) for key in var_info.keys()}

        total_steps = 0

        for sequence in self.sequences:
            for x, y in sequence:
                total_steps += 1
                # Combine static and dynamic features
                static_features = self.base_graph.ndata['static']
                combined_features = torch.cat((static_features, torch.tensor(x, dtype=torch.float32)), dim=1)
                # Create a temporary graph for this step
                g = self.base_graph.clone()
                g.ndata.pop('static')
                g.ndata['x'] = combined_features
                g.ndata['y'] = torch.tensor(y, dtype=torch.float32)

                for var_name, info in var_info.items():
                    source = g.ndata[info["source"]]
                    value = source[:, info['index']:info['index']+1].double()
                    mean_value = value.mean()
                    stats[var_name] += mean_value
                    meansqr_stats[f"{var_name}_meansqr"] += (value ** 2).mean()

        # Compute mean and std
        for var_name in var_info.keys():
            stats[var_name] /= total_steps
            meansqr_stats[f"{var_name}_meansqr"] /= total_steps
            mean = stats[var_name]
            meansqr = meansqr_stats[f"{var_name}_meansqr"]
            variance = meansqr - mean ** 2
            variance = torch.clamp(variance, min=0.0)
            stats[f"{var_name}_std"] = torch.sqrt(variance)

            # Remove intermediate meansqr stats
            del meansqr_stats[f"{var_name}_meansqr"]

        return stats

    def _get_edge_stats(self, var_info):
        """
        Compute statistics (mean and std) for edge variables.
        """
        # Initialize stats dictionary with float64 tensors
        stats = {key: torch.zeros(1, dtype=torch.float64) for key in var_info.keys()}
        meansqr_stats = {f"{key}_meansqr": torch.zeros(1, dtype=torch.float64) for key in var_info.keys()}

        # Use the first graph to compute edge stats (edges are static)
        graph = self.__getitem__(0)[0]  # Get the first graph in the first sequence
        for var_name, info in var_info.items():
            source = graph.edata[info["source"]]
            value = source[:, info["index"]:info["index"]+1].double()  # Convert to float64
            mean_value = value.mean()
            stats[var_name] += mean_value
            meansqr_stats[f"{var_name}_meansqr"] += (value ** 2).mean()

        # Compute mean and std
        for var_name in var_info.keys():
            mean = stats[var_name]
            meansqr = meansqr_stats[f"{var_name}_meansqr"]
            variance = meansqr - mean ** 2
            variance = torch.clamp(variance, min=0.0)
            stats[f"{var_name}_std"] = torch.sqrt(variance)

            # Remove intermediate meansqr stats
            del meansqr_stats[f"{var_name}_meansqr"]

        return stats
