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

def get_node_outputs(x, x_future, dt):
    """
    Contains the fluid acceleration between the current graph and the graph 
    in the next time step. These are the features used for training: y=(v_t_next-v_t_curr)/dt [num_nodes x 2]

    Returns:
        np.array: array with fluid acceleration features [num_nodes, 3]
    """
    # Indices for h, u, v in the dynamic features
    h_idx = 0
    u_idx = 1
    v_idx = 2
    
    result = (x_future[:, h_idx:v_idx+1] - x[:, h_idx:v_idx+1]) / dt
    
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
        print(number_ts)

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
                y = get_node_outputs(dynamic_node_features, dynamic_node_features_future, dt)
                y = put_boundary_infos_on_changes(y,static_node_features) #put 0 on changes  

                dynamic_data_list.append((dynamic_node_features, y))

            # Save dynamic data for this chunk
            with open(os.path.join(data_folder, f"{dataset_name}_{traj}_{start_ts}-{end_ts}.pkl"), 'wb') as f:
                pickle.dump(dynamic_data_list, f)

    # Save the base graphs separately
    dgl.save_graphs(os.path.join(data_folder, f"{dataset_name}_base.bin"), base_graph_list)
    return True
    
    
    
#################################

class TelemacDataset(DGLDataset):
    """In-memory MeshGraphNet Dataset for stationary mesh
    Notes:
        - This dataset prepares and processes the data available in MeshGraphNet's repo:
            https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
        - A single adj matrix is used for each transient simulation.
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
        force_reload=False,
        verbose=False,
        normalize=True
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
        self.length = num_samples * (num_steps - 1)

        # Load base graph (assuming a single graph)
        self.base_graph, _ = dgl.load_graphs(data_dir)
        self.base_graph = self.base_graph[0]

        # Load dynamic data
        with open(dynamic_data_file, 'rb') as f:
            all_dynamic_data = pickle.load(f)
            self.dynamic_data_list = all_dynamic_data[:self.length]

        
        # Define dictionaries for nodes and edges
        self.node_var_info = {
            "h": {"source": "x", "index": 6},
            "u": {"source": "x", "index": 7},
            "v": {"source": "x", "index": 8},
            
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
            # Calculate statistics
            node_stats = self._get_node_stats(self.node_var_info)
            edge_stats = self._get_edge_stats(self.edge_var_info)
            print(node_stats)
            print(edge_stats)
            # Normalize node and edge data
            self._normalize_data(node_stats, edge_stats, self.node_var_info, self.edge_var_info)

    def _normalize_data(self, node_stats, edge_stats, node_var_info, edge_var_info):
        """Normalize node and edge data in all graphs based on computed statistics."""
        
        # normalize static values 
        for var_name, info in node_var_info.items():
            if var_name in ["strickler","z"]:
                print(var_name)
                mean = node_stats[var_name].item()
                std = node_stats[f"{var_name}_std"].item()
                if std != 0.0 :
                    data_tensor = self.base_graph.ndata['static'][:, info["index"]:info["index"]+1]
                    self.base_graph.ndata['static'][:, info["index"]:info["index"]+1] = (data_tensor - mean) / std
                    
        for var_name, info in edge_var_info.items():
                print(var_name)
                mean = edge_stats[var_name].item()
                std = edge_stats[f"{var_name}_std"].item()
                if std != 0.0 :
                    data_tensor = self.base_graph.edata[info["source"]][:, info["index"]:info["index"]+1]
                    self.base_graph.edata[info["source"]][:, info["index"]:info["index"]+1] = (data_tensor - mean) / std
                    
                
        for index,dynamic_data in enumerate(self.dynamic_data_list):
                x,y = dynamic_data
                for var_name, info in edge_var_info.items():
                    if var_name in ["h","u","v"]:
                        mean = node_stats[var_name].item()
                        std = node_stats[f"{var_name}_std"].item()
                        if std != 0.0 :
                            data_tensor = x[:, info["index"]:info["index"]+1]
                            self.dynamic_data_list[index][0][:, info["index"]:info["index"]+1]= (data_tensor - mean) / std
                            
                    if var_name in ["delta_h","delta_u","delta_v"]:
                        mean = node_stats[var_name].item()
                        std = node_stats[f"{var_name}_std"].item()
                        if std != 0.0 :
                            data_tensor = y[:, info["index"]:info["index"]+1]
                            self.dynamic_data_list[index][1][:, info["index"]:info["index"]+1]= (data_tensor - mean) / std
                            
            
            # Normalize edge data
            


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
        var_info: A dictionary containing information about variables. 
        """

        # Initialize stats dictionary
        stats = {key: torch.zeros(1, dtype=torch.float32) for key in var_info.keys()}
        meansqr_keys = [f"{key}_meansqr" for key in var_info.keys()]
        stats.update({key: torch.zeros(1, dtype=torch.float32) for key in meansqr_keys})

        for i in range(self.length):
            graph = self.__getitem__(i)
            for var_name, info in var_info.items():
                source = graph.ndata[info["source"]]
                value = source[:, info["index"]:info["index"]+1]

                stats[var_name] += torch.mean(value)
                stats[f"{var_name}_meansqr"] += torch.mean(value ** 2)

        # Compute actual means, mean squares, and standard deviations
        for var_name in var_info.keys():
            stats[var_name] /= self.length
            stats[f"{var_name}_meansqr"] /= self.length
            mean = stats[var_name]
            meansqr = stats[f"{var_name}_meansqr"]
            stats[f"{var_name}_std"] = torch.sqrt(meansqr - mean ** 2)

            # Cleanup intermediate meansqr stats
            del stats[f"{var_name}_meansqr"]

        return stats


    def _get_edge_stats(self, var_info):
        """
        Computes statistics for edge data based on the provided variable information.

        Parameters:
        var_info (dict): Information about variables, including their names, storage locations ('x' or 'y'), and indices.
                         Example format:
                         {
                             "x relative pose": {"source": "x", "index": 0},
                             # Add more variables as needed
                         }
        """
        
        # Initialize stats dictionary
        stats = {key: torch.zeros(1, dtype=torch.float32) for key in var_info.keys()}
        meansqr_keys = [f"{key}_meansqr" for key in var_info.keys()]
        stats.update({key: torch.zeros(1, dtype=torch.float32) for key in meansqr_keys})

        
        graph = self.__getitem__(0)
        for var_name, info in var_info.items():
            source = graph.edata[info["source"]]
            value = source[:, info["index"]:info["index"]+1]
            stats[var_name] += torch.mean(value)
            stats[f"{var_name}_meansqr"] += torch.mean(value ** 2)

        # Finalize stats by computing mean, mean square, and standard deviation
        for var_name in var_info.keys():
            mean = stats[var_name]
            meansqr = stats[f"{var_name}_meansqr"]
            stats[f"{var_name}_std"] = torch.sqrt(meansqr - mean ** 2)

            # Cleanup intermediate meansqr stats
            del stats[f"{var_name}_meansqr"]

        return stats
    
