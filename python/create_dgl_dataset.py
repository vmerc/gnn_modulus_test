import numpy as np 
import enum
import os 
import torch
import dgl 
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
    
    bnd_one_hot_dict = {'[2, 2, 2]':np.array([0,0,0,3]),
                     '[5, 4, 4]':np.array([0,1,0,0]),
                      '[4, 5, 5]':np.array([0,0,2,0]),
                      }
    # on crée un one hot vectot de la taille de tous les pts ou tous les points sont normaux
    output = np.zeros((nb_points,NodeType.SIZE))
    output[:,0] = 0
    
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


def get_node_features(res,res_mesh,timestep):
    tri = res.tri
    bnd_info = res.get_bnd_info()
    node_type = extract_node_type(tri,bnd_info)
    
    huv = extract_h_u_v(res,timestep)
    
    cf = extract_coeff(res_mesh,0)
    
    z = extract_fond(res_mesh,0)
    
    return np.concatenate([node_type,huv,cf,z],axis=1).astype('float32')

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


def get_node_outputs(res,dt,timestep):
    """
    Contains the fluid acceleration between the current graph and the graph 
    in the next time step. These are the features used for training: y=(v_t_next-v_t_curr)/dt [num_nodes x 2]

    Returns:
        _type_: _description_
    """
    result = (extract_h_u_v(res,timestep+1)-extract_h_u_v(res,timestep))/dt
    
    return result



def put_boundary_infos(x,x_future):
    """
    Args:
        x (_type_): _description_
        y (_type_): _description_
    """
    #Q imposé
    x[(x[:,:4] == [0,0,2,0]).all(axis=1),4:7] = x_future[(x[:,:4] == [0,0,2,0]).all(axis=1),4:7]
    #print((x[:,:4] == [0,0,2,0]).all(axis=1).sum())
    # on impose hauteur à l'instant t+1 
    x[(x[:,:4] == [0,1,0,0]).all(axis=1),4:5] = x_future[(x[:,:4] == [0,1,0,0]).all(axis=1),4:5]
    #print((x[:,:4] == [0,1,0,0]).all(axis=1).sum())
    return x


def get_dgl_graph(tri):
    """
    Create a DGL graph from the triangulation information.
    """
    coo_edges = get_edge_index(tri)  # get connectivity
    g = dgl.graph((coo_edges[0], coo_edges[1]))  # Create a DGL graph
    return g

def add_features_to_graph(g, node_features, edge_features):
    """
    Add node and edge features to the DGL graph.
    """
    g.ndata['x'] = torch.tensor(node_features, dtype=torch.float32)  # Add node features
    g.edata['x'] = torch.tensor(edge_features, dtype=torch.float32)  # Add edge features


def create_dgl_dataset(mesh_list, res_list, cli_list, dt_list, data_folder, dataset_name):
    """
    Generate a DGL dataset for all the results in res_list.
    """
    # Define the list that will store the DGL graphs
    graph_list = []
    assert(len(mesh_list) == len(res_list))
    assert(len(dt_list) == len(res_list))
    assert(len(cli_list) == len(res_list))
    number_trajectories = len(mesh_list)

    for traj in range(number_trajectories):
        mesh_path = mesh_list[traj]
        res_path = res_list[traj]
        cli_path = cli_list[traj]
        dt = dt_list[traj]
        
        res = TelemacFile(res_path, bnd_file=cli_path)
        res_mesh = TelemacFile(mesh_path)
        
        number_ts = int(res.times.shape[0])
        #local_graphs = []

        for ts in range(number_ts - 1):
            # Create DGL graph
            g = get_dgl_graph(res.tri)

            # Get node and edge features
            x = get_node_features(res, res_mesh, ts)
            edge_features = get_edges_features(res.tri, g.edges(), res_mesh)

            # Get outputs for training
            y = get_node_outputs(res, dt, ts)
            
            x_future = get_node_features(res,res_mesh,ts+1)
            x = put_boundary_infos(x,x_future)
            
            add_features_to_graph(g, x, edge_features)
            
            g.ndata['y'] = torch.tensor(y, dtype=torch.float32)

            graph_list.append(g)

        #graph_list.append(local_graphs)

    # Save the list of DGL graphs
    dgl.save_graphs(os.path.join(data_folder, dataset_name + '.bin'), graph_list)
    return True


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
        self.split = split
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.length = num_samples * (num_steps - 1)
        self.graphs,_ =  dgl.load_graphs(data_dir)
        
        # Define dictionaries for nodes and edges
        self.node_var_info = {
            "h": {"source": "x", "index": 4},
            "u": {"source": "x", "index": 5},
            "v": {"source": "x", "index": 6},
            "strickler": {"source": "x", "index": 7},
            "z": {"source": "x", "index": 8},
            "delta_h": {"source": "y", "index": 0},
            "delta_u": {"source": "y", "index": 1},
            "delta_v": {"source": "y", "index": 2},
        }
        self.edge_var_info = {
            "xrel": {"source": "x", "index": 0},
            "yrel": {"source": "x", "index": 1},
            "norm": {"source": "x", "index": 3},
        }
        
        
        if normalize:
            # Calculate statistics
            node_stats = self._get_node_stats(self.node_var_info)
            edge_stats = self._get_edge_stats(self.edge_var_info)
            print(node_stats)
            #print(edge_stats)
            # Normalize node and edge data
            self._normalize_data(node_stats, edge_stats, self.node_var_info, self.edge_var_info)

    def _normalize_data(self, node_stats, edge_stats, node_var_info, edge_var_info):
        """Normalize node and edge data in all graphs based on computed statistics."""
        for graph in self.graphs:
            # Normalize node data
            for var_name, info in node_var_info.items():
                mean = node_stats[var_name].item()
                std = node_stats[f"{var_name}_std"].item()
                if std != 0.0 :
                    data_tensor = graph.ndata[info["source"]][:, info["index"]:info["index"]+1]
                    graph.ndata[info["source"]][:, info["index"]:info["index"]+1] = (data_tensor - mean) / std
            
            # Normalize edge data
            for var_name, info in edge_var_info.items():
                mean = edge_stats[var_name].item()
                std = edge_stats[f"{var_name}_std"].item()
                if std != 0.0 :
                    data_tensor = graph.edata[info["source"]][:, info["index"]:info["index"]+1]
                    graph.edata[info["source"]][:, info["index"]:info["index"]+1] = (data_tensor - mean) / std


    def __getitem__(self, idx):
        gidx = idx // (self.num_steps - 1)  # graph index
        tidx = idx % (self.num_steps - 1)  # time step index
        graph = self.graphs[idx]
        return graph

    def __len__(self):
        return self.length 
    
    
    def _get_node_stats(self, var_info):
        """
        var_info: A dictionary containing information about variables. Example format:
        {
            "h": {"source": "x", "index": 4},
            "u": {"source": "x", "index": 5},
            "v": {"source": "x", "index": 6},
            "strickler": {"source": "x", "index": 7},
            "h_diff": {"source": "y", "index": 0},
            "u_diff": {"source": "y", "index": 1},
            "v_diff": {"source": "y", "index": 2}
        }
        """

        # Initialize stats dictionary
        stats = {key: torch.zeros(1, dtype=torch.float32) for key in var_info.keys()}
        meansqr_keys = [f"{key}_meansqr" for key in var_info.keys()]
        stats.update({key: torch.zeros(1, dtype=torch.float32) for key in meansqr_keys})

        for i in range(self.length):
            graph = self.graphs[i]
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

        for i in range(self.length):
            graph = self.graphs[i]
            for var_name, info in var_info.items():
                source = graph.edata[info["source"]]
                value = source[:, info["index"]:info["index"]+1]

                stats[var_name] += torch.mean(value)
                stats[f"{var_name}_meansqr"] += torch.mean(value ** 2)

        # Finalize stats by computing mean, mean square, and standard deviation
        for var_name in var_info.keys():
            stats[var_name] /= self.length
            stats[f"{var_name}_meansqr"] /= self.length
            mean = stats[var_name]
            meansqr = stats[f"{var_name}_meansqr"]
            stats[f"{var_name}_std"] = torch.sqrt(meansqr - mean ** 2)

            # Cleanup intermediate meansqr stats
            del stats[f"{var_name}_meansqr"]

        return stats

    
