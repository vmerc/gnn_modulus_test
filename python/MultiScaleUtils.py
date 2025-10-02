import dgl
import torch
import numpy as np
import matplotlib.tri as tri

def create_downscale_graph(X_fin, X_coarse, Triangles):
    """
    Construct a directed DGL graph where each fine-mesh node connects to the vertices 
    of the coarse-mesh triangle it belongs to. In the resulting graph, the fine nodes 
    have indices [0, N_fin-1] and the coarse nodes have indices [N_fin, N_fin+N_coarse-1].

    Parameters:
    - X_fin: np.array (N_fin, 2)  
        Coordinates of fine-mesh nodes.
    - X_coarse: np.array (N_coarse, 2)  
        Coordinates of coarse-mesh nodes.
    - Triangles: np.array (ntri, 3)  
        Triangle vertex indices (counterclockwise order) referring to the coarse nodes.

    Returns:
    - graph: DGL graph with N_fin*3 edges.
      Edge features include:
         - relative_position (absolute difference in x and y)
         - distance (Euclidean L2 norm)
    """
    # Triangulation to find which coarse triangle each fine node belongs to.
    triangulation = tri.Triangulation(X_coarse[:, 0], X_coarse[:, 1], Triangles)
    trifinder = triangulation.get_trifinder()
    triangle_indices = trifinder(X_fin[:, 0], X_fin[:, 1])

    # Remove out-of-bounds points (-1 indicates no triangle found)
    valid_mask = triangle_indices >= 0
    valid_fine_indices = np.where(valid_mask)[0]
    valid_triangles = triangle_indices[valid_mask]

    # For each valid fine node, connect it to the 3 vertices of its triangle.
    src_nodes = np.repeat(valid_fine_indices, 3)
    # Triangle vertices are originally indexed for X_coarse. 
    # We shift them by len(X_fin) so that coarse nodes occupy indices [N_fin, ...]
    dst_nodes = Triangles[valid_triangles].flatten() + len(X_fin)

    # Compute absolute relative positions
    # Pour le calcul, on retire le décalage sur les indices de X_coarse.
    rel_positions = np.abs(X_fin[valid_fine_indices].repeat(3, axis=0) - X_coarse[dst_nodes - len(X_fin)])
    
    # Compute Euclidean distances (L2 norm)
    distances = np.linalg.norm(rel_positions, axis=1, keepdims=True)

    # Conversion en tenseurs PyTorch
    src_nodes = torch.from_numpy(src_nodes).long()
    dst_nodes = torch.from_numpy(dst_nodes).long()
    
    
    rel_positions = torch.from_numpy(rel_positions).float()
    distances = torch.from_numpy(distances).float()

    # Création du graphe dirigé
    num_nodes = len(X_fin) + len(X_coarse)
    
    
    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)

    # Ajout des features d'arêtes : concaténation de la position relative et de la distance
    graph.edata["x"] = torch.cat((rel_positions, distances), axis=1)  # Shape: (3N, 3)

    return graph


def create_transition_graph(X_1, X_2, Triangles_1):
    """
    Construct a directed DGL graph where each 2-mesh node connects to the vertices 
    of the 1-mesh triangle it belongs to. In the resulting graph, the 1-mesh nodes 
    have indices [0, N_1-1] and the 2-mesh nodes have indices [N_1, N_1+N_2-1].

    Parameters:
    - X_1: np.array (N_1, 2)  
        Coordinates of 1-mesh nodes.
    - X_2: np.array (N_2, 2)  
        Coordinates of 2-mesh nodes.
    - Triangles_1: np.array (ntri, 3)  
        Triangle vertex indices (counterclockwise order) referring to the 1-mesh nodes.

    Returns:
    - graph: DGL graph with N_2*3 edges.
      Edge features include:
         - relative_position (absolute difference in x and y)
         - distance (Euclidean L2 norm)
    """
    # Create a triangulation on the 1-mesh nodes.
    triangulation = tri.Triangulation(X_1[:, 0], X_1[:, 1], Triangles_1)
    trifinder = triangulation.get_trifinder()
    
    # For each 2-mesh node, determine in which triangle (from the 1-mesh) it lies.
    triangle_indices = trifinder(X_2[:, 0], X_2[:, 1])
    
    # Keep only the valid points (triangle index not equal to -1)
    valid_mask = triangle_indices >= 0
    valid_fine_indices = np.where(valid_mask)[0]
    valid_triangles = triangle_indices[valid_mask]
    
    # For each valid 2-mesh node, connect it to the 3 vertices of its triangle.
    # Source nodes: 2-mesh nodes, but re-indexed by adding len(X_1)
    src_nodes = np.repeat(valid_fine_indices, 3) + len(X_1)
    
    # Destination nodes: vertices of the triangle (they are indices in X_1, so no shift)
    dst_nodes = Triangles_1[valid_triangles].flatten()
    
    # Compute the relative position (absolute difference) between each 2-mesh node and 
    # each of the triangle vertices it connects to.
    rel_positions = np.abs(X_2[valid_fine_indices].repeat(3, axis=0) - X_1[dst_nodes])
    
    # Compute the Euclidean distance (L2 norm)
    distances = np.linalg.norm(rel_positions, axis=1, keepdims=True)
    
    # Convert all arrays to PyTorch tensors.
    src_nodes = torch.from_numpy(src_nodes).long()
    dst_nodes = torch.from_numpy(dst_nodes).long()
    rel_positions = torch.from_numpy(rel_positions).float()
    distances = torch.from_numpy(distances).float()
    
    # Total number of nodes: 1-mesh nodes + 2-mesh nodes.
    num_nodes = len(X_1) + len(X_2)
    
    # Create the directed graph using DGL.
    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
    
    # Attach edge features: concatenation of the relative positions and the distances.
    graph.edata["x"] = torch.cat((rel_positions, distances), axis=1)  # Shape: (3*N_valid, 3)
    
    return graph
