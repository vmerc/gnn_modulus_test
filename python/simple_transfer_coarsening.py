#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components


NORMAL = 0
PRESCRIBED_H = 1
PRESCRIBED_Q = 2
WALL_BOUNDARY = 3


@dataclass
class CoarseMesh:
    cluster: np.ndarray
    xy: np.ndarray
    z: np.ndarray
    strickler: np.ndarray | None
    representative_fine_node: np.ndarray
    undirected_edges: np.ndarray
    region_id: np.ndarray
    coarse_region_id: np.ndarray
    boundary_type: np.ndarray
    restriction: csr_matrix
    topo_barrier_node_mask: np.ndarray
    preserve_node_mask: np.ndarray


def unique_edges_from_triangles(triangles: np.ndarray) -> np.ndarray:
    """Extract unique undirected edges from fine triangular connectivity."""
    tri = np.asarray(triangles, dtype=np.int64)
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def expand_node_mask(mask: np.ndarray, edges: np.ndarray, n_iter: int = 1) -> np.ndarray:
    """Expand a node mask by fine-mesh one-ring neighborhoods."""
    expanded = np.asarray(mask, dtype=bool).copy()
    for _ in range(n_iter):
        touched = expanded[edges[:, 0]] | expanded[edges[:, 1]]
        expanded[edges[touched].ravel()] = True
    return expanded


def sharp_topography_edges(
    xy: np.ndarray,
    edges: np.ndarray,
    z: np.ndarray,
    dz_min: float | None = None,
    slope_min: float | None = None,
    dz_quantile: float = 0.98,
    slope_quantile: float = 0.98,
    dz_floor: float = 0.5,
    slope_floor: float = 0.05,
) -> tuple[np.ndarray, float, float]:
    """Detect fine edges with strong local bed-elevation jumps."""
    u = edges[:, 0]
    v = edges[:, 1]
    length = np.linalg.norm(xy[v] - xy[u], axis=1)
    dz = np.abs(z[v] - z[u])
    slope = dz / np.maximum(length, 1e-12)

    if dz_min is None:
        dz_min = max(dz_floor, float(np.quantile(dz, dz_quantile)))
    if slope_min is None:
        slope_min = max(slope_floor, float(np.quantile(slope, slope_quantile)))

    return (dz >= dz_min) & (slope >= slope_min), dz_min, slope_min


def get_topo_barrier_node_mask(
    xy: np.ndarray,
    edges: np.ndarray,
    z: np.ndarray,
    dz_min: float | None = None,
    slope_min: float | None = None,
    expand: int = 1,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Approximate dikes/levees/topographic breaks from strong elevation jumps.

    Returns
    -------
    topo_barrier_node_mask:
        Fine nodes close to strong topographic jumps.
    topo_edge_mask:
        Fine edges where the topographic jump criterion is active.
    used_dz_min:
        Effective dz threshold.
    used_slope_min:
        Effective slope threshold.
    """
    topo_edge_mask, used_dz_min, used_slope_min = sharp_topography_edges(
        xy=xy,
        edges=edges,
        z=z,
        dz_min=dz_min,
        slope_min=slope_min,
    )

    topo_barrier_node_mask = np.zeros(len(xy), dtype=bool)
    topo_barrier_node_mask[edges[topo_edge_mask].ravel()] = True
    topo_barrier_node_mask = expand_node_mask(topo_barrier_node_mask, edges, expand)

    return topo_barrier_node_mask, topo_edge_mask, used_dz_min, used_slope_min


def get_preserve_node_mask(
    boundary_type: np.ndarray,
    topo_barrier_node_mask: np.ndarray,
) -> np.ndarray:
    """
    Mark fine nodes that must stay isolated during coarse point construction.

    We preserve:
    - Telemac boundary nodes,
    - strong topographic-break nodes.
    """
    boundary_type = np.asarray(boundary_type, dtype=np.int64)
    return (boundary_type != NORMAL) | topo_barrier_node_mask


def connected_regions_after_cuts(
    n_nodes: int,
    edges: np.ndarray,
    cut_edge_mask: np.ndarray,
) -> np.ndarray:
    """Connected components after virtually removing cut fine edges."""
    kept_edges = edges[~cut_edge_mask]
    row = kept_edges[:, 0]
    col = kept_edges[:, 1]
    data = np.ones(2 * len(row), dtype=np.float64)
    adj = coo_matrix((data, (np.r_[row, col], np.r_[col, row])), shape=(n_nodes, n_nodes))
    _, labels = connected_components(adj, directed=False)
    return labels.astype(np.int64)


def constrained_grid_clusters(
    xy: np.ndarray,
    spacing: float,
    region_id: np.ndarray,
    boundary_type: np.ndarray,
    preserve_node_mask: np.ndarray,
) -> np.ndarray:
    """Grid clustering constrained by region and boundary type."""
    origin = xy.min(axis=0)
    cell = np.floor((xy - origin) / spacing).astype(np.int64)

    keys = np.column_stack([cell[:, 0], cell[:, 1], region_id, boundary_type])
    _, cluster = np.unique(keys, axis=0, return_inverse=True)
    cluster = cluster.astype(np.int64)

    if preserve_node_mask.any():
        first_singleton = cluster.max() + 1
        singleton_ids = np.arange(
            first_singleton,
            first_singleton + preserve_node_mask.sum(),
            dtype=np.int64,
        )
        cluster[preserve_node_mask] = singleton_ids

    _, cluster = np.unique(cluster, return_inverse=True)
    return cluster.astype(np.int64)


def build_restriction_matrix(
    cluster: np.ndarray,
    weights: np.ndarray | None = None,
) -> csr_matrix:
    """Sparse matrix P such that coarse_values = P @ fine_values."""
    cluster = np.asarray(cluster, dtype=np.int64)
    n_fine = len(cluster)
    n_coarse = int(cluster.max()) + 1

    if weights is None:
        weights = np.ones(n_fine, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    denom = np.bincount(cluster, weights=weights, minlength=n_coarse)
    data = weights / denom[cluster]
    rows = cluster
    cols = np.arange(n_fine)
    return csr_matrix((data, (rows, cols)), shape=(n_coarse, n_fine))


def coarse_label_from_cluster(cluster: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Assign each coarse point the majority fine label of its cluster."""
    n_coarse = int(cluster.max()) + 1
    coarse_labels = np.zeros(n_coarse, dtype=np.int64)

    for coarse_id in range(n_coarse):
        values = labels[cluster == coarse_id]
        unique, counts = np.unique(values, return_counts=True)
        coarse_labels[coarse_id] = unique[np.argmax(counts)]

    return coarse_labels


def coarse_edges_from_fine_edges(fine_edges: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    """
    Contract fine edges into coarse edges.

    For each fine edge (i, j), let ci = cluster[i] and cj = cluster[j].
    If ci != cj, add the undirected coarse edge (ci, cj).
    """
    coarse_edges = np.column_stack([cluster[fine_edges[:, 0]], cluster[fine_edges[:, 1]]])
    coarse_edges = coarse_edges[coarse_edges[:, 0] != coarse_edges[:, 1]]
    if len(coarse_edges) == 0:
        return np.empty((0, 2), dtype=np.int64)
    coarse_edges = np.sort(coarse_edges, axis=1)
    return np.unique(coarse_edges, axis=0).astype(np.int64)


def representative_fine_nodes_from_clusters(
    xy: np.ndarray,
    cluster: np.ndarray,
    centroid_xy: np.ndarray,
) -> np.ndarray:
    """
    Pick one fine node per cluster: the node closest to the cluster centroid.

    The centroid can be weighted through the restriction matrix used upstream.
    The returned coordinates stay exactly on the fine mesh, which gives a
    geometrically cleaner support than raw barycenters.
    """
    xy = np.asarray(xy, dtype=np.float64)
    cluster = np.asarray(cluster, dtype=np.int64)
    centroid_xy = np.asarray(centroid_xy, dtype=np.float64)

    n_coarse = int(cluster.max()) + 1
    if len(centroid_xy) != n_coarse:
        raise ValueError("centroid_xy must contain one centroid per cluster")

    order = np.argsort(cluster, kind="stable")
    sorted_cluster = cluster[order]
    split_idx = np.flatnonzero(np.diff(sorted_cluster)) + 1
    groups = np.split(order, split_idx)

    representative = np.empty(n_coarse, dtype=np.int64)
    for coarse_id, members in enumerate(groups):
        pts = xy[members]
        center = centroid_xy[coarse_id]
        dist2 = np.sum((pts - center) ** 2, axis=1)
        representative[coarse_id] = members[np.argmin(dist2)]

    return representative


def node_type_one_hot(boundary_type: np.ndarray, size: int = 4) -> np.ndarray:
    """Convert integer node-type codes into one-hot vectors."""
    boundary_type = np.asarray(boundary_type, dtype=np.int64)
    if boundary_type.ndim != 1:
        raise ValueError("boundary_type must be a 1D array")
    if boundary_type.size == 0:
        return np.zeros((0, size), dtype=np.float32)
    if boundary_type.min() < 0 or boundary_type.max() >= size:
        raise ValueError(f"boundary_type values must lie in [0, {size - 1}]")

    one_hot = np.zeros((len(boundary_type), size), dtype=np.float32)
    one_hot[np.arange(len(boundary_type)), boundary_type] = 1.0
    return one_hot


def directed_edges_and_features(
    xy: np.ndarray,
    undirected_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build directed edges and edge features compatible with create_dgl_dataset.

    Edge features follow the existing convention:
    [dx, dy, norm] with dx, dy computed as pos[src] - pos[dst].
    """
    xy = np.asarray(xy, dtype=np.float64)
    undirected_edges = np.asarray(undirected_edges, dtype=np.int64)

    if len(undirected_edges) == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, 3), dtype=np.float32),
        )

    directed_edges = np.vstack([undirected_edges, undirected_edges[:, ::-1]]).astype(np.int64)
    src = directed_edges[:, 0]
    dst = directed_edges[:, 1]
    delta = xy[src] - xy[dst]
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    edge_features = np.concatenate([delta, norm], axis=1).astype(np.float32)
    return directed_edges, edge_features


def build_coarse_dgl_graph(coarse: CoarseMesh):
    """
    Convert the coarse mesh into a DGL base graph saved later as `.bin`.

    The node and edge feature names match the conventions already used in
    python/create_dgl_dataset.py.
    """
    try:
        import dgl
        import torch
    except ImportError as exc:
        raise ImportError(
            "build_coarse_dgl_graph requires both dgl and torch to be installed."
        ) from exc

    directed_edges, edge_features = directed_edges_and_features(
        xy=coarse.xy,
        undirected_edges=coarse.undirected_edges,
    )

    src = torch.as_tensor(directed_edges[:, 0], dtype=torch.int64)
    dst = torch.as_tensor(directed_edges[:, 1], dtype=torch.int64)
    graph = dgl.graph((src, dst), num_nodes=len(coarse.xy))

    if coarse.strickler is None:
        strickler = np.zeros(len(coarse.xy), dtype=np.float32)
    else:
        strickler = np.asarray(coarse.strickler, dtype=np.float32)

    static = np.concatenate(
        [
            node_type_one_hot(coarse.boundary_type),
            strickler[:, None],
            np.asarray(coarse.z, dtype=np.float32)[:, None],
        ],
        axis=1,
    )

    graph.edata["x"] = torch.as_tensor(edge_features, dtype=torch.float32)
    graph.ndata["static"] = torch.as_tensor(static, dtype=torch.float32)
    graph.ndata["pos"] = torch.as_tensor(coarse.xy, dtype=torch.float32)
    graph.ndata["boundary_type"] = torch.as_tensor(coarse.boundary_type, dtype=torch.int64)
    graph.ndata["region_id"] = torch.as_tensor(coarse.coarse_region_id, dtype=torch.int64)
    graph.ndata["representative_fine_node"] = torch.as_tensor(
        coarse.representative_fine_node,
        dtype=torch.int64,
    )
    return graph


def write_coarse_bin(
    output_path: str | Path,
    coarse: CoarseMesh,
    overwrite: bool = False,
) -> Path:
    """
    Save the coarse mesh as a DGL `.bin` file.

    This writes a single base graph, in the same spirit as the
    `*_base.bin` files created by create_dgl_dataset.
    """
    try:
        import dgl
    except ImportError as exc:
        raise ImportError("write_coarse_bin requires dgl to be installed.") from exc

    output_path = Path(output_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".bin")
    if output_path.suffix != ".bin":
        raise ValueError("output_path must end with .bin")

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use overwrite=True.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = build_coarse_dgl_graph(coarse)
    dgl.save_graphs(str(output_path), [graph])
    return output_path


def restrict_time_series(P: csr_matrix, values: np.ndarray) -> np.ndarray:
    """
    Restrict fine time series to the coarse points.

    values shape: (T, N, F)
    output shape: (T, C, F)
    """
    t_size, n_nodes, n_features = values.shape
    if n_nodes != P.shape[1]:
        raise ValueError(f"Expected {P.shape[1]} fine nodes, got {n_nodes}")

    flat = values.transpose(0, 2, 1).reshape(t_size * n_features, n_nodes)
    coarse = flat @ P.T
    return coarse.reshape(t_size, n_features, P.shape[0]).transpose(0, 2, 1)


def build_simple_coarse_mesh(
    xy: np.ndarray,
    triangles: np.ndarray,
    z: np.ndarray,
    boundary_type: np.ndarray | None = None,
    strickler: np.ndarray | None = None,
    spacing: float = 80.0,
    preserve_expand: int = 1,
    dz_min: float | None = None,
    slope_min: float | None = None,
    weights: np.ndarray | None = None,
) -> CoarseMesh:
    """
    Build only the coarse points, without reconstructing coarse edges.

    boundary_type follows the existing dataset convention:
    0 = normal, 1 = prescribed H, 2 = prescribed Q, 3 = wall.

    The method keeps:
    - Telemac boundary nodes,
    - strong topographic jumps as a bank/dike/levee proxy,
    - coarse clustering elsewhere.
    """
    xy = np.asarray(xy, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    z = np.asarray(z, dtype=np.float64)

    if boundary_type is None:
        boundary_type = np.zeros(len(xy), dtype=np.int64)
    else:
        boundary_type = np.asarray(boundary_type, dtype=np.int64)

    fine_edges = unique_edges_from_triangles(triangles)

    (
        topo_barrier_node_mask,
        topo_edge_mask,
        used_dz_min,
        used_slope_min,
    ) = get_topo_barrier_node_mask(
        xy=xy,
        edges=fine_edges,
        z=z,
        dz_min=dz_min,
        slope_min=slope_min,
        expand=preserve_expand,
    )

    u = fine_edges[:, 0]
    v = fine_edges[:, 1]
    boundary_change_edge_mask = boundary_type[u] != boundary_type[v]
    cut_edge_mask = topo_edge_mask | boundary_change_edge_mask

    region_id = connected_regions_after_cuts(
        n_nodes=len(xy),
        edges=fine_edges,
        cut_edge_mask=cut_edge_mask,
    )

    preserve_node_mask = get_preserve_node_mask(
        boundary_type=boundary_type,
        topo_barrier_node_mask=topo_barrier_node_mask,
    )

    cluster = constrained_grid_clusters(
        xy=xy,
        spacing=spacing,
        region_id=region_id,
        boundary_type=boundary_type,
        preserve_node_mask=preserve_node_mask,
    )

    restriction = build_restriction_matrix(cluster, weights=weights)
    centroid_xy = np.asarray(restriction @ xy)
    representative_fine_node = representative_fine_nodes_from_clusters(
        xy=xy,
        cluster=cluster,
        centroid_xy=centroid_xy,
    )
    coarse_xy = np.asarray(xy[representative_fine_node])
    coarse_z = np.asarray(restriction @ z)
    coarse_strickler = None if strickler is None else np.asarray(restriction @ strickler)
    coarse_undirected_edges = coarse_edges_from_fine_edges(fine_edges, cluster)
    coarse_region_id = coarse_label_from_cluster(cluster, region_id)
    coarse_boundary_type = coarse_label_from_cluster(cluster, boundary_type)

    print(
        "Simple Telemac-style coarse points:",
        f"fine_nodes={len(xy)}",
        f"coarse_nodes={len(coarse_xy)}",
        f"undirected_edges={len(coarse_undirected_edges)}",
        f"spacing={spacing}",
        f"dz_min={used_dz_min:.3g}",
        f"slope_min={used_slope_min:.3g}",
        f"regions={len(np.unique(coarse_region_id))}",
        "geometry=fine_representative_nearest_centroid",
    )

    return CoarseMesh(
        cluster=cluster,
        xy=coarse_xy,
        z=coarse_z,
        strickler=coarse_strickler,
        representative_fine_node=representative_fine_node,
        undirected_edges=coarse_undirected_edges,
        region_id=region_id,
        coarse_region_id=coarse_region_id,
        boundary_type=coarse_boundary_type,
        restriction=restriction,
        topo_barrier_node_mask=topo_barrier_node_mask,
        preserve_node_mask=preserve_node_mask,
    )


def mesh_scale_stats(xy: np.ndarray, undirected_edges: np.ndarray) -> dict[str, float]:
    """Quick diagnostics on coarse edge lengths."""
    if len(undirected_edges) == 0:
        return {
            "nodes": float(len(xy)),
            "undirected_edges": 0.0,
            "mean_edge_m": 0.0,
            "median_edge_m": 0.0,
            "max_edge_m": 0.0,
            "r10_median_m": 0.0,
            "r10_mean_m": 0.0,
            "r10_max_m": 0.0,
        }

    src = undirected_edges[:, 0]
    dst = undirected_edges[:, 1]
    length = np.linalg.norm(xy[dst] - xy[src], axis=1)

    both_src = np.r_[src, dst]
    both_length = np.r_[length, length]
    degree = np.bincount(both_src, minlength=len(xy))
    sum_length = np.bincount(both_src, weights=both_length, minlength=len(xy))
    valid = degree > 0
    mean_out_length = sum_length[valid] / degree[valid]
    r10 = 10.0 * mean_out_length

    return {
        "nodes": float(len(xy)),
        "undirected_edges": float(len(undirected_edges)),
        "mean_edge_m": float(length.mean()),
        "median_edge_m": float(np.median(length)),
        "max_edge_m": float(length.max()),
        "r10_median_m": float(np.median(r10)),
        "r10_mean_m": float(r10.mean()),
        "r10_max_m": float(r10.max()),
    }


if __name__ == "__main__":
    print(
        "Import this module and call build_simple_coarse_mesh(xy, triangles, z, "
        "boundary_type) to build coarse points, inherited coarse edges, and the "
        "restriction matrix. Use write_coarse_bin(...) to export a DGL .bin."
    )
