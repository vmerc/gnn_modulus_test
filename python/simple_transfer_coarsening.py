#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, QhullError


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
    triangles: np.ndarray
    undirected_edges: np.ndarray
    region_id: np.ndarray
    coarse_region_id: np.ndarray
    boundary_type: np.ndarray
    restriction: csr_matrix
    topo_barrier_node_mask: np.ndarray
    preserve_node_mask: np.ndarray


def unique_edges_from_triangles(triangles: np.ndarray) -> np.ndarray:
    """Extract unique undirected edges from triangular connectivity."""
    tri = np.asarray(triangles, dtype=np.int64)
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def expand_node_mask(mask: np.ndarray, edges: np.ndarray, n_iter: int = 1) -> np.ndarray:
    """Expand a node mask by mesh one-ring neighborhoods."""
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
    """Detect edges with strong local bed-elevation jumps."""
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
        Nodes close to strong topographic jumps.
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
    Mark nodes that should not be merged during coarsening.

    We preserve all boundary nodes and strong topographic breaks.
    Preserved nodes become singleton clusters.
    """
    boundary_type = np.asarray(boundary_type, dtype=np.int64)
    return (boundary_type != NORMAL) | topo_barrier_node_mask


def connected_regions_after_cuts(
    n_nodes: int,
    edges: np.ndarray,
    cut_edge_mask: np.ndarray,
) -> np.ndarray:
    """Connected components after virtually removing cut edges."""
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
    """Grid clustering constrained by hydraulic region and boundary type."""
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
    """Sparse matrix P such that coarse = P @ fine."""
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


def coarse_edges_from_fine_edges(fine_edges: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    """Collapse fine edges into unique undirected coarse edges."""
    coarse_edges = np.column_stack([cluster[fine_edges[:, 0]], cluster[fine_edges[:, 1]]])
    coarse_edges = coarse_edges[coarse_edges[:, 0] != coarse_edges[:, 1]]
    coarse_edges = np.sort(coarse_edges, axis=1)
    return np.unique(coarse_edges, axis=0)


def coarse_label_from_cluster(cluster: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Assign each coarse node the majority fine label in its cluster."""
    n_coarse = int(cluster.max()) + 1
    coarse_labels = np.zeros(n_coarse, dtype=np.int64)

    for coarse_id in range(n_coarse):
        values = labels[cluster == coarse_id]
        unique, counts = np.unique(values, return_counts=True)
        coarse_labels[coarse_id] = unique[np.argmax(counts)]

    return coarse_labels


def min_positive_edge_length(xy: np.ndarray, edges: np.ndarray) -> float | None:
    """Return the smallest strictly positive edge length, or None if absent."""
    if len(edges) == 0:
        return None

    src = edges[:, 0]
    dst = edges[:, 1]
    length = np.linalg.norm(np.asarray(xy)[dst] - np.asarray(xy)[src], axis=1)
    positive = length[length > 0.0]
    if len(positive) == 0:
        return None
    return float(positive.min())


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    """Remap integer labels to a compact 0..K-1 range."""
    _, compact = np.unique(np.asarray(labels, dtype=np.int64), return_inverse=True)
    return compact.astype(np.int64)


def merge_components_from_edges(n_nodes: int, edges: np.ndarray) -> np.ndarray:
    """
    Merge nodes connected by the provided undirected edges.

    Returns a compact label per input node describing the merged components.
    """
    if len(edges) == 0:
        return np.arange(n_nodes, dtype=np.int64)

    row = edges[:, 0]
    col = edges[:, 1]
    data = np.ones(2 * len(row), dtype=np.float64)
    adj = coo_matrix((data, (np.r_[row, col], np.r_[col, row])), shape=(n_nodes, n_nodes))
    _, labels = connected_components(adj, directed=False)
    return _compact_labels(labels)


def float32_duplicate_labels(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Group nodes that collapse to the same coordinates in float32.

    This matches what happens when exporting a mesh to SERAFIN single precision.
    """
    xy32 = np.asarray(xy, dtype=np.float64).astype(np.float32).astype(np.float64)
    _, inverse, counts = np.unique(xy32, axis=0, return_inverse=True, return_counts=True)
    return _compact_labels(inverse), counts


def _delaunay_triangles_for_points(xy: np.ndarray) -> np.ndarray:
    """Run a robust Delaunay triangulation on one point cloud."""
    xy = np.asarray(xy, dtype=np.float64)

    if len(xy) < 3:
        return np.empty((0, 3), dtype=np.int64)

    finite_mask = np.isfinite(xy).all(axis=1)
    valid_nodes = np.flatnonzero(finite_mask)
    valid_xy = xy[valid_nodes]

    if len(valid_xy) < 3:
        return np.empty((0, 3), dtype=np.int64)

    unique_xy, unique_idx = np.unique(valid_xy, axis=0, return_index=True)
    unique_nodes = valid_nodes[unique_idx]

    if len(unique_xy) < 3:
        return np.empty((0, 3), dtype=np.int64)

    center = unique_xy.mean(axis=0)
    scale = np.ptp(unique_xy, axis=0).max()

    if scale <= 0.0:
        return np.empty((0, 3), dtype=np.int64)

    scaled_xy = (unique_xy - center) / scale

    try:
        delaunay = Delaunay(scaled_xy, qhull_options="Qbb Qc Qz Q12 QJ")
    except QhullError as exc:
        raise RuntimeError(
            "Delaunay triangulation failed. Check if coarse points are nearly "
            "collinear or if the mesh contains invalid coordinates."
        ) from exc

    simplices = delaunay.simplices.astype(np.int64)
    simplices = simplices[(simplices < len(unique_xy)).all(axis=1)]

    return unique_nodes[simplices]


def _filter_long_triangles(
    xy: np.ndarray,
    triangles: np.ndarray,
    max_edge_length: float | None,
) -> np.ndarray:
    """Remove triangles with at least one edge longer than the threshold."""
    if max_edge_length is None or len(triangles) == 0:
        return triangles

    p = xy[triangles]
    l01 = np.linalg.norm(p[:, 0] - p[:, 1], axis=1)
    l12 = np.linalg.norm(p[:, 1] - p[:, 2], axis=1)
    l20 = np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
    keep = np.maximum.reduce([l01, l12, l20]) <= max_edge_length

    return triangles[keep]


def filter_degenerate_triangles(
    xy: np.ndarray,
    triangles: np.ndarray,
    min_edge_length: float = 1e-9,
    min_area: float = 1e-12,
) -> np.ndarray:
    """Remove triangles with repeated nodes, near-zero edges, or near-zero area."""
    xy = np.asarray(xy, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)

    if len(triangles) == 0:
        return triangles

    p = xy[triangles]
    repeated_node = (
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 2] == triangles[:, 0])
    )
    finite = np.isfinite(p).all(axis=(1, 2))

    edge01 = p[:, 1] - p[:, 0]
    edge12 = p[:, 2] - p[:, 1]
    edge20 = p[:, 0] - p[:, 2]
    l01 = np.linalg.norm(edge01, axis=1)
    l12 = np.linalg.norm(edge12, axis=1)
    l20 = np.linalg.norm(edge20, axis=1)
    min_length = np.minimum.reduce([l01, l12, l20])

    twice_area = np.abs(edge01[:, 0] * (p[:, 2, 1] - p[:, 0, 1]) - edge01[:, 1] * (p[:, 2, 0] - p[:, 0, 0]))

    keep = finite & ~repeated_node & (min_length > min_edge_length) & (0.5 * twice_area > min_area)
    return triangles[keep]


def _points_inside_triangular_domain(
    points: np.ndarray,
    domain_xy: np.ndarray,
    domain_triangles: np.ndarray,
) -> np.ndarray:
    """Return True for points inside the original triangular Telemac domain."""
    try:
        import matplotlib.tri as mtri
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to filter Delaunay triangles against the "
            "original Telemac triangular domain."
        ) from exc

    points = np.asarray(points, dtype=np.float64)
    domain_xy = np.asarray(domain_xy, dtype=np.float64)
    domain_triangles = np.asarray(domain_triangles, dtype=np.int64)

    triangulation = mtri.Triangulation(
        domain_xy[:, 0],
        domain_xy[:, 1],
        domain_triangles,
    )
    tri_finder = triangulation.get_trifinder()
    return tri_finder(points[:, 0], points[:, 1]) >= 0


def filter_triangles_inside_domain(
    xy: np.ndarray,
    triangles: np.ndarray,
    domain_xy: np.ndarray,
    domain_triangles: np.ndarray,
) -> np.ndarray:
    """
    Remove coarse Delaunay triangles outside the original Telemac domain.

    Delaunay fills the convex hull of the coarse points. For a river mesh this
    can create triangles over concave exterior areas or holes. A triangle is
    kept only if its vertices, edge midpoints, and centroid all lie inside the
    original fine triangular mesh.
    """
    xy = np.asarray(xy, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)

    if len(triangles) == 0:
        return triangles

    p = xy[triangles]
    samples = np.stack(
        [
            p[:, 0],
            p[:, 1],
            p[:, 2],
            0.5 * (p[:, 0] + p[:, 1]),
            0.5 * (p[:, 1] + p[:, 2]),
            0.5 * (p[:, 2] + p[:, 0]),
            p.mean(axis=1),
        ],
        axis=1,
    )

    inside = _points_inside_triangular_domain(
        samples.reshape(-1, 2),
        domain_xy=domain_xy,
        domain_triangles=domain_triangles,
    )
    keep = inside.reshape(len(triangles), samples.shape[1]).all(axis=1)
    return triangles[keep]


def delaunay_triangles(
    xy: np.ndarray,
    max_edge_length: float | None = None,
    region_labels: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build a coarse triangulation from coarse points using Delaunay.

    When region_labels is provided, Delaunay is run independently inside each
    region. This prevents the triangulation from reconnecting areas separated
    by detected topographic barriers or boundary cuts.

    If max_edge_length is provided, triangles containing an edge longer than
    this threshold are removed. This avoids the most obvious long triangles
    across sparse or concave parts of the domain.
    """
    xy = np.asarray(xy, dtype=np.float64)

    if region_labels is None:
        triangles = _delaunay_triangles_for_points(xy)
        return _filter_long_triangles(xy, triangles, max_edge_length)

    region_labels = np.asarray(region_labels)
    if len(region_labels) != len(xy):
        raise ValueError("region_labels must have one value per point")

    region_triangles = []
    for region in np.unique(region_labels):
        global_nodes = np.flatnonzero(region_labels == region)
        local_triangles = _delaunay_triangles_for_points(xy[global_nodes])
        if len(local_triangles) > 0:
            region_triangles.append(global_nodes[local_triangles])

    if not region_triangles:
        return np.empty((0, 3), dtype=np.int64)

    triangles = np.vstack(region_triangles)
    return _filter_long_triangles(xy, triangles, max_edge_length)


def restrict_time_series(P: csr_matrix, values: np.ndarray) -> np.ndarray:
    """
    Restrict fine time series to the coarse support.

    values shape: (T, N, F)
    output shape: (T, C, F)
    """
    t_size, n_nodes, n_features = values.shape
    if n_nodes != P.shape[1]:
        raise ValueError(f"Expected {P.shape[1]} fine nodes, got {n_nodes}")

    flat = values.transpose(0, 2, 1).reshape(t_size * n_features, n_nodes)
    coarse = flat @ P.T
    return coarse.reshape(t_size, n_features, P.shape[0]).transpose(0, 2, 1)


def _rebuild_coarse_support(
    cluster: np.ndarray,
    xy: np.ndarray,
    triangles: np.ndarray,
    z: np.ndarray,
    region_id: np.ndarray,
    boundary_type: np.ndarray,
    strickler: np.ndarray | None,
    weights: np.ndarray | None,
    fine_edges: np.ndarray,
) -> tuple[
    csr_matrix,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Recompute coarse geometry and labels from a fine->coarse cluster map.
    """
    restriction = build_restriction_matrix(cluster, weights=weights)
    coarse_xy = np.asarray(restriction @ xy)
    coarse_z = np.asarray(restriction @ z)
    coarse_strickler = None if strickler is None else np.asarray(restriction @ strickler)
    coarse_region = coarse_label_from_cluster(cluster, region_id)
    coarse_boundary_type = coarse_label_from_cluster(cluster, boundary_type)
    coarse_edges = coarse_edges_from_fine_edges(fine_edges, cluster)

    coarse_triangles = delaunay_triangles(
        coarse_xy,
        region_labels=coarse_region,
    )
    coarse_triangles = filter_triangles_inside_domain(
        xy=coarse_xy,
        triangles=coarse_triangles,
        domain_xy=xy,
        domain_triangles=triangles,
    )
    coarse_triangles = filter_degenerate_triangles(coarse_xy, coarse_triangles)

    return (
        restriction,
        coarse_xy,
        coarse_z,
        coarse_strickler,
        coarse_region,
        coarse_boundary_type,
        coarse_edges,
        coarse_triangles,
    )


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
    min_coarse_edge_from_fine_factor: float | None = 1.0,
    max_short_edge_merge_iter: int = 8,
    enforce_serafin_safe_nodes: bool = True,
    max_serafin_safe_merge_iter: int = 8,
) -> CoarseMesh:
    """
    Build a first-transfer coarse mesh support without GIS data.

    boundary_type follows the existing dataset convention:
    0 = normal, 1 = prescribed H, 2 = prescribed Q, 3 = wall.

    The method protects:
    - Telemac boundary nodes,
    - strong topographic jumps as a bank/dike/levee proxy.

    Additional control:
    - if min_coarse_edge_from_fine_factor is not None, iteratively merge coarse
      nodes involved in triangulation edges shorter than
      min_coarse_edge_from_fine_factor * min_positive_edge_length(fine_mesh).
    - if enforce_serafin_safe_nodes is True, iteratively merge coarse nodes
      that would collapse to identical coordinates in float32, to avoid zero
      edges after SERAFIN export.
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

    base_cluster = constrained_grid_clusters(
        xy=xy,
        spacing=spacing,
        region_id=region_id,
        boundary_type=boundary_type,
        preserve_node_mask=preserve_node_mask,
    )
    base_cluster = _compact_labels(base_cluster)

    fine_min_edge = min_positive_edge_length(xy, fine_edges)
    short_edge_threshold = None
    if (
        min_coarse_edge_from_fine_factor is not None
        and fine_min_edge is not None
        and min_coarse_edge_from_fine_factor > 0.0
    ):
        short_edge_threshold = float(min_coarse_edge_from_fine_factor) * fine_min_edge

    short_edge_merge_iter = 0
    removed_short_edge_nodes = 0
    serafin_safe_merge_iter = 0
    removed_serafin_duplicate_nodes = 0

    while True:
        (
            base_P,
            base_xy,
            base_z,
            base_strickler,
            base_region,
            coarse_boundary_type,
            base_edges,
            base_triangles,
        ) = _rebuild_coarse_support(
            cluster=base_cluster,
            xy=xy,
            triangles=triangles,
            z=z,
            region_id=region_id,
            boundary_type=boundary_type,
            strickler=strickler,
            weights=weights,
            fine_edges=fine_edges,
        )

        if enforce_serafin_safe_nodes and serafin_safe_merge_iter < max_serafin_safe_merge_iter:
            float32_labels, _ = float32_duplicate_labels(base_xy)
            serafin_safe_n_coarse = int(float32_labels.max()) + 1
            current_n_coarse = len(base_xy)
            if serafin_safe_n_coarse < current_n_coarse:
                removed_serafin_duplicate_nodes += current_n_coarse - serafin_safe_n_coarse
                serafin_safe_merge_iter += 1
                base_cluster = float32_labels[base_cluster]
                base_cluster = _compact_labels(base_cluster)
                continue

        if short_edge_threshold is None or short_edge_merge_iter >= max_short_edge_merge_iter:
            break

        tri_edges = unique_edges_from_triangles(base_triangles)
        if len(tri_edges) == 0:
            break

        tri_edge_lengths = np.linalg.norm(base_xy[tri_edges[:, 1]] - base_xy[tri_edges[:, 0]], axis=1)
        short_edge_mask = tri_edge_lengths + 1e-12 < short_edge_threshold
        if not short_edge_mask.any():
            break

        merged_labels = merge_components_from_edges(len(base_xy), tri_edges[short_edge_mask])
        new_n_coarse = int(merged_labels.max()) + 1
        old_n_coarse = len(base_xy)
        if new_n_coarse >= old_n_coarse:
            break

        removed_short_edge_nodes += old_n_coarse - new_n_coarse
        short_edge_merge_iter += 1
        base_cluster = merged_labels[base_cluster]
        base_cluster = _compact_labels(base_cluster)

    print(
        "Simple Telemac-style coarsening:",
        f"fine_nodes={len(xy)}",
        f"coarse_nodes={len(base_xy)}",
        f"triangles={len(base_triangles)}",
        f"undirected_edges={len(base_edges)}",
        f"spacing={spacing}",
        f"dz_min={used_dz_min:.3g}",
        f"slope_min={used_slope_min:.3g}",
        f"fine_min_edge={fine_min_edge if fine_min_edge is not None else 'none'}",
        f"short_edge_threshold={short_edge_threshold if short_edge_threshold is not None else 'disabled'}",
        f"short_edge_merge_iter={short_edge_merge_iter}",
        f"removed_short_edge_nodes={removed_short_edge_nodes}",
        f"serafin_safe_merge_iter={serafin_safe_merge_iter}",
        f"removed_serafin_duplicate_nodes={removed_serafin_duplicate_nodes}",
    )

    return CoarseMesh(
        cluster=base_cluster,
        xy=base_xy,
        z=base_z,
        strickler=base_strickler,
        triangles=base_triangles,
        undirected_edges=base_edges,
        region_id=region_id,
        coarse_region_id=base_region,
        boundary_type=coarse_boundary_type,
        restriction=base_P,
        topo_barrier_node_mask=topo_barrier_node_mask,
        preserve_node_mask=preserve_node_mask,
    )


def mesh_scale_stats(xy: np.ndarray, undirected_edges: np.ndarray) -> dict[str, float]:
    """Quick diagnostics on the coarse mesh-support edge lengths."""
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
        "boundary_type)."
    )
