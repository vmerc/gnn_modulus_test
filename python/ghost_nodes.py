from __future__ import annotations

from pathlib import Path


PRESCRIBED_Q_BC = (4, 5, 5)


def _parse_conlim_rows(cli_path):
    rows = []
    for line in Path(cli_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 13:
            continue

        try:
            lih = int(parts[0])
            liu = int(parts[1])
            liv = int(parts[2])
            node_id = int(parts[11]) - 1
            line_id = int(parts[12]) - 1
        except ValueError:
            continue

        rows.append(
            {
                "bc_type": (lih, liu, liv),
                "node_id": node_id,
                "line_id": line_id,
            }
        )

    if not rows:
        raise ValueError(f"No valid boundary rows found in {cli_path}")

    return rows


def _unique_preserve_order(values):
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def extract_inlet_node_lists_from_conlim(
    cli_path,
    boundary_type=PRESCRIBED_Q_BC,
):
    rows = _parse_conlim_rows(cli_path)
    inlet_node_lists = []
    current_nodes = []

    for row in rows:
        if row["bc_type"] == tuple(boundary_type):
            current_nodes.append(row["node_id"])
            continue

        if current_nodes:
            inlet_node_lists.append(_unique_preserve_order(current_nodes))
            current_nodes = []

    if current_nodes:
        inlet_node_lists.append(_unique_preserve_order(current_nodes))

    if not inlet_node_lists:
        raise ValueError(
            f"No boundary segment with type {tuple(boundary_type)} found in {cli_path}"
        )

    return inlet_node_lists


def normalize_inlet_node_lists(inlet_node_lists):
    normalized = []
    for inlet_nodes in inlet_node_lists:
        nodes = [int(node_id) for node_id in inlet_nodes]
        nodes = _unique_preserve_order(nodes)
        if not nodes:
            raise ValueError("Each inlet must contain at least one physical node.")
        normalized.append(nodes)

    if not normalized:
        raise ValueError("At least one inlet must be provided.")

    return normalized


def add_ghost_source_nodes(graph, inlet_node_lists, edge_feature_dim):
    import dgl
    import torch

    inlet_node_lists = normalize_inlet_node_lists(inlet_node_lists)

    num_physical_nodes = graph.num_nodes()
    num_source_nodes = len(inlet_node_lists)
    total_nodes = num_physical_nodes + num_source_nodes

    src_old, dst_old = graph.edges(order="eid")
    src_parts = [src_old]
    dst_parts = [dst_old]
    source_edge_count = 0

    for source_id, inlet_nodes in enumerate(inlet_node_lists):
        source_node_id = num_physical_nodes + source_id
        inlet_tensor = torch.tensor(inlet_nodes, dtype=src_old.dtype, device=src_old.device)
        source_tensor = torch.full_like(inlet_tensor, source_node_id)
        src_parts.append(source_tensor)
        dst_parts.append(inlet_tensor)
        source_edge_count += len(inlet_nodes)

    src_all = torch.cat(src_parts, dim=0)
    dst_all = torch.cat(dst_parts, dim=0)

    augmented_graph = dgl.graph(
        (src_all, dst_all),
        num_nodes=total_nodes,
        idtype=graph.idtype,
        device=graph.device,
    )

    if "x" in graph.edata:
        edge_features = graph.edata["x"]
        zeros = torch.zeros(
            (source_edge_count, edge_feature_dim),
            dtype=edge_features.dtype,
            device=edge_features.device,
        )
        augmented_graph.edata["x"] = torch.cat((edge_features, zeros), dim=0)

    is_physical = torch.zeros((total_nodes, 1), dtype=torch.bool, device=graph.device)
    is_source = torch.zeros((total_nodes, 1), dtype=torch.bool, device=graph.device)
    source_id = torch.full((total_nodes, 1), -1, dtype=torch.int64, device=graph.device)

    is_physical[:num_physical_nodes] = True
    is_source[num_physical_nodes:] = True
    source_id[num_physical_nodes:, 0] = torch.arange(
        num_source_nodes,
        dtype=torch.int64,
        device=graph.device,
    )

    augmented_graph.ndata["is_physical"] = is_physical
    augmented_graph.ndata["is_source"] = is_source
    augmented_graph.ndata["source_id"] = source_id

    return augmented_graph
