#!/usr/bin/env python3
from __future__ import annotations

import argparse
import filecmp
import os
import sys
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import dgl
import numpy as np
import torch
from omegaconf import OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from prepare_x8_short_ghost_dataset import (  # noqa: E402
    X8_BASE_BIN_NAME,
    boundary_edges,
    build_interpolation_plan,
    build_x8_base_dataset,
    find_single_file,
)
from python.create_dgl_dataset import add_mesh_info, create_multimesh, load_liq_q_series  # noqa: E402
from python.ghost_nodes import PRESCRIBED_Q_BC, read_conlim_rows  # noqa: E402
from python.python_code.data_manip.extraction.telemac_file import TelemacFile  # noqa: E402


EXPECTED_NODES = 16_222
EXPECTED_BASE_EDGES = 96_984
EXPECTED_MULTIMESH_EDGES = 130_601
EXPECTED_SELF_LOOPS = 39


def load_graph(path: Path):
    graphs, _ = dgl.load_graphs(str(path))
    if len(graphs) != 1:
        raise ValueError(f"{path}: expected one graph, found {len(graphs)}")
    return graphs[0]


def sorted_edges_and_features(graph):
    edges = np.column_stack([ids.numpy() for ids in graph.edges(order="eid")])
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    return edges[order], graph.edata["x"].numpy()[order]


def check_same_edges(generated, historical, label: str) -> None:
    generated_edges, generated_features = sorted_edges_and_features(generated)
    historical_edges, historical_features = sorted_edges_and_features(historical)
    if not np.array_equal(generated_edges, historical_edges):
        generated_set = set(map(tuple, generated_edges))
        historical_set = set(map(tuple, historical_edges))
        generated_only = generated_set - historical_set
        historical_only = historical_set - generated_set
        raise ValueError(
            f"{label}: different topology: "
            f"generated_only={len(generated_only)} "
            f"(self_loops={sum(src == dst for src, dst in generated_only)}, "
            f"sample={sorted(generated_only)[:10]}), "
            f"historical_only={len(historical_only)} "
            f"(self_loops={sum(src == dst for src, dst in historical_only)}, "
            f"sample={sorted(historical_only)[:10]})"
        )
    np.testing.assert_allclose(
        generated_features, historical_features, rtol=1e-5, atol=1e-4
    )


def check_graphs(base_graph, multimesh, cfg) -> None:
    historical_base = load_graph(Path(cfg.historical_base_graph_path))

    check_same_edges(base_graph, historical_base, "x8 graph")
    if base_graph.num_nodes() != EXPECTED_NODES or base_graph.num_edges() != EXPECTED_BASE_EDGES:
        raise ValueError(
            f"Unexpected x8 graph size: {base_graph.num_nodes()} nodes, "
            f"{base_graph.num_edges()} edges"
        )
    if multimesh.num_nodes() != EXPECTED_NODES or multimesh.num_edges() != EXPECTED_MULTIMESH_EDGES:
        raise ValueError(
            f"Unexpected multimesh size: {multimesh.num_nodes()} nodes, "
            f"{multimesh.num_edges()} edges"
        )

    if not torch.equal(base_graph.ndata["static"][:, :4], historical_base.ndata["static"][:, :4]):
        raise ValueError("Regenerated and historical x8 boundary types differ.")
    torch.testing.assert_close(
        base_graph.ndata["static"][:, 4:], historical_base.ndata["static"][:, 4:]
    )
    torch.testing.assert_close(multimesh.ndata["static"], base_graph.ndata["static"])

    src, dst = multimesh.edges()
    self_loops = int(torch.count_nonzero(src == dst).item())
    if self_loops != EXPECTED_SELF_LOOPS:
        raise ValueError(
            f"Unexpected multimesh self-loop count: {self_loops} != {EXPECTED_SELF_LOOPS}"
        )
    print(
        f"Validated regenerated topology: nodes={multimesh.num_nodes()}, "
        f"directed_edges={multimesh.num_edges()}, self_loops={self_loops}"
    )


def find_fine_boundary_files(source_case_dir: Path) -> tuple[Path, Path]:
    """Retrouve le .cli et le résultat fin qui partage sa numérotation."""
    cli_path = find_single_file(
        source_case_dir, ["*.cli", "*.conlim", "cli", "conlim"], "boundary file"
    )
    first_result_path = find_single_file(
        source_case_dir,
        ["Group_1_peak_1000.res", "Group_1_peak_1000_Group_1_peak_1000.res"],
        "sample result",
    )
    return cli_path, first_result_path


def fine_inlet_nodes(edges, boundary_types) -> tuple[list[int], list[int]]:
    """Ordonne les nœuds de l'entrée et repère le voisin extérieur à chaque bout."""
    boundary_nodes = {node for edge in edges for node in edge}
    if boundary_nodes != boundary_types.keys():
        raise ValueError("Les nœuds de bord du .cli et de la triangulation fine diffèrent.")

    neighbors = {node: [] for node, code in boundary_types.items() if code == PRESCRIBED_Q_BC}
    for a, b in edges:
        if a in neighbors:
            neighbors[a].append(b)
        if b in neighbors:
            neighbors[b].append(a)

    # Les vrais voisins de bord permettent de suivre l'ouverture même si la liste .cli boucle.
    if any(len(adjacent) != 2 for adjacent in neighbors.values()):
        raise ValueError("Un nœud de l'entrée fine doit avoir deux voisins sur le contour.")
    inlet_neighbors = {
        node: [other for other in adjacent if other in neighbors]
        for node, adjacent in neighbors.items()
    }
    ends = [node for node, adjacent in inlet_neighbors.items() if len(adjacent) == 1]
    if len(ends) != 2 or any(not adjacent for adjacent in inlet_neighbors.values()):
        raise ValueError("La Têt doit avoir une ligne d'entrée fine à deux extrémités.")

    ordered = [ends[0]]
    previous = None
    while ordered[-1] != ends[1]:
        node = ordered[-1]
        ordered.append(next(other for other in inlet_neighbors[node] if other != previous))
        previous = node
    if len(ordered) != len(neighbors):
        raise ValueError("Les nœuds à débit prescrit ne forment pas une seule ouverture fine.")

    outside = [next(other for other in neighbors[node] if other not in neighbors) for node in ends]
    return ordered, outside


def inspect_fine_inlet(source_case_dir: Path) -> np.ndarray:
    """Affiche la section fine et renvoie ses coordonnées ordonnées."""
    cli_path, result_path = find_fine_boundary_files(source_case_dir)
    with closing(TelemacFile(str(result_path), bnd_file=str(cli_path))) as fine_res:
        xy, triangles = add_mesh_info(fine_res)

    boundary_types = {row["node_id"]: row["bc_type"] for row in read_conlim_rows(cli_path)}
    nodes, outside = fine_inlet_nodes(boundary_edges(triangles), boundary_types)

    # Somme des longueurs en plan des arêtes entre deux nœuds à débit prescrit.
    fine_line = xy[nodes].astype(float)
    lengths = np.linalg.norm(np.diff(fine_line, axis=0), axis=1)
    if not np.all(np.isfinite(lengths) & (lengths > 0)):
        raise ValueError("La section fine contient une arête de longueur invalide.")
    print(f"Section fine (indices Python) : {cli_path}")
    print(f"{len(nodes)} nœuds, {len(lengths)} arêtes, longueur Q–Q = {lengths.sum():.3f} m")
    for a, b, length in zip(nodes, nodes[1:], lengths):
        print(f"  arête {a}–{b} : {length:.3f} m")
    for node, adjacent in zip((nodes[0], nodes[-1]), outside):
        x, y = xy[node]
        print(
            f"  extrémité {node} ({x:.3f}, {y:.3f}) {boundary_types[node]}, "
            f"voisin {adjacent} {boundary_types[adjacent]}"
        )
    return fine_line


def inspect_x8_inlet(fine_line: np.ndarray, mesh_path: Path) -> dict[int, float]:
    """Projette la section fine sur le bord ×8 et calcule les contributions P1."""
    with closing(TelemacFile(str(mesh_path))) as mesh:
        xy, triangles = add_mesh_info(mesh)

    edges = boundary_edges(triangles)
    if not edges:
        raise ValueError("Aucune arête de bord sur le ×8.")
    edge_xy = xy[edges].astype(float)
    starts = edge_xy[:, 0]
    directions = edge_xy[:, 1] - starts
    squared_lengths = np.sum(directions**2, axis=1)
    if not np.all(np.isfinite(squared_lengths) & (squared_lengths > 0)):
        raise ValueError("Le bord ×8 contient une arête de longueur invalide.")

    # Subdiviser les arêtes fines pour intégrer les changements d'arête ou de sommet projeté.
    subdivisions = 32
    midpoints = (np.arange(subdivisions) + 0.5) / subdivisions
    node_lengths = np.zeros(len(xy))
    visited_edges = []
    max_distance = 0.0
    for start, end in zip(fine_line, fine_line[1:]):
        segment = end - start
        part_length = np.linalg.norm(segment) / subdivisions
        points = start + midpoints[:, None] * segment
        offsets = points[:, None, :] - starts
        fractions = np.clip(np.sum(offsets * directions, axis=2) / squared_lengths, 0, 1)
        distances = np.linalg.norm(offsets - fractions[:, :, None] * directions, axis=2)
        closest = distances.argmin(axis=1)
        max_distance = max(max_distance, distances[np.arange(subdivisions), closest].max())

        # Chaque point dépose une part de longueur fine selon les fonctions de forme 1−ξ et ξ.
        for row, index in enumerate(closest):
            a, b = edges[index]
            fraction = fractions[row, index]
            node_lengths[a] += part_length * (1 - fraction)
            node_lengths[b] += part_length * fraction
            visited_edges.append((a, b))

    fine_length = np.linalg.norm(np.diff(fine_line, axis=0), axis=1).sum()
    np.testing.assert_allclose(node_lengths.sum(), fine_length, rtol=1e-10, atol=1e-10)

    print(f"Projection de l'entrée sur {mesh_path}")
    print(f"  écart maximal aux points de quadrature = {max_distance:.3f} m")
    print(f"  arêtes ×8 retenues : {list(dict.fromkeys(visited_edges))}")
    support = {node: float(length) for node, length in enumerate(node_lengths) if length > 0}
    for node, length in support.items():
        print(
            f"  sommet {node}, longueur contributive = {length:.3f} m, "
            f"poids = {length / fine_length:.6f}"
        )
    if any(not set(left).intersection(right) for left, right in zip(visited_edges, visited_edges[1:])):
        raise ValueError("La projection de la section saute entre des arêtes ×8 non adjacentes.")
    if len(support) < 2:
        raise ValueError("La projection de la section est concentrée sur un seul sommet ×8.")
    return support


def build_graphs(output_dir: Path, cfg) -> None:
    source_case_dir = Path(cfg.fine_res_dir)
    fine_mesh_path = find_single_file(source_case_dir, ["*.slf", "*.geo"], "fine mesh")
    cli_path, first_result_path = find_fine_boundary_files(source_case_dir)
    x8_mesh_path = Path(cfg.x8_mesh_path)

    if not filecmp.cmp(x8_mesh_path, Path(cfg.coarse_mesh_path), shallow=False):
        raise ValueError("The source x8 mesh differs from the mesh used by the historical dataset.")

    with (
        closing(TelemacFile(str(fine_mesh_path))) as fine_mesh,
        closing(TelemacFile(str(first_result_path), bnd_file=str(cli_path))) as fine_res,
        closing(TelemacFile(str(x8_mesh_path))) as x8_mesh,
    ):
        fine_xy, _ = add_mesh_info(fine_mesh)
        x8_xy, x8_triangles = add_mesh_info(x8_mesh)
        interpolation_plan = build_interpolation_plan(fine_xy, x8_xy)
        base_graph = build_x8_base_dataset(
            fine_mesh,
            fine_res,
            cli_path,
            x8_mesh,
            interpolation_plan,
            x8_xy,
            x8_triangles,
            output_dir,
        )

    multimesh_path = output_dir / Path(cfg.multimesh_path).name
    create_multimesh(
        base_graph_path=output_dir / X8_BASE_BIN_NAME,
        fine_mesh_path=x8_mesh_path,
        coarse_mesh_paths=[Path(cfg.x16_mesh_path), Path(cfg.x32_mesh_path)],
        output_path=multimesh_path,
        keep_self_loops=True,
    )
    multimesh = load_graph(multimesh_path)
    check_graphs(base_graph, multimesh, cfg)


def inspect_inlet_discharge(
    hydro_path: Path, start_hour: float, dt_seconds: float, node_lengths: dict[int, float]
) -> None:
    """Intègre le débit du .liq sur un pas et affiche sa répartition sur le ×8."""
    times, flows, labels = load_liq_q_series(hydro_path)
    times = times.astype(float, copy=False)
    if flows.shape[1] != 1:
        raise ValueError(f"La Têt attend un seul débit amont dans {hydro_path}, trouvé {labels}.")
    if not (np.all(np.isfinite(times)) and np.all(np.diff(times) > 0)):
        raise ValueError(f"Les temps dans {hydro_path} doivent être finis et strictement croissants.")
    if not np.all(np.isfinite(flows)):
        raise ValueError(f"Les débits dans {hydro_path} doivent être finis.")

    start = start_hour * 3600.0
    stop = start + dt_seconds
    if not times[0] <= start < stop <= times[-1]:
        raise ValueError(f"Pas [{start:g}, {stop:g}] s hors de l’hydrogramme {hydro_path}.")

    # Intégrer la ligne brisée du .liq en conservant chaque jalon intérieur au pas.
    interior_times = times[(start < times) & (times < stop)]
    step_times = np.concatenate(([start], interior_times, [stop]))
    step_flows = np.interp(step_times, times, flows[:, 0])
    volume = np.sum(0.5 * (step_flows[:-1] + step_flows[1:]) * np.diff(step_times))
    mean_flow = volume / dt_seconds
    linear_flow = mean_flow / sum(node_lengths.values())
    node_flows = {node: linear_flow * length for node, length in node_lengths.items()}
    distributed_volume = sum(node_flows.values()) * dt_seconds
    if not np.isclose(distributed_volume, volume, rtol=1e-12, atol=1e-9):
        raise ValueError("La somme des apports aux nœuds ×8 diffère du volume entrant.")

    print(f"Hydrogramme {hydro_path} ({labels[0]}), pas [{start_hour:g}, {stop / 3600:g}] h")
    print(f"  Q(t) = {step_flows[0]:.3f} m³/s, Q(t+Δt) = {step_flows[-1]:.3f} m³/s")
    print(
        f"  Q moyen = {mean_flow:.3f} m³/s, débit linéique = {linear_flow:.3f} m²/s, "
        f"volume = {volume:.3f} m³"
    )
    for node, flow in node_flows.items():
        print(f"  nœud ×8 {node}, apport = {flow:.3f} m³/s, volume = {flow * dt_seconds:.3f} m³")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prépare les graphes de la Têt.")
    inspection = parser.add_mutually_exclusive_group()
    inspection.add_argument("--inspect-fine-inlet", action="store_true", help="Affiche la section d'entrée fine.")
    inspection.add_argument("--inspect-x8-inlet", action="store_true", help="Affiche les contributions P1 sur le ×8.")
    inspection.add_argument(
        "--inspect-inlet-discharge", nargs="?", const=17.5, type=float, metavar="HEURE",
        help="Affiche aussi les apports par sommet sur un pas (heure de début par défaut 17,5).",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(SCRIPT_DIR / "conf" / "papier_ghost" / "base.yaml")
    if args.inspect_fine_inlet:
        inspect_fine_inlet(Path(cfg.fine_res_dir))
        return
    if args.inspect_x8_inlet or args.inspect_inlet_discharge is not None:
        fine_line = inspect_fine_inlet(Path(cfg.fine_res_dir))
        node_lengths = inspect_x8_inlet(fine_line, Path(cfg.coarse_mesh_path))
        if args.inspect_inlet_discharge is not None:
            inspect_inlet_discharge(
                Path(cfg.hydro_dir[0]), args.inspect_inlet_discharge, cfg.dt_seconds, node_lengths
            )
        return

    graph_dir = Path(cfg.graph_dir)
    if os.path.lexists(graph_dir):
        raise FileExistsError(f"Graph directory already exists: {graph_dir}")

    graph_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-graphs-", dir=graph_dir.parent) as tmp_dir:
        output_dir = Path(tmp_dir) / graph_dir.name
        output_dir.mkdir()
        build_graphs(output_dir, cfg)
        output_dir.rename(graph_dir)
    print(f"Graphs ready: {graph_dir}")


if __name__ == "__main__":
    main()
