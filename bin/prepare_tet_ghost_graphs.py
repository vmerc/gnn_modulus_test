#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
from python.create_dgl_dataset import (  # noqa: E402
    add_mesh_info,
    create_multimesh,
    integrate_liq_step,
    load_liq_q_series,
    validate_liq_for_integration,
)
from python.ghost_nodes import PRESCRIBED_Q_BC, read_conlim_rows  # noqa: E402
from python.python_code.data_manip.extraction.telemac_file import TelemacFile  # noqa: E402


EXPECTED_NODES = 16_222
EXPECTED_BASE_EDGES = 96_984
EXPECTED_MULTIMESH_EDGES = 130_601
EXPECTED_SELF_LOOPS = 39
INLET_MIDPOINTS = (np.arange(32) + 0.5) / 32


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


def inspect_fine_inlet(source_case_dir: Path):
    """Affiche la section fine et renvoie ses nœuds ordonnés et son maillage."""
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
    return nodes, xy, triangles


def inspect_x8_inlet(fine_line: np.ndarray, mesh_path: Path):
    """Renvoie les longueurs P1 et leur répartition aux points d'intégration."""
    with closing(TelemacFile(str(mesh_path))) as mesh:
        xy, triangles = add_mesh_info(mesh)

    edges = boundary_edges(triangles)
    if not edges:
        raise ValueError("Aucune arête de bord sur le ×8.")
    edges = np.asarray(edges)
    edge_xy = xy[edges].astype(float)
    starts = edge_xy[:, 0]
    directions = edge_xy[:, 1] - starts
    squared_lengths = np.sum(directions**2, axis=1)
    if not np.all(np.isfinite(squared_lengths) & (squared_lengths > 0)):
        raise ValueError("Le bord ×8 contient une arête de longueur invalide.")

    # Subdiviser les arêtes fines pour intégrer les changements d'arête ou de sommet projeté.
    sample_nodes = []
    sample_lengths = []
    max_distance = 0.0
    for start, end in zip(fine_line, fine_line[1:]):
        segment = end - start
        part_length = np.linalg.norm(segment) / len(INLET_MIDPOINTS)
        points = start + INLET_MIDPOINTS[:, None] * segment
        offsets = points[:, None, :] - starts
        fractions = np.clip(np.sum(offsets * directions, axis=2) / squared_lengths, 0, 1)
        distances = np.linalg.norm(offsets - fractions[:, :, None] * directions, axis=2)
        closest = distances.argmin(axis=1)
        rows = np.arange(len(INLET_MIDPOINTS))
        max_distance = max(max_distance, distances[rows, closest].max())

        # Chaque point dépose une part de longueur fine selon les fonctions de forme 1−ξ et ξ.
        fraction = fractions[rows, closest]
        sample_nodes.append(edges[closest])
        sample_lengths.append(part_length * np.column_stack((1 - fraction, fraction)))

    sample_nodes = np.stack(sample_nodes)
    sample_lengths = np.stack(sample_lengths)
    node_lengths = np.bincount(sample_nodes.ravel(), weights=sample_lengths.ravel())
    visited_edges = list(map(tuple, sample_nodes.reshape(-1, 2)))
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
    return support, sample_nodes, sample_lengths


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
    if flows.shape[1] != 1:
        raise ValueError(f"La Têt attend un seul débit amont dans {hydro_path}, trouvé {labels}.")
    validate_liq_for_integration(times, flows, hydro_path)

    start = start_hour * 3600.0
    stop = start + dt_seconds
    step_flows, volumes = integrate_liq_step(times, flows, start, stop)
    volume = volumes[0]
    mean_flow = volume / dt_seconds
    linear_flow = mean_flow / sum(node_lengths.values())
    node_flows = {node: linear_flow * length for node, length in node_lengths.items()}
    distributed_volume = sum(node_flows.values()) * dt_seconds
    if not np.isclose(distributed_volume, volume, rtol=1e-12, atol=1e-9):
        raise ValueError("La somme des apports aux nœuds ×8 diffère du volume entrant.")

    print(f"Hydrogramme {hydro_path} ({labels[0]}), pas [{start_hour:g}, {stop / 3600:g}] h")
    print(f"  Q(t) = {step_flows[0, 0]:.3f} m³/s, Q(t+Δt) = {step_flows[-1, 0]:.3f} m³/s")
    print(
        f"  Q moyen = {mean_flow:.3f} m³/s, débit linéique = {linear_flow:.3f} m²/s, "
        f"volume = {volume:.3f} m³"
    )
    for node, flow in node_flows.items():
        print(f"  nœud ×8 {node}, apport = {flow:.3f} m³/s, volume = {flow * dt_seconds:.3f} m³")


def inward_inlet_normals(nodes, xy, triangles, segments, lengths):
    """Oriente la normale de chaque arête vers son triangle incident."""
    normals = np.column_stack((-segments[:, 1], segments[:, 0])) / lengths[:, None]
    for edge, (a, b) in enumerate(zip(nodes, nodes[1:])):
        incident = (triangles == a).any(axis=1) & (triangles == b).any(axis=1)
        triangle, = triangles[incident]
        opposite, = set(triangle) - {a, b}
        orientation = normals[edge] @ (xy[opposite] - xy[a])
        if orientation == 0:
            raise ValueError(f"Triangle incident dégénéré sur l'arête fine {a}–{b}.")
        if orientation < 0:
            normals[edge] *= -1
    return normals


def inlet_flux_snapshots(result_path, hydro_path, nodes, xy, triangles, normals):
    """Reconstruit le flux normal lors de la montée et au pic échantillonné."""
    liq_times, liq_flows, _ = load_liq_q_series(hydro_path)
    validate_liq_for_integration(liq_times, liq_flows, hydro_path)
    liq_flow, = liq_flows.T
    with closing(TelemacFile(str(result_path))) as result:
        result_xy, result_triangles = add_mesh_info(result)
        if not (np.array_equal(result_xy, xy) and np.array_equal(result_triangles, triangles)):
            raise ValueError(f"{result_path}: maillage différent de celui qui porte les indices du .cli.")
        times = result.times
        records = np.flatnonzero((times >= liq_times[0]) & (times <= liq_times[-1]))
        if len(records) < 2 or not times[records[0]] <= liq_times[liq_flow.argmax()] <= times[records[-1]]:
            raise ValueError(f"{result_path}: les sorties couvertes par le .liq ne couvrent pas le pic.")
        discharges = np.interp(times[records], liq_times, liq_flow)
        peak = int(discharges.argmax())
        if peak == 0:
            raise ValueError(f"{result_path}: aucun enregistrement de montée avant le pic.")
        rise = discharges[peak] - discharges[0]
        stages = []
        for part in (0.5, 0.75):
            target = discharges[0] + part * rise
            stages.append((f"Montée {part:.0%}", int(np.argmin(abs(discharges[:peak] - target)))))
        stages.append(("Pic", peak))
        variables = (
            ("WATER DEPTH", "VELOCITY U", "VELOCITY V")
            if "WATER DEPTH" in result.varnames else ("HAUTEUR D'EAU", "VITESSE U", "VITESSE V")
        )
        for stage, index in stages:
            record = int(records[index])
            # Interpoler les états nodaux bruts aux mêmes milieux que la projection d'E8.
            fields = np.column_stack([result.get_data_value(var, record)[nodes] for var in variables])
            at_midpoints = fields[:-1, None] + INLET_MIDPOINTS[None, :, None] * np.diff(fields, axis=0)[:, None]
            normal_velocity = (at_midpoints[..., 1:] * normals[:, None]).sum(axis=2)
            flux = at_midpoints[..., 0] * normal_velocity
            if not np.isfinite(flux).all():
                raise ValueError(f"{result_path}: flux normal non fini à t={times[record]:g} s.")
            yield stage, times[record] / 3600, discharges[index], flux


def inspect_inlet_flux(cfg, nodes, xy, triangles, node_lengths, sample_nodes, sample_lengths):
    """Compare le débit normal TELEMAC et les poids de section déjà utilisés par E8."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(cfg.article_results_dir) / "inlet_flux"
    if output_dir.exists():
        raise FileExistsError(f"Diagnostic déjà présent : {output_dir}")

    section_path = Path(cfg.graph_dir) / "inlet_section.yaml"
    section, = OmegaConf.load(section_path).inlet_sections
    e8_lengths = dict(section)
    if e8_lengths.keys() != node_lengths.keys():
        raise ValueError(f"{section_path}: les sommets de la section diffèrent de la projection.")
    x8_nodes = sorted(node_lengths)
    lengths_e8 = np.array([e8_lengths[node] for node in x8_nodes])
    np.testing.assert_allclose(lengths_e8, [node_lengths[node] for node in x8_nodes], rtol=1e-10, atol=1e-8)
    shares_e8 = lengths_e8 / lengths_e8.sum()

    segments = np.diff(xy[nodes].astype(float), axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    normals = inward_inlet_normals(nodes, xy, triangles, segments, lengths)
    distances = np.cumsum(lengths) - lengths / 2
    training_hydrographs = {Path(path).name: Path(path) for path in cfg.hydro_dir}
    result_dir = Path(cfg.fine_res_dir)

    node_rows, profile_rows = [], []
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True, sharey="col", constrained_layout=True)
    for group, panels in enumerate(axes, start=1):
        for peak, ax in zip((1800, 3200), panels):
            event = f"Group_{group}_peak_{peak}"
            result_path = find_single_file(result_dir, [f"{event}.res", f"{event}_{event}.res"], event)
            hydro_path = training_hydrographs[f"generated_hydrographs_{event}.liq"]
            snapshots = inlet_flux_snapshots(result_path, hydro_path, nodes, xy, triangles, normals)
            for stage, hour, q_liq, flux in snapshots:
                edge_flows = flux.mean(axis=1) * lengths
                q_fine = edge_flows.sum()
                apportioned = np.bincount(
                    sample_nodes.ravel(), weights=(sample_lengths * flux[..., None]).ravel(),
                )[x8_nodes]
                np.testing.assert_allclose(apportioned.sum(), q_fine, rtol=1e-12, atol=1e-9)
                shares = apportioned / q_fine if q_fine > 0 else np.full(len(x8_nodes), np.nan)
                reverse_flow = np.maximum(-flux, 0).mean(axis=1) @ lengths
                delta_q = q_fine - q_liq
                delta_q_pct = 100 * delta_q / q_liq if q_liq != 0 else np.nan
                print(
                    f"{event}, {stage} à {hour:g} h, Q .res = {q_fine:.3f}, Q .liq = {q_liq:.3f} m³/s, "
                    f"écart = {delta_q:+.3f} m³/s ({delta_q_pct:+.2f} %), flux sortant = {reverse_flow:.3f} m³/s, "
                    f"parts ×8 = {dict(zip(x8_nodes, np.round(100 * shares, 2).tolist()))} %"
                )
                for node, q_node, share, share_e8 in zip(x8_nodes, apportioned, shares, shares_e8):
                    node_rows.append((
                        event, stage, hour, q_liq, q_fine, delta_q, delta_q_pct, reverse_flow,
                        node, q_node, share, share_e8, 100 * (share - share_e8),
                    ))
                for a, b, distance, q_edge, length in zip(nodes, nodes[1:], distances, edge_flows, lengths):
                    profile_rows.append((event, stage, hour, a, b, distance, q_edge / length, q_edge))
                ax.plot(distances, edge_flows / lengths, label=f"{stage} ({hour:g} h)")
                if stage == "Pic":
                    ax.axhline(q_liq / lengths.sum(), color="0.45", linestyle="--", label="E8 uniforme au pic")
            ax.set_title(f"Groupe {group}, pic {peak} m³/s")
            ax.legend(fontsize=7)
        panels[0].set_ylabel("Flux normal (m²/s)")
    for ax in axes[-1]:
        ax.set_xlabel(f"Abscisse sur la section fine depuis le nœud {nodes[0]} (m)")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".inlet-flux-", dir=output_dir.parent) as temp_dir:
        staged = Path(temp_dir)
        with (staged / "inlet_flux_by_node.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow((
                "event", "stage", "time_h", "q_liq_m3s", "q_fine_m3s", "delta_q_m3s", "delta_q_pct", "reverse_flow_m3s",
                "x8_node", "q_node_m3s", "share", "e8_share", "delta_share_pp",
            ))
            writer.writerows(node_rows)
        with (staged / "inlet_flux_profile.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("event", "stage", "time_h", "start_node", "end_node", "s_mid_m", "q_normal_m2s", "q_edge_m3s"))
            writer.writerows(profile_rows)
        fig.savefig(staged / "inlet_flux_profile.pdf")
        plt.close(fig)
        staged.rename(output_dir)
    print(f"Diagnostic du flux sur la section d'entrée : {output_dir}")


def export_inlet_section(path: Path, node_lengths: dict[int, float]) -> None:
    """Enregistre les longueurs P1 à pleine précision pour E8."""
    section = {"inlet_sections": [[[node, length] for node, length in node_lengths.items()]]}
    with TemporaryDirectory(prefix=".inlet-section-", dir=path.parent) as temp_dir:
        temporary = Path(temp_dir) / path.name
        OmegaConf.save(OmegaConf.create(section), temporary)
        # Publier le fichier complet et accepter une relance si la géométrie est identique.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if OmegaConf.to_container(OmegaConf.load(path)) != section:
                raise FileExistsError(f"La section existe déjà avec une autre géométrie : {path}") from None
    print(f"Section ×8 enregistrée dans {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prépare les graphes de la Têt.")
    inspection = parser.add_mutually_exclusive_group()
    inspection.add_argument("--inspect-fine-inlet", action="store_true", help="Affiche la section d'entrée fine.")
    inspection.add_argument("--inspect-x8-inlet", action="store_true", help="Affiche les contributions P1 sur le ×8.")
    inspection.add_argument("--export-inlet-section", action="store_true", help="Enregistre la section ×8 pour E8.")
    inspection.add_argument("--inspect-inlet-flux", action="store_true", help="Compare le flux normal TELEMAC aux poids d'E8.")
    inspection.add_argument(
        "--inspect-inlet-discharge", nargs="?", const=17.5, type=float, metavar="HEURE",
        help="Affiche aussi les apports par sommet sur un pas (heure de début par défaut 17,5).",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(SCRIPT_DIR / "conf" / "papier_ghost" / "base.yaml")
    if args.inspect_fine_inlet:
        inspect_fine_inlet(Path(cfg.fine_res_dir))
        return
    if any((args.inspect_x8_inlet, args.export_inlet_section, args.inspect_inlet_discharge is not None, args.inspect_inlet_flux)):
        nodes, xy, triangles = inspect_fine_inlet(Path(cfg.fine_res_dir))
        node_lengths, sample_nodes, sample_lengths = inspect_x8_inlet(
            xy[nodes].astype(float), Path(cfg.coarse_mesh_path),
        )
        if args.inspect_inlet_flux:
            inspect_inlet_flux(cfg, nodes, xy, triangles, node_lengths, sample_nodes, sample_lengths)
        if args.export_inlet_section:
            export_inlet_section(Path(cfg.graph_dir) / "inlet_section.yaml", node_lengths)
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
