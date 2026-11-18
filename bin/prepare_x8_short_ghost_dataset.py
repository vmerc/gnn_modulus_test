#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import dgl
import numpy as np
import torch
from scipy.spatial import Delaunay, KDTree


SOURCE_CASE_DIR = Path("/work/m24046/m24046mrcr/results_data_30min")
X8_CASE_DIR = Path("/work/m24046/m24046mrcr/results_data_30min_35_70_maillagex8")
X8_MESH_NAME = "Mesh8_corrige.slf"
MULTIMESH_NAME = "Multimesh_8_32.bin"

FULL_STOP = 80
SHORT_START = 35
SHORT_STOP = 65
DT_SECONDS = 1800.0
EXPECTED_INLETS = [[12770]]

X8_BASE_BIN_NAME = "Mesh8_base.bin"
INLET_JSON_NAME = "x8_inlet_node_lists.json"
INLET_YAML_NAME = "x8_inlet_node_lists.yaml"
DYNAMIC_YAML_NAME = "dynamic_dir.yaml"


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

os.chdir(SCRIPT_DIR)
sys.path.append(str(PROJECT_ROOT))

from python.create_dgl_dataset import (  # noqa: E402
    add_mesh_info,
    create_dgl_dataset_chunked,
    extract_node_type,
    get_dgl_graph,
    get_edge_features_from_positions,
)
from python.ghost_nodes import (  # noqa: E402
    extract_inlet_node_lists_from_conlim,
    normalize_inlet_node_lists,
)
from python.python_code.data_manip.extraction.telemac_file import TelemacFile  # noqa: E402

os.chdir(PROJECT_ROOT)


def find_single_file(directory: Path, patterns: list[str], label: str) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(directory.glob(pattern)))

    unique_matches: list[Path] = []
    seen = set()
    for path in matches:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_matches.append(path)

    if not unique_matches:
        raise FileNotFoundError(f"No {label} found in {directory}")
    if len(unique_matches) > 1:
        raise ValueError(f"Expected one {label} in {directory}, found: {unique_matches}")
    return unique_matches[0]


def find_paper_results(directory: Path) -> dict[str, Path]:
    results = {}
    for group in range(1, 5):
        for peak in range(1000, 3601, 200):
            event = f"Group_{group}_peak_{peak}"
            results[event] = find_single_file(
                directory, [f"{event}.res", f"{event}_{event}.res"], event
            )
    return results


def unique_preserve_order(values: list[int]) -> list[int]:
    seen = set()
    output: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def boundary_node_indices(triangles: np.ndarray) -> np.ndarray:
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for tri in triangles:
        a, b, c = [int(node_id) for node_id in tri]
        edge_count[tuple(sorted((a, b)))] += 1
        edge_count[tuple(sorted((b, c)))] += 1
        edge_count[tuple(sorted((c, a)))] += 1

    boundary_nodes = set()
    for edge, count in edge_count.items():
        if count != 1:
            continue
        boundary_nodes.update(edge)

    if not boundary_nodes:
        raise ValueError("Could not detect boundary nodes on x8 mesh.")

    return np.asarray(sorted(boundary_nodes), dtype=np.int64)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(path: Path, data) -> None:
    with path.open("wb") as handle:
        pickle.dump(data, handle)


def build_interpolation_plan(fine_xy: np.ndarray, coarse_xy: np.ndarray) -> dict[str, np.ndarray]:
    triangulation = Delaunay(fine_xy)
    simplex_ids = triangulation.find_simplex(coarse_xy)
    valid_mask = simplex_ids >= 0

    num_points = coarse_xy.shape[0]
    num_vertices = fine_xy.shape[1] + 1
    vertices = np.full((num_points, num_vertices), -1, dtype=np.int64)
    weights = np.zeros((num_points, num_vertices), dtype=np.float32)

    if np.any(valid_mask):
        valid_simplex_ids = simplex_ids[valid_mask]
        vertices[valid_mask] = triangulation.simplices[valid_simplex_ids]

        transform = triangulation.transform[valid_simplex_ids]
        delta = coarse_xy[valid_mask] - transform[:, -1, :]
        bary = np.einsum("nij,nj->ni", transform[:, :-1, :], delta)

        weights[valid_mask, :-1] = bary
        weights[valid_mask, -1] = 1.0 - bary.sum(axis=1)

    return {
        "vertices": vertices,
        "weights": weights,
        "valid_mask": valid_mask,
    }


def interpolate_field(fine_values: np.ndarray, interpolation_plan: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(fine_values, dtype=np.float32)
    squeeze_output = values.ndim == 1
    if squeeze_output:
        values = values[:, None]

    output = np.zeros(
        (interpolation_plan["weights"].shape[0], values.shape[1]),
        dtype=np.float32,
    )

    valid_mask = interpolation_plan["valid_mask"]
    if np.any(valid_mask):
        vertices = interpolation_plan["vertices"][valid_mask]
        weights = interpolation_plan["weights"][valid_mask]
        gathered_values = values[vertices]
        output[valid_mask] = np.einsum("ij,ijk->ik", weights, gathered_values)

    if squeeze_output:
        return output[:, 0]
    return output


def read_scalar_field(mesh: TelemacFile, field_name: str) -> np.ndarray:
    values = np.asarray(mesh.get_data_value(field_name, 0), dtype=np.float32)
    return values[:, None]


def project_boundary_node_types(
    fine_res: TelemacFile,
    x8_xy: np.ndarray,
    x8_boundary_nodes: np.ndarray,
) -> np.ndarray:
    fine_xy, _ = add_mesh_info(fine_res)
    fine_bnd_info = fine_res.get_bnd_info()
    fine_node_types = extract_node_type(fine_res.tri, fine_bnd_info)
    fine_boundary_mask = fine_node_types[:, 0] == 0.0

    if not np.any(fine_boundary_mask):
        raise ValueError("No boundary node detected on fine mesh.")

    fine_boundary_xy = fine_xy[fine_boundary_mask]
    fine_boundary_types = fine_node_types[fine_boundary_mask]
    tree = KDTree(fine_boundary_xy)
    _, indices = tree.query(x8_xy[x8_boundary_nodes])

    x8_node_types = np.zeros((x8_xy.shape[0], fine_node_types.shape[1]), dtype=np.float32)
    x8_node_types[:, 0] = 1.0
    x8_node_types[x8_boundary_nodes] = fine_boundary_types[np.asarray(indices)]
    return x8_node_types


def project_inlet_node_lists(
    fine_res: TelemacFile,
    cli_path: Path,
    x8_xy: np.ndarray,
    x8_boundary_nodes: np.ndarray,
) -> list[list[int]]:
    fine_xy, _ = add_mesh_info(fine_res)
    x8_boundary_xy = x8_xy[x8_boundary_nodes]
    tree = KDTree(x8_boundary_xy)

    fine_inlets = extract_inlet_node_lists_from_conlim(cli_path)
    projected_inlets: list[list[int]] = []
    for inlet_nodes in fine_inlets:
        inlet_xy = fine_xy[np.asarray(inlet_nodes, dtype=np.int64)]
        _, indices = tree.query(inlet_xy)
        projected_nodes = [int(x8_boundary_nodes[int(idx)]) for idx in np.atleast_1d(indices)]
        projected_nodes = unique_preserve_order(projected_nodes)
        projected_inlets.append(projected_nodes)

    return normalize_inlet_node_lists(projected_inlets)


def write_yaml_list(path: Path, key: str, values: list[list[int]] | list[Path]) -> None:
    lines = [f"{key}:\n"]
    for value in values:
        if isinstance(value, Path):
            lines.append(f"  - '{value}'\n")
            continue
        joined = ", ".join(str(item) for item in value)
        lines.append(f"  - [{joined}]\n")
    path.write_text("".join(lines), encoding="utf-8")


def build_x8_base_dataset(
    fine_mesh: TelemacFile,
    fine_res: TelemacFile,
    cli_path: Path,
    x8_mesh: TelemacFile,
    interpolation_plan: dict[str, np.ndarray],
    x8_xy: np.ndarray,
    x8_triangles: np.ndarray,
    output_dir: Path,
):
    x8_boundary_nodes = boundary_node_indices(x8_triangles)
    x8_node_types = project_boundary_node_types(fine_res, x8_xy, x8_boundary_nodes)
    inlet_node_lists = project_inlet_node_lists(fine_res, cli_path, x8_xy, x8_boundary_nodes)
    if inlet_node_lists != EXPECTED_INLETS:
        raise ValueError(f"Projected inlet {inlet_node_lists} differs from {EXPECTED_INLETS}.")
    if not np.any(x8_node_types[:, 1]):
        raise ValueError("No prescribed-stage boundary node on the projected ×8 mesh.")
    print("Inlet ×8 (zero-based):", inlet_node_lists)

    try:
        friction = read_scalar_field(x8_mesh, "FROTTEMENT")
    except Exception:
        friction = interpolate_field(
            read_scalar_field(fine_mesh, "FROTTEMENT"),
            interpolation_plan,
        )

    try:
        bottom = read_scalar_field(x8_mesh, "FOND")
    except Exception:
        bottom = interpolate_field(
            read_scalar_field(fine_mesh, "FOND"),
            interpolation_plan,
        )

    graph, edge_features = get_dgl_graph(x8_mesh.tri)
    graph.edata["x"] = torch.tensor(edge_features, dtype=torch.float32)
    static_features = np.concatenate([x8_node_types, friction, bottom], axis=1).astype(np.float32)
    graph.ndata["static"] = torch.tensor(static_features, dtype=torch.float32)

    dgl.save_graphs(str(output_dir / X8_BASE_BIN_NAME), [graph])
    (output_dir / INLET_JSON_NAME).write_text(
        json.dumps(inlet_node_lists, indent=2),
        encoding="utf-8",
    )
    write_yaml_list(output_dir / INLET_YAML_NAME, "inlet_node_lists", inlet_node_lists)
    return graph


def write_historical_multimesh(
    base_graph, x8_xy: np.ndarray, historical_path: Path, output_path: Path
) -> None:
    [historical], labels = dgl.load_graphs(str(historical_path))
    if historical.num_nodes() != base_graph.num_nodes():
        raise ValueError(f"{historical_path}: incompatible ×8 node count.")
    historical_edges = np.stack([ids.numpy() for ids in historical.edges(order="eid")])
    expected_features = get_edge_features_from_positions(x8_xy, historical_edges)
    if not np.allclose(historical.edata["x"].numpy(), expected_features, rtol=1e-5, atol=1e-4):
        raise ValueError(f"{historical_path}: edge geometry differs from the projected ×8 mesh.")
    native_edges = set(zip(*(ids.tolist() for ids in base_graph.edges())))
    if not native_edges.issubset(zip(*historical_edges)):
        raise ValueError(f"{historical_path}: missing native ×8 edges.")
    historical.ndata["static"] = base_graph.ndata["static"]
    dgl.save_graphs(str(output_path), [historical], labels)
    print(f"Directed edges: ×8={base_graph.num_edges()}, ×8/×32={historical.num_edges()}")


def main() -> None:
    output_root = Path(os.environ["TURPAN_RUN_DIR"]) / "data"
    fine_mesh_path = find_single_file(SOURCE_CASE_DIR, ["*.slf", "*.geo"], "fine mesh")
    cli_path = find_single_file(
        SOURCE_CASE_DIR,
        ["*.cli", "*.conlim", "cli", "conlim"],
        "boundary file",
    )
    x8_mesh_path = X8_CASE_DIR / X8_MESH_NAME
    results = find_paper_results(SOURCE_CASE_DIR)

    output_root.mkdir()
    x8_full_dir = output_root / "x8_full"
    x8_short_dir = output_root / "x8_short"
    x8_full_dir.mkdir()
    x8_short_dir.mkdir()

    first_result_path = next(iter(results.values()))
    with (
        closing(TelemacFile(str(fine_mesh_path))) as fine_mesh,
        closing(TelemacFile(str(x8_mesh_path))) as x8_mesh,
        closing(TelemacFile(str(first_result_path), bnd_file=str(cli_path))) as fine_res,
    ):
        fine_xy, _ = add_mesh_info(fine_mesh)
        x8_xy, x8_triangles = add_mesh_info(x8_mesh)
        interpolation_plan = build_interpolation_plan(fine_xy, x8_xy)
        base_graph = build_x8_base_dataset(
            fine_mesh, fine_res, cli_path, x8_mesh, interpolation_plan,
            x8_xy, x8_triangles, x8_full_dir,
        )

    write_historical_multimesh(
        base_graph, x8_xy, X8_CASE_DIR / MULTIMESH_NAME, x8_short_dir / MULTIMESH_NAME
    )
    shutil.copy2(x8_mesh_path, x8_full_dir / X8_MESH_NAME)
    for filename in (X8_BASE_BIN_NAME, X8_MESH_NAME, INLET_JSON_NAME, INLET_YAML_NAME):
        shutil.copy2(x8_full_dir / filename, x8_short_dir / filename)

    expected_times = DT_SECONDS * np.arange(FULL_STOP + 1)
    full_files = []
    short_files = []
    for event, res_path in results.items():
        with closing(TelemacFile(str(res_path))) as result:
            if len(result.times) != len(expected_times) or not np.allclose(
                result.times, expected_times, rtol=0, atol=1e-3
            ):
                raise ValueError(f"{res_path}: expected {len(expected_times)} states 1800 s apart from t=0.")
            result_xy = np.column_stack((result.meshx, result.meshy))
            if result_xy.shape != fine_xy.shape or not np.allclose(result_xy, fine_xy, rtol=0, atol=1e-4):
                raise ValueError(f"{res_path}: fine mesh coordinates or node order differ.")

        dataset_name = f"{event}_{event}"
        full_name = f"{dataset_name}_0_0-{FULL_STOP}"
        short_name = f"{dataset_name}_0_{SHORT_START}-{SHORT_STOP - 1}"
        with TemporaryDirectory(dir=output_root) as fine_dir:
            create_dgl_dataset_chunked(
                mesh_list=[str(fine_mesh_path)],
                res_list=[str(res_path)],
                cli_list=[str(cli_path)],
                dt_list=[1],
                data_folder=fine_dir,
                dataset_name=dataset_name,
                chunk_size=FULL_STOP,
                enforce_q_boundary=False,
                enforce_h_boundary=True,
            )
            projected = [
                (interpolate_field(x, interpolation_plan), interpolate_field(y, interpolation_plan), ts)
                for x, y, ts in load_pickle(Path(fine_dir) / f"{full_name}.pkl")
            ]

        if [ts for _, _, ts in projected] != list(range(FULL_STOP)):
            raise ValueError(f"{event}: unexpected set or order of stored time indices.")
        full_path = x8_full_dir / f"{full_name}_interpolated.pkl"
        short_path = x8_short_dir / f"{short_name}_interpolated.pkl"
        save_pickle(full_path, projected)
        save_pickle(short_path, projected[SHORT_START:SHORT_STOP])
        full_files.append(full_path)
        short_files.append(short_path)
        print(f"{event}: {len(projected)} full pairs, {SHORT_STOP - SHORT_START} short pairs")

    write_yaml_list(x8_full_dir / DYNAMIC_YAML_NAME, "dynamic_dir", full_files)
    write_yaml_list(x8_short_dir / DYNAMIC_YAML_NAME, "dynamic_dir", short_files)
    print(f"Dataset ready ({len(results)} crues): {output_root}")


if __name__ == "__main__":
    main()
