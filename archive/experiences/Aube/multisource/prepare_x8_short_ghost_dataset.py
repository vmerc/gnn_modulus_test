#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import dgl
import numpy as np
import torch
from scipy.spatial import Delaunay, KDTree


SOURCE_CASE_DIR = Path("/work/m24046/m24046mrcr/results_data_30min")
X8_CASE_DIR = Path("/work/m24046/m24046mrcr/results_data_30min_35_70_maillagex8")
OUTPUT_ROOT = Path("/work/m24046/m24046mrcr/dataset_Tet_short_ghost")

CHUNK_SIZE = 100
SHORT_START = 35
SHORT_STOP = 65

ENFORCE_Q_BOUNDARY = False
ENFORCE_H_BOUNDARY = True

X8_BASE_BIN_NAME = "Mesh8_base.bin"
INLET_JSON_NAME = "x8_inlet_node_lists.json"
INLET_YAML_NAME = "x8_inlet_node_lists.yaml"
DYNAMIC_YAML_NAME = "dynamic_dir.yaml"


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BIN_DIR = PROJECT_ROOT / "bin"

os.chdir(BIN_DIR)
sys.path.append(str(PROJECT_ROOT))

from python.create_dgl_dataset import (  # noqa: E402
    add_mesh_info,
    create_dgl_dataset_chunked,
    extract_node_type,
    get_dgl_graph,
)
from python.ghost_nodes import (  # noqa: E402
    extract_inlet_node_lists_from_conlim,
    normalize_inlet_node_lists,
)
from python.python_code.data_manip.extraction.telemac_file import TelemacFile  # noqa: E402

os.chdir(PROJECT_ROOT)


CHUNK_PATTERN = re.compile(
    r"^(?P<event>.+)_(?P<traj>\d+)_(?P<start>\d+)-(?P<end>\d+)(?P<suffix>_interpolated)?\.pkl$"
)


def require_directory(path: Path, name: str) -> Path:
    value = Path(path)
    if str(value) in {"", ".", "/path/to/source_case_dir", "/path/to/x8_case_dir"}:
        raise ValueError(f"{name} must be set.")
    value = value.expanduser()
    if not value.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {value}")
    return value


def prepare_output_root(path: Path) -> Path:
    value = Path(path)
    if str(value) in {"", ".", "/path/to/output_root"}:
        raise ValueError("OUTPUT_ROOT must be set.")
    value = value.expanduser()
    value.mkdir(parents=True, exist_ok=True)
    return value


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


def list_res_files(directory: Path) -> list[Path]:
    res_files = sorted(directory.glob("*.res"))
    if not res_files:
        raise FileNotFoundError(f"No .res files found in {directory}")
    return res_files


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def unpack_sample(sample):
    if isinstance(sample, dict):
        return sample["x"], sample["y"], sample.get("ts", None)
    if len(sample) == 2:
        x, y = sample
        return x, y, None
    if len(sample) == 3:
        x, y, ts = sample
        return x, y, ts
    raise ValueError("Unsupported sample format.")


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
    fine_mesh_path: Path,
    sample_res_path: Path,
    cli_path: Path,
    x8_mesh_path: Path,
    output_dir: Path,
) -> None:
    fine_res = TelemacFile(str(sample_res_path), bnd_file=str(cli_path))
    fine_mesh = TelemacFile(str(fine_mesh_path))
    x8_mesh = TelemacFile(str(x8_mesh_path))

    fine_xy, _ = add_mesh_info(fine_mesh)
    x8_xy, x8_triangles = add_mesh_info(x8_mesh)
    interpolation_plan = build_interpolation_plan(fine_xy, x8_xy)
    x8_boundary_nodes = boundary_node_indices(x8_triangles)
    x8_node_types = project_boundary_node_types(fine_res, x8_xy, x8_boundary_nodes)
    inlet_node_lists = project_inlet_node_lists(fine_res, cli_path, x8_xy, x8_boundary_nodes)

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
    shutil.copy2(x8_mesh_path, output_dir / x8_mesh_path.name)


def interpolate_pkl_file(
    input_path: Path,
    output_path: Path,
    interpolation_plan: dict[str, np.ndarray],
) -> None:
    samples = load_pickle(input_path)
    interpolated_samples = []

    for sample in samples:
        x, y, ts = unpack_sample(sample)
        x_interp = interpolate_field(x, interpolation_plan)
        y_interp = interpolate_field(y, interpolation_plan)
        if ts is None:
            interpolated_samples.append((x_interp, y_interp))
        else:
            interpolated_samples.append((x_interp, y_interp, int(ts)))

    save_pickle(output_path, interpolated_samples)


def interpolate_fine_dataset(
    fine_mesh_path: Path,
    x8_mesh_path: Path,
    fine_dataset_dir: Path,
    x8_full_dir: Path,
) -> None:
    fine_mesh = TelemacFile(str(fine_mesh_path))
    x8_mesh = TelemacFile(str(x8_mesh_path))
    fine_xy, _ = add_mesh_info(fine_mesh)
    x8_xy, _ = add_mesh_info(x8_mesh)
    interpolation_plan = build_interpolation_plan(fine_xy, x8_xy)

    input_files = sorted(fine_dataset_dir.glob("*.pkl"))
    if not input_files:
        raise FileNotFoundError(f"No .pkl files found in {fine_dataset_dir}")

    for input_path in input_files:
        output_path = x8_full_dir / f"{input_path.stem}_interpolated.pkl"
        interpolate_pkl_file(input_path, output_path, interpolation_plan)


def parse_chunk_path(path: Path) -> tuple[str, int, int, int]:
    match = CHUNK_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unsupported chunk name: {path.name}")
    event = match.group("event")
    traj = int(match.group("traj"))
    start = int(match.group("start"))
    end = int(match.group("end"))
    return event, traj, start, end


def build_short_dataset(
    x8_full_dir: Path,
    x8_short_dir: Path,
    short_start: int,
    short_stop: int,
) -> list[Path]:
    grouped_files: dict[tuple[str, int], list[tuple[int, Path]]] = defaultdict(list)
    for path in sorted(x8_full_dir.glob("*_interpolated.pkl")):
        event, traj, start, _ = parse_chunk_path(path)
        grouped_files[(event, traj)].append((start, path))

    if not grouped_files:
        raise FileNotFoundError(f"No interpolated .pkl files found in {x8_full_dir}")

    output_files: list[Path] = []
    for (event, traj), items in sorted(grouped_files.items()):
        combined_samples = []
        for _, path in sorted(items):
            combined_samples.extend(load_pickle(path))

        if len(combined_samples) < short_stop:
            raise ValueError(
                f"{event}: {len(combined_samples)} samples, need at least {short_stop}"
            )

        reduced_samples = combined_samples[short_start:short_stop]
        output_path = x8_short_dir / f"{event}_{traj}_{short_start}-{short_stop - 1}_interpolated.pkl"
        save_pickle(output_path, reduced_samples)
        output_files.append(output_path)

    shutil.copy2(x8_full_dir / X8_BASE_BIN_NAME, x8_short_dir / X8_BASE_BIN_NAME)
    shutil.copy2(x8_full_dir / INLET_JSON_NAME, x8_short_dir / INLET_JSON_NAME)
    shutil.copy2(x8_full_dir / INLET_YAML_NAME, x8_short_dir / INLET_YAML_NAME)
    write_yaml_list(x8_short_dir / DYNAMIC_YAML_NAME, "dynamic_dir", output_files)
    return output_files


def create_fine_dataset(
    source_case_dir: Path,
    fine_mesh_path: Path,
    cli_path: Path,
    fine_dataset_dir: Path,
) -> list[Path]:
    res_files = list_res_files(source_case_dir)

    for res_path in res_files:
        dataset_name = res_path.stem
        create_dgl_dataset_chunked(
            mesh_list=[str(fine_mesh_path)],
            res_list=[str(res_path)],
            cli_list=[str(cli_path)],
            dt_list=[1],
            data_folder=str(fine_dataset_dir),
            dataset_name=dataset_name,
            chunk_size=CHUNK_SIZE,
            enforce_q_boundary=ENFORCE_Q_BOUNDARY,
            enforce_h_boundary=ENFORCE_H_BOUNDARY,
        )

    return res_files


def main() -> None:
    source_case_dir = require_directory(SOURCE_CASE_DIR, "SOURCE_CASE_DIR")
    x8_case_dir = require_directory(X8_CASE_DIR, "X8_CASE_DIR")
    output_root = prepare_output_root(OUTPUT_ROOT)

    fine_mesh_path = find_single_file(source_case_dir, ["*.slf", "*.geo"], "fine mesh")
    cli_path = find_single_file(
        source_case_dir,
        ["*.cli", "*.conlim", "cli", "conlim"],
        "boundary file",
    )
    x8_mesh_path = find_single_file(x8_case_dir, ["*.slf", "*.geo"], "x8 mesh")

    fine_dataset_dir = ensure_dir(output_root / "fine_dataset")
    x8_full_dir = ensure_dir(output_root / "x8_full")
    x8_short_dir = ensure_dir(output_root / "x8_short")
    #On a déjà les fulls
    res_files = list_res_files(source_case_dir)
    #res_files = create_fine_dataset(source_case_dir, fine_mesh_path, cli_path, fine_dataset_dir)
    build_x8_base_dataset(fine_mesh_path, res_files[0], cli_path, x8_mesh_path, x8_full_dir)
    interpolate_fine_dataset(fine_mesh_path, x8_mesh_path, fine_dataset_dir, x8_full_dir)
    short_files = build_short_dataset(x8_full_dir, x8_short_dir, SHORT_START, SHORT_STOP)

    print("fine_dataset:", fine_dataset_dir)
    print("x8_full:", x8_full_dir)
    print("x8_short:", x8_short_dir)
    print("x8_base_bin:", x8_short_dir / X8_BASE_BIN_NAME)
    print("inlet_node_lists:", x8_short_dir / INLET_YAML_NAME)
    print("short_files:")
    for path in short_files:
        print(" -", path)


if __name__ == "__main__":
    main()
