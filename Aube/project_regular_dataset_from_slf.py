#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import pickle
import sys

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from tqdm import tqdm

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_path)

from python.python_code.data_manip.extraction.telemac_file import TelemacFile
from python.create_dgl_dataset import (
    NodeType,
    add_mesh_info,
    extract_node_type,
    get_dgl_graph,
    get_dynamic_node_features,
    get_node_outputs,
    get_static_node_features,
    put_boundary_infos,
    put_boundary_infos_on_changes,
)

# create_dgl_dataset.py changes cwd during import
os.chdir(project_path)

import dgl
import torch


FINE_MESH_SLF = "/work/m24046/m24046mrcr/Aube/07_EA04_T3_V12_Q100_v1/T3_V12_v2_EA04.slf"
FINE_CLI = "/work/m24046/m24046mrcr/Aube/07_EA04_T3_V12_Q100_v1/cas.conlim"
FINE_RES_DIR = "/work/m24046/m24046mrcr/Aube/07_EA04_T3_V12_Q100_v1/"
FINE_RES_FILES = [
    "/work/m24046/m24046mrcr/Aube/07_EA04_T3_V12_Q100_v1/T3_V12_EA04_Q100_v1.res",
]

REGULAR_MESH_SLF = "/work/m24046/m24046mrcr/Aube/Aube_T1_regulier20m.slf"
OUTPUT_DIR = "/work/m24046/m24046mrcr/Aube/Test_regular_Q100/"
DATASET_NAME = "Aube_regular"
CHUNK_SIZE = 80
BOUNDARY_MATCH_TOL = 20.0
OVERWRITE = True


def read_first_available(telemac, names, timestep=0, default=None):
    for name in names:
        try:
            return telemac.get_data_value(name, timestep)
        except Exception:
            pass
    if default is not None:
        return default
    raise KeyError(f"None of these variables were found: {names}")


def build_fast_interpolator(triangulation, target_points):
    simplex = triangulation.find_simplex(target_points)
    valid = simplex >= 0

    barycentric = np.zeros((valid.sum(), 3), dtype="float32")
    vertices = np.zeros((valid.sum(), 3), dtype=np.int64)

    if valid.any():
        transform = triangulation.transform[simplex[valid], :2]
        offset = triangulation.transform[simplex[valid], 2]
        delta = target_points[valid] - offset

        barycentric[:, :2] = np.einsum("ijk,ik->ij", transform, delta)
        barycentric[:, 2] = 1.0 - barycentric[:, 0] - barycentric[:, 1]
        vertices = triangulation.simplices[simplex[valid]]

    invalid_count = len(target_points) - int(valid.sum())
    print("fast interpolate valid points:", int(valid.sum()), "/", len(target_points))
    print("fast interpolate invalid points:", invalid_count)

    return {
        "valid": valid,
        "vertices": vertices,
        "barycentric": barycentric,
        "n_target": len(target_points),
    }


def fast_interpolate(values, interp_data):
    values = np.ascontiguousarray(values)
    input_was_1d = values.ndim == 1
    if input_was_1d:
        values = values[:, None]

    out = np.zeros((interp_data["n_target"], values.shape[1]), dtype="float32")

    if interp_data["valid"].any():
        gathered = values[interp_data["vertices"]]
        out[interp_data["valid"]] = np.sum(
            gathered * interp_data["barycentric"][:, :, None],
            axis=1,
        ).astype("float32")

    if input_was_1d:
        return out[:, 0]
    return out


def coarse_node_type_from_fine_boundary(fine_with_cli, fine_points, coarse_points, tolerance):
    fine_node_type = extract_node_type(
        fine_with_cli.tri,
        fine_with_cli.get_bnd_info(),
    ).astype("float32")

    coarse_node_type = np.zeros((len(coarse_points), NodeType.SIZE), dtype="float32")
    coarse_node_type[:, NodeType.NORMAL] = 1.0

    tree = cKDTree(fine_points)
    distances, indices = tree.query(coarse_points)
    matched = distances <= tolerance
    coarse_node_type[matched] = fine_node_type[indices[matched]]

    print("matched boundary/support nodes:", int(matched.sum()), "/", len(coarse_points))
    print("coarse node type counts:", coarse_node_type.sum(axis=0).astype(int).tolist())
    return coarse_node_type


def write_base_graph(coarse_mesh, fine_mesh, fine_with_cli, interp_data, x_fine, x_coarse, output_path):
    if os.path.exists(output_path) and not OVERWRITE:
        raise FileExistsError(f"Output already exists: {output_path}")

    node_type = coarse_node_type_from_fine_boundary(
        fine_with_cli,
        x_fine,
        x_coarse,
        BOUNDARY_MATCH_TOL,
    )

    try:
        z = read_first_available(
            coarse_mesh,
            ["FOND", "BOTTOM"],
            timestep=0,
        ).astype("float32")
        print("z read from regular mesh")
    except Exception:
        z_fine = read_first_available(
            fine_mesh,
            ["FOND", "BOTTOM"],
            timestep=0,
        ).astype("float32")
        z = fast_interpolate(z_fine, interp_data)
        print("z projected from fine mesh")

    try:
        strickler = read_first_available(
            coarse_mesh,
            ["FROTTEMENT", "STRICKLER", "FRICTION"],
            timestep=0,
        ).astype("float32")
        print("strickler read from regular mesh")
    except Exception:
        try:
            strickler_fine = read_first_available(
                fine_mesh,
                ["FROTTEMENT", "STRICKLER", "FRICTION"],
                timestep=0,
            ).astype("float32")
            strickler = fast_interpolate(strickler_fine, interp_data)
            print("strickler projected from fine mesh")
        except Exception:
            strickler = np.zeros(len(x_coarse), dtype="float32")
            print("strickler not found, using zeros")

    static = np.concatenate(
        [node_type, strickler[:, None], z[:, None]],
        axis=1,
    ).astype("float32")

    graph, edge_features = get_dgl_graph(coarse_mesh.tri)
    graph.edata["x"] = torch.tensor(edge_features, dtype=torch.float32)
    graph.ndata["static"] = torch.tensor(static, dtype=torch.float32)

    dgl.save_graphs(output_path, [graph])
    print("base graph written:", output_path)


if not FINE_MESH_SLF:
    raise ValueError("Fill FINE_MESH_SLF at the top of the script.")
if not FINE_CLI:
    raise ValueError("Fill FINE_CLI at the top of the script.")
if not REGULAR_MESH_SLF:
    raise ValueError("Fill REGULAR_MESH_SLF at the top of the script.")
if not OUTPUT_DIR:
    raise ValueError("Fill OUTPUT_DIR at the top of the script.")

if FINE_RES_FILES:
    res_files = FINE_RES_FILES
else:
    res_files = sorted(glob.glob(os.path.join(FINE_RES_DIR, "*.res")))

if len(res_files) == 0:
    raise ValueError("No .res files found. Fill FINE_RES_DIR or FINE_RES_FILES.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("fine mesh:", FINE_MESH_SLF)
print("fine cli:", FINE_CLI)
print("regular mesh:", REGULAR_MESH_SLF)
print("output dir:", OUTPUT_DIR)
print("res files:")
for res_file in res_files:
    print(" -", res_file)

fine_mesh = TelemacFile(FINE_MESH_SLF)
fine_with_cli = TelemacFile(FINE_MESH_SLF, bnd_file=FINE_CLI)
regular_mesh = TelemacFile(REGULAR_MESH_SLF)

X_fine, triangles_fine = add_mesh_info(fine_mesh)
X_regular, triangles_regular = add_mesh_info(regular_mesh)

print("fine nodes:", len(X_fine), "fine triangles:", len(triangles_fine))
print("regular nodes:", len(X_regular), "regular triangles:", len(triangles_regular))

print("Building fine Delaunay...")
triangulation = Delaunay(X_fine)
print("Triangulation complete.")

interp_data = build_fast_interpolator(triangulation, X_regular)

base_graph_path = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_base.bin")
write_base_graph(
    regular_mesh,
    fine_mesh,
    fine_with_cli,
    interp_data,
    X_fine,
    X_regular,
    base_graph_path,
)

for traj_index, res_path in enumerate(res_files):
    print(f"Processing {res_path}")
    res = TelemacFile(res_path, bnd_file=FINE_CLI)
    fine_static = get_static_node_features(res, fine_mesh)
    n_times = int(res.times.shape[0])

    for start_ts in tqdm(range(0, n_times - 1, CHUNK_SIZE), desc=os.path.basename(res_path)):
        end_ts = min(start_ts + CHUNK_SIZE, n_times - 1)
        dynamic_data = []

        for ts in range(start_ts, end_ts):
            x_fine = get_dynamic_node_features(res, ts)
            x_future_fine = get_dynamic_node_features(res, ts + 1)

            x_fine = put_boundary_infos(x_fine, x_future_fine, fine_static)
            y_fine = get_node_outputs(x_fine, x_future_fine)
            y_fine = put_boundary_infos_on_changes(y_fine, fine_static)

            xy_fine = np.concatenate([x_fine, y_fine], axis=1)
            xy_regular = fast_interpolate(xy_fine, interp_data)
            x_regular = xy_regular[:, :3]
            y_regular = xy_regular[:, 3:]

            dynamic_data.append((x_regular, y_regular, int(ts)))

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{DATASET_NAME}_{traj_index}_{start_ts}-{end_ts}_interpolated.pkl",
        )
        if os.path.exists(output_path) and not OVERWRITE:
            raise FileExistsError(f"Output already exists: {output_path}")

        with open(output_path, "wb") as fp:
            pickle.dump(dynamic_data, fp)

        print("saved:", output_path)

    if hasattr(res, "close"):
        res.close()

for telemac in [fine_mesh, fine_with_cli, regular_mesh]:
    if hasattr(telemac, "close"):
        telemac.close()

print("Projection finished.")
