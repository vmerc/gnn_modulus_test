#!/usr/bin/env python3
from __future__ import annotations

import filecmp
import os
import shutil
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
    build_interpolation_plan,
    build_x8_base_dataset,
    find_paper_results,
    find_single_file,
)
from python.create_dgl_dataset import add_mesh_info, create_multimesh  # noqa: E402
from python.python_code.data_manip.extraction.telemac_file import TelemacFile  # noqa: E402


EXPECTED_NODES = 16_222
EXPECTED_BASE_EDGES = 96_984
EXPECTED_MULTIMESH_EDGES = 130_597


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


def remove_surplus_self_loops(multimesh, historical):
    src, dst = multimesh.edges(order="eid")
    edge_pairs = list(zip(src.tolist(), dst.tolist()))
    generated_edges = set(edge_pairs)
    historical_edges = set(zip(*(ids.tolist() for ids in historical.edges())))
    generated_only = generated_edges - historical_edges
    historical_only = historical_edges - generated_edges
    if not generated_only and not historical_only:
        return multimesh
    if (
        len(generated_only) != 4
        or historical_only
        or any(source != destination for source, destination in generated_only)
    ):
        raise ValueError(
            "The regenerated graph does not differ from the published graph "
            f"only by four self-loops: generated_only={sorted(generated_only)[:10]}, "
            f"historical_only={sorted(historical_only)[:10]}"
        )

    keep = torch.tensor(
        [edge not in generated_only for edge in edge_pairs], dtype=torch.bool
    )
    filtered = dgl.graph((src[keep], dst[keep]), num_nodes=multimesh.num_nodes())
    for name, values in multimesh.ndata.items():
        filtered.ndata[name] = values.clone()
    for name, values in multimesh.edata.items():
        filtered.edata[name] = values[keep].clone()
    print(
        "Removed four regenerated self-loops absent from the published topology: "
        f"{sorted(generated_only)}"
    )
    return filtered


def check_graphs(base_graph, multimesh, cfg) -> None:
    historical_base = load_graph(Path(cfg.historical_base_graph_path))
    historical_multimesh = load_graph(Path(cfg.historical_multimesh_path))

    for label, graph in (
        ("regenerated multimesh", multimesh),
        ("historical multimesh", historical_multimesh),
    ):
        src, dst = graph.edges()
        print(
            f"{label}: edges={graph.num_edges()}, "
            f"self_loops={int(torch.count_nonzero(src == dst).item())}"
        )

    check_same_edges(base_graph, historical_base, "x8 graph")
    check_same_edges(multimesh, historical_multimesh, "x8-x16-x32 graph")
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
    if self_loops == 0:
        raise ValueError("The historical multimesh topology requires self-loops.")
    print(
        f"Validated manuscript topology: nodes={multimesh.num_nodes()}, "
        f"directed_edges={multimesh.num_edges()}, self_loops={self_loops}"
    )


def build_graphs(output_dir: Path, cfg) -> None:
    source_case_dir = Path(cfg.fine_res_dir)
    fine_mesh_path = find_single_file(source_case_dir, ["*.slf", "*.geo"], "fine mesh")
    cli_path = find_single_file(
        source_case_dir, ["*.cli", "*.conlim", "cli", "conlim"], "boundary file"
    )
    first_result_path = next(iter(find_paper_results(source_case_dir).values()))
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
    historical_multimesh = load_graph(Path(cfg.historical_multimesh_path))
    multimesh = remove_surplus_self_loops(multimesh, historical_multimesh)
    dgl.save_graphs(str(multimesh_path), [multimesh])
    check_graphs(base_graph, multimesh, cfg)
    shutil.copy2(x8_mesh_path, output_dir / x8_mesh_path.name)


def main() -> None:
    cfg = OmegaConf.load(SCRIPT_DIR / "conf" / "papier_ghost" / "base.yaml")
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
