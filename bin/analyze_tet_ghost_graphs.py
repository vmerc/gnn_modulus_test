#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from contextlib import closing
from pathlib import Path

import dgl
import numpy as np
from omegaconf import OmegaConf


BIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BIN_DIR.parent))

from prepare_x8_short_ghost_dataset import find_single_file  # noqa: E402
from python.create_dgl_dataset import get_dgl_graph  # noqa: E402
from python.python_code.data_manip.extraction.telemac_file import TelemacFile  # noqa: E402


EXPECTED_SIZES = {
    "original": (412_844, 2_464_330),
    "x8": (16_222, 96_984),
    "x8_multimesh": (16_222, 130_601),
}


def load_graph(path: Path):
    graphs, _ = dgl.load_graphs(str(path))
    if len(graphs) != 1:
        raise ValueError(f"{path}: attendu un graphe, trouvé {len(graphs)}.")
    return graphs[0]


def summarize(name, graph, lengths):
    sources = graph.edges(order="eid")[0].numpy()
    degree = np.bincount(sources, minlength=graph.num_nodes())
    if np.any(degree == 0):
        raise ValueError(f"{name}: nœud sans arête sortante.")
    mean_outgoing = np.bincount(
        sources, weights=lengths, minlength=graph.num_nodes()
    ) / degree
    r10 = 10 * mean_outgoing
    return {
        "graph": name,
        "nodes": graph.num_nodes(),
        "directed_edges": graph.num_edges(),
        "mean_edge_m": lengths.mean(),
        "r10_median_m": np.median(r10),
        "r10_mean_m": r10.mean(),
        "r10_max_m": r10.max(),
    }


def main() -> None:
    cfg = OmegaConf.load(BIN_DIR / "conf" / "papier_ghost" / "base.yaml")
    output_dir = Path(cfg.article_results_dir) / "graph"

    fine_res_dir = Path(cfg.fine_res_dir)
    fine_mesh_path = find_single_file(fine_res_dir, ["*.slf", "*.geo"], "fine mesh")
    with closing(TelemacFile(str(fine_mesh_path))) as fine_mesh:
        original_graph, original_features = get_dgl_graph(fine_mesh.tri)

    x8_graph = load_graph(Path(cfg.data_dir))
    multimesh_graph = load_graph(Path(cfg.multimesh_path))
    graphs = {
        "original": (original_graph, original_features[:, 2]),
        "x8": (x8_graph, x8_graph.edata["x"][:, 2].numpy()),
        "x8_multimesh": (
            multimesh_graph,
            multimesh_graph.edata["x"][:, 2].numpy(),
        ),
    }
    rows = [summarize(name, graph, lengths) for name, (graph, lengths) in graphs.items()]
    for row in rows:
        size = (row["nodes"], row["directed_edges"])
        if size != EXPECTED_SIZES[row["graph"]]:
            raise ValueError(
                f"{row['graph']}: taille {size}, attendu {EXPECTED_SIZES[row['graph']]}."
            )
    unique_lengths = {
        name: np.unique(lengths) for name, (_, lengths) in graphs.items()
    }

    output_dir.mkdir(parents=True)
    with (output_dir / "graph_summary_table1.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / "edge_lengths_figure2.npz",
        **unique_lengths,
    )

    for row in rows:
        print(
            f"{row['graph']}: nodes={row['nodes']}, "
            f"edges={row['directed_edges']}, mean_edge={row['mean_edge_m']:.3f} m, "
            f"r10 median/mean/max={row['r10_median_m']:.3f}/"
            f"{row['r10_mean_m']:.3f}/{row['r10_max_m']:.3f} m"
        )
    print(f"Résultats : {output_dir}")


if __name__ == "__main__":
    main()
