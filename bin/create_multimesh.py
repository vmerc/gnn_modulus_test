import argparse
import os
import sys
from pathlib import Path

start_directory = Path.cwd()
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_path)

from python.create_dgl_dataset import create_multimesh


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add projected coarse-mesh edges to a DGL base graph."
    )
    parser.add_argument("--base-graph", required=True)
    parser.add_argument("--fine-mesh", required=True)
    parser.add_argument("--coarse-mesh", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    def absolute_path(path):
        path = Path(path).expanduser()
        return path if path.is_absolute() else start_directory / path

    output_path = create_multimesh(
        base_graph_path=absolute_path(args.base_graph),
        fine_mesh_path=absolute_path(args.fine_mesh),
        coarse_mesh_paths=[absolute_path(path) for path in args.coarse_mesh],
        output_path=absolute_path(args.output),
    )
    print("written:", output_path)


if __name__ == "__main__":
    main()
