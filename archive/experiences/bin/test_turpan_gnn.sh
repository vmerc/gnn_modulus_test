#!/bin/bash
#SBATCH --job-name=test-turpan-gnn
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:03:00

set -euo pipefail

repo_dir="${TURPAN_REPO_DIR:-$(cd "$SLURM_SUBMIT_DIR/.." && pwd -P)}"
image="${1:-/work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif}"

srun apptainer exec --nv --no-home --bind "$repo_dir" --pwd "$repo_dir" "$image" \
    python -u - <<'PY'
import dgl
import torch

from python.CustomMeshGraphNet import MeshGraphNet

assert torch.cuda.is_available(), "CUDA indisponible dans le conteneur"
torch.manual_seed(0)
device = torch.device("cuda")
graph = dgl.graph(([0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]), num_nodes=3, device=device)
model = MeshGraphNet(
    input_dim_nodes=10,
    input_dim_edges=3,
    output_dim=3,
    processor_size=2,
    hidden_dim_processor=16,
    hidden_dim_node_encoder=16,
    hidden_dim_edge_encoder=16,
    hidden_dim_node_decoder=16,
).to(device)

prediction = model(torch.ones((3, 10), device=device), torch.ones((6, 3), device=device), graph)
assert prediction.shape == (3, 3), prediction.shape
assert torch.isfinite(prediction).all()
loss = prediction.square().mean()
loss.backward()
gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
print(f"GPU {torch.cuda.get_device_name(0)}")
print(f"Sortie {tuple(prediction.shape)} | loss {loss.item():.6g} | backward OK")
PY
