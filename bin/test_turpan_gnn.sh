#!/bin/bash
#SBATCH --job-name=test-tet-ghost
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

set -euo pipefail

submit_dir=${SLURM_SUBMIT_DIR:?Répertoire de soumission Slurm absent}
if [ -d "$submit_dir/conf/papier_ghost" ]; then
  bin_dir=$submit_dir
else
  bin_dir=$submit_dir/bin
fi
cd -- "$bin_dir"
image=/work/conteneurs/sessions-interactives/modulus-24.01-calmip-si.sif

srun apptainer exec --bind /tmpdir,/work --pwd "$bin_dir" --nv "$image" python -u - "$bin_dir" <<'PY'
import pickle
import sys
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

bin_dir = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(bin_dir))

import dgl
import hydra
import numpy as np
import torch

import train_push_source_nodes as training
from eval_tet_ghost import TEST_EVENTS, check_projected_geometry
from train_push_source_nodes import SourceNodeTrainer, seed_everything
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.launch.utils import save_checkpoint
from python.python_code.data_manip.extraction.telemac_file import TelemacFile

DistributedManager.initialize()
dist = DistributedManager()
assert dist.device.type == "cuda", "CUDA indisponible dans le conteneur"
logger = RankZeroLoggingWrapper(PythonLogger("test"), dist)

conf_dir = Path(training.__file__).resolve().parent / "conf"
print(f"Configuration Hydra: {conf_dir}")
if not conf_dir.is_dir():
    raise FileNotFoundError(f"Dossier de configuration introuvable: {conf_dir}")

with hydra.initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
    configs = [hydra.compose(config_name=f"papier_ghost/{name}") for name in ("E1", "E5", "E6")]

dataset_dir = Path(configs[0].dataset_dir)
if not dataset_dir.is_dir():
    raise FileNotFoundError(f"Dataset historique introuvable: {dataset_dir}")

def load_samples(path, expected_timestamps):
    with open(path, "rb") as data_file:
        samples = pickle.load(data_file)
    timestamps = [int(ts) for _, _, ts in samples]
    assert timestamps == list(expected_timestamps), (path, timestamps)
    return samples

for path in configs[0].dynamic_dir:
    load_samples(path, range(35, 65))
for event in TEST_EVENTS:
    load_samples(dataset_dir / "x8_short" / f"{event}_{event}_0_35-64_interpolated.pkl", range(35, 65))
    load_samples(dataset_dir / "x8_full" / f"{event}_{event}_0_0-80_interpolated.pkl", range(80))

[simple_graph], _ = dgl.load_graphs(str(configs[0].data_dir))
[multimesh_graph], _ = dgl.load_graphs(str(configs[1].data_dir))
assert (simple_graph.num_nodes(), simple_graph.num_edges()) == (16222, 96984), (
    "Taille du graphe ×8 inattendue"
)
assert (multimesh_graph.num_nodes(), multimesh_graph.num_edges()) == (16222, 130601), (
    "Taille du multimaillage inattendue"
)
if not torch.equal(simple_graph.ndata["static"][:, :4], multimesh_graph.ndata["static"][:, :4]):
    raise ValueError("Les types de frontière diffèrent entre les graphes ×8 et multimesh.")
torch.testing.assert_close(simple_graph.ndata["static"][:, 4:], multimesh_graph.ndata["static"][:, 4:])
simple_edges = set(zip(*(ids.tolist() for ids in simple_graph.edges())))
multimesh_edges = set(zip(*(ids.tolist() for ids in multimesh_graph.edges())))
assert simple_edges.issubset(multimesh_edges), "Le multimaillage ne contient pas toutes les arêtes ×8"
src, dst = multimesh_graph.edges()
assert torch.count_nonzero(src == dst) == 39, "Nombre de self-loops multimesh inattendu"

with closing(TelemacFile(str(configs[0].coarse_mesh_path))) as coarse_mesh:
    coarse_tri = coarse_mesh.tri

with TemporaryDirectory(prefix=".check-") as temp_dir:
    run_dir = Path(temp_dir)
    for cfg in configs:
        experiment = cfg.experiment
        for path in [cfg.data_dir, *cfg.dynamic_dir, *cfg.hydro_dir]:
            assert Path(path).is_file(), path

        samples = load_samples(cfg.dynamic_dir[0], range(35, 65))
        timestamps = [int(ts) for _, _, ts in samples]
        inlet = cfg.inlet_node_lists[0][0]
        inlet_states = np.stack([x[inlet] for x, _, _ in samples])
        inlet_deltas = np.stack([y[inlet] for _, y, _ in samples])
        assert np.any(inlet_deltas[:, 1:] != 0), "Incréments de vitesse nuls à l'entrée Q"
        np.testing.assert_allclose(inlet_states[:-1] + inlet_deltas[:-1], inlet_states[1:], atol=1e-5)

        cfg.dynamic_dir = [cfg.dynamic_dir[0]]
        cfg.hydro_dir = [cfg.hydro_dir[0]]
        cfg.ckpt_path = str(run_dir / f"test-{experiment}")
        if experiment == "E6":
            cfg.init_ckpt_path = str(run_dir / "test-E5")
        cfg.num_dataloader_workers = 0
        seed_everything(cfg.seed)
        trainer = SourceNodeTrainer(cfg, logger)
        dataset = trainer.dataset
        assert len(dataset) == len(samples) - cfg.sequence_length + 1
        check_projected_geometry(dataset, coarse_tri)

        hydro = dataset.hydrographs[cfg.hydro_dir[0]]
        expected_q = np.interp(
            np.array(timestamps) * cfg.dt_seconds, hydro["time_seconds"], hydro["q_values"][:, 0]
        )
        mean = dataset.source_stats["q"].item()
        std = dataset.source_stats["q_std"].item()
        assert std > 0, "Hydrogramme de contrôle constant"
        seen_q = [dataset[i][0]["x_src"][0, 0].item() * std + mean for i in range(len(dataset))]
        np.testing.assert_allclose(seen_q, expected_q[:len(dataset)], rtol=1e-5, atol=1e-4)

        test_epoch = trainer.pf_start_epoch + cfg.get("pf_warmup_epochs", 0)
        if experiment == "E6":
            assert trainer.epoch_init == cfg.init_epoch + 1
            assert trainer.p_tf(trainer.epoch_init) == 1 and trainer.p_tf(test_epoch) == 0
            for filename in ("node_stats.json", "edge_stats.json", "source_stats.json"):
                assert (Path(cfg.ckpt_path) / filename).read_bytes() == (Path(cfg.init_ckpt_path) / filename).read_bytes()

        loss = trainer.train_batch(next(iter(trainer.dataloader)), epoch=test_epoch)
        assert torch.isfinite(loss), loss
        print(f"{experiment}: {len(dataset)} fenêtres, Q(t) et états amont vérifiés, loss={loss.item():.6g}")
        if experiment == "E5":
            save_checkpoint(
                cfg.ckpt_path, models=trainer.model, optimizer=trainer.optimizer,
                scheduler=trainer.scheduler, scaler=trainer.scaler, epoch=600,
            )
PY
