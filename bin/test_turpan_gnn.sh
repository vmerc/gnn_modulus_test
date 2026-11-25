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

srun apptainer exec --bind /tmpdir,/work --pwd "$bin_dir" --nv "$image" python -u - "$bin_dir" "$@" <<'PY'
import argparse
import pickle
import sys
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

parser = argparse.ArgumentParser()
parser.add_argument("bin_dir", type=Path)
parser.add_argument("--e8", action="store_true", help="Contrôler E8 et son témoin ponctuel.")
args = parser.parse_args()
bin_dir = args.bin_dir.resolve()
sys.path.insert(0, str(bin_dir))

import dgl
import hydra
import numpy as np
import torch

import train_push_source_nodes as training
from eval_tet_ghost import TEST_EVENTS, check_projected_geometry, rollout
from train_push_source_nodes import SourceNodeTrainer, seed_everything
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.launch.utils import save_checkpoint
from python.create_dgl_dataset import NodeType, collate_source_sequences
from python.eval_rollout import build_rollout_context
from python.ghost_nodes import edge_features_with_ghost_forcing
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
    e1 = hydra.compose(config_name="papier_ghost/E1")
    e5 = hydra.compose(config_name="papier_ghost/E5")
    if args.e8:
        names = ("E8_one_step", "E8", "E8_point_one_step", "E8_point")
        configs = [hydra.compose(config_name=f"papier_ghost/{name}") for name in names]
    else:
        names = ("E6", "E7_one_step", "E7")
        configs = [e1, e5, *(hydra.compose(config_name=f"papier_ghost/{name}") for name in names)]

dataset_dir = Path(e1.dataset_dir)
if not dataset_dir.is_dir():
    raise FileNotFoundError(f"Dataset historique introuvable: {dataset_dir}")

def load_samples(path, expected_timestamps):
    with open(path, "rb") as data_file:
        samples = pickle.load(data_file)
    timestamps = [int(ts) for _, _, ts in samples]
    assert timestamps == list(expected_timestamps), (path, timestamps)
    return samples

for path in e1.dynamic_dir:
    load_samples(path, range(35, 65))
for event in TEST_EVENTS:
    load_samples(dataset_dir / "x8_short" / f"{event}_{event}_0_35-64_interpolated.pkl", range(35, 65))
    load_samples(dataset_dir / "x8_full" / f"{event}_{event}_0_0-80_interpolated.pkl", range(80))

[simple_graph], _ = dgl.load_graphs(str(e1.data_dir))
[multimesh_graph], _ = dgl.load_graphs(str(e5.data_dir))
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

with closing(TelemacFile(str(e1.coarse_mesh_path))) as coarse_mesh:
    coarse_tri = coarse_mesh.tri


def check_e8_forcing(dataset, cfg, event_size, start):
    first_sequence = dataset[0]
    graph = dataset.base_graph_with_sources.to(dist.device)
    src, dst = graph.edges(order="eid")
    mask = graph.ndata["is_source"][src, 0]
    nodes = dst[mask].tolist()
    expected_q = {3751: 89.417, 3756: 1.820, 12770: 489.305}
    if cfg.experiment == "E8_point":
        expected_q = {12770: 580.543}
    assert set(nodes) == expected_q.keys(), nodes
    types = dataset.base_graph.ndata["static"][nodes, :NodeType.SIZE]
    print(f"{cfg.experiment}: types des sommets d’entrée {list(zip(nodes, types.tolist()))}")
    assert not torch.any(types[:, NodeType.PRESCRIBED_H]), "Une destination impose une hauteur"

    rows = first_sequence[0]["ghost_edge_features"]
    np.testing.assert_allclose(rows[:, 0].numpy() * 1000, [expected_q[node] for node in nodes], atol=0.001, rtol=0)
    np.testing.assert_allclose(rows[:, 1].sum().item() * 100, 159.987, atol=0.001, rtol=0)

    # VÉRIFIER le volume de chaque crue après insertion des lignes dans le batch DGL.
    batched = collate_source_sequences([first_sequence, dataset[event_size]])[0]
    batch_graph = batched["graph"].to(dist.device)
    batch_src, batch_dst = batch_graph.edges(order="eid")
    batch_mask = batch_graph.ndata["is_source"][batch_src, 0]
    features = edge_features_with_ghost_forcing(batch_graph, batched["ghost_edge_features"])
    destinations = batch_dst[batch_mask].reshape(2, -1)
    assert destinations.tolist() == [nodes, [node + graph.num_nodes() for node in nodes]]
    volumes = features[batch_mask, 0].reshape(2, -1).sum(dim=1).cpu().numpy() * 1000 * cfg.dt_seconds
    stop = start + cfg.dt_seconds
    expected_volumes = []
    for path in cfg.hydro_dir:
        hydro = dataset.hydrographs[path]
        times = hydro["time_seconds"]
        step = np.r_[start, times[(times > start) & (times < stop)], stop]
        expected_volumes.append(np.trapz(np.interp(step, times, hydro["q_values"][:, 0]), step))
    assert not np.isclose(*expected_volumes), "Les deux crues doivent avoir des volumes différents"
    np.testing.assert_allclose(volumes, expected_volumes, rtol=1e-6)

    # CONTRÔLER les attributs vus par le modèle à deux pas successifs, y compris le dernier.
    sequence = [
        (step["x_phys"].to(dist.device), step["x_src"].to(dist.device), step["y_phys"], step["ghost_edge_features"])
        for step in first_sequence[:2]
    ]
    seen = []

    def record_forcing(*, graph, x_phys, x_src, edge_features):
        seen.append(edge_features[mask].cpu())
        return torch.zeros_like(x_phys[:, :3])

    context = build_rollout_context(dataset, dist.device, dataset.use_q_feature)
    predictions = list(rollout(record_forcing, graph, sequence, context, n_steps=2, report_every=1))
    expected = torch.stack([step[3] for step in sequence])
    assert len(predictions) == 2 and not torch.allclose(expected[0], expected[1])
    torch.testing.assert_close(torch.stack(seen), expected)
    print(f"{cfg.experiment}: apports vérifiés par sommet, par crue dans le batch et aux deux pas du rollout")


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

        event_count = 2 if args.e8 else 1
        cfg.dynamic_dir = cfg.dynamic_dir[:event_count]
        cfg.hydro_dir = cfg.hydro_dir[:event_count]
        cfg.ckpt_path = str(run_dir / f"test-{experiment}")
        if experiment == "E6":
            cfg.init_ckpt_path = str(run_dir / "test-E5")
        elif cfg.sequence_length > 1:
            resume_epoch = training.configure_resume(cfg, Path(cfg.ckpt_path))
            assert resume_epoch == 600
        cfg.num_dataloader_workers = 0
        seed_everything(cfg.seed)
        trainer = SourceNodeTrainer(cfg, logger)
        dataset = trainer.dataset
        event_size = len(samples) - cfg.sequence_length + 1
        assert len(dataset) == event_size * event_count
        check_projected_geometry(dataset, coarse_tri)

        hydro = dataset.hydrographs[cfg.hydro_dir[0]]
        seen_q = np.array([dataset[i][0]["x_src"][0].tolist() for i in range(event_size)])
        for offset, feature in enumerate(dataset.source_feature_names):
            times = (np.array(timestamps) + offset) * cfg.dt_seconds
            expected_q = np.interp(times, hydro["time_seconds"], hydro["q_values"][:, 0])
            mean = dataset.source_stats[feature].item()
            std = dataset.source_stats[f"{feature}_std"].item()
            assert std > 0, "Hydrogramme de contrôle constant"
            np.testing.assert_allclose(
                seen_q[:, offset] * std + mean, expected_q[:event_size], rtol=1e-5, atol=1e-4
            )

        test_epoch = trainer.pf_start_epoch + cfg.get("pf_warmup_epochs", 0)
        if cfg.sequence_length > 1:
            assert trainer.epoch_init == cfg.init_epoch + 1 == cfg.pf_start_epoch
            assert trainer.p_tf(trainer.epoch_init) == 1 and trainer.p_tf(test_epoch) == 0
            if args.e8:
                check_e8_forcing(dataset, cfg, event_size, timestamps[0] * cfg.dt_seconds)
        if experiment == "E6":
            for filename in ("node_stats.json", "edge_stats.json", "source_stats.json"):
                assert (Path(cfg.ckpt_path) / filename).read_bytes() == (Path(cfg.init_ckpt_path) / filename).read_bytes()

        loss = trainer.train_batch(next(iter(trainer.dataloader)), epoch=test_epoch)
        assert torch.isfinite(loss), loss
        print(f"{experiment}: {len(dataset)} fenêtres, {dataset.source_feature_names} et états amont vérifiés, "
              f"loss={loss.item():.6g}")
        if experiment != "E1" and cfg.sequence_length == 1:
            save_checkpoint(
                cfg.ckpt_path, models=trainer.model, optimizer=trainer.optimizer,
                scheduler=trainer.scheduler, scaler=trainer.scaler, epoch=600,
            )
        if experiment == "E5":
            (Path(cfg.ckpt_path) / f"{trainer.model.meta.name}.0.610.mdlus").touch()
            resume_epoch = training.configure_resume(cfg, Path(cfg.ckpt_path))
            assert resume_epoch == 600
            resumed = SourceNodeTrainer(cfg, logger)
            assert resumed.epoch_init == 601
            print("E5: reprise du dernier checkpoint complet vérifiée")
            del resumed
PY
