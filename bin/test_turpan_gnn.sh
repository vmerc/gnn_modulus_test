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

srun apptainer exec --bind /tmpdir,/work --pwd "$bin_dir" --nv "$image" python -u - <<'PY'
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import torch

import train_push_source_nodes as training
from train_push_source_nodes import SourceNodeTrainer, seed_everything
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.launch.utils import save_checkpoint

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

with TemporaryDirectory(prefix=".check-", dir=configs[0].campaign_dir) as temp_dir:
    run_dir = Path(temp_dir)
    for cfg in configs:
        experiment = cfg.experiment
        for path in [cfg.data_dir, *cfg.dynamic_dir, *cfg.hydro_dir]:
            assert Path(path).is_file(), path

        with open(cfg.dynamic_dir[0], "rb") as data_file:
            samples = pickle.load(data_file)
        timestamps = [int(ts) for _, _, ts in samples]
        assert timestamps == list(range(35, 65)), timestamps
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
