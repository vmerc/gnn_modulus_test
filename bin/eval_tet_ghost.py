import argparse
import csv
import sys
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from hydra import compose, initialize_config_dir

BIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BIN_DIR.parent))

from python.CustomMeshGraphNet import MeshGraphNetWithSourceNodes
from python.create_dgl_dataset import (
    TelemacDatasetWithSourceNodes,
    get_edge_features_from_positions,
    unpack_dynamic_sample,
)
from python.eval_rollout import (
    build_rollout_context,
    denormalize_delta,
    denormalize_state,
    renormalize_state,
)
from python.ghost_nodes import edge_features_with_ghost_forcing, load_inlet_sections
from python.python_code.data_manip.extraction.telemac_file import TelemacFile
from python.python_code.data_manip.formats.regular_grid import interpolate_on_grid


TEST_EVENTS = (
    "Group_1_peak_2600", "Group_2_peak_1000", "Group_2_peak_1200", "Group_2_peak_1600",
    "Group_4_peak_2000", "Group_1_peak_1200", "Group_1_peak_2400", "Group_3_peak_3400",
    "Group_1_peak_1400", "Group_1_peak_2000", "Group_1_peak_2200", "Group_2_peak_3600",
    "Group_3_peak_2200", "Group_3_peak_2800", "Group_4_peak_1200", "Group_4_peak_3000",
)
SHORT_FIRST_STEP = 35
MAIN_HOURS = 12
FULL_HOURS = 40
GRID_STEP_M = 25.0
EXPECTED_GRID_SHAPE = (689, 748)
EXPECTED_FINE_GRID_POINTS = 276_001
EXPECTED_COMMON_GRID_POINTS = 275_827
THRESHOLDS = (0.05, 0.30)
STATE_METRICS = ("mae_h", "mae_u", "mae_v")
CSI_METRICS = ("csi_5cm", "csi_30cm")
MAP_EVENT = "Group_3_peak_3400"
LONG_MAP_EVENT = "Group_3_peak_2800"
LONG_MAP_HOURS = (6, 12, 24, 36)


def build_test_dataset(cfg, sequence_length, full=False):
    # CHARGER les 16 crues de test, leurs états ×8 et leurs hydrogrammes Q(t).
    hydro_root = Path(cfg.get("hydro_data_dir", Path(cfg.hydro_dir[0]).parent))
    hydro_files = [hydro_root / f"generated_hydrographs_{event}.liq" for event in TEST_EVENTS]
    # VÉRIFIER qu'aucune crue de test ne figure dans l'apprentissage.
    if {path.name for path in hydro_files} & {Path(path).name for path in cfg.hydro_dir}:
        raise ValueError("Une crue de test apparaît dans la liste des hydrogrammes d’entraînement.")

    data_root = Path(cfg.dataset_dir) / "x8_full" if full else Path(cfg.short_data_dir)
    window = "0-80" if full else "35-64"
    # CHARGER le graphe ×8 ou multimesh, puis lui ajouter le ghost node amont.
    ds = TelemacDatasetWithSourceNodes(
        data_dir=cfg.data_dir,
        dynamic_data_files=[
            data_root / f"{event}_{event}_0_{window}_interpolated.pkl"
            for event in TEST_EVENTS
        ],
        hydro_data_files=hydro_files,
        inlet_node_lists=cfg.get("inlet_node_lists"),
        inlet_sections=load_inlet_sections(cfg),
        cli_file=cfg.get("cli_file"),
        use_q_feature=cfg.use_q_feature,
        use_next_q=cfg.get("use_next_q", False),
        split="test",
        ckpt_path=cfg.ckpt_path,
        sequence_length=sequence_length,
        overlap=0,
        dt_seconds=cfg.dt_seconds,
    )
    if [meta["hydro_path"] for meta in ds.sequence_meta] != [str(path) for path in hydro_files]:
        raise ValueError(f"Chaque crue de test doit fournir une seule séquence pour la fenêtre {window}.")
    return ds


def make_grid(tri):
    # CONSTRUIRE la grille commune de 25 m utilisée pour comparer GNN et TELEMAC.
    axes = [
        coords.min() + GRID_STEP_M * np.arange(int(np.ceil(np.ptp(coords) / GRID_STEP_M)) + 1)
        for coords in (tri.x, tri.y)
    ]
    grid = np.meshgrid(*axes)
    if tuple(sorted(grid[0].shape)) != EXPECTED_GRID_SHAPE:
        raise ValueError(
            f"Grille fine inattendue : {grid[0].shape}, attendu {EXPECTED_GRID_SHAPE}."
        )
    return grid


def depth_grids(prediction, target_grid, coarse_tri, grid):
    # INTERPOLER la prédiction ×8 et conserver l'intersection des points valides.
    predicted_grid, _ = interpolate_on_grid(coarse_tri, prediction.astype(np.float64), grid=grid)
    predicted_values = predicted_grid.filled(np.nan)
    target_values = target_grid.filled(np.nan)
    fine_valid = np.isfinite(target_values)
    fine_points = np.count_nonzero(fine_valid)
    if fine_points != EXPECTED_FINE_GRID_POINTS:
        raise ValueError(
            f"Domaine TELEMAC inattendu : {fine_points} points valides, "
            f"attendu {EXPECTED_FINE_GRID_POINTS}."
        )
    valid = np.isfinite(predicted_values) & fine_valid
    common_points = np.count_nonzero(valid)
    if common_points != EXPECTED_COMMON_GRID_POINTS:
        raise ValueError(
            f"Support commun ×8/TELEMAC inattendu : {common_points} points valides, "
            f"attendu {EXPECTED_COMMON_GRID_POINTS}."
        )
    return predicted_values, target_values, valid


def compute_csi(wet_pred, wet_target):
    union = np.count_nonzero(wet_pred | wet_target)
    return np.count_nonzero(wet_pred & wet_target) / union if union else np.nan


def check_projected_geometry(ds, coarse_tri):
    if ds.num_physical_nodes != len(coarse_tri.x):
        raise ValueError("Le graphe et le maillage ×8 ont des nombres de nœuds différents.")

    edges = np.stack([endpoint.numpy() for endpoint in ds.base_graph.edges(order="eid")])
    coordinates = np.column_stack((coarse_tri.x, coarse_tri.y))
    expected = get_edge_features_from_positions(coordinates, edges)
    for index, feature in enumerate(("xrel", "yrel", "norm")):
        mean = ds.edge_stats[feature].item()
        std = ds.edge_stats[f"{feature}_std"].item()
        if std != 0.0:
            expected[:, index] = (expected[:, index] - mean) / std
    if not np.allclose(expected, ds.base_graph.edata["x"].numpy(), rtol=1e-5, atol=2e-5):
        raise ValueError("Les déplacements d’arêtes du graphe ne correspondent pas au maillage ×8.")


def read_reference_depths(event, fine_res_dir, fine_tri, grid, times, dt_seconds):
    candidates = [fine_res_dir / name for name in (f"{event}.res", f"{event}_{event}.res")]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"{event}: attendu un seul fichier Telemac parmi {candidates}, trouvé {matches}.")

    with closing(TelemacFile(str(matches[0]))) as reference:
        if fine_tri is None:
            fine_tri = reference.tri
            grid = make_grid(fine_tri)
            print(f"Grille fine {grid[0].shape}, pas de {GRID_STEP_M:g} m")
        elif not (
            np.array_equal(reference.tri.x, fine_tri.x)
            and np.array_equal(reference.tri.y, fine_tri.y)
            and np.array_equal(reference.tri.triangles, fine_tri.triangles)
        ):
            raise ValueError(f"{event}: maillage différent de la première référence Telemac.")

        depth_name = "WATER DEPTH" if "WATER DEPTH" in reference.varnames else "HAUTEUR D'EAU"
        depth_grids = []
        for record in times:
            if not np.isclose(reference.get_data_time(record), record * dt_seconds, rtol=0, atol=1e-3):
                raise ValueError(f"{event}: temps Telemac inattendu pour l’indice {record}.")
            depth = reference.get_data_value(depth_name, record)
            depth_grid, _ = interpolate_on_grid(fine_tri, depth.astype(np.float64), grid=grid)
            depth_grids.append(depth_grid)
    return fine_tri, grid, depth_grids


def predict_delta(model, graph, x_phys, x_src, ghost_edge_features):
    edge_features = edge_features_with_ghost_forcing(graph, ghost_edge_features)
    return model(graph=graph, x_phys=x_phys, x_src=x_src, edge_features=edge_features)


def evaluation_sequence(ds, index, expected_times, device):
    times = [unpack_dynamic_sample(sample)[2] for sample in ds.sequences[index]]
    if times != expected_times:
        raise ValueError(f"{TEST_EVENTS[index]}: indices temporels incorrects dans la fenêtre ×8.")
    return [
        (
            sample["x_phys"].to(device),
            sample["x_src"].to(device),
            sample["y_phys"],
            sample.get("ghost_edge_features"),
        )
        for sample in ds[index]
    ]


def one_step_normalized_error(model, graph, sample, device):
    # À chaque pas indépendant, x_phys contient l'état de référence.
    x_phys, x_src, target_delta, ghost_edge_features = sample
    delta = predict_delta(model, graph, x_phys, x_src, ghost_edge_features)
    return delta - target_delta.to(device)


def rollout(model, graph, sequence, context, n_steps, report_every):
    start = context.dyn_start
    x_phys = sequence[0][0]
    static = x_phys[:, :start]
    h_mask = (static[:, :4] == static.new_tensor([0, 1, 0, 0])).all(dim=1)

    # POUR chaque pas : prédire un incrément, le dénormaliser et mettre à jour l'état.
    for step in range(1, n_steps + 1):
        previous_phys, src, delta_target, ghost_edge_features = sequence[step - 1]
        state = denormalize_state(x_phys[:, start:start + 3], context)
        delta = predict_delta(model, graph, x_phys, src, ghost_edge_features)
        prediction = state + denormalize_delta(delta, context)
        # IMPOSER la hauteur aval prescrite portée par l'entrée au temps de prévision.
        prediction[h_mask, 0] = state[h_mask, 0]

        if report_every and step % report_every == 0:
            if step < len(sequence):
                target = denormalize_state(sequence[step][0][:, start:start + 3], context)
            else:
                # The last full-event record stores the transition from 39.5 to 40 h.
                target = denormalize_state(previous_phys[:, start:start + 3], context)
                target += denormalize_delta(delta_target.to(context.device), context)
            target[h_mask, 0] = state[h_mask, 0]
            yield step, prediction, target

        if step < n_steps:
            # CONSTRUIRE l'entrée suivante avec la prédiction et la prochaine hauteur aval.
            next_phys = sequence[step][0]
            next_depth = next_phys[h_mask, start] * context.state_std[0] + context.state_mean[0]
            prediction[h_mask, 0] = next_depth
            x_phys = torch.cat((static, renormalize_state(prediction, context), next_phys[:, start + 3:]), dim=1)


def write_rollout_timing(model, graph, sequence, context, steps_per_hour, output_dir):
    gpu = torch.cuda.get_device_name(context.device)
    n_steps = 6 * steps_per_hour
    # E3/E6 : ÉCHAUFFER puis mesurer trois fois le rollout autorégressif de 6 h.
    for _ in rollout(model, graph, sequence, context, n_steps, report_every=0):
        pass
    torch.cuda.synchronize(context.device)
    times = []
    for _ in range(3):
        start = perf_counter()
        for _ in rollout(model, graph, sequence, context, n_steps, report_every=0):
            pass
        torch.cuda.synchronize(context.device)
        times.append(perf_counter() - start)
    print(f"Rollout 6 h sur {gpu}: médiane {np.median(times):.6f} s après un échauffement")
    with (output_dir / "timing_6h.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("event", "gpu", "horizon_h", "repeat", "seconds"))
        writer.writerows((MAP_EVENT, gpu, 6, index, seconds) for index, seconds in enumerate(times, start=1))


@torch.inference_mode()
def evaluate(model, ds, fine_res_dir, coarse_tri, steps_per_hour, device, experiment, output_dir, full=False):
    # VÉRIFIER que le graphe chargé correspond bien à la géométrie du maillage ×8.
    check_projected_geometry(ds, coarse_tri)
    context = build_rollout_context(ds, device, ds.use_q_feature)
    graph = ds.base_graph_with_sources.to(device)
    fine_csi = not full and ds.use_q_feature
    figure8 = not full and experiment in {"E3", "E6"}
    figure10 = full and experiment == "E5"
    first_step = 0 if full else SHORT_FIRST_STEP
    n_steps = (FULL_HOURS if full else MAIN_HOURS) * steps_per_hour
    report_every = 1 if full else steps_per_hour
    expected_times = list(range(first_step, first_step + n_steps + (0 if full else 1)))

    fine_tri = None
    grid = None
    long_maps = []
    rows = []
    # POUR chaque crue de test, dérouler le modèle sans réinjecter les états cibles.
    for index, event in enumerate(TEST_EVENTS):
        print(f"Rollout {'complet' if full else 'sur fenêtre courte'} {event}")
        sequence = evaluation_sequence(ds, index, expected_times, device)

        if fine_csi:
            # CHARGER les hauteurs TELEMAC sur la grille commune.
            fine_tri, grid, fine_depths = read_reference_depths(
                event, fine_res_dir, fine_tri, grid,
                expected_times[steps_per_hour::steps_per_hour], ds.dt_seconds,
            )
        if figure8 and event == MAP_EVENT:
            write_rollout_timing(model, graph, sequence, context, steps_per_hour, output_dir)

        for step, prediction, target in rollout(model, graph, sequence, context, n_steps, report_every):
            hour = step * ds.dt_seconds / 3600
            # CALCULER la MAE (norme L1 du manuscrit) séparément sur h, u et v.
            mae = (prediction - target).abs().mean(dim=0)
            if not torch.isfinite(mae).all():
                raise ValueError(f"{event}: erreur absolue non finie à {hour:g} h.")
            csi = []
            if fine_csi:
                # CALCULER le CSI à 5 cm et 30 cm sur les 275 827 points communs.
                predicted_depth, target_depth, valid = depth_grids(
                    prediction[:, 0].cpu().numpy(), fine_depths[step // steps_per_hour - 1], coarse_tri, grid
                )
                for threshold in THRESHOLDS:
                    flooded = (predicted_depth >= threshold) & valid
                    reference_flooded = (target_depth >= threshold) & valid
                    score = compute_csi(flooded, reference_flooded)
                    csi.append(score)
                    if figure8 and event == MAP_EVENT and hour == 6 and threshold == 0.30:
                        # E3/E6 : CONSERVER les masques nécessaires à la figure 8.
                        np.savez_compressed(
                            output_dir / "map_figure8.npz", event=event, hour=hour, threshold_m=threshold,
                            x=grid[0][0], y=grid[1][:, 0], valid=valid,
                            flooded=flooded, reference_flooded=reference_flooded, csi=score,
                        )
            if figure10 and event == LONG_MAP_EVENT and hour in LONG_MAP_HOURS:
                # E5 : CONSERVER les masques nécessaires à la figure 10.
                flooded = prediction[:, 0].cpu().numpy() >= 0.10
                reference_flooded = target[:, 0].cpu().numpy() >= 0.10
                long_maps.append((flooded, reference_flooded, compute_csi(flooded, reference_flooded)))
            rows.append((event, hour, *mae.tolist(), *csi))
    if figure10:
        flooded, reference_flooded, csi = zip(*long_maps)
        np.savez_compressed(
            output_dir / "map_figure10.npz", event=LONG_MAP_EVENT, hours=LONG_MAP_HOURS, threshold_m=0.10,
            x=coarse_tri.x, y=coarse_tri.y, triangles=coarse_tri.triangles,
            flooded=np.stack(flooded), reference_flooded=np.stack(reference_flooded), csi=csi,
        )
    return rows


@torch.inference_mode()
def evaluate_one_step(model, ds, coarse_tri, n_steps, device):
    check_projected_geometry(ds, coarse_tri)
    context = build_rollout_context(ds, device, ds.use_q_feature)
    graph = ds.base_graph_with_sources.to(device)
    types = ds.physical_static[:, :4].to(device)
    h_mask = (types == types.new_tensor([0, 1, 0, 0])).all(dim=1)
    expected_times = list(range(SHORT_FIRST_STEP, SHORT_FIRST_STEP + n_steps))

    rows = []
    for index, event in enumerate(TEST_EVENTS):
        sequence = evaluation_sequence(ds, index, expected_times, device)
        for ts, sample in zip(expected_times, sequence):
            error_normalized = one_step_normalized_error(model, graph, sample, device)
            mse = error_normalized.square().mean()

            # À état initial commun, l'erreur d'état est celle de l'incrément.
            # La hauteur aval est imposée dans les deux états, comme dans le rollout.
            error = error_normalized.abs() * context.delta_std
            error[h_mask, 0] = 0
            metrics = torch.cat((error.mean(dim=0), mse.view(1)))
            start_time_h = ts * ds.dt_seconds / 3600
            if not torch.isfinite(metrics).all():
                raise ValueError(f"{event}: erreur non finie pour un départ à {start_time_h:g} h.")
            rows.append((event, start_time_h, *metrics.tolist()))
    return rows


@torch.inference_mode()
def spatial_errors(model, ds, coarse_tri, steps_per_hour, last_hour, device):
    """Calcule les erreurs par sommet ×8 pour chaque crue de test."""
    check_projected_geometry(ds, coarse_tri)
    context = build_rollout_context(ds, device, ds.use_q_feature)
    graph = ds.base_graph_with_sources.to(device)
    n_steps = MAIN_HOURS * steps_per_hour
    expected_times = list(range(SHORT_FIRST_STEP, SHORT_FIRST_STEP + n_steps + 1))

    mae_u, bias_u, mae_h = [], [], []
    for index, event in enumerate(TEST_EVENTS):
        sequence = evaluation_sequence(ds, index, expected_times, device)
        error_u = torch.stack([
            one_step_normalized_error(model, graph, sample, device)[:, 1]
            for sample in sequence[:-1]
        ]) * context.delta_std[1]
        # L'écart est signé avant de calculer le biais et la moyenne des erreurs absolues.
        mae_u.append(error_u.abs().mean(dim=0).cpu().numpy())
        bias_u.append(error_u.mean(dim=0).cpu().numpy())

        error_h = torch.stack([
            (prediction[:, 0] - target[:, 0]).abs()
            for _, prediction, target in rollout(
                model, graph, sequence, context, last_hour * steps_per_hour, steps_per_hour
            )
        ])
        mae_h.append(error_h.cpu().numpy())
        print(f"Erreurs spatiales calculées pour {event}")

    fields = tuple(np.stack(series) for series in (mae_u, bias_u, mae_h))
    if any(not np.isfinite(field).all() for field in fields):
        raise ValueError("Erreurs non finies dans le diagnostic spatial.")
    return fields


def write_results(rows, output_dir, prefix, metrics, time_column="horizon_h"):
    # ÉCRIRE une ligne par crue et par instant évalué.
    with (output_dir / f"{prefix}_by_event.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("event", time_column, *metrics))
        writer.writerows(rows)

    # AGRÉGER ensuite la moyenne et l'écart-type population sur les 16 crues.
    by_hour = defaultdict(list)
    for _, horizon, *values in rows:
        by_hour[horizon].append(values)
    header = [time_column, *(f"{name}_{stat}" for name in metrics for stat in ("mean", "std"))]
    with (output_dir / f"{prefix}_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        print(f"{prefix}_summary.csv")
        print(",".join(header))
        for hour, values in sorted(by_hour.items()):
            summary = [hour]
            for column in np.asarray(values, dtype=float).T:
                finite = column[np.isfinite(column)]
                summary.extend((finite.mean(), finite.std()) if len(finite) else ("", ""))
            writer.writerow(summary)
            print(",".join(str(value) for value in summary))


def load_source_model(cfg, config_name, epoch, device):
    hidden_dim = cfg.get("hidden_dim", 64)
    model = MeshGraphNetWithSourceNodes(
        input_dim_nodes_phys=cfg.num_input_features,
        input_dim_nodes_src=cfg.num_input_features_source,
        input_dim_edges=cfg.num_edge_features,
        output_dim=cfg.num_output_features,
        processor_size=cfg.mp_layers,
        hidden_dim_processor=hidden_dim,
        hidden_dim_node_encoder=hidden_dim,
        hidden_dim_edge_encoder=hidden_dim,
        hidden_dim_node_decoder=hidden_dim,
        do_concat_trick=cfg.do_concat_trick,
        num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
    )
    checkpoint_dir = Path(cfg.ckpt_path)
    weights = checkpoint_dir / f"{model.meta.name}.0.{epoch}.mdlus"
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {weights}")
    print(f"Évaluation {config_name}, checkpoint {weights}")
    model.load(str(weights))
    return model.to(device).eval()


def run_spatial_diagnostic(configs, config_names, epoch, steps_per_hour, device, output_dir):
    from plot_tet_ghost_diagnostics import write_diagnostics

    cfg, reference = configs
    if cfg.experiment == reference.experiment:
        raise ValueError("Choisir deux expériences distinctes pour le diagnostic.")
    for key in ("dataset_dir", "short_data_dir", "hydro_data_dir", "coarse_mesh_path", "data_dir", "dt_seconds", "seed"):
        if cfg.get(key) != reference.get(key):
            raise ValueError(f"Le diagnostic spatial exige la même valeur de {key} dans les deux configurations.")

    horizons = tuple(range(1, 7))
    inlet_nodes = set()
    errors = []
    with closing(TelemacFile(cfg.coarse_mesh_path)) as coarse_mesh:
        for config, name in zip(configs, config_names):
            model = load_source_model(config, name, epoch, device)
            ds = build_test_dataset(config, MAIN_HOURS * steps_per_hour + 1)
            inlet_nodes.update(node for section in ds.inlet_node_lists for node in section)
            errors.append(spatial_errors(model, ds, coarse_mesh.tri, steps_per_hour, horizons[-1], device))
            del model, ds

        mae_u, bias_u, mae_h = (np.stack(pair) for pair in zip(*errors))
        experiments = (cfg.experiment, reference.experiment)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        # Publier ensemble les cartes et les données une fois le calcul terminé.
        with TemporaryDirectory(prefix=".diagnostic-", dir=output_dir.parent) as temp_dir:
            write_diagnostics(
                Path(temp_dir), coarse_mesh.tri, sorted(inlet_nodes), TEST_EVENTS, horizons,
                experiments, mae_u, bias_u, mae_h, step_hours=cfg.dt_seconds / 3600,
            )
            Path(temp_dir).rename(output_dir)
    print(f"Diagnostic spatial : {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Évalue un checkpoint ghost Têt sur les 16 crues de test.")
    parser.add_argument("config", help="Nom du YAML relatif à bin/conf, sans extension")
    parser.add_argument("epoch", type=int, nargs="?", default=900)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--one-step", action="store_true", help="Repartir de l'état de référence à chaque pas.")
    modes.add_argument("--diagnostic-against", metavar="CONFIG", help="Comparer les erreurs spatiales de deux modèles.")
    args = parser.parse_args()
    print(f"Epoch: {args.epoch}")
    with initialize_config_dir(config_dir=str(BIN_DIR / "conf"), version_base=None):
        cfg = compose(config_name=args.config)
        if args.diagnostic_against:
            reference = compose(config_name=args.diagnostic_against)

    steps_per_hour = int(3600 / cfg.dt_seconds)
    if steps_per_hour * cfg.dt_seconds != 3600 or steps_per_hour < 1:
        raise ValueError("Le pas de temps doit diviser une heure pour l’évaluation à horizons fixes.")
    device = torch.device("cuda")

    experiment = cfg.experiment
    result_root = Path(cfg.article_results_dir) / experiment / f"Seed{cfg.seed}"
    if args.diagnostic_against:
        output_dir = result_root / f"diagnostic-{args.epoch}-vs-{reference.experiment}"
        if output_dir.exists():
            raise FileExistsError(f"Diagnostic déjà présent : {output_dir}")
        run_spatial_diagnostic(
            (cfg, reference), (args.config, args.diagnostic_against),
            args.epoch, steps_per_hour, device, output_dir,
        )
        return

    model = load_source_model(cfg, args.config, args.epoch, device)
    n_steps = MAIN_HOURS * steps_per_hour
    if args.one_step:
        ds = build_test_dataset(cfg, n_steps)
        with closing(TelemacFile(cfg.coarse_mesh_path)) as coarse_mesh:
            rows = evaluate_one_step(model, ds, coarse_mesh.tri, n_steps, device)
        output_dir = result_root / f"eval-{args.epoch}-one-step"
        output_dir.mkdir(parents=True)
        print(f"Résultats à un pas ({cfg.dt_seconds / 60:g} min) dans {output_dir}")
        write_results(rows, output_dir, "metrics", STATE_METRICS + ("mse_delta_norm",), time_column="start_time_h")
        return

    fine_res_dir = Path(cfg.fine_res_dir)
    # ÉVALUATION PRINCIPALE : 16 crues et horizons horaires de 1 à 12 h.
    ds = build_test_dataset(cfg, n_steps + 1)
    output_dir = result_root / f"eval-{args.epoch}"
    # ÉCRIRE les métriques séparément des checkpoints, sans remplacer un résultat existant.
    output_dir.mkdir(parents=True)
    print(f"Résultats : {output_dir}")
    with closing(TelemacFile(cfg.coarse_mesh_path)) as coarse_mesh:
        coarse_tri = coarse_mesh.tri
        rows = evaluate(model, ds, fine_res_dir, coarse_tri, steps_per_hour, device, experiment, output_dir)
        # AJOUTER les deux CSI pour les expériences avec Q global.
        metrics = STATE_METRICS + CSI_METRICS if ds.use_q_feature else STATE_METRICS
        write_results(rows, output_dir, "metrics", metrics)
        del ds

        if experiment in {"E5", "E6", "E7", "E8", "E8_point"}:
            # ANNEXE A : reprendre les mêmes poids pour les rollouts complets de 40 h.
            ds = build_test_dataset(cfg, FULL_HOURS * steps_per_hour, full=True)
            rows = evaluate(model, ds, fine_res_dir, coarse_tri, steps_per_hour, device, experiment, output_dir, full=True)
            write_results(rows, output_dir, "metrics_long", STATE_METRICS)


if __name__ == "__main__":
    main()
