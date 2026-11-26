from __future__ import annotations

import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def inlet_regions(coarse_tri, inlet_nodes):
    # Seules les arêtes des triangles ×8 définissent le voisinage.
    edges = coarse_tri.edges
    front = np.zeros(len(coarse_tri.x), dtype=bool)
    front[inlet_nodes] = True
    visited = front.copy()
    regions = {"maillage": np.arange(len(front)), "entree": inlet_nodes}
    for ring in (1, 2):
        neighbors = np.zeros_like(front)
        neighbors[edges[front[edges].any(axis=1)].ravel()] = True
        front = neighbors & ~visited
        visited |= front
        regions[f"anneau_{ring}"] = np.flatnonzero(front)
    return regions


def write_region_metrics(path, regions, events, horizons, experiments, mae_u, bias_u, mae_h, step_hours):
    diagnostics = [("mae_u", "m/s", step_hours, mae_u), ("bias_u", "m/s", step_hours, bias_u)]
    diagnostics.extend(("mae_h", "m", hour, mae_h[:, :, index]) for index, hour in enumerate(horizons))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("event", "metric", "unit", "horizon_h", "region", "n_nodes", *experiments, "delta"))
        for metric, unit, hour, values in diagnostics:
            for name, nodes in regions.items():
                if not len(nodes):
                    continue
                means = values[:, :, nodes].mean(axis=-1)
                for event, (value, reference) in zip(events, means.T):
                    writer.writerow((event, metric, unit, hour, name, len(nodes), value, reference, value - reference))
                value, reference = means.mean(axis=1)
                writer.writerow(("all", metric, unit, hour, name, len(nodes), value, reference, value - reference))


def draw_error_map(ax, coarse_tri, values, limit, inlet_nodes):
    image = ax.tripcolor(
        coarse_tri, values, shading="gouraud", cmap="RdBu_r",
        vmin=-limit, vmax=limit, rasterized=True,
    )
    ax.scatter(
        coarse_tri.x[inlet_nodes], coarse_tri.y[inlet_nodes],
        s=26, facecolors="none", edgecolors="black", linewidths=0.8,
    )
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    ax.ticklabel_format(style="plain")
    return image


def plot_u_error(path, coarse_tri, inlet_nodes, regions, experiment_pair, values, step_minutes):
    limit = max(abs(values).max(), np.finfo(float).eps)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), constrained_layout=True)
    for ax, title in zip(axes, ("Maillage ×8", "Zoom sur l’entrée")):
        image = draw_error_map(ax, coarse_tri, values, limit, inlet_nodes)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")

    zoom_nodes = np.concatenate([nodes for name, nodes in regions.items() if name != "maillage"])
    x, y = coarse_tri.x[zoom_nodes], coarse_tri.y[zoom_nodes]
    padding = 0.1 * max(np.ptp(x), np.ptp(y), 1.0)
    axes[1].set(xlim=(x.min() - padding, x.max() + padding), ylim=(y.min() - padding, y.max() + padding))
    for ring, marker, label in (
        ("anneau_1", "+", "À 1 arête ×8"),
        ("anneau_2", "x", "À 2 arêtes ×8"),
    ):
        nodes = regions[ring]
        axes[1].scatter(
            coarse_tri.x[nodes], coarse_tri.y[nodes], marker=marker,
            color="0.25", linewidths=0.6, s=13, label=label,
        )
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False)
    for index, node in enumerate(inlet_nodes):
        axes[1].annotate(
            str(node), (coarse_tri.x[node], coarse_tri.y[node]),
            xytext=(4, 5 + 8 * index), textcoords="offset points", fontsize=8,
        )
    fig.suptitle(f"Écart de MAE sur u à {step_minutes:g} min", fontsize=13)
    fig.colorbar(image, ax=axes, label=f"{experiment_pair} (m/s)", shrink=0.8)
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_h_error(path, coarse_tri, inlet_nodes, horizons, experiment_pair, values):
    # Les panneaux partagent la même échelle en centimètres.
    limit = max(abs(values).max(), np.finfo(float).eps)
    n_rows = (len(horizons) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(11.5, 3.5 * n_rows), constrained_layout=True)
    for ax, hour, field in zip(axes.flat, horizons, values):
        image = draw_error_map(ax, coarse_tri, field, limit, inlet_nodes)
        ax.set_title(f"{hour:g} h")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(horizons):]:
        ax.axis("off")
    fig.suptitle("Écart de MAE sur la hauteur en rollout", fontsize=13)
    fig.colorbar(image, ax=axes, label=f"{experiment_pair} (cm)", shrink=0.85)
    fig.savefig(path, dpi=250)
    plt.close(fig)


def write_diagnostics(output_dir, coarse_tri, inlet_nodes, events, horizons, experiments, mae_u, bias_u, mae_h, *, step_hours):
    """Écrit les écarts appariés sur le maillage ×8 et les métriques par région."""
    n_events, n_nodes = len(events), len(coarse_tri.x)
    if mae_u.shape != (2, n_events, n_nodes) or bias_u.shape != mae_u.shape:
        raise ValueError("Dimensions inattendues pour les erreurs à un pas.")
    if mae_h.shape != (2, n_events, len(horizons), n_nodes):
        raise ValueError("Dimensions inattendues pour les erreurs en rollout.")

    inlet_nodes = np.unique(np.asarray(inlet_nodes, dtype=int))
    regions = inlet_regions(coarse_tri, inlet_nodes)
    experiment, reference = experiments
    experiment_pair = f"{experiment} − {reference}"
    step_minutes = step_hours * 60
    plot_u_error(
        output_dir / f"diagnostic_u_{step_minutes:g}min.pdf", coarse_tri, inlet_nodes, regions,
        experiment_pair, (mae_u[0] - mae_u[1]).mean(axis=0), step_minutes,
    )
    plot_h_error(
        output_dir / "diagnostic_h_rollout.pdf", coarse_tri, inlet_nodes, horizons,
        experiment_pair, 100.0 * (mae_h[0] - mae_h[1]).mean(axis=0),
    )
    write_region_metrics(
        output_dir / "metrics_by_region.csv", regions, events, horizons, experiments,
        mae_u, bias_u, mae_h, step_hours,
    )
    np.savez_compressed(
        output_dir / "spatial_errors.npz",
        x=coarse_tri.x, y=coarse_tri.y, triangles=coarse_tri.triangles,
        inlet_nodes=inlet_nodes, events=events, horizons=horizons, experiments=experiments,
        mae_u=mae_u, bias_u=bias_u, mae_h=mae_h, one_step_h=step_hours,
    )
