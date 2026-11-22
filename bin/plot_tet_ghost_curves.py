#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = {
    "E1": "#1f77b4",
    "E2": "#ff7f0e",
    "E3": "#2ca02c",
    "E4": "#d62728",
    "E5": "#9467bd",
    "E6": "#8c564b",
}
STATE_METRICS = (
    ("mae_h", r"$L_1(h)$ (m)", False),
    ("mae_u", r"$L_1(u)$ (m s$^{-1}$)", True),
    ("mae_v", r"$L_1(v)$ (m s$^{-1}$)", True),
)


def load_summary(results_dir: Path, experiment: str, long: bool = False):
    name = "metrics_long_summary.csv" if long else "metrics_summary.csv"
    path = results_dir / experiment / "Seed0" / "eval-900" / name
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} ne contient aucun horizon d'évaluation.")

    summary = {}
    for field in reader.fieldnames:
        values = [row[field] for row in rows]
        if field.startswith("csi_"):
            # L'évaluation laisse la cellule vide quand le CSI est indéfini.
            values = ["nan" if value == "" else value for value in values]
        summary[field] = np.asarray([float(value) for value in values])
    return summary


def plot_curve(ax, summaries, experiments, metric, markers=True, unit_interval=False):
    for experiment in experiments:
        data = summaries[experiment]
        hours = data["horizon_h"]
        mean = data[f"{metric}_mean"]
        std = data[f"{metric}_std"]
        lower = mean - std
        upper = mean + std
        if unit_interval:
            lower = np.clip(lower, 0, 1)
            upper = np.clip(upper, 0, 1)
        elif ax.get_yscale() == "log":
            lower = np.where(lower > 0, lower, np.nan)
        else:
            lower = np.maximum(lower, 0)
        ax.plot(
            hours,
            mean,
            label=f"Experiment {experiment[1:]}",
            color=COLORS[experiment],
            marker="o" if markers else None,
            markersize=3,
            linewidth=1.5,
        )
        ax.fill_between(
            hours,
            lower,
            upper,
            color=COLORS[experiment],
            alpha=0.18,
            linewidth=0,
        )
    ax.grid(alpha=0.3)


def save_figure(fig, output_dir: Path, number: int):
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"figure{number}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_l1_figure(summaries, output_dir, number, title, columns):
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 8.4), sharex=True, sharey="row")
    for column, (pair, column_title) in enumerate(columns):
        axes[0, column].set_title(column_title)
        for row, (metric, ylabel, log_scale) in enumerate(STATE_METRICS):
            ax = axes[row, column]
            if log_scale:
                ax.set_yscale("log")
            plot_curve(ax, summaries, pair, metric)
            if column == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel("Horizon (hours)")
    fig.suptitle(title)
    handles = [line for ax in axes[0] for line in ax.lines]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.96))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_dir, number)


def plot_csi(summaries, output_dir, number, metric, threshold):
    experiments = ("E2", "E3", "E5", "E6")
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    plot_curve(ax, summaries, experiments, metric, unit_interval=True)
    ax.set_xlabel("Horizon (hours)")
    ax.set_ylabel("Fine-grid CSI")
    ax.set_title(f"Fine-grid CSI vs horizon (threshold={threshold:.3f} m)")
    ax.legend(loc="lower left", frameon=False)
    save_figure(fig, output_dir, number)


def plot_long_rollout(summaries, output_dir):
    experiments = ("E5", "E6")
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=True)
    for ax, (metric, ylabel, log_scale) in zip(axes, STATE_METRICS):
        if log_scale:
            ax.set_yscale("log")
        plot_curve(ax, summaries, experiments, metric, markers=False)
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="upper left", frameon=False)
    axes[-1].set_xlabel("Horizon (hours)")
    fig.suptitle(r"$L_1$ mean $\pm$ std on full floods")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_dir, 9)


def main():
    parser = argparse.ArgumentParser(description="Trace les courbes du manuscrit ghost Têt.")
    parser.add_argument("results_dir", type=Path, help="Dossier article_results")
    parser.add_argument("--output-dir", type=Path, help="Dossier de sortie (défaut : RESULTS/figures)")
    args = parser.parse_args()

    summaries = {exp: load_summary(args.results_dir, exp) for exp in COLORS}
    long_summaries = {exp: load_summary(args.results_dir, exp, long=True) for exp in ("E5", "E6")}
    output_dir = args.output_dir or args.results_dir / "figures"
    output_dir.mkdir(parents=True)

    plot_l1_figure(
        summaries,
        output_dir,
        3,
        r"Ablation of global discharge forcing $Q(t)$",
        ((('E1', 'E2'), "Standard mesh"), (('E4', 'E5'), "Multimesh")),
    )
    plot_l1_figure(
        summaries,
        output_dir,
        4,
        "Ablation of multimesh",
        ((('E2', 'E5'), "With Q / Without PF"), (('E3', 'E6'), "With Q / With PF")),
    )
    plot_l1_figure(
        summaries,
        output_dir,
        5,
        "Ablation of pushforward training",
        ((('E2', 'E3'), "Standard mesh"), (('E5', 'E6'), "Multimesh")),
    )
    plot_csi(summaries, output_dir, 6, "csi_5cm", 0.05)
    plot_csi(summaries, output_dir, 7, "csi_30cm", 0.30)
    plot_long_rollout(long_summaries, output_dir)
    print(f"Courbes : {output_dir}")


if __name__ == "__main__":
    main()
