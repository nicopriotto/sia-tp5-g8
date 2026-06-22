from __future__ import annotations

import csv
import json
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRIC_SPECS = [
    ("full_success_rate_mean", "full_success_rate_std", "Full Success Rate", "tab:green"),
    ("exact_reconstruction_rate_mean", "exact_reconstruction_rate_std", "Exact Reconstruction Rate", "tab:blue"),
    ("mean_pixel_error_mean", "mean_pixel_error_std", "Mean Pixel Error", "tab:orange"),
    ("max_pixel_error_mean", "max_pixel_error_std", "Max Pixel Error", "tab:red"),
]

BASIC_METRIC_SPECS = [
    ("full_success_rate_mean", "full_success_rate_std", "Full Success Rate (↑)", "tab:green"),
    ("max_pixel_error_mean", "max_pixel_error_std", "Max Pixel Error (↓)", "tab:red"),
]

SUMMARY_FIELDS = [
    "experiment",
    "variant",
    "mode",
    "master_seed",
    "n_runs",
    "seeds",
    "full_success_rate_mean",
    "full_success_rate_std",
    "within_one_pixel_rate_mean",
    "within_one_pixel_rate_std",
    "exact_reconstruction_rate_mean",
    "exact_reconstruction_rate_std",
    "max_pixel_error_mean",
    "max_pixel_error_std",
    "mean_pixel_error_mean",
    "mean_pixel_error_std",
]


def run_ranking_key(metrics: dict) -> tuple:
    return (
        int(bool(metrics["all_patterns_within_one_pixel"])),
        float(metrics["exact_reconstruction_rate"]),
        -float(metrics["max_pixel_error"]),
        -float(metrics["mean_pixel_error"]),
        -float(metrics["seed"]),
    )


def variant_ranking_key(summary: dict) -> tuple:
    return (
        float(summary["full_success_rate_mean"]),
        float(summary["exact_reconstruction_rate_mean"]),
        -float(summary["max_pixel_error_mean"]),
        -float(summary["mean_pixel_error_mean"]),
        -float(summary["exact_reconstruction_rate_std"]),
    )


def select_best_variant_summary(summary_rows: list[dict]) -> dict:
    if not summary_rows:
        raise ValueError("summary_rows cannot be empty")
    return max(summary_rows, key=variant_ranking_key)


def select_best_run_metrics(run_metrics: list[dict]) -> dict:
    if not run_metrics:
        raise ValueError("run_metrics cannot be empty")
    return max(run_metrics, key=run_ranking_key)


def aggregate_variant_summary(
    experiment: str,
    variant: str,
    mode: str,
    master_seed: int,
    seeds: list[int],
    run_metrics: list[dict],
) -> dict:
    full_success = np.array([float(bool(item["all_patterns_within_one_pixel"])) for item in run_metrics], dtype=float)
    within_one = np.array([float(item["within_one_pixel_rate"]) for item in run_metrics], dtype=float)
    exact = np.array([float(item["exact_reconstruction_rate"]) for item in run_metrics], dtype=float)
    max_error = np.array([float(item["max_pixel_error"]) for item in run_metrics], dtype=float)
    mean_error = np.array([float(item["mean_pixel_error"]) for item in run_metrics], dtype=float)

    return {
        "experiment": experiment,
        "variant": variant,
        "mode": mode,
        "master_seed": int(master_seed),
        "n_runs": len(run_metrics),
        "seeds": json.dumps([int(seed) for seed in seeds]),
        "full_success_rate_mean": float(np.mean(full_success)),
        "full_success_rate_std": float(np.std(full_success)),
        "within_one_pixel_rate_mean": float(np.mean(within_one)),
        "within_one_pixel_rate_std": float(np.std(within_one)),
        "exact_reconstruction_rate_mean": float(np.mean(exact)),
        "exact_reconstruction_rate_std": float(np.std(exact)),
        "max_pixel_error_mean": float(np.mean(max_error)),
        "max_pixel_error_std": float(np.std(max_error)),
        "mean_pixel_error_mean": float(np.mean(mean_error)),
        "mean_pixel_error_std": float(np.std(mean_error)),
    }


def write_summary_csv(output_path: str | Path, row: dict) -> None:
    path = Path(output_path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def write_comparison_csv(output_path: str | Path, rows: list[dict]) -> None:
    path = Path(output_path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(
    output_path: str | Path,
    rows: list[dict],
    experiment: str,
    metric_specs: list[tuple] | None = None,
) -> None:
    if metric_specs is None:
        metric_specs = DEFAULT_METRIC_SPECS

    names = [row["variant"] for row in rows]
    x = np.arange(len(names))
    n = len(metric_specs)
    ncols = 2 if n > 1 else 1
    nrows = ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(max(12, 1.4 * len(names)), 4.0 * nrows))
    axes = np.array(axes).ravel()

    for ax, (mean_key, std_key, title, color) in zip(axes, metric_specs):
        means = [float(row[mean_key]) for row in rows]
        stds = [float(row[std_key]) for row in rows]
        ax.bar(x, means, yerr=stds, color=color, capsize=5, error_kw={"linewidth": 1.2})
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}" if v % 1 else f"{v:.0f}"))

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(experiment)
    fig.tight_layout()
    fig.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
