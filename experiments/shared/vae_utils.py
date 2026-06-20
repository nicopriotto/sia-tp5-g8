from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autoencoder_vae.main import load_config, run_vae

from .seeds import MASTER_SEED, get_mode_seeds


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "autoencoder_vae" / "configs" / "base.json"
EXPERIMENT_OUTPUT_ROOT = REPO_ROOT / "experiments" / "output" / "vae"

SUMMARY_FIELDS = [
    "experiment",
    "variant",
    "mode",
    "master_seed",
    "n_runs",
    "seeds",
    "latent_dim",
    "beta",
    "reconstruction_loss_mean",
    "reconstruction_loss_std",
    "kl_loss_mean",
    "kl_loss_std",
    "total_loss_mean",
    "total_loss_std",
]


def parse_mode_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=["formal", "quick"], default="formal")
    return parser.parse_args()


def deep_update(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_base_config() -> tuple[dict, Path]:
    return load_config(BASE_CONFIG_PATH)


def prepare_variant_output(experiment: str, variant: str) -> Path:
    output_dir = EXPERIMENT_OUTPUT_ROOT / experiment / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def aggregate_variant_summary(
    experiment: str,
    variant: str,
    mode: str,
    master_seed: int,
    seeds: list[int],
    run_metrics: list[dict],
) -> dict:
    recon = np.array([float(item["reconstruction_loss"]) for item in run_metrics], dtype=float)
    kl = np.array([float(item["kl_loss"]) for item in run_metrics], dtype=float)
    total = np.array([float(item["total_loss"]) for item in run_metrics], dtype=float)

    return {
        "experiment": experiment,
        "variant": variant,
        "mode": mode,
        "master_seed": int(master_seed),
        "n_runs": len(run_metrics),
        "seeds": json.dumps([int(seed) for seed in seeds]),
        "latent_dim": int(run_metrics[0]["latent_dim"]),
        "beta": float(run_metrics[0]["beta"]),
        "reconstruction_loss_mean": float(np.mean(recon)),
        "reconstruction_loss_std": float(np.std(recon)),
        "kl_loss_mean": float(np.mean(kl)),
        "kl_loss_std": float(np.std(kl)),
        "total_loss_mean": float(np.mean(total)),
        "total_loss_std": float(np.std(total)),
    }


def write_summary_csv(output_path: str | Path, row: dict) -> None:
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def write_comparison_csv(output_path: str | Path, rows: list[dict]) -> None:
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_variant_multi_seed(
    experiment: str,
    variant: str,
    base_config: dict,
    base_config_path: Path,
    override: dict,
    mode: str,
) -> tuple[dict, list[dict]]:
    seeds = get_mode_seeds(mode)
    variant_output_dir = prepare_variant_output(experiment, variant)
    run_metrics: list[dict] = []

    for run_idx, seed in enumerate(seeds):
        run_dir = variant_output_dir / f"run_{run_idx:02d}"
        config = deep_update(base_config, override)
        config["dataset"]["seed"] = int(seed)
        result = run_vae(config=config, config_path=base_config_path, output_dir=run_dir)
        metrics = result["metrics"]
        run_metrics.append(metrics)
        print(
            f"[{experiment}/{variant}] run {run_idx + 1}/{len(seeds)}"
            f" seed={seed}"
            f" latent={metrics['latent_dim']}"
            f" beta={metrics['beta']}"
            f" recon={metrics['reconstruction_loss']:.2f}"
            f" kl={metrics['kl_loss']:.2f}"
            f" total={metrics['total_loss']:.2f}"
        )

    summary_row = aggregate_variant_summary(
        experiment=experiment,
        variant=variant,
        mode=mode,
        master_seed=MASTER_SEED,
        seeds=seeds,
        run_metrics=run_metrics,
    )
    write_summary_csv(variant_output_dir / "summary.csv", summary_row)
    return summary_row, run_metrics


def plot_comparison(output_path: str | Path, rows: list[dict], experiment: str) -> None:
    names = [row["variant"] for row in rows]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(max(12, 1.6 * len(names)), 4))
    metric_specs = [
        ("reconstruction_loss_mean", "reconstruction_loss_std", "Reconstruction loss", "tab:blue"),
        ("kl_loss_mean", "kl_loss_std", "KL divergence", "tab:orange"),
        ("total_loss_mean", "total_loss_std", "Total loss (recon + beta*KL)", "tab:green"),
    ]
    for ax, (mean_key, std_key, title, color) in zip(axes, metric_specs):
        means = [row[mean_key] for row in rows]
        stds = [row[std_key] for row in rows]
        ax.bar(x, means, yerr=stds, color=color, capsize=5, error_kw={"linewidth": 1.2})
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")

    fig.suptitle(experiment)
    fig.tight_layout()
    fig.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def finalize_experiment(experiment: str, summary_rows: list[dict]) -> None:
    experiment_root = EXPERIMENT_OUTPUT_ROOT / experiment
    write_comparison_csv(experiment_root / "comparison.csv", summary_rows)
    plot_comparison(experiment_root / "comparison.png", summary_rows, experiment=experiment)
