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
from autoencoder_vae.generation import load_vae_npz

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
    "selection_metric",
    "n_train_samples_mean",
    "n_validation_samples_mean",
    "latent_dim",
    "beta",
    "reconstruction_loss_mean",
    "reconstruction_loss_std",
    "kl_loss_mean",
    "kl_loss_std",
    "total_loss_mean",
    "total_loss_std",
    "train_reconstruction_loss_mean",
    "train_reconstruction_loss_std",
    "train_kl_loss_mean",
    "train_kl_loss_std",
    "train_total_loss_mean",
    "train_total_loss_std",
    "validation_reconstruction_loss_mean",
    "validation_reconstruction_loss_std",
    "validation_kl_loss_mean",
    "validation_kl_loss_std",
    "validation_total_loss_mean",
    "validation_total_loss_std",
]


def parse_mode_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=["formal", "quick"], default="formal")
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Cap the number of seeds per variant (e.g. 3 for a fast sweep). Defaults to the full set for the mode.",
    )
    return parser.parse_args()


def resolve_seeds(mode: str, max_seeds: int | None = None) -> list[int]:
    """Seeds for a mode, optionally capped to the first ``max_seeds`` of them."""
    seeds = get_mode_seeds(mode)
    if max_seeds is not None:
        if max_seeds < 1:
            raise ValueError("max_seeds must be >= 1")
        seeds = seeds[:max_seeds]
    return seeds


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


def _metric_array(run_metrics: list[dict], key: str, fallback_key: str | None = None) -> np.ndarray:
    values = []
    for item in run_metrics:
        if key in item:
            values.append(float(item[key]))
        elif fallback_key is not None:
            values.append(float(item[fallback_key]))
        else:
            raise KeyError(f"Missing metric {key!r}")
    return np.array(values, dtype=float)


def aggregate_variant_summary(
    experiment: str,
    variant: str,
    mode: str,
    master_seed: int,
    seeds: list[int],
    run_metrics: list[dict],
) -> dict:
    recon = _metric_array(run_metrics, "reconstruction_loss")
    kl = _metric_array(run_metrics, "kl_loss")
    total = _metric_array(run_metrics, "total_loss")
    train_recon = _metric_array(run_metrics, "train_reconstruction_loss", fallback_key="reconstruction_loss")
    train_kl = _metric_array(run_metrics, "train_kl_loss", fallback_key="kl_loss")
    train_total = _metric_array(run_metrics, "train_total_loss", fallback_key="total_loss")
    validation_recon = _metric_array(run_metrics, "validation_reconstruction_loss", fallback_key="reconstruction_loss")
    validation_kl = _metric_array(run_metrics, "validation_kl_loss", fallback_key="kl_loss")
    validation_total = _metric_array(run_metrics, "validation_total_loss", fallback_key="total_loss")
    n_train = np.array([int(item["n_train_samples"]) for item in run_metrics], dtype=float)
    n_validation = np.array([int(item["n_validation_samples"]) for item in run_metrics], dtype=float)

    return {
        "experiment": experiment,
        "variant": variant,
        "mode": mode,
        "master_seed": int(master_seed),
        "n_runs": len(run_metrics),
        "seeds": json.dumps([int(seed) for seed in seeds]),
        "selection_metric": str(run_metrics[0]["selection_metric"]),
        "n_train_samples_mean": float(np.mean(n_train)),
        "n_validation_samples_mean": float(np.mean(n_validation)),
        "latent_dim": int(run_metrics[0]["latent_dim"]),
        "beta": float(run_metrics[0]["beta"]),
        "reconstruction_loss_mean": float(np.mean(recon)),
        "reconstruction_loss_std": float(np.std(recon)),
        "kl_loss_mean": float(np.mean(kl)),
        "kl_loss_std": float(np.std(kl)),
        "total_loss_mean": float(np.mean(total)),
        "total_loss_std": float(np.std(total)),
        "train_reconstruction_loss_mean": float(np.mean(train_recon)),
        "train_reconstruction_loss_std": float(np.std(train_recon)),
        "train_kl_loss_mean": float(np.mean(train_kl)),
        "train_kl_loss_std": float(np.std(train_kl)),
        "train_total_loss_mean": float(np.mean(train_total)),
        "train_total_loss_std": float(np.std(train_total)),
        "validation_reconstruction_loss_mean": float(np.mean(validation_recon)),
        "validation_reconstruction_loss_std": float(np.std(validation_recon)),
        "validation_kl_loss_mean": float(np.mean(validation_kl)),
        "validation_kl_loss_std": float(np.std(validation_kl)),
        "validation_total_loss_mean": float(np.mean(validation_total)),
        "validation_total_loss_std": float(np.std(validation_total)),
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
    max_seeds: int | None = None,
) -> tuple[dict, list[dict]]:
    seeds = resolve_seeds(mode, max_seeds)
    variant_output_dir = prepare_variant_output(experiment, variant)
    run_metrics: list[dict] = []

    for run_idx, seed in enumerate(seeds):
        run_dir = variant_output_dir / f"run_{run_idx:02d}"
        config = deep_update(base_config, override)
        config["dataset"]["seed"] = int(seed)
        config["dataset"]["split_seed"] = int(seed)
        result = run_vae(config=config, config_path=base_config_path, output_dir=run_dir)
        metrics = result["metrics"]
        run_metrics.append(metrics)

        if int(metrics["n_validation_samples"]) > 0:
            detail = (
                f" train_total={metrics['train_total_loss']:.2f}"
                f" val_total={metrics['validation_total_loss']:.2f}"
            )
        else:
            detail = (
                f" recon={metrics['reconstruction_loss']:.2f}"
                f" kl={metrics['kl_loss']:.2f}"
                f" total={metrics['total_loss']:.2f}"
            )

        print(
            f"[{experiment}/{variant}] run {run_idx + 1}/{len(seeds)}"
            f" seed={seed}"
            f" latent={metrics['latent_dim']}"
            f" beta={metrics['beta']}"
            f"{detail}"
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


def variant_ranking_key(summary: dict) -> tuple:
    return (
        float(summary["validation_total_loss_mean"]),
        float(summary["validation_total_loss_std"]),
        float(summary["validation_reconstruction_loss_mean"]),
    )


def select_best_variant_summary(summary_rows: list[dict]) -> dict:
    if not summary_rows:
        raise ValueError("summary_rows cannot be empty")
    return min(summary_rows, key=variant_ranking_key)


def plot_comparison(output_path: str | Path, rows: list[dict], experiment: str) -> None:
    names = [row["variant"] for row in rows]
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(max(12, 1.6 * len(names)), 4))
    metric_specs = [
        (
            "validation_reconstruction_loss_mean",
            "validation_reconstruction_loss_std",
            "Validation reconstruction loss",
            "tab:blue",
        ),
        (
            "validation_kl_loss_mean",
            "validation_kl_loss_std",
            "Validation KL divergence",
            "tab:orange",
        ),
        (
            "validation_total_loss_mean",
            "validation_total_loss_std",
            "Validation total loss",
            "tab:green",
        ),
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

    best = select_best_variant_summary(summary_rows)
    (experiment_root / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(
        f"[{experiment}] ganador: {best['variant']}"
        f" val_total={best['validation_total_loss_mean']:.2f}"
        f" val_total_std={best['validation_total_loss_std']:.2f}"
        f" val_recon={best['validation_reconstruction_loss_mean']:.2f}"
    )


# --- VAE-specific reporting: loss curves vs. a swept value + generated-sample montages ---


def variant_run_dir(experiment: str, variant: str, run_idx: int = 0) -> Path:
    """Path of a single multi-seed run inside a variant (run_00 is the first seed)."""
    return EXPERIMENT_OUTPUT_ROOT / experiment / variant / f"run_{run_idx:02d}"


def plot_metric_curves(
    output_path: str | Path,
    x_values: list[float],
    rows: list[dict],
    x_label: str,
    title: str,
    logx: bool = False,
) -> None:
    """Three side-by-side curves (val reconstruction / KL / total) vs. the swept value.

    Replaces the bar chart for ordered numeric sweeps: it shows the trade-off as a
    trend instead of disconnected bars, and keeps each metric on its own axis so a
    large KL (e.g. at beta=0) does not crush the reconstruction scale.
    """
    x = np.asarray(x_values, dtype=float)
    metric_specs = [
        ("validation_reconstruction_loss_mean", "validation_reconstruction_loss_std", "Reconstrucción (val)", "tab:blue"),
        ("validation_kl_loss_mean", "validation_kl_loss_std", "Divergencia KL (val)", "tab:orange"),
        ("validation_total_loss_mean", "validation_total_loss_std", "Loss total (val)", "tab:green"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (mean_key, std_key, ax_title, color) in zip(axes, metric_specs):
        means = np.array([row[mean_key] for row in rows], dtype=float)
        stds = np.array([row[std_key] for row in rows], dtype=float)
        ax.errorbar(x, means, yerr=stds, color=color, marker="o", capsize=4, linewidth=1.8)
        ax.set_title(ax_title)
        ax.set_xlabel(x_label)
        if logx:
            ax.set_xscale("symlog")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:g}" for value in x])
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(Path(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _reshape_samples(flat: np.ndarray, channels: int) -> np.ndarray:
    n = flat.shape[0]
    side = int(round((flat.shape[1] // channels) ** 0.5))
    if channels == 1:
        return flat.reshape(n, side, side)
    return flat.reshape(n, side, side, channels)


def plot_sample_montage(
    output_path: str | Path,
    entries: list[tuple[str, Path, int]],
    title: str,
    n_per_row: int = 8,
    seed: int = 2024,
    share_z: bool = False,
) -> None:
    """One row of prior samples (z ~ N(0, I) -> decode) per variant.

    ``entries`` is a list of (row_label, run_dir, channels). ``share_z=True`` reuses
    the same eps across rows (only valid when every variant has the same latent_dim,
    e.g. the beta sweep) so differences come purely from the trained decoder.
    """
    nrows = len(entries)
    fig, axes = plt.subplots(nrows, n_per_row, figsize=(n_per_row * 1.15, nrows * 1.25))
    axes = np.atleast_2d(axes)
    for r, (label, run_dir, channels) in enumerate(entries):
        model = load_vae_npz(Path(run_dir) / "model.npz")
        row_rng = np.random.default_rng(seed if share_z else seed + r)
        z = row_rng.standard_normal((n_per_row, model.latent_dim))
        images = _reshape_samples(np.clip(model.decode(z), 0.0, 1.0), channels)
        for col in range(n_per_row):
            ax = axes[r, col]
            if channels == 1:
                ax.imshow(images[col], cmap="gray", vmin=0.0, vmax=1.0)
            else:
                ax.imshow(images[col])
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=10, rotation=0, ha="right", va="center", labelpad=18)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(Path(output_path), dpi=130, bbox_inches="tight")
    plt.close(fig)


def select_knee_variant(summary_rows: list[dict], x_values: list[float], tol: float = 0.05) -> dict:
    """Smallest swept value whose val reconstruction is within ``tol`` of the best.

    For capacity sweeps (latent_dim) the goal is the *minimum sufficient* value, not
    the absolute lowest loss (which always favours the largest capacity). We take the
    best reconstruction and return the cheapest variant within ``tol`` relative of it.
    """
    if not summary_rows:
        raise ValueError("summary_rows cannot be empty")
    recon = np.array([row["validation_reconstruction_loss_mean"] for row in summary_rows], dtype=float)
    best = float(recon.min())
    threshold = best * (1.0 + tol)
    order = np.argsort(np.asarray(x_values, dtype=float))
    for idx in order:
        if recon[idx] <= threshold:
            return summary_rows[idx]
    return summary_rows[int(recon.argmin())]
