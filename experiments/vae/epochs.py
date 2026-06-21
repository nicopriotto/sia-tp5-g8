from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoencoder_vae.main import load_config, run_vae
from experiments.shared.seeds import MASTER_SEED, get_mode_seeds
from experiments.shared.vae_utils import deep_update, prepare_variant_output


EXPERIMENT = "epochs"
EPOCHS_MAX = 120
VALIDATION_FRACTION = 0.1
DATASET_LIMIT = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate the best training duration for a chosen VAE config.")
    parser.add_argument("--config", required=True, help="Path to the chosen VAE config JSON.")
    parser.add_argument("--mode", choices=["formal", "quick"], default="formal")
    return parser.parse_args()


def build_epoch_override() -> dict:
    return {
        "dataset": {
            "limit": DATASET_LIMIT,
            "validation_fraction": VALIDATION_FRACTION,
        },
        "training": {
            "epochs_max": EPOCHS_MAX,
            "early_stopping_patience": EPOCHS_MAX,
        },
    }


def write_epoch_curve_csv(
    output_path: Path,
    seeds: list[int],
    validation_total: np.ndarray,
    validation_reconstruction: np.ndarray,
    validation_kl: np.ndarray,
) -> None:
    fieldnames = [
        "epoch",
        "validation_total_loss_mean",
        "validation_total_loss_std",
        "validation_reconstruction_loss_mean",
        "validation_reconstruction_loss_std",
        "validation_kl_loss_mean",
        "validation_kl_loss_std",
    ]
    fieldnames.extend([f"seed_{seed}_validation_total_loss" for seed in seeds])

    rows = []
    for epoch_idx in range(validation_total.shape[1]):
        row = {
            "epoch": epoch_idx + 1,
            "validation_total_loss_mean": float(np.mean(validation_total[:, epoch_idx])),
            "validation_total_loss_std": float(np.std(validation_total[:, epoch_idx])),
            "validation_reconstruction_loss_mean": float(np.mean(validation_reconstruction[:, epoch_idx])),
            "validation_reconstruction_loss_std": float(np.std(validation_reconstruction[:, epoch_idx])),
            "validation_kl_loss_mean": float(np.mean(validation_kl[:, epoch_idx])),
            "validation_kl_loss_std": float(np.std(validation_kl[:, epoch_idx])),
        }
        for seed_idx, seed in enumerate(seeds):
            row[f"seed_{seed}_validation_total_loss"] = float(validation_total[seed_idx, epoch_idx])
        rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_epoch_curve(output_path: Path, seeds: list[int], validation_total: np.ndarray, recommended_epoch: int) -> None:
    epochs = np.arange(1, validation_total.shape[1] + 1)
    mean_curve = validation_total.mean(axis=0)
    std_curve = validation_total.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    for seed_idx, seed in enumerate(seeds):
        ax.plot(epochs, validation_total[seed_idx], linewidth=1.1, alpha=0.25, label=f"seed {seed}")
    ax.plot(epochs, mean_curve, color="tab:green", linewidth=2.4, label="mean")
    ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, color="tab:green", alpha=0.18)
    ax.axvline(recommended_epoch, color="tab:red", linestyle="--", linewidth=1.4, label=f"recommended={recommended_epoch}")
    ax.set_title("Validation total loss by epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation total loss")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    seeds = get_mode_seeds(args.mode)
    output_dir = prepare_variant_output(EXPERIMENT, config_path.stem)
    override = build_epoch_override()

    histories: list[list[dict]] = []
    for run_idx, seed in enumerate(seeds):
        run_dir = output_dir / f"run_{run_idx:02d}"
        run_config = deep_update(config, override)
        run_config["dataset"]["seed"] = int(seed)
        run_config["dataset"]["split_seed"] = int(seed)
        result = run_vae(config=run_config, config_path=config_path, output_dir=run_dir)
        history = result["history"]
        histories.append(history)

        best_metrics = result["metrics"]
        print(
            f"[{EXPERIMENT}/{config_path.stem}] run {run_idx + 1}/{len(seeds)}"
            f" seed={seed}"
            f" best_epoch={best_metrics['best_epoch']}"
            f" val_total={best_metrics['validation_total_loss']:.2f}"
        )

    validation_total = np.array(
        [[float(row["validation_total_loss"]) for row in history] for history in histories],
        dtype=float,
    )
    validation_reconstruction = np.array(
        [[float(row["validation_reconstruction_loss"]) for row in history] for history in histories],
        dtype=float,
    )
    validation_kl = np.array(
        [[float(row["validation_kl_loss"]) for row in history] for history in histories],
        dtype=float,
    )

    mean_curve = validation_total.mean(axis=0)
    recommended_epoch = int(np.argmin(mean_curve) + 1)

    write_epoch_curve_csv(
        output_dir / "epoch_curve.csv",
        seeds=seeds,
        validation_total=validation_total,
        validation_reconstruction=validation_reconstruction,
        validation_kl=validation_kl,
    )
    plot_epoch_curve(output_dir / "epoch_curve.png", seeds=seeds, validation_total=validation_total, recommended_epoch=recommended_epoch)

    recommended = {
        "experiment": EXPERIMENT,
        "variant": config_path.stem,
        "mode": args.mode,
        "master_seed": MASTER_SEED,
        "seeds": [int(seed) for seed in seeds],
        "config_path": str(config_path.resolve()),
        "epochs_max": EPOCHS_MAX,
        "dataset_limit": DATASET_LIMIT,
        "validation_fraction": VALIDATION_FRACTION,
        "selection_metric": "validation_total_loss_mean",
        "recommended_epoch": recommended_epoch,
        "suggested_training_override": {
            "epochs_max": recommended_epoch,
            "early_stopping_patience": recommended_epoch,
        },
        "validation_total_loss_mean": float(mean_curve[recommended_epoch - 1]),
        "validation_total_loss_std": float(validation_total.std(axis=0)[recommended_epoch - 1]),
        "validation_reconstruction_loss_mean": float(validation_reconstruction.mean(axis=0)[recommended_epoch - 1]),
    }
    (output_dir / "recommended_epoch.json").write_text(json.dumps(recommended, indent=2), encoding="utf-8")

    print(
        f"[{EXPERIMENT}] recommended_epoch={recommended_epoch}"
        f" val_total={recommended['validation_total_loss_mean']:.2f}"
        f" val_total_std={recommended['validation_total_loss_std']:.2f}"
    )


if __name__ == "__main__":
    main()
