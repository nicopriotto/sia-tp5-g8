from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.shared.vae_utils import (
    EXPERIMENT_OUTPUT_ROOT,
    load_base_config,
    parse_mode_args,
    plot_sample_montage,
    run_variant_multi_seed,
    variant_run_dir,
    write_comparison_csv,
)


EXPERIMENT = "channels"
TENSOR_DIR = REPO_ROOT / "data" / "data_punks_bundle" / "tensors"

# Each representation flattens to a different input_dim, so the summed BCE is not
# comparable across them directly; we report it per value (BCE / input_dim) below.
REPRESENTATIONS = [
    {"name": "gray", "tensor": "punks_gray.npy", "input_dim": 24 * 24 * 1, "channels": 1, "label": "gris"},
    {"name": "rgb", "tensor": "punks_rgb.npy", "input_dim": 24 * 24 * 3, "channels": 3, "label": "RGB"},
    {"name": "rgba", "tensor": "punks_rgba.npy", "input_dim": 24 * 24 * 4, "channels": 4, "label": "RGBA"},
]
RECOMMENDED = "rgb"  # color fidelity without the (near-constant) alpha channel of RGBA.

# Reduced dataset plus validation split so the sweep is feasible and comparable.
COMMON_OVERRIDE = {
    "dataset": {"limit": 2000, "validation_fraction": 0.1},
    "training": {"epochs_max": 60, "early_stopping_patience": 15},
}


def plot_normalized_reconstruction(output_path: Path, rows: list[dict], reps: list[dict]) -> None:
    """Bar chart of validation reconstruction per value (BCE / input_dim).

    Dividing by input_dim makes the three representations comparable: a raw summed
    BCE is mechanically larger for RGBA simply because it has more values.
    """
    labels = [rep["label"] for rep in reps]
    input_dims = np.array([rep["input_dim"] for rep in reps], dtype=float)
    recon = np.array([row["validation_reconstruction_loss_mean"] for row in rows], dtype=float)
    recon_std = np.array([row["validation_reconstruction_loss_std"] for row in rows], dtype=float)
    per_value = recon / input_dims
    per_value_std = recon_std / input_dims

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["tab:gray", "tab:blue", "tab:purple"]
    ax.bar(x, per_value, yerr=per_value_std, color=colors[: len(labels)], capsize=5)
    for xi, value in zip(x, per_value):
        ax.text(xi, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("VAE — reconstrucción por valor (BCE / input_dim) · menor es mejor")
    ax.set_ylabel("BCE por valor (val)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_mode_args("Compare image representations (gray / RGB / RGBA) for the VAE.")
    base_config, base_config_path = load_base_config()

    summary_rows = []
    montage_entries = []
    for rep in REPRESENTATIONS:
        tensor_path = TENSOR_DIR / rep["tensor"]
        if not tensor_path.exists():
            raise FileNotFoundError(
                f"Missing tensor {tensor_path}. Generate it with "
                f"`python3 {REPO_ROOT / 'data' / 'data_punks_bundle' / 'preprocess.py'}`."
            )
        variant = rep["name"]
        override = {
            **COMMON_OVERRIDE,
            "dataset": {**COMMON_OVERRIDE["dataset"], "tensor_path": str(tensor_path)},
            "model": {"input_dim": rep["input_dim"]},
        }
        summary, _ = run_variant_multi_seed(
            experiment=EXPERIMENT,
            variant=variant,
            base_config=base_config,
            base_config_path=base_config_path,
            override=override,
            mode=args.mode,
            max_seeds=args.max_seeds,
        )
        summary_rows.append(summary)
        montage_entries.append((rep["label"], variant_run_dir(EXPERIMENT, variant), rep["channels"]))

    experiment_root = EXPERIMENT_OUTPUT_ROOT / EXPERIMENT
    experiment_root.mkdir(parents=True, exist_ok=True)
    write_comparison_csv(experiment_root / "comparison.csv", summary_rows)
    plot_normalized_reconstruction(experiment_root / "comparison.png", summary_rows, REPRESENTATIONS)
    plot_sample_montage(
        experiment_root / "samples_by_channels.png",
        entries=montage_entries,
        title="VAE — punks generados desde z ~ N(0, I) por representación",
    )

    recommended = next(row for row in summary_rows if row["variant"] == RECOMMENDED)
    (experiment_root / "best.json").write_text(json.dumps(recommended, indent=2), encoding="utf-8")

    print(f"[{EXPERIMENT}] reconstrucción por valor (BCE / input_dim, val):")
    for rep, row in zip(REPRESENTATIONS, summary_rows):
        per_value = row["validation_reconstruction_loss_mean"] / rep["input_dim"]
        print(
            f"  {rep['label']:>4s}  input_dim={rep['input_dim']:>4d}"
            f"  recon_total={row['validation_reconstruction_loss_mean']:8.2f}"
            f"  recon/val={per_value:.5f}"
        )
    print(f"[{EXPERIMENT}] representación recomendada={RECOMMENDED} (color sin el alpha casi constante de RGBA)")


if __name__ == "__main__":
    main()
