from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoencoder_vae.generation import (
    flat_to_images,
    interpolate_latent,
    load_vae_npz,
    reconstruct,
    sample_from_prior,
)
from autoencoder_vae.model import VariationalAutoencoder


def _show_row(axes, images: np.ndarray, ylabel: str | None = None) -> None:
    for col, ax in enumerate(axes):
        ax.imshow(np.clip(images[col], 0.0, 1.0))
        ax.set_xticks([])
        ax.set_yticks([])
    if ylabel is not None:
        axes[0].set_ylabel(ylabel, fontsize=11)


def plot_reconstructions(model: VariationalAutoencoder, X_flat: np.ndarray, n: int, output_path: Path) -> None:
    originals = flat_to_images(X_flat[:n])
    recons = reconstruct(model, X_flat[:n])
    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 2.6))
    _show_row(axes[0], originals, "original")
    _show_row(axes[1], recons, "reconstrucción")
    fig.suptitle("VAE — reconstrucción de punks de entrenamiento")
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_generated_grid(images: np.ndarray, output_path: Path, cols: int = 8) -> None:
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.1))
    for idx, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.set_xticks([])
        ax.set_yticks([])
        if idx < n:
            ax.imshow(np.clip(images[idx], 0.0, 1.0))
        else:
            ax.axis("off")
    fig.suptitle("VAE — punks generados desde z ~ N(0, I)")
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_interpolation(model: VariationalAutoencoder, x_a: np.ndarray, x_b: np.ndarray, steps: int, output_path: Path) -> None:
    images = interpolate_latent(model, x_a, x_b, steps)
    fig, axes = plt.subplots(1, steps, figsize=(steps * 1.1, 1.4))
    _show_row(axes, images)
    fig.suptitle("VAE — interpolación en el espacio latente (mu_a → mu_b)")
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_latent_pca(model: VariationalAutoencoder, X_flat: np.ndarray, output_path: Path) -> None:
    """Project the latent means mu to 2D via PCA (numpy SVD) and scatter them."""
    mu = model.encode(X_flat)
    centered = mu - mu.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.4, color="tab:blue")
    ax.set_title(f"VAE — proyección PCA del latente ({model.latent_dim}D → 2D)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def load_run_dataset(run_dir: Path) -> np.ndarray:
    """Load the same dataset the run was trained on, from its resolved_config.json."""
    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    dataset_cfg = config["dataset"]
    raw = np.load(dataset_cfg["tensor_path"]).astype(np.float32) / 255.0
    X = raw.reshape(raw.shape[0], -1)
    limit = dataset_cfg.get("limit")
    if limit is not None:
        X = X[: int(limit)]
    return X


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot VAE reconstructions, samples and latent structure.")
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=str(THIS_DIR / "output" / "base"),
        help="Run directory containing model.npz. Defaults to autoencoder_vae/output/base.",
    )
    parser.add_argument("--samples", type=int, default=32, help="Number of generated samples to plot.")
    parser.add_argument("--seed", type=int, default=2024, help="Sampling seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    model = load_vae_npz(run_dir / "model.npz")
    X = load_run_dataset(run_dir)
    rng = np.random.default_rng(args.seed)

    plot_reconstructions(model, X, min(8, X.shape[0]), run_dir / "reconstructions.png")
    plot_generated_grid(sample_from_prior(model, args.samples, rng), run_dir / "generated_grid.png")
    plot_interpolation(model, X[0], X[1], steps=8, output_path=run_dir / "interpolation.png")
    plot_latent_pca(model, X, run_dir / "latent_pca.png")

    print(f"Wrote reconstructions, generated_grid, interpolation and latent_pca to {run_dir}")


if __name__ == "__main__":
    main()
