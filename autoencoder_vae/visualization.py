from __future__ import annotations

import argparse
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
from autoencoder_vae.dataset import load_run_dataset_split
from autoencoder_vae.model import VariationalAutoencoder


def _show_row(axes, images: np.ndarray, ylabel: str | None = None) -> None:
    for col, ax in enumerate(axes):
        ax.imshow(np.clip(images[col], 0.0, 1.0))
        ax.set_xticks([])
        ax.set_yticks([])
    if ylabel is not None:
        axes[0].set_ylabel(ylabel, fontsize=11)


def plot_reconstructions(
    model: VariationalAutoencoder,
    X_flat: np.ndarray,
    n: int,
    output_path: Path,
    title: str,
) -> None:
    originals = flat_to_images(X_flat[:n])
    recons = reconstruct(model, X_flat[:n])
    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 2.6))
    _show_row(axes[0], originals, "original")
    _show_row(axes[1], recons, "reconstrucción")
    fig.suptitle(title)
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
    _, dataset_split = load_run_dataset_split(run_dir)
    rng = np.random.default_rng(args.seed)

    train_flat = dataset_split.train_flat
    validation_flat = dataset_split.validation_flat
    all_flat = dataset_split.flat

    plot_reconstructions(
        model,
        train_flat,
        min(8, train_flat.shape[0]),
        run_dir / "reconstructions.png",
        "VAE — reconstrucción de punks de entrenamiento",
    )
    if dataset_split.validation_size:
        plot_reconstructions(
            model,
            train_flat,
            min(8, train_flat.shape[0]),
            run_dir / "reconstructions_train.png",
            "VAE — reconstrucción del split de entrenamiento",
        )
        plot_reconstructions(
            model,
            validation_flat,
            min(8, validation_flat.shape[0]),
            run_dir / "reconstructions_validation.png",
            "VAE — reconstrucción del split de validación",
        )
    plot_generated_grid(sample_from_prior(model, args.samples, rng), run_dir / "generated_grid.png")
    interpolation_source = train_flat if train_flat.shape[0] >= 2 else all_flat
    plot_interpolation(
        model,
        interpolation_source[0],
        interpolation_source[1],
        steps=8,
        output_path=run_dir / "interpolation.png",
    )
    plot_latent_pca(model, all_flat, run_dir / "latent_pca.png")

    print(f"Wrote reconstructions, generated_grid, interpolation and latent_pca to {run_dir}")


if __name__ == "__main__":
    main()
