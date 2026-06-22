from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .dataset import flat_to_grid


def _draw_grid(ax, grid: np.ndarray, title: str) -> None:
    ax.imshow(grid, cmap="binary", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(5))
    ax.set_yticks(range(7))
    ax.set_xticklabels(range(1, 6), fontsize=7)
    ax.set_yticklabels(range(1, 8), fontsize=7)
    ax.grid(False)


def plot_latent_scatter(latent_codes: np.ndarray, labels: list[str], output_path: str | Path) -> None:
    path = Path(output_path)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(latent_codes[:, 0], latent_codes[:, 1], color="tab:blue")
    for idx, label in enumerate(labels):
        ax.annotate(label, (latent_codes[idx, 0], latent_codes[idx, 1]), xytext=(5, 5), textcoords="offset points")
    ax.set_title("Latent Space Projection")
    ax.set_xlabel("Latent dimension 1")
    ax.set_ylabel("Latent dimension 2")
    ax.axhline(0.0, color="#cccccc", linewidth=0.8)
    ax.axvline(0.0, color="#cccccc", linewidth=0.8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reconstructions(
    original_X: np.ndarray,
    reconstruction_binary: np.ndarray,
    labels: list[str],
    output_path: str | Path,
    reconstruction_probabilities: np.ndarray | None = None,
) -> None:
    path = Path(output_path)
    include_probabilities = reconstruction_probabilities is not None
    n_rows = original_X.shape[0]
    n_cols = 3 if include_probabilities else 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, max(2, 1.7 * n_rows)))
    axes = np.atleast_2d(axes)

    for row_idx, label in enumerate(labels):
        original_grid = flat_to_grid(original_X[row_idx]).astype(int)
        binary_grid = flat_to_grid(reconstruction_binary[row_idx]).astype(int)
        _draw_grid(axes[row_idx, 0], original_grid, f"{label} original")
        if include_probabilities:
            prob_grid = reconstruction_probabilities[row_idx].reshape(7, 5)
            axes[row_idx, 1].imshow(prob_grid, cmap="viridis", vmin=0.0, vmax=1.0)
            axes[row_idx, 1].set_title(f"{label} prob.", fontsize=9)
            axes[row_idx, 1].set_xticks(range(5))
            axes[row_idx, 1].set_yticks(range(7))
        _draw_grid(axes[row_idx, n_cols - 1], binary_grid, f"{label} recon")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_denoising_triplets(
    original_X: np.ndarray,
    noisy_X: np.ndarray,
    reconstruction_binary: np.ndarray,
    labels: list[str],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    n_rows = original_X.shape[0]
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, max(2, 1.7 * n_rows)))
    axes = np.atleast_2d(axes)

    for row_idx, label in enumerate(labels):
        original_grid = flat_to_grid(original_X[row_idx]).astype(int)
        noisy_grid = noisy_X[row_idx].reshape(7, 5)
        recon_grid = flat_to_grid(reconstruction_binary[row_idx]).astype(int)
        _draw_grid(axes[row_idx, 0], original_grid, f"{label} original")
        axes[row_idx, 1].imshow(noisy_grid, cmap="binary", vmin=0.0, vmax=1.0)
        axes[row_idx, 1].set_title(f"{label} noisy", fontsize=9)
        axes[row_idx, 1].set_xticks(range(5))
        axes[row_idx, 1].set_yticks(range(7))
        _draw_grid(axes[row_idx, 2], recon_grid, f"{label} denoised")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_denoising_curve(
    noise_levels: list[float],
    input_mean_pixel_error: list[float],
    output_mean_pixel_error: list[float],
    exact_reconstruction_rate: list[float],
    output_path: str | Path,
    title: str = "Denoising capacity vs noise level",
) -> None:
    path = Path(output_path)
    fig, (ax_err, ax_exact) = plt.subplots(1, 2, figsize=(12, 5))

    ax_err.plot(noise_levels, input_mean_pixel_error, marker="o", label="Input (noisy)")
    ax_err.plot(noise_levels, output_mean_pixel_error, marker="o", label="Output (denoised)")
    ax_err.set_title("Mean pixel error")
    ax_err.set_xlabel("noise_level")
    ax_err.set_ylabel("mean pixel error per pattern")
    ax_err.legend(loc="best")
    ax_err.grid(alpha=0.3)

    ax_exact.plot(noise_levels, exact_reconstruction_rate, marker="o", color="tab:green")
    ax_exact.set_title("Exact reconstruction rate")
    ax_exact.set_xlabel("noise_level")
    ax_exact.set_ylabel("fraction recovered exactly")
    ax_exact.set_ylim(-0.02, 1.02)
    ax_exact.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_binary_pattern(
    pattern_binary: np.ndarray,
    output_path: str | Path,
    title: str,
    subtitle: str | None = None,
) -> None:
    path = Path(output_path)
    fig, ax = plt.subplots(figsize=(4, 5))
    _draw_grid(ax, flat_to_grid(pattern_binary).astype(int), title)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_generated_letter(
    latent_codes: np.ndarray,
    labels: list[str],
    generated_latent_point: list[float],
    generated_binary: np.ndarray,
    output_path: str | Path,
    legend: str,
) -> None:
    path = Path(output_path)
    fig, (ax_scatter, ax_grid) = plt.subplots(1, 2, figsize=(10, 5))

    ax_scatter.scatter(latent_codes[:, 0], latent_codes[:, 1], color="tab:blue", label="Training patterns")
    for idx, label in enumerate(labels):
        ax_scatter.annotate(label, (latent_codes[idx, 0], latent_codes[idx, 1]), xytext=(4, 4), textcoords="offset points")
    ax_scatter.scatter(
        [generated_latent_point[0]],
        [generated_latent_point[1]],
        color="tab:red",
        s=90,
        marker="*",
        label="Generated point",
    )
    ax_scatter.set_title("Latent Grid Search")
    ax_scatter.set_xlabel("Latent dimension 1")
    ax_scatter.set_ylabel("Latent dimension 2")
    ax_scatter.legend(loc="best")
    ax_scatter.grid(alpha=0.3)

    _draw_grid(ax_grid, flat_to_grid(generated_binary).astype(int), "Generated 5x7 pattern")
    ax_grid.set_xlabel(legend, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
