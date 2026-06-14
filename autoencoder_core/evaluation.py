from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import flat_to_grid


def threshold_reconstruction(reconstruction_probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (reconstruction_probabilities >= threshold).astype(int)


def compute_reconstruction_metrics(
    target_X: np.ndarray,
    reconstruction_probabilities: np.ndarray,
    pixel_threshold: float,
) -> dict:
    reconstruction_binary = threshold_reconstruction(reconstruction_probabilities, pixel_threshold)
    target_binary = target_X.astype(int)
    pixel_error_per_pattern = np.abs(reconstruction_binary - target_binary).sum(axis=1)

    return {
        "pixel_error_per_pattern": pixel_error_per_pattern.astype(int).tolist(),
        "mean_pixel_error": float(np.mean(pixel_error_per_pattern)),
        "max_pixel_error": int(np.max(pixel_error_per_pattern)),
        "exact_reconstruction_rate": float(np.mean(pixel_error_per_pattern == 0)),
        "within_one_pixel_rate": float(np.mean(pixel_error_per_pattern <= 1)),
        "all_patterns_within_one_pixel": bool(np.all(pixel_error_per_pattern <= 1)),
        "reconstruction_binary": reconstruction_binary,
        "reconstruction_probabilities": reconstruction_probabilities,
    }


def evaluate_autoencoder(
    model,
    X: np.ndarray,
    config: dict,
    target_X: np.ndarray | None = None,
) -> dict:
    target = X if target_X is None else target_X
    pixel_threshold = float(config["evaluation"]["pixel_threshold"])
    probabilities = model.reconstruct(X)
    metrics = compute_reconstruction_metrics(target, probabilities, pixel_threshold)
    metrics["latent_codes"] = model.encode(X)
    return metrics


def checkpoint_ranking_tuple(metrics: dict, train_loss: float) -> tuple:
    return (
        int(bool(metrics["all_patterns_within_one_pixel"])),
        float(metrics["exact_reconstruction_rate"]),
        -float(metrics["max_pixel_error"]),
        -float(metrics["mean_pixel_error"]),
        -float(train_loss),
    )


def run_ranking_tuple(metrics: dict) -> tuple:
    return (
        int(bool(metrics["all_patterns_within_one_pixel"])),
        float(metrics["exact_reconstruction_rate"]),
        -float(metrics["max_pixel_error"]),
        -float(metrics["mean_pixel_error"]),
        -float(metrics["seed"]),
    )


def _latent_grid_points(latent_codes: np.ndarray, grid_size: int = 41) -> np.ndarray:
    mins = latent_codes.min(axis=0)
    maxs = latent_codes.max(axis=0)
    ranges = maxs - mins
    ranges = np.where(ranges == 0.0, 1.0, ranges)
    lower = mins - 0.25 * ranges
    upper = maxs + 0.25 * ranges
    axis_0 = np.linspace(lower[0], upper[0], grid_size)
    axis_1 = np.linspace(lower[1], upper[1], grid_size)
    mesh_x, mesh_y = np.meshgrid(axis_0, axis_1)
    return np.column_stack([mesh_x.ravel(), mesh_y.ravel()])


def generate_novel_letter(
    model,
    X: np.ndarray,
    labels: list[str],
    pixel_threshold: float,
) -> dict:
    latent_codes = model.encode(X)
    grid_points = _latent_grid_points(latent_codes)
    decoded_probabilities = model.decode(grid_points)
    decoded_binary = threshold_reconstruction(decoded_probabilities, pixel_threshold)
    known_patterns = {tuple(pattern.astype(int).tolist()) for pattern in X}
    confidence = np.mean(np.abs(decoded_probabilities - 0.5), axis=1)

    candidate_indices = [
        idx
        for idx, pattern in enumerate(decoded_binary)
        if tuple(pattern.astype(int).tolist()) not in known_patterns
    ]

    if candidate_indices:
        best_idx = max(candidate_indices, key=lambda idx: (confidence[idx], -idx))
        method = "latent_grid_search"
        legend = "Novel pattern found via latent grid search."
    else:
        best_idx = int(np.argmax(confidence))
        method = "latent_grid_search_no_novel_pattern"
        legend = "No novel pattern found; best approximation shown."

    return {
        "generation_method": method,
        "generated_latent_point": grid_points[best_idx].astype(float).tolist(),
        "generated_probabilities": decoded_probabilities[best_idx],
        "generated_binary": decoded_binary[best_idx],
        "latent_codes": latent_codes,
        "labels": labels,
        "legend": legend,
    }
