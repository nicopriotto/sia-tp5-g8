from __future__ import annotations

import numpy as np


SUPPORTED_NOISE_TYPES = {"bit_flip", "salt_and_pepper", "gaussian_clipped"}


def apply_noise(
    X: np.ndarray,
    noise_type: str,
    noise_level: float,
    seed: int,
) -> np.ndarray:
    if noise_type not in SUPPORTED_NOISE_TYPES:
        raise ValueError(f"Unsupported noise_type: {noise_type}")
    if noise_level < 0.0:
        raise ValueError("noise_level must be non-negative")
    if noise_level == 0.0:
        return X.copy()

    rng = np.random.default_rng(seed)

    if noise_type == "bit_flip":
        mask = rng.random(X.shape) < noise_level
        return np.where(mask, 1.0 - X, X).astype(float)

    if noise_type == "salt_and_pepper":
        mask = rng.random(X.shape) < noise_level
        salt = rng.random(X.shape) < 0.5
        noisy = X.copy()
        noisy[mask] = salt[mask].astype(float)
        return np.clip(noisy, 0.0, 1.0)

    noisy = X + rng.normal(0.0, noise_level, size=X.shape)
    return np.clip(noisy, 0.0, 1.0)
