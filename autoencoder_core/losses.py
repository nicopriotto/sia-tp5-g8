from __future__ import annotations

import numpy as np


EPSILON = 1e-8


def clip_probabilities(y_pred: np.ndarray) -> np.ndarray:
    return np.clip(y_pred, EPSILON, 1.0 - EPSILON)


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_prob = clip_probabilities(y_pred)
    loss = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(np.mean(np.sum(loss, axis=1)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sum((y_true - y_pred) ** 2, axis=1)))


def compute_reconstruction_loss(y_true: np.ndarray, y_pred: np.ndarray, loss_name: str) -> float:
    if loss_name == "binary_cross_entropy":
        return binary_cross_entropy(y_true, y_pred)
    if loss_name == "mean_squared_error":
        return mean_squared_error(y_true, y_pred)
    raise ValueError(f"Unsupported loss function: {loss_name}")


def kl_divergence(mu: np.ndarray, logvar: np.ndarray) -> float:
    """KL[ N(mu, exp(logvar)) || N(0, I) ], summed per sample and averaged over the batch."""
    per_sample = -0.5 * np.sum(1.0 + logvar - mu ** 2 - np.exp(logvar), axis=1)
    return float(np.mean(per_sample))


def kl_divergence_grads(mu: np.ndarray, logvar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradients of kl_divergence w.r.t. mu and logvar, scaled by 1/batch.

    Matches the 1/batch convention of output_delta so the VAE backprop can add
    these directly to the reconstruction deltas.
    """
    scale = float(mu.shape[0])
    grad_mu = mu / scale
    grad_logvar = 0.5 * (np.exp(logvar) - 1.0) / scale
    return grad_mu, grad_logvar


def output_delta(y_true: np.ndarray, y_pred: np.ndarray, loss_name: str) -> np.ndarray:
    scale = float(y_true.shape[0])
    if loss_name == "binary_cross_entropy":
        return (y_pred - y_true) / scale
    if loss_name == "mean_squared_error":
        return (2.0 * (y_pred - y_true) * y_pred * (1.0 - y_pred)) / scale
    raise ValueError(f"Unsupported loss function: {loss_name}")
