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


def output_delta(y_true: np.ndarray, y_pred: np.ndarray, loss_name: str) -> np.ndarray:
    scale = float(y_true.shape[0])
    if loss_name == "binary_cross_entropy":
        return (y_pred - y_true) / scale
    if loss_name == "mean_squared_error":
        return (2.0 * (y_pred - y_true) * y_pred * (1.0 - y_pred)) / scale
    raise ValueError(f"Unsupported loss function: {loss_name}")
