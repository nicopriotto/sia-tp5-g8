from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def sigmoid_derivative(sigmoid_x: np.ndarray) -> np.ndarray:
    return sigmoid_x * (1.0 - sigmoid_x)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    y = np.tanh(x)
    return 1.0 - y * y


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


def leaky_relu(x: np.ndarray, negative_slope: float = 0.01) -> np.ndarray:
    return np.where(x > 0.0, x, negative_slope * x)


def leaky_relu_derivative(x: np.ndarray, negative_slope: float = 0.01) -> np.ndarray:
    return np.where(x > 0.0, 1.0, negative_slope)


def apply_activation(name: str | None, x: np.ndarray) -> np.ndarray:
    if name is None or name == "linear":
        return x
    if name == "sigmoid":
        return sigmoid(x)
    if name == "tanh":
        return tanh(x)
    if name == "relu":
        return relu(x)
    if name == "leaky_relu":
        return leaky_relu(x)
    raise ValueError(f"Unsupported activation: {name}")


def activation_derivative(name: str | None, pre_activation: np.ndarray, output: np.ndarray) -> np.ndarray:
    if name is None or name == "linear":
        return np.ones_like(pre_activation)
    if name == "sigmoid":
        return sigmoid_derivative(output)
    if name == "tanh":
        return tanh_derivative(pre_activation)
    if name == "relu":
        return relu_derivative(pre_activation)
    if name == "leaky_relu":
        return leaky_relu_derivative(pre_activation)
    raise ValueError(f"Unsupported activation: {name}")
