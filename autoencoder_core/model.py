from __future__ import annotations

from copy import deepcopy

import numpy as np

from .activations import activation_derivative, apply_activation


class Autoencoder:
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_hidden_layers: list[int],
        decoder_hidden_layers: list[int],
        hidden_activation: str,
        output_activation: str = "sigmoid",
        weight_init: str = "xavier_uniform",
        dropout: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if output_activation != "sigmoid":
            raise ValueError("output_activation must be sigmoid")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder_hidden_layers = [int(v) for v in encoder_hidden_layers]
        self.decoder_hidden_layers = [int(v) for v in decoder_hidden_layers]
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.weight_init = weight_init
        self.dropout = float(dropout)
        self.rng = rng or np.random.default_rng(0)

        layer_dims = (
            [self.input_dim]
            + self.encoder_hidden_layers
            + [self.latent_dim]
            + self.decoder_hidden_layers
            + [self.input_dim]
        )
        encoder_hidden_count = len(self.encoder_hidden_layers)
        decoder_hidden_count = len(self.decoder_hidden_layers)
        self.layer_activations = (
            [self.hidden_activation] * encoder_hidden_count
            + [None]
            + [self.hidden_activation] * decoder_hidden_count
            + [self.output_activation]
        )
        self.latent_layer_index = encoder_hidden_count

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(layer_dims[:-1], layer_dims[1:]):
            self.weights.append(self._init_weight_matrix(fan_in, fan_out))
            self.biases.append(np.zeros((1, fan_out), dtype=float))

    def _init_weight_matrix(self, fan_in: int, fan_out: int) -> np.ndarray:
        if self.weight_init == "xavier_uniform":
            limit = np.sqrt(6.0 / (fan_in + fan_out))
        elif self.weight_init == "he_uniform":
            limit = np.sqrt(6.0 / fan_in)
        else:
            raise ValueError(f"Unsupported weight_init: {self.weight_init}")
        return self.rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(float)

    def set_rng(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def parameters(self) -> dict[str, np.ndarray]:
        params: dict[str, np.ndarray] = {}
        for idx, weight in enumerate(self.weights):
            params[f"W{idx}"] = weight
            params[f"b{idx}"] = self.biases[idx]
        return params

    def clone_parameters(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self.parameters().items()}

    def load_parameters(self, params: dict[str, np.ndarray]) -> None:
        for idx in range(len(self.weights)):
            self.weights[idx][...] = params[f"W{idx}"]
            self.biases[idx][...] = params[f"b{idx}"]

    def architecture_config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "encoder_hidden_layers": deepcopy(self.encoder_hidden_layers),
            "decoder_hidden_layers": deepcopy(self.decoder_hidden_layers),
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
            "weight_init": self.weight_init,
            "dropout": self.dropout,
        }

    def forward(self, X: np.ndarray, training: bool = False) -> dict:
        layer_inputs: list[np.ndarray] = []
        pre_activations: list[np.ndarray] = []
        post_activations: list[np.ndarray] = []
        dropout_masks: list[np.ndarray | None] = []

        current = X
        latent = None

        for layer_idx, activation_name in enumerate(self.layer_activations):
            layer_inputs.append(current)
            z = current @ self.weights[layer_idx] + self.biases[layer_idx]
            a = apply_activation(activation_name, z)
            dropout_mask = None
            if training and activation_name not in (None, "sigmoid") and self.dropout > 0.0:
                dropout_mask = (self.rng.random(a.shape) >= self.dropout).astype(float)
                dropout_mask /= 1.0 - self.dropout
                current = a * dropout_mask
            else:
                current = a
            if layer_idx == self.latent_layer_index:
                latent = a.copy()
            pre_activations.append(z)
            post_activations.append(a)
            dropout_masks.append(dropout_mask)

        return {
            "output": current,
            "latent": latent if latent is not None else post_activations[self.latent_layer_index],
            "layer_inputs": layer_inputs,
            "pre_activations": pre_activations,
            "post_activations": post_activations,
            "dropout_masks": dropout_masks,
        }

    def backward(self, X: np.ndarray, output_delta: np.ndarray) -> dict[str, np.ndarray]:
        cache = self.forward(X, training=True)
        grads: dict[str, np.ndarray] = {}
        delta = output_delta

        for layer_idx in reversed(range(len(self.weights))):
            layer_input = cache["layer_inputs"][layer_idx]
            grads[f"W{layer_idx}"] = layer_input.T @ delta
            grads[f"b{layer_idx}"] = np.sum(delta, axis=0, keepdims=True)

            if layer_idx == 0:
                continue

            delta = delta @ self.weights[layer_idx].T
            previous_idx = layer_idx - 1
            dropout_mask = cache["dropout_masks"][previous_idx]
            if dropout_mask is not None:
                delta = delta * dropout_mask
            prev_activation = self.layer_activations[previous_idx]
            prev_pre_activation = cache["pre_activations"][previous_idx]
            prev_post_activation = cache["post_activations"][previous_idx]
            delta = delta * activation_derivative(prev_activation, prev_pre_activation, prev_post_activation)

        return grads

    def compute_gradients(self, X: np.ndarray, target: np.ndarray, loss_name: str, output_delta_fn) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cache = self.forward(X, training=True)
        delta = output_delta_fn(target, cache["output"], loss_name)
        grads: dict[str, np.ndarray] = {}

        for layer_idx in reversed(range(len(self.weights))):
            layer_input = cache["layer_inputs"][layer_idx]
            grads[f"W{layer_idx}"] = layer_input.T @ delta
            grads[f"b{layer_idx}"] = np.sum(delta, axis=0, keepdims=True)

            if layer_idx == 0:
                continue

            delta = delta @ self.weights[layer_idx].T
            previous_idx = layer_idx - 1
            dropout_mask = cache["dropout_masks"][previous_idx]
            if dropout_mask is not None:
                delta = delta * dropout_mask
            prev_activation = self.layer_activations[previous_idx]
            prev_pre_activation = cache["pre_activations"][previous_idx]
            prev_post_activation = cache["post_activations"][previous_idx]
            delta = delta * activation_derivative(prev_activation, prev_pre_activation, prev_post_activation)

        return cache["output"], grads

    def encode(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, training=False)["latent"]

    def decode(self, Z: np.ndarray) -> np.ndarray:
        current = Z
        latent_to_output_start = self.latent_layer_index + 1
        for layer_idx in range(latent_to_output_start, len(self.weights)):
            activation_name = self.layer_activations[layer_idx]
            current = apply_activation(activation_name, current @ self.weights[layer_idx] + self.biases[layer_idx])
        return current

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, training=False)["output"]
