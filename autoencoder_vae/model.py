from __future__ import annotations

import numpy as np

from autoencoder_core.activations import activation_derivative, apply_activation
from autoencoder_core.losses import kl_divergence_grads
from autoencoder_core.model import Autoencoder


class VariationalAutoencoder(Autoencoder):
    """Variational autoencoder built on top of the deterministic Autoencoder.

    The base "latent layer" (a linear projection of the last encoder hidden
    activation into latent_dim) is reused as the mu head. A parallel logvar head
    is added, and the latent code is sampled with the reparametrization trick:

        z = mu + exp(0.5 * logvar) * eps,   eps ~ N(0, I)

    The training objective is reconstruction + beta * KL. beta is stored on the
    model so compute_gradients keeps the base signature and the existing training
    loop can be reused unchanged.
    """

    def __init__(self, *args, beta: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if beta < 0.0:
            raise ValueError("beta must be >= 0")
        self.beta = float(beta)
        # The mu head reuses self.weights[latent_layer_index] / biases. The logvar
        # head shares its fan-in/out and starts at zero so logvar=0 (sigma=1) at init.
        fan_in, fan_out = self.weights[self.latent_layer_index].shape
        self.logvar_weight = np.zeros((fan_in, fan_out), dtype=float)
        self.logvar_bias = np.zeros((1, fan_out), dtype=float)

    # -- parameter bookkeeping (extends the base with the logvar head) --

    def parameters(self) -> dict[str, np.ndarray]:
        params = super().parameters()
        params["W_logvar"] = self.logvar_weight
        params["b_logvar"] = self.logvar_bias
        return params

    def load_parameters(self, params: dict[str, np.ndarray]) -> None:
        super().load_parameters(params)
        self.logvar_weight[...] = params["W_logvar"]
        self.logvar_bias[...] = params["b_logvar"]

    def architecture_config(self) -> dict:
        config = super().architecture_config()
        config["beta"] = self.beta
        config["variational"] = True
        return config

    # -- forward with reparametrization --

    def _encode_hidden(self, X: np.ndarray, training: bool) -> dict:
        layer_inputs: list[np.ndarray] = []
        pre_activations: list[np.ndarray] = []
        post_activations: list[np.ndarray] = []
        dropout_masks: list[np.ndarray | None] = []

        current = X
        for layer_idx in range(self.latent_layer_index):
            activation_name = self.layer_activations[layer_idx]
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
            pre_activations.append(z)
            post_activations.append(a)
            dropout_masks.append(dropout_mask)

        return {
            "encoder_output": current,
            "layer_inputs": layer_inputs,
            "pre_activations": pre_activations,
            "post_activations": post_activations,
            "dropout_masks": dropout_masks,
        }

    def forward(self, X: np.ndarray, training: bool = False) -> dict:
        enc = self._encode_hidden(X, training)
        h = enc["encoder_output"]

        li = self.latent_layer_index
        mu = h @ self.weights[li] + self.biases[li]
        logvar = h @ self.logvar_weight + self.logvar_bias

        if training:
            eps = self.rng.standard_normal(mu.shape)
            z = mu + np.exp(0.5 * logvar) * eps
        else:
            eps = np.zeros_like(mu)
            z = mu

        decoder_inputs: list[np.ndarray] = []
        decoder_pre: list[np.ndarray] = []
        decoder_post: list[np.ndarray] = []
        decoder_masks: list[np.ndarray | None] = []
        current = z
        for layer_idx in range(li + 1, len(self.weights)):
            activation_name = self.layer_activations[layer_idx]
            decoder_inputs.append(current)
            z_dec = current @ self.weights[layer_idx] + self.biases[layer_idx]
            a = apply_activation(activation_name, z_dec)
            dropout_mask = None
            if training and activation_name not in (None, "sigmoid") and self.dropout > 0.0:
                dropout_mask = (self.rng.random(a.shape) >= self.dropout).astype(float)
                dropout_mask /= 1.0 - self.dropout
                current = a * dropout_mask
            else:
                current = a
            decoder_pre.append(z_dec)
            decoder_post.append(a)
            decoder_masks.append(dropout_mask)

        return {
            "output": current,
            "latent": mu,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "eps": eps,
            "encoder_output": h,
            "encoder_layer_inputs": enc["layer_inputs"],
            "encoder_pre": enc["pre_activations"],
            "encoder_post": enc["post_activations"],
            "encoder_masks": enc["dropout_masks"],
            "decoder_inputs": decoder_inputs,
            "decoder_pre": decoder_pre,
            "decoder_post": decoder_post,
            "decoder_masks": decoder_masks,
        }

    # -- backprop: decoder -> reparametrization -> KL -> encoder --

    def compute_gradients(self, X: np.ndarray, target: np.ndarray, loss_name: str, output_delta_fn) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cache = self.forward(X, training=True)
        grads: dict[str, np.ndarray] = {}
        li = self.latent_layer_index

        # Decoder: from the output pre-activation delta down to dL/dz.
        delta = output_delta_fn(target, cache["output"], loss_name)
        n_dec = len(self.weights) - (li + 1)
        for k in reversed(range(n_dec)):
            layer_idx = li + 1 + k
            layer_input = cache["decoder_inputs"][k]
            grads[f"W{layer_idx}"] = layer_input.T @ delta
            grads[f"b{layer_idx}"] = np.sum(delta, axis=0, keepdims=True)
            delta = delta @ self.weights[layer_idx].T
            if k > 0:
                mask = cache["decoder_masks"][k - 1]
                if mask is not None:
                    delta = delta * mask
                prev_activation = self.layer_activations[layer_idx - 1]
                delta = delta * activation_derivative(prev_activation, cache["decoder_pre"][k - 1], cache["decoder_post"][k - 1])
        delta_z = delta  # latent layer is linear -> derivative is 1

        # Reparametrization trick + KL term (both deltas already 1/batch scaled).
        std = np.exp(0.5 * cache["logvar"])
        grad_mu_kl, grad_logvar_kl = kl_divergence_grads(cache["mu"], cache["logvar"])
        delta_mu = delta_z + self.beta * grad_mu_kl
        delta_logvar = delta_z * (0.5 * std * cache["eps"]) + self.beta * grad_logvar_kl

        # mu / logvar heads.
        h = cache["encoder_output"]
        grads[f"W{li}"] = h.T @ delta_mu
        grads[f"b{li}"] = np.sum(delta_mu, axis=0, keepdims=True)
        grads["W_logvar"] = h.T @ delta_logvar
        grads["b_logvar"] = np.sum(delta_logvar, axis=0, keepdims=True)

        # Encoder hidden layers: dL/d(post-activation of last hidden) flows back.
        delta = delta_mu @ self.weights[li].T + delta_logvar @ self.logvar_weight.T
        for k in reversed(range(li)):
            mask = cache["encoder_masks"][k]
            if mask is not None:
                delta = delta * mask
            activation_name = self.layer_activations[k]
            delta = delta * activation_derivative(activation_name, cache["encoder_pre"][k], cache["encoder_post"][k])
            layer_input = cache["encoder_layer_inputs"][k]
            grads[f"W{k}"] = layer_input.T @ delta
            grads[f"b{k}"] = np.sum(delta, axis=0, keepdims=True)
            if k > 0:
                delta = delta @ self.weights[k].T

        return cache["output"], grads

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Return the latent mean mu (deterministic encoding)."""
        return self.forward(X, training=False)["mu"]
