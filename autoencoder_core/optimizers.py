from __future__ import annotations

import numpy as np


class SGDWithMomentum:
    def __init__(self, learning_rate: float, parameter_names: list[str], momentum: float = 0.9) -> None:
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.velocity = {name: 0.0 for name in parameter_names}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        for name, grad in grads.items():
            previous_velocity = self.velocity[name]
            velocity = self.momentum * previous_velocity - self.learning_rate * grad
            self.velocity[name] = velocity
            params[name][...] = params[name] + velocity


class Adam:
    def __init__(
        self,
        learning_rate: float,
        parameter_names: list[str],
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.m = {name: 0.0 for name in parameter_names}
        self.v = {name: 0.0 for name in parameter_names}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for name, grad in grads.items():
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad ** 2)
            m_hat = self.m[name] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1.0 - self.beta2 ** self.t)
            params[name][...] = params[name] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


def build_optimizer(name: str, learning_rate: float, parameter_names: list[str]):
    if name == "sgd_momentum":
        return SGDWithMomentum(learning_rate=learning_rate, parameter_names=parameter_names)
    if name == "adam":
        return Adam(learning_rate=learning_rate, parameter_names=parameter_names)
    raise ValueError(f"Unsupported optimizer: {name}")
