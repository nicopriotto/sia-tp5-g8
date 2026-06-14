from __future__ import annotations

from copy import deepcopy

import numpy as np

from .evaluation import checkpoint_ranking_tuple, evaluate_autoencoder
from .losses import compute_reconstruction_loss, output_delta
from .noise import apply_noise
from .optimizers import build_optimizer


def _global_gradient_norm(grads: dict[str, np.ndarray]) -> float:
    total = 0.0
    for grad in grads.values():
        total += float(np.sum(grad ** 2))
    return float(np.sqrt(total))


def _apply_gradient_clipping(grads: dict[str, np.ndarray], clip_norm: float | None) -> None:
    if clip_norm is None:
        return
    grad_norm = _global_gradient_norm(grads)
    if grad_norm == 0.0 or grad_norm <= clip_norm:
        return
    scale = clip_norm / grad_norm
    for key in grads:
        grads[key] = grads[key] * scale


def _denoising_train_inputs(X: np.ndarray, config: dict, epoch_seed: int) -> np.ndarray:
    denoising_cfg = config["denoising"]
    noise_level = float(denoising_cfg["noise_level"])
    if noise_level == 0.0:
        return X.copy()
    return apply_noise(
        X,
        noise_type=str(denoising_cfg["noise_type"]),
        noise_level=noise_level,
        seed=epoch_seed,
    )


def _selection_inputs(X: np.ndarray, config: dict, epoch_seed: int) -> np.ndarray:
    denoising_cfg = config["denoising"]
    if float(denoising_cfg["noise_level"]) == 0.0:
        return X
    if bool(denoising_cfg["noise_on_train_only"]):
        return X
    return apply_noise(
        X,
        noise_type=str(denoising_cfg["noise_type"]),
        noise_level=float(denoising_cfg["noise_level"]),
        seed=epoch_seed + 1_000_000,
    )


def train_autoencoder(model, X: np.ndarray, config: dict, rng: np.random.Generator) -> dict:
    model.set_rng(rng)

    training_cfg = config["training"]
    model_cfg = config["model"]
    batch_size = int(training_cfg["batch_size"])
    epochs_max = int(training_cfg["epochs_max"])
    patience = int(training_cfg["early_stopping_patience"])
    learning_rate = float(training_cfg["learning_rate"])
    l2_weight_decay = float(training_cfg["l2_weight_decay"])
    gradient_clip_norm = training_cfg["gradient_clip_norm"]
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)

    optimizer = build_optimizer(
        name=str(model_cfg["optimizer"]),
        learning_rate=learning_rate,
        parameter_names=list(model.parameters().keys()),
    )

    best_state = model.clone_parameters()
    best_epoch = 1
    best_ranking = None
    best_metrics = None
    best_train_loss = None
    epochs_without_improvement = 0
    history: list[dict] = []

    target_X = X.copy() if bool(config["denoising"]["train_with_clean_target"]) else None
    train_target = X if target_X is None else target_X

    for epoch in range(1, epochs_max + 1):
        train_input = _denoising_train_inputs(X, config, epoch_seed=int(rng.integers(0, 1_000_000_000)))
        permutation = rng.permutation(train_input.shape[0])
        train_input = train_input[permutation]
        shuffled_target = train_target[permutation]

        reconstruction_losses = []
        l2_penalties = []

        for batch_start in range(0, train_input.shape[0], batch_size):
            batch_end = batch_start + batch_size
            batch_X = train_input[batch_start:batch_end]
            batch_target = shuffled_target[batch_start:batch_end]

            reconstruction_output, grads = model.compute_gradients(
                batch_X,
                batch_target,
                loss_name=str(model_cfg["loss_function"]),
                output_delta_fn=output_delta,
            )
            reconstruction_loss = compute_reconstruction_loss(batch_target, reconstruction_output, str(model_cfg["loss_function"]))
            l2_penalty = 0.5 * l2_weight_decay * sum(float(np.sum(weight ** 2)) for weight in model.weights)
            for layer_idx, weight in enumerate(model.weights):
                grads[f"W{layer_idx}"] = grads[f"W{layer_idx}"] + l2_weight_decay * weight
            _apply_gradient_clipping(grads, gradient_clip_norm)
            optimizer.step(model.parameters(), grads)

            reconstruction_losses.append(reconstruction_loss)
            l2_penalties.append(l2_penalty)

        mean_reconstruction_loss = float(np.mean(reconstruction_losses))
        mean_l2_penalty = float(np.mean(l2_penalties))
        train_loss = mean_reconstruction_loss + mean_l2_penalty

        selection_input = _selection_inputs(X, config, epoch_seed=int(rng.integers(0, 1_000_000_000)))
        epoch_metrics = evaluate_autoencoder(model, selection_input, config, target_X=X)
        ranking = checkpoint_ranking_tuple(epoch_metrics, train_loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "reconstruction_loss": mean_reconstruction_loss,
                "l2_penalty": mean_l2_penalty,
                "mean_pixel_error": epoch_metrics["mean_pixel_error"],
                "max_pixel_error": epoch_metrics["max_pixel_error"],
                "exact_reconstruction_rate": epoch_metrics["exact_reconstruction_rate"],
                "within_one_pixel_rate": epoch_metrics["within_one_pixel_rate"],
                "all_patterns_within_one_pixel": epoch_metrics["all_patterns_within_one_pixel"],
            }
        )

        if best_ranking is None or ranking > best_ranking:
            best_ranking = ranking
            best_state = model.clone_parameters()
            best_epoch = epoch
            best_metrics = deepcopy(epoch_metrics)
            best_train_loss = train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_parameters(best_state)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "best_metrics": best_metrics,
        "best_train_loss": best_train_loss,
    }
