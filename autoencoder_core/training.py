from __future__ import annotations

from copy import deepcopy

import numpy as np

from .evaluation import checkpoint_ranking_tuple, evaluate_autoencoder
from .losses import compute_reconstruction_loss, kl_divergence, output_delta
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


def _vae_epoch_metrics(model, X: np.ndarray, loss_name: str) -> dict:
    eval_cache = model.forward(X, training=False)
    reconstruction_loss = compute_reconstruction_loss(X, eval_cache["output"], loss_name)
    kl_loss = kl_divergence(eval_cache["mu"], eval_cache["logvar"])
    return {
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
        "total_loss": reconstruction_loss + model.beta * kl_loss,
    }


def train_autoencoder(
    model,
    X: np.ndarray,
    config: dict,
    rng: np.random.Generator,
    selection_X: np.ndarray | None = None,
    selection_target_X: np.ndarray | None = None,
) -> dict:
    model.set_rng(rng)

    training_cfg = config["training"]
    model_cfg = config["model"]
    is_variational = hasattr(model, "beta")
    batch_size = int(training_cfg["batch_size"])
    epochs_max = int(training_cfg["epochs_max"])
    patience = int(training_cfg["early_stopping_patience"])
    learning_rate = float(training_cfg["learning_rate"])
    l2_weight_decay = float(training_cfg["l2_weight_decay"])
    gradient_clip_norm = training_cfg["gradient_clip_norm"]
    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)
    progress_every = int(training_cfg.get("progress_every", 0))
    selection_metric = "validation_total_loss" if is_variational and selection_X is not None else "total_loss"
    if not is_variational:
        selection_metric = "checkpoint_ranking_tuple"

    optimizer = build_optimizer(
        name=str(model_cfg["optimizer"]),
        learning_rate=learning_rate,
        parameter_names=list(model.parameters().keys()),
    )

    best_state = model.clone_parameters()
    best_epoch = 1
    best_ranking = None
    best_early_stopping_score = None
    best_metrics = None
    best_train_loss = None
    epochs_without_improvement = 0
    history: list[dict] = []

    target_X = X.copy() if bool(config["denoising"]["train_with_clean_target"]) else None
    train_target = X if target_X is None else target_X
    selection_target = (X if selection_X is None else selection_X) if selection_target_X is None else selection_target_X

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

        if is_variational:
            train_metrics = _vae_epoch_metrics(model, X, str(model_cfg["loss_function"]))
            selected_metrics = train_metrics
            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "reconstruction_loss": train_metrics["reconstruction_loss"],
                "kl_loss": train_metrics["kl_loss"],
                "total_loss": train_metrics["total_loss"],
                "train_reconstruction_loss": train_metrics["reconstruction_loss"],
                "train_kl_loss": train_metrics["kl_loss"],
                "train_total_loss": train_metrics["total_loss"],
            }
            if selection_X is not None:
                validation_metrics = _vae_epoch_metrics(model, selection_X, str(model_cfg["loss_function"]))
                selected_metrics = validation_metrics
                history_row["reconstruction_loss"] = validation_metrics["reconstruction_loss"]
                history_row["kl_loss"] = validation_metrics["kl_loss"]
                history_row["total_loss"] = validation_metrics["total_loss"]
                history_row["validation_reconstruction_loss"] = validation_metrics["reconstruction_loss"]
                history_row["validation_kl_loss"] = validation_metrics["kl_loss"]
                history_row["validation_total_loss"] = validation_metrics["total_loss"]
            ranking = (-selected_metrics["total_loss"],)
            early_stopping_score = -selected_metrics["total_loss"]
            epoch_metrics = deepcopy(history_row)
            history.append(history_row)
        else:
            selection_base = X if selection_X is None else selection_X
            selection_input = _selection_inputs(
                selection_base,
                config,
                epoch_seed=int(rng.integers(0, 1_000_000_000)),
            )
            epoch_metrics = evaluate_autoencoder(model, selection_input, config, target_X=selection_target)
            ranking = checkpoint_ranking_tuple(epoch_metrics, train_loss)
            early_stopping_score = -train_loss

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

        if progress_every and (epoch == 1 or epoch % progress_every == 0):
            if is_variational:
                if selection_X is not None:
                    status = (
                        f"train_total={epoch_metrics['train_total_loss']:.2f}"
                        f" val_total={epoch_metrics['validation_total_loss']:.2f}"
                    )
                else:
                    status = (
                        f"recon={epoch_metrics['reconstruction_loss']:.2f}"
                        f" kl={epoch_metrics['kl_loss']:.2f}"
                        f" total={epoch_metrics['total_loss']:.2f}"
                    )
            else:
                status = (
                    f"exact={epoch_metrics['exact_reconstruction_rate']:.3f}"
                    f" max_err={epoch_metrics['max_pixel_error']}"
                )
            print(f"epoch {epoch}/{epochs_max} train_loss={train_loss:.4f} {status}", flush=True)

        checkpoint_improved = best_ranking is None or ranking > best_ranking
        early_stopping_improved = (
            best_early_stopping_score is None
            or early_stopping_score > best_early_stopping_score
        )

        if checkpoint_improved:
            best_ranking = ranking
            best_state = model.clone_parameters()
            best_epoch = epoch
            best_metrics = deepcopy(epoch_metrics)
            best_train_loss = train_loss

        if early_stopping_improved:
            best_early_stopping_score = early_stopping_score

        if checkpoint_improved or early_stopping_improved:
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
        "best_history": deepcopy(history[best_epoch - 1]),
        "best_train_loss": best_train_loss,
        "selection_metric": selection_metric,
    }
