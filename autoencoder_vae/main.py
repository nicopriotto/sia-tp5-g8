from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoencoder_core.serialization import save_model_npz
from autoencoder_core.training import train_autoencoder
from autoencoder_vae.dataset import (
    normalize_dataset_config,
    prepare_dataset_split,
    write_dataset_manifest,
)
from autoencoder_vae.model import VariationalAutoencoder

DEFAULT_CONFIG = THIS_DIR / "configs" / "base.json"


def load_config(config_path: str | Path | None) -> tuple[dict, Path]:
    path = DEFAULT_CONFIG if config_path is None else Path(config_path)
    resolved_path = path if path.is_absolute() else (Path.cwd() / path).resolve()
    config = json.loads(resolved_path.read_text(encoding="utf-8"))
    return config, resolved_path


def resolve_config_paths(config: dict, config_path: Path) -> dict:
    resolved = deepcopy(config)
    resolved["dataset"] = normalize_dataset_config(resolved["dataset"])
    tensor_path = Path(resolved["dataset"]["tensor_path"])
    if not tensor_path.is_absolute():
        tensor_path = (config_path.parent / tensor_path).resolve()
    resolved["dataset"]["tensor_path"] = str(tensor_path)
    return resolved


def default_output_dir(config_path: Path) -> Path:
    return THIS_DIR / "output" / config_path.stem


def build_model(config: dict, seed: int) -> VariationalAutoencoder:
    model_cfg = config["model"]
    return VariationalAutoencoder(
        input_dim=int(model_cfg["input_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        encoder_hidden_layers=list(model_cfg["encoder_hidden_layers"]),
        decoder_hidden_layers=list(model_cfg["decoder_hidden_layers"]),
        hidden_activation=str(model_cfg["hidden_activation"]),
        output_activation=str(model_cfg["output_activation"]),
        weight_init=str(model_cfg["weight_init"]),
        dropout=float(model_cfg["dropout"]),
        beta=float(model_cfg["beta"]),
        rng=np.random.default_rng(seed),
    )


def write_history_csv(history: list[dict], output_path: str | Path) -> None:
    path = Path(output_path)
    if not history:
        raise ValueError("Training history is empty")
    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def run_vae(config: dict, config_path: Path, output_dir: str | Path) -> dict:
    resolved_config = resolve_config_paths(config, config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_seed = int(resolved_config["dataset"]["seed"])
    rng = np.random.default_rng(run_seed)

    dataset_split = prepare_dataset_split(resolved_config)
    model = build_model(resolved_config, seed=run_seed)
    validation_X = dataset_split.validation_flat if dataset_split.validation_size else None
    training_result = train_autoencoder(
        model,
        dataset_split.train_flat,
        resolved_config,
        rng,
        selection_X=validation_X,
    )
    best_history = training_result["best_history"]

    resolved_config_json = deepcopy(resolved_config)
    resolved_config_json["output_dir"] = str(output_path.resolve())
    (output_path / "resolved_config.json").write_text(json.dumps(resolved_config_json, indent=2), encoding="utf-8")
    write_dataset_manifest(dataset_split, output_path / "dataset_manifest.json")

    metrics = {
        "seed": run_seed,
        "n_samples": int(dataset_split.n_samples),
        "n_train_samples": int(dataset_split.train_size),
        "n_validation_samples": int(dataset_split.validation_size),
        "input_dim": int(dataset_split.flat.shape[1]),
        "latent_dim": int(resolved_config["model"]["latent_dim"]),
        "beta": float(resolved_config["model"]["beta"]),
        "epochs_trained": training_result["epochs_trained"],
        "best_epoch": training_result["best_epoch"],
        "selection_metric": training_result["selection_metric"],
        "reconstruction_loss": best_history["reconstruction_loss"],
        "kl_loss": best_history["kl_loss"],
        "total_loss": best_history["total_loss"],
        "resolved_hyperparameters": {
            "dataset": resolved_config["dataset"],
            "model": resolved_config["model"],
            "training": resolved_config["training"],
        },
    }
    for key in (
        "train_reconstruction_loss",
        "train_kl_loss",
        "train_total_loss",
        "validation_reconstruction_loss",
        "validation_kl_loss",
        "validation_total_loss",
    ):
        if key in best_history:
            metrics[key] = best_history[key]

    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_history_csv(training_result["history"], output_path / "training_history.csv")
    save_model_npz(model, output_path / "model.npz")

    return {"metrics": metrics, "history": training_result["history"], "output_dir": str(output_path.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TP5 variational autoencoder (2).")
    parser.add_argument("config_path", nargs="?", default=None, help="Path to the JSON config file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to autoencoder_vae/output/<config_stem>/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config_path)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(config_path)
    result = run_vae(config=config, config_path=config_path, output_dir=output_dir)
    metrics = result["metrics"]
    print(
        "Finished VAE run:"
        f" output={result['output_dir']}"
        f" seed={metrics['seed']}"
        f" n={metrics['n_samples']}"
        f" latent={metrics['latent_dim']}"
        f" beta={metrics['beta']}"
        f" recon={metrics['reconstruction_loss']:.2f}"
        f" kl={metrics['kl_loss']:.2f}"
        f" total={metrics['total_loss']:.2f}"
    )


if __name__ == "__main__":
    main()
