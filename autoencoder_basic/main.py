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

from autoencoder_core.dataset import load_font_dataset, write_subset_manifest
from autoencoder_core.evaluation import evaluate_autoencoder, generate_novel_letter
from autoencoder_core.model import Autoencoder
from autoencoder_core.serialization import save_model_npz
from autoencoder_core.training import train_autoencoder
from autoencoder_core.visualization import (
    plot_generated_letter,
    plot_latent_scatter,
    plot_reconstructions,
)

DEFAULT_CONFIG = THIS_DIR / "configs" / "base.json"


def load_config(config_path: str | Path | None) -> tuple[dict, Path]:
    path = DEFAULT_CONFIG if config_path is None else Path(config_path)
    resolved_path = path if path.is_absolute() else (Path.cwd() / path).resolve()
    config = json.loads(resolved_path.read_text(encoding="utf-8"))
    return config, resolved_path


def _deep_update(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def resolve_config_paths(config: dict, config_path: Path) -> dict:
    resolved = deepcopy(config)
    dataset_path = Path(resolved["dataset"]["source_path"])
    if not dataset_path.is_absolute():
        dataset_path = (config_path.parent / dataset_path).resolve()
    resolved["dataset"]["source_path"] = str(dataset_path)
    return resolved


def default_output_dir(config_path: Path) -> Path:
    return THIS_DIR / "output" / config_path.stem


def build_model(config: dict, seed: int) -> Autoencoder:
    model_cfg = config["model"]
    return Autoencoder(
        input_dim=int(model_cfg["input_dim"]),
        latent_dim=int(model_cfg["latent_dim"]),
        encoder_hidden_layers=list(model_cfg["encoder_hidden_layers"]),
        decoder_hidden_layers=list(model_cfg["decoder_hidden_layers"]),
        hidden_activation=str(model_cfg["hidden_activation"]),
        output_activation=str(model_cfg["output_activation"]),
        weight_init=str(model_cfg["weight_init"]),
        dropout=float(model_cfg["dropout"]),
        rng=np.random.default_rng(seed),
    )


def write_history_csv(history: list[dict], output_path: str | Path) -> None:
    path = Path(output_path)
    if not history:
        raise ValueError("Training history is empty")
    fieldnames = [
        "epoch",
        "train_loss",
        "reconstruction_loss",
        "l2_penalty",
        "mean_pixel_error",
        "max_pixel_error",
        "exact_reconstruction_rate",
        "within_one_pixel_rate",
        "all_patterns_within_one_pixel",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def run_basic_autoencoder(
    config: dict,
    config_path: Path,
    output_dir: str | Path,
) -> dict:
    resolved_config = resolve_config_paths(config, config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_seed = int(resolved_config["dataset"]["seed"])
    rng = np.random.default_rng(run_seed)
    dataset = load_font_dataset(
        source_path=resolved_config["dataset"]["source_path"],
        subset=resolved_config["dataset"]["subset"],
        seed=run_seed,
    )
    write_subset_manifest(dataset, output_path / "subset_manifest.json")

    model = build_model(resolved_config, seed=run_seed)
    training_result = train_autoencoder(model, dataset.X, resolved_config, rng)
    evaluation = evaluate_autoencoder(model, dataset.X, resolved_config)
    generation = generate_novel_letter(
        model=model,
        X=dataset.X,
        labels=dataset.labels,
        pixel_threshold=float(resolved_config["evaluation"]["pixel_threshold"]),
    )

    resolved_config_json = deepcopy(resolved_config)
    resolved_config_json["output_dir"] = str(output_path.resolve())
    (output_path / "resolved_config.json").write_text(json.dumps(resolved_config_json, indent=2), encoding="utf-8")

    metrics = {
        "seed": run_seed,
        "subset_name": dataset.subset_name,
        "subset_mode": dataset.subset_mode,
        "selection_metric": resolved_config["evaluation"]["selection_metric"],
        "mean_pixel_error": evaluation["mean_pixel_error"],
        "max_pixel_error": evaluation["max_pixel_error"],
        "exact_reconstruction_rate": evaluation["exact_reconstruction_rate"],
        "within_one_pixel_rate": evaluation["within_one_pixel_rate"],
        "all_patterns_within_one_pixel": evaluation["all_patterns_within_one_pixel"],
        "pixel_error_por_patron": evaluation["pixel_error_per_pattern"],
        "epochs_trained": training_result["epochs_trained"],
        "best_epoch": training_result["best_epoch"],
        "generation_method": generation["generation_method"],
        "generated_latent_point": generation["generated_latent_point"],
        "resolved_hyperparameters": {
            "dataset": resolved_config["dataset"],
            "model": resolved_config["model"],
            "training": resolved_config["training"],
            "evaluation": resolved_config["evaluation"],
            "denoising": resolved_config["denoising"],
        },
    }
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    write_history_csv(training_result["history"], output_path / "training_history.csv")
    plot_latent_scatter(evaluation["latent_codes"], dataset.labels, output_path / "latent_scatter.png")
    plot_reconstructions(
        original_X=dataset.X,
        reconstruction_binary=evaluation["reconstruction_binary"],
        reconstruction_probabilities=evaluation["reconstruction_probabilities"],
        labels=dataset.labels,
        output_path=output_path / "reconstructions.png",
    )
    plot_generated_letter(
        latent_codes=generation["latent_codes"],
        labels=dataset.labels,
        generated_latent_point=generation["generated_latent_point"],
        generated_binary=generation["generated_binary"],
        output_path=output_path / "generated_letters.png",
        legend=generation["legend"],
    )
    save_model_npz(model, output_path / "model.npz")

    return {
        "metrics": metrics,
        "history": training_result["history"],
        "output_dir": str(output_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TP5 basic autoencoder.")
    parser.add_argument("config_path", nargs="?", default=None, help="Path to the JSON config file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to autoencoder_basic/output/<config_stem>/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config_path)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(config_path)
    result = run_basic_autoencoder(config=config, config_path=config_path, output_dir=output_dir)
    metrics = result["metrics"]
    print(
        "Finished run:"
        f" output={result['output_dir']}"
        f" seed={metrics['seed']}"
        f" exact_reconstruction_rate={metrics['exact_reconstruction_rate']:.4f}"
        f" max_pixel_error={metrics['max_pixel_error']}"
    )


if __name__ == "__main__":
    main()
