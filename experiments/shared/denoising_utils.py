from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from autoencoder_denoising.main import load_config, run_denoising_autoencoder

from .reporting import aggregate_variant_summary, plot_comparison, write_comparison_csv, write_summary_csv
from .seeds import MASTER_SEED, get_mode_seeds


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "autoencoder_denoising" / "configs" / "base.json"
EXPERIMENT_OUTPUT_ROOT = REPO_ROOT / "experiments" / "output" / "denoising"


def parse_mode_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--mode", choices=["formal", "quick"], default="formal")
    return parser.parse_args()


def deep_update(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_base_config() -> tuple[dict, Path]:
    return load_config(BASE_CONFIG_PATH)


def prepare_variant_output(experiment: str, variant: str) -> Path:
    output_dir = EXPERIMENT_OUTPUT_ROOT / experiment / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_variant_multi_seed(
    experiment: str,
    variant: str,
    base_config: dict,
    base_config_path: Path,
    override: dict,
    mode: str,
) -> tuple[dict, list[dict]]:
    seeds = get_mode_seeds(mode)
    variant_output_dir = prepare_variant_output(experiment, variant)
    run_metrics: list[dict] = []

    for run_idx, seed in enumerate(seeds):
        run_dir = variant_output_dir / f"run_{run_idx:02d}"
        config = deep_update(base_config, override)
        config["dataset"]["seed"] = int(seed)
        result = run_denoising_autoencoder(config=config, config_path=base_config_path, output_dir=run_dir)
        run_metrics.append(result["metrics"])
        metrics = result["metrics"]
        print(
            f"[{experiment}/{variant}] run {run_idx + 1}/{len(seeds)}"
            f" seed={seed}"
            f" noise={metrics['noise_type']}@{metrics['noise_level']}"
            f" in={metrics['input_mean_pixel_error']:.2f}"
            f" out={metrics['mean_pixel_error']:.2f}"
            f" exact={metrics['exact_reconstruction_rate']:.3f}"
        )

    summary_row = aggregate_variant_summary(
        experiment=experiment,
        variant=variant,
        mode=mode,
        master_seed=MASTER_SEED,
        seeds=seeds,
        run_metrics=run_metrics,
    )
    write_summary_csv(variant_output_dir / "summary.csv", summary_row)
    return summary_row, run_metrics


def finalize_experiment(experiment: str, summary_rows: list[dict]) -> None:
    experiment_root = EXPERIMENT_OUTPUT_ROOT / experiment
    write_comparison_csv(experiment_root / "comparison.csv", summary_rows)
    plot_comparison(experiment_root / "comparison.png", summary_rows, experiment=experiment)
