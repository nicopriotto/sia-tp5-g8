from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoencoder_core.visualization import plot_denoising_curve
from experiments.shared.denoising_utils import (
    EXPERIMENT_OUTPUT_ROOT,
    finalize_experiment,
    load_base_config,
    parse_mode_args,
    run_variant_multi_seed,
)


EXPERIMENT = "noise_level"
NOISE_TYPES = ["bit_flip", "salt_and_pepper", "gaussian_clipped"]
NOISE_LEVELS = [0.05, 0.1, 0.15, 0.2, 0.3]


def main() -> None:
    args = parse_mode_args("Sweep noise_level for each noise_type and study denoising capacity.")
    base_config, base_config_path = load_base_config()

    summary_rows = []
    for noise_type in NOISE_TYPES:
        curve = {"levels": [], "input_err": [], "output_err": [], "exact": []}
        for level in NOISE_LEVELS:
            variant = f"{noise_type}_{level:0.2f}"
            override = {"denoising": {"noise_type": noise_type, "noise_level": level}}
            summary, run_metrics = run_variant_multi_seed(
                experiment=EXPERIMENT,
                variant=variant,
                base_config=base_config,
                base_config_path=base_config_path,
                override=override,
                mode=args.mode,
            )
            summary_rows.append(summary)
            curve["levels"].append(level)
            curve["input_err"].append(float(np.mean([m["input_mean_pixel_error"] for m in run_metrics])))
            curve["output_err"].append(float(np.mean([m["mean_pixel_error"] for m in run_metrics])))
            curve["exact"].append(float(np.mean([m["exact_reconstruction_rate"] for m in run_metrics])))

        curve_path = EXPERIMENT_OUTPUT_ROOT / EXPERIMENT / f"curve_{noise_type}.png"
        plot_denoising_curve(
            noise_levels=curve["levels"],
            input_mean_pixel_error=curve["input_err"],
            output_mean_pixel_error=curve["output_err"],
            exact_reconstruction_rate=curve["exact"],
            output_path=curve_path,
            title=f"Denoising capacity vs noise level ({noise_type})",
        )

    finalize_experiment(EXPERIMENT, summary_rows)


if __name__ == "__main__":
    main()
