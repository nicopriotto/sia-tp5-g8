from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.shared.vae_utils import (
    finalize_experiment,
    load_base_config,
    parse_mode_args,
    run_variant_multi_seed,
)


EXPERIMENT = "latent_dim"
LATENT_DIMS = [2, 4, 8, 16, 32]

# Reduced dataset plus validation split so the full sweep is feasible and comparable.
COMMON_OVERRIDE = {
    "dataset": {"limit": 2000, "validation_fraction": 0.1},
    "training": {"epochs_max": 60, "early_stopping_patience": 15},
}


def main() -> None:
    args = parse_mode_args("Sweep latent_dim and study reconstruction quality vs. capacity.")
    base_config, base_config_path = load_base_config()

    summary_rows = []
    for latent_dim in LATENT_DIMS:
        override = {**COMMON_OVERRIDE, "model": {"latent_dim": latent_dim}}
        summary, _ = run_variant_multi_seed(
            experiment=EXPERIMENT,
            variant=f"latent_{latent_dim}",
            base_config=base_config,
            base_config_path=base_config_path,
            override=override,
            mode=args.mode,
        )
        summary_rows.append(summary)

    finalize_experiment(EXPERIMENT, summary_rows)


if __name__ == "__main__":
    main()
