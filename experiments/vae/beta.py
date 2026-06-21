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


EXPERIMENT = "beta"
BETA_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0]

# Reduced dataset plus validation split so the full sweep is feasible and comparable.
COMMON_OVERRIDE = {
    "dataset": {"limit": 2000, "validation_fraction": 0.1},
    "training": {"epochs_max": 60, "early_stopping_patience": 15},
}


def main() -> None:
    args = parse_mode_args("Sweep beta and study the reconstruction vs. KL trade-off.")
    base_config, base_config_path = load_base_config()

    summary_rows = []
    for beta in BETA_VALUES:
        override = {**COMMON_OVERRIDE, "model": {"beta": beta}}
        summary, _ = run_variant_multi_seed(
            experiment=EXPERIMENT,
            variant=f"beta_{beta}",
            base_config=base_config,
            base_config_path=base_config_path,
            override=override,
            mode=args.mode,
        )
        summary_rows.append(summary)

    finalize_experiment(EXPERIMENT, summary_rows)


if __name__ == "__main__":
    main()
