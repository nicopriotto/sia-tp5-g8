from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.shared.experiment_utils import finalize_experiment, load_base_config, parse_mode_args, run_variant_multi_seed


EXPERIMENT = "epochs"
EPOCH_OPTIONS = [500, 1000, 2000, 3000, 4000]


def main() -> None:
    args = parse_mode_args("Compare training budgets by maximum epochs ceteris paribus.")
    base_config, base_config_path = load_base_config(args.mode)

    variants = []
    for epochs_max in EPOCH_OPTIONS:
        variants.append(
            (
                f"ep_{epochs_max}",
                {
                    "training": {
                        "epochs_max": epochs_max,
                        "early_stopping_patience": epochs_max,
                    }
                },
            )
        )

    summary_rows = []
    for variant_name, override in variants:
        summary, _ = run_variant_multi_seed(
            experiment=EXPERIMENT,
            variant=variant_name,
            base_config=base_config,
            base_config_path=base_config_path,
            override=override,
            mode=args.mode,
        )
        summary_rows.append(summary)

    finalize_experiment(EXPERIMENT, summary_rows)


if __name__ == "__main__":
    main()
