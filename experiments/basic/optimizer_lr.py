from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.shared.experiment_utils import finalize_experiment, load_base_config, parse_mode_args, run_variant_multi_seed


EXPERIMENT = "optimizer_lr"
LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def main() -> None:
    args = parse_mode_args("Compare optimizer and learning-rate variants.")
    base_config, base_config_path = load_base_config(args.mode)

    variants = []
    for optimizer in ("adam", "sgd_momentum"):
        for learning_rate in LEARNING_RATES:
            lr_token = f"{learning_rate:.0e}".replace("-0", "-").replace("+0", "")
            variant_name = f"{optimizer}_lr_{lr_token}"
            variants.append(
                (
                    variant_name,
                    {"model": {"optimizer": optimizer}, "training": {"learning_rate": learning_rate}},
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
