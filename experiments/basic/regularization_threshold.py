from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.shared.experiment_utils import finalize_experiment, load_base_config, parse_mode_args, run_variant_multi_seed


EXPERIMENT = "regularization_threshold"


def main() -> None:
    args = parse_mode_args("Compare L2, gradient clipping and pixel-threshold variants.")
    base_config, base_config_path = load_base_config(args.mode)
    variants = [
        ("l2_0", {"training": {"l2_weight_decay": 0.0}}),
        ("l2_1e5", {"training": {"l2_weight_decay": 1e-5}}),
        ("l2_1e4", {"training": {"l2_weight_decay": 1e-4}}),
        ("l2_1e3", {"training": {"l2_weight_decay": 1e-3}}),
        ("clip_none", {"training": {"gradient_clip_norm": None}}),
        ("clip_1", {"training": {"gradient_clip_norm": 1.0}}),
        ("clip_5", {"training": {"gradient_clip_norm": 5.0}}),
        ("thr_04", {"evaluation": {"pixel_threshold": 0.4}}),
        ("thr_05", {"evaluation": {"pixel_threshold": 0.5}}),
        ("thr_06", {"evaluation": {"pixel_threshold": 0.6}}),
    ]

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
