from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.shared.denoising_utils import (
    finalize_experiment,
    load_base_config,
    parse_mode_args,
    run_variant_multi_seed,
)


EXPERIMENT = "noise_type"
NOISE_LEVEL = 0.1


def main() -> None:
    args = parse_mode_args("Compare bit_flip vs salt_and_pepper vs gaussian_clipped at a fixed noise level.")
    base_config, base_config_path = load_base_config()

    variants = [
        ("bit_flip", {"denoising": {"noise_type": "bit_flip", "noise_level": NOISE_LEVEL}}),
        ("salt_and_pepper", {"denoising": {"noise_type": "salt_and_pepper", "noise_level": NOISE_LEVEL}}),
        ("gaussian_clipped", {"denoising": {"noise_type": "gaussian_clipped", "noise_level": NOISE_LEVEL}}),
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
