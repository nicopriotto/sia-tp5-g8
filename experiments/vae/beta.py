from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.shared.vae_utils import (
    EXPERIMENT_OUTPUT_ROOT,
    load_base_config,
    parse_mode_args,
    plot_metric_curves,
    plot_sample_montage,
    run_variant_multi_seed,
    variant_run_dir,
    write_comparison_csv,
)


EXPERIMENT = "beta"
BETA_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0]
RECOMMENDED_BETA = 1.0  # chosen by the reconstruction/KL trade-off, not by lowest total loss.

# Reduced dataset plus validation split so the full sweep is feasible and comparable.
COMMON_OVERRIDE = {
    "dataset": {"limit": 2000, "validation_fraction": 0.1},
    "training": {"epochs_max": 60, "early_stopping_patience": 15},
}


def main() -> None:
    args = parse_mode_args("Sweep beta and study the reconstruction vs. KL trade-off.")
    base_config, base_config_path = load_base_config()

    summary_rows = []
    montage_entries = []
    for beta in BETA_VALUES:
        variant = f"beta_{beta}"
        override = {**COMMON_OVERRIDE, "model": {"beta": beta}}
        summary, _ = run_variant_multi_seed(
            experiment=EXPERIMENT,
            variant=variant,
            base_config=base_config,
            base_config_path=base_config_path,
            override=override,
            mode=args.mode,
            max_seeds=args.max_seeds,
        )
        summary_rows.append(summary)
        montage_entries.append((f"β={beta:g}", variant_run_dir(EXPERIMENT, variant), 3))

    experiment_root = EXPERIMENT_OUTPUT_ROOT / EXPERIMENT
    experiment_root.mkdir(parents=True, exist_ok=True)
    write_comparison_csv(experiment_root / "comparison.csv", summary_rows)

    # The story is the trade-off, so we plot trends (not bars) and show generated
    # samples per beta. We do NOT crown the lowest-total-loss variant: total loss
    # always favours beta=0 (it pays no KL), which is exactly the model that cannot
    # generate. beta is picked from the trade-off below.
    plot_metric_curves(
        experiment_root / "comparison.png",
        x_values=BETA_VALUES,
        rows=summary_rows,
        x_label="beta",
        title="VAE — trade-off reconstrucción ↔ KL al variar beta",
        logx=True,
    )
    plot_sample_montage(
        experiment_root / "samples_by_beta.png",
        entries=montage_entries,
        title="VAE — punks generados desde z ~ N(0, I) por beta (mismo z por fila)",
        share_z=True,
    )

    recommended = next(row for row in summary_rows if float(row["beta"]) == RECOMMENDED_BETA)
    (experiment_root / "best.json").write_text(json.dumps(recommended, indent=2), encoding="utf-8")

    print(f"[{EXPERIMENT}] trade-off por beta (val):")
    for row in summary_rows:
        print(
            f"  beta={float(row['beta']):>4g}"
            f"  recon={row['validation_reconstruction_loss_mean']:8.2f}"
            f"  kl={row['validation_kl_loss_mean']:8.2f}"
            f"  total={row['validation_total_loss_mean']:8.2f}"
        )
    print(
        f"[{EXPERIMENT}] beta recomendado={RECOMMENDED_BETA:g} (por trade-off; revisar samples_by_beta.png)"
    )


if __name__ == "__main__":
    main()
