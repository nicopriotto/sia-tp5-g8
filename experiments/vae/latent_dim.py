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
    select_knee_variant,
    variant_run_dir,
    write_comparison_csv,
)


EXPERIMENT = "latent_dim"
LATENT_DIMS = [2, 4, 8, 16, 32]
KNEE_TOLERANCE = 0.05  # accept the smallest latent within 5% of the best reconstruction.

# Reduced dataset plus validation split so the full sweep is feasible and comparable.
COMMON_OVERRIDE = {
    "dataset": {"limit": 2000, "validation_fraction": 0.1},
    "training": {"epochs_max": 60, "early_stopping_patience": 15},
}


def main() -> None:
    args = parse_mode_args("Sweep latent_dim and study reconstruction quality vs. capacity.")
    base_config, base_config_path = load_base_config()

    summary_rows = []
    montage_entries = []
    for latent_dim in LATENT_DIMS:
        variant = f"latent_{latent_dim}"
        override = {**COMMON_OVERRIDE, "model": {"latent_dim": latent_dim}}
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
        montage_entries.append((f"d={latent_dim}", variant_run_dir(EXPERIMENT, variant), 3))

    experiment_root = EXPERIMENT_OUTPUT_ROOT / EXPERIMENT
    experiment_root.mkdir(parents=True, exist_ok=True)
    write_comparison_csv(experiment_root / "comparison.csv", summary_rows)

    plot_metric_curves(
        experiment_root / "comparison.png",
        x_values=LATENT_DIMS,
        rows=summary_rows,
        x_label="latent_dim",
        title="VAE — reconstrucción vs. capacidad del espacio latente",
    )
    plot_sample_montage(
        experiment_root / "samples_by_latent.png",
        entries=montage_entries,
        title="VAE — punks generados desde z ~ N(0, I) por dimensión latente",
    )

    # Pick the minimum sufficient capacity (knee), not the absolute lowest loss
    # (which always favours the largest latent).
    recommended = select_knee_variant(summary_rows, LATENT_DIMS, tol=KNEE_TOLERANCE)
    (experiment_root / "best.json").write_text(json.dumps(recommended, indent=2), encoding="utf-8")

    print(f"[{EXPERIMENT}] reconstrucción por latente (val):")
    for row in summary_rows:
        print(
            f"  latent={int(row['latent_dim']):>3d}"
            f"  recon={row['validation_reconstruction_loss_mean']:8.2f}"
            f"  kl={row['validation_kl_loss_mean']:8.2f}"
            f"  total={row['validation_total_loss_mean']:8.2f}"
        )
    print(
        f"[{EXPERIMENT}] latente recomendado (codo, tol={KNEE_TOLERANCE:.0%})="
        f"{int(recommended['latent_dim'])}"
        f" val_recon={recommended['validation_reconstruction_loss_mean']:.2f}"
    )


if __name__ == "__main__":
    main()
