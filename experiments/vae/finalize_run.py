"""End-to-end final step for the VAE study (ejercicio 2).

Reads the sweep winners (channels / beta / latent_dim), chooses the final
configuration, estimates the training length with epochs.py, trains the final
model on the full dataset, and produces every figure the presentation needs:
reconstructions, generated grid, latent interpolation, PCA scatter and the
originality report.

Idempotent: a completed run leaves a DONE marker and is skipped on re-entry, so
it is safe to trigger from a watcher that may fire more than once.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.shared.vae_utils import EXPERIMENT_OUTPUT_ROOT, load_base_config, deep_update

VAE_DIR = REPO_ROOT / "autoencoder_vae"
CONFIGS_DIR = VAE_DIR / "configs"
OUTPUT_FINAL = VAE_DIR / "output" / "final"
DONE_MARKER = OUTPUT_FINAL / "FINALIZE_DONE"

# Representation: the channels sweep recommends RGB (color without the near-constant
# alpha channel of RGBA), which is also the base config, so the final model is RGB.
FINAL_TENSOR = "../../data/data_punks_bundle/tensors/punks_rgb.npy"
FINAL_INPUT_DIM = 24 * 24 * 3


def _read_best(experiment: str, key: str, default):
    path = EXPERIMENT_OUTPUT_ROOT / experiment / "best.json"
    if not path.exists():
        print(f"[finalize] WARN: {path} no existe, uso default {key}={default}")
        return default
    value = json.loads(path.read_text(encoding="utf-8")).get(key, default)
    print(f"[finalize] {experiment}: {key}={value}")
    return value


def _run(cmd: list[str]) -> None:
    print(f"[finalize] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    if DONE_MARKER.exists():
        print(f"[finalize] ya completado (marker {DONE_MARKER}); nada que hacer.")
        return

    base_config, base_config_path = load_base_config()

    beta = float(_read_best("beta", "beta", 1.0))
    latent_dim = int(_read_best("latent_dim", "latent_dim", base_config["model"]["latent_dim"]))
    channels_winner = _read_best("channels", "variant", "rgb")
    print(f"[finalize] representación ganadora del sweep={channels_winner}; uso RGB para el modelo final.")

    # 1) Config para estimar las épocas (epochs.py vuelve a fijar dataset 2000 + val 0.1).
    search_override = {
        "model": {"beta": beta, "latent_dim": latent_dim, "input_dim": FINAL_INPUT_DIM},
        "dataset": {"tensor_path": FINAL_TENSOR},
    }
    search_config = deep_update(base_config, search_override)
    search_path = CONFIGS_DIR / "final_search.json"
    search_path.write_text(json.dumps(search_config, indent=2), encoding="utf-8")

    _run([sys.executable, "experiments/vae/epochs.py", "--config", str(search_path), "--mode", "quick"])

    rec_path = EXPERIMENT_OUTPUT_ROOT / "epochs" / search_path.stem / "recommended_epoch.json"
    recommended_epoch = int(json.loads(rec_path.read_text(encoding="utf-8"))["recommended_epoch"])
    print(f"[finalize] recommended_epoch={recommended_epoch}")

    # 2) Config final: dataset COMPLETO, sin validación, entrenando las épocas recomendadas.
    final_override = {
        "model": {"beta": beta, "latent_dim": latent_dim, "input_dim": FINAL_INPUT_DIM},
        "dataset": {"tensor_path": FINAL_TENSOR, "limit": None, "validation_fraction": 0.0},
        "training": {
            "epochs_max": recommended_epoch,
            "early_stopping_patience": recommended_epoch,
        },
    }
    final_config = deep_update(base_config, final_override)
    final_path = CONFIGS_DIR / "final.json"
    final_path.write_text(json.dumps(final_config, indent=2), encoding="utf-8")

    # 3) Entrenar el modelo final y generar todo el material de la presentación.
    _run([sys.executable, "autoencoder_vae/main.py", str(final_path), "--output-dir", str(OUTPUT_FINAL)])
    _run([sys.executable, "autoencoder_vae/visualization.py", str(OUTPUT_FINAL)])
    _run([sys.executable, "autoencoder_vae/generation.py", str(OUTPUT_FINAL), "--n", "64"])
    _run([sys.executable, "autoencoder_vae/originality.py", str(OUTPUT_FINAL)])

    summary = {
        "beta": beta,
        "latent_dim": latent_dim,
        "channels_winner": channels_winner,
        "recommended_epoch": recommended_epoch,
        "final_config": str(final_path),
        "output_dir": str(OUTPUT_FINAL),
    }
    DONE_MARKER.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[finalize] LISTO. {json.dumps(summary)}")


if __name__ == "__main__":
    main()
