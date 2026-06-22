"""Joint 2D hyperparameter search over (beta x latent_dim) for the VAE (ej. 2).

Why this exists: the per-axis sweeps (beta at fixed latent=4, latent at fixed
beta=2) are coordinate descent and never prove that the CHOSEN pair is jointly
optimal. This trains the full grid under identical conditions and scores each
cell by GENERATION quality, so the final (beta, latent) is justified as the
joint optimum, not as two stitched-together 1-D arguments.

Selection metric (a generative model must be judged by its samples, not by the
training loss — total loss is invalid because beta=0 always minimises it):
  * realism   = mean distance from each z~N(0,I) sample to its nearest REAL
                punk  (lower = fewer ghosts / artifacts)
  * diversity = mean pairwise distance among generated samples
                (higher = no mode collapse)
Reconstruction is reported too: it explains the LATENT axis (capacity) but is
monotone in beta so it cannot choose beta.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoencoder_vae.main import run_vae, load_config
from autoencoder_vae.generation import load_vae_npz, sample_from_prior

OUT_ROOT = REPO_ROOT / "experiments" / "output" / "vae" / "grid"
TENSOR = "../../data/data_punks_bundle/tensors/punks_rgb.npy"

BETAS = [0.0, 1.0, 2.0, 4.0, 8.0]
LATENTS = [2, 4, 8, 16]

N_GEN = 256       # samples drawn from the prior per cell
GEN_SEED = 2024
LIMIT = 2000      # real punks used for training + as realism reference
EPOCHS = 60


def cell_tag(beta: float, latent: int) -> str:
    return f"b{beta:g}_l{latent}"


def realism_diversity(model, real_flat: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    gen = sample_from_prior(model, n, rng).reshape(n, -1)          # (n, 1728) in [0,1]
    # realism: nearest real punk for each generated sample (lower = better)
    nn = cdist(gen, real_flat).min(axis=1)
    realism = float(nn.mean())
    # diversity: mean pairwise distance among generated (higher = better)
    dd = cdist(gen, gen)
    iu = np.triu_indices(n, k=1)
    diversity = float(dd[iu].mean())
    return realism, diversity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny grid + few epochs to validate the pipeline")
    args = ap.parse_args()

    betas, latents, epochs, n_gen = BETAS, LATENTS, EPOCHS, N_GEN
    if args.smoke:
        betas, latents, epochs, n_gen = [1.0, 8.0], [4, 8], 3, 64

    base_config, base_path = load_config(REPO_ROOT / "autoencoder_vae" / "configs" / "base.json")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(betas) * len(latents)
    i = 0
    for latent in latents:
        for beta in betas:
            i += 1
            tag = cell_tag(beta, latent)
            cfg = deepcopy(base_config)
            cfg["model"].update({"beta": beta, "latent_dim": latent, "input_dim": 1728})
            cfg["dataset"].update({"tensor_path": TENSOR, "limit": LIMIT, "validation_fraction": 0.0})
            # identical training budget for every cell (no early stop -> comparable)
            cfg["training"].update({"epochs_max": epochs, "early_stopping_patience": epochs})
            out_dir = OUT_ROOT / tag
            print(f"[grid {i}/{total}] training {tag} (beta={beta}, latent={latent}, epochs={epochs})", flush=True)
            res = run_vae(cfg, base_path, out_dir)
            recon = float(res["metrics"]["reconstruction_loss"])
            kl = float(res["metrics"]["kl_loss"])

            model = load_vae_npz(out_dir / "model.npz")
            # realism reference = the same real punks the cell trained on
            from autoencoder_vae.dataset import prepare_dataset_split, normalize_dataset_config
            resolved = deepcopy(cfg)
            resolved["dataset"] = normalize_dataset_config(resolved["dataset"])
            resolved["dataset"]["tensor_path"] = str((base_path.parent / TENSOR).resolve())
            real_flat = prepare_dataset_split(resolved).train_flat
            realism, diversity = realism_diversity(model, real_flat, n_gen, GEN_SEED)

            row = {"beta": beta, "latent": latent, "tag": tag,
                   "recon": recon, "kl": kl, "realism": realism, "diversity": diversity}
            results.append(row)
            print(f"           recon={recon:.1f} kl={kl:.2f} realism={realism:.3f} diversity={diversity:.3f}", flush=True)

    payload = {"betas": betas, "latents": latents, "epochs": epochs, "n_gen": n_gen,
               "limit": LIMIT, "results": results}
    out_json = OUT_ROOT / ("grid_results_smoke.json" if args.smoke else "grid_results.json")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[grid] DONE -> {out_json}")


if __name__ == "__main__":
    main()
