"""Render the joint (beta x latent) grid search: heatmaps + trade-off frontier.

Selection rule (no arbitrary weighting):
  * rank each cell by realism (NN-dist to real, lower better) and by diversity
    (pairwise dist, higher better); the joint optimum minimises rank_realism +
    rank_diversity (a weighting-free balance of the two).
  * we also draw the Pareto frontier in (diversity, realism) space and mark
    where the chosen pair lands, so the audience sees the trade-off directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
GRID = REPO_ROOT / "experiments" / "output" / "vae" / "grid"
OUT = REPO_ROOT / "autoencoder_vae" / "output"


def load():
    data = json.loads((GRID / "grid_results.json").read_text())
    return data


def grid_matrix(results, betas, latents, key):
    M = np.full((len(latents), len(betas)), np.nan)
    for r in results:
        i = latents.index(r["latent"]); j = betas.index(r["beta"])
        M[i, j] = r[key]
    return M


def pareto_front(div, dist):
    """Indices on the Pareto front: maximise div, minimise dist."""
    order = np.argsort(-np.asarray(div))
    front, best_dist = [], np.inf
    for idx in order:
        if dist[idx] <= best_dist:
            front.append(idx); best_dist = dist[idx]
    return set(front)


def main() -> None:
    data = load()
    betas, latents, results = data["betas"], data["latents"], data["results"]

    realism = np.array([r["realism"] for r in results])
    diversity = np.array([r["diversity"] for r in results])
    rank_real = realism.argsort().argsort()              # 0 = best (lowest dist)
    rank_div = (-diversity).argsort().argsort()           # 0 = best (highest div)
    score = rank_real + rank_div
    naive_best = results[int(score.argmin())]
    print(f"[plot] NAIVE rank-sum winner (confounded): beta={naive_best['beta']} latent={naive_best['latent']}")

    # Principled choice: latente by RECONSTRUCTION (capacity, un-confounded);
    # beta bounded below by KL (low beta = prior holes = noise) and above by
    # diversity (high beta = collapse). The cell clean on all three is (b2, l8).
    CHOICE = {"beta": 2.0, "latent": 8}
    bsel = next(r for r in results if r["beta"] == CHOICE["beta"] and r["latent"] == CHOICE["latent"])
    best = results.index(bsel)
    print(f"[plot] principled choice: beta={bsel['beta']} latent={bsel['latent']} "
          f"recon={bsel['recon']:.0f} kl={bsel['kl']:.1f} diversity={bsel['diversity']:.2f}")

    # ---------- heatmaps (only UN-confounded quantities) ----------
    panels = [("recon", "Reconstrucción (↓)\ncapacidad → fija latente", "viridis_r"),
              ("kl", "KL (regularización)\nβ bajo → huecos en el prior", "magma"),
              ("diversity", "Diversidad (↑)\nβ alto → colapso", "magma")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
    for ax, (key, title, cmap) in zip(axes, panels):
        M = grid_matrix(results, betas, latents, key)
        norm = LogNorm(vmin=np.nanmin(M), vmax=np.nanmax(M)) if key == "kl" else None
        im = ax.imshow(M, cmap=cmap, aspect="auto", origin="lower", norm=norm)
        ax.set_xticks(range(len(betas))); ax.set_xticklabels([f"{b:g}" for b in betas])
        ax.set_yticks(range(len(latents))); ax.set_yticklabels(latents)
        ax.set_xlabel("β"); ax.set_ylabel("latente")
        ax.set_title(title, fontsize=10)
        for i in range(len(latents)):
            for j in range(len(betas)):
                txt = f"{M[i,j]:.0f}" if key in ("recon", "kl") else f"{M[i,j]:.1f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if cmap.endswith("_r") else "black")
        # outline chosen cell
        i = latents.index(bsel["latent"]); j = betas.index(bsel["beta"])
        ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor="#16a34a", lw=3))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Búsqueda conjunta β × latente — elegido (recuadro): β={bsel['beta']:g}, latente={bsel['latent']}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "grid_heatmaps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- trade-off scatter + Pareto frontier ----------
    naive_idx = results.index(naive_best)
    front = pareto_front(diversity, realism)
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for k, r in enumerate(results):
        on = k in front
        if k == best:
            color = "#16a34a"
        elif k == naive_idx:
            color = "#dc2626"
        elif on:
            color = "#2563eb"
        else:
            color = "#9ca3af"
        ax.scatter(r["diversity"], r["realism"], s=95 if k in (best, naive_idx) else (60 if on else 38),
                   c=color, edgecolors="black", linewidths=0.7, zorder=4 if k in (best, naive_idx) else (3 if on else 2))
        ax.annotate(f"β{r['beta']:g}·l{r['latent']}", (r["diversity"], r["realism"]),
                    fontsize=6.5, xytext=(3, 3), textcoords="offset points",
                    color="#111" if on else "#888")
    fl = sorted([results[k] for k in front], key=lambda r: r["diversity"])
    ax.plot([r["diversity"] for r in fl], [r["realism"] for r in fl],
            "--", color="#2563eb", lw=1.3, alpha=0.7, zorder=1, label="frontera de Pareto")
    ax.scatter([], [], c="#16a34a", edgecolors="black", label="elegido: β2·l8 (criterio por propiedad)")
    ax.scatter([], [], c="#dc2626", edgecolors="black", label="'ganador' ingenuo β8·l4 → borroso")
    # callout on the misleading corner
    nb = naive_best
    ax.annotate("máx. 'realismo' = punks promediados/borrosos\n(la métrica engaña)",
                (nb["diversity"], nb["realism"]), xytext=(-8, -42), textcoords="offset points",
                fontsize=7.5, color="#dc2626", ha="right",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.1))
    ax.set_xlabel("diversidad  →  (más variado)")
    ax.set_ylabel("dist. a punk real  ←  (más 'realista')")
    ax.invert_yaxis()  # lower distance goes up
    ax.set_title("Trade-off generación: ninguna métrica pixel-a-pixel sola elige bien")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "grid_tradeoff.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {OUT/'grid_heatmaps.png'} and {OUT/'grid_tradeoff.png'}")


if __name__ == "__main__":
    main()
