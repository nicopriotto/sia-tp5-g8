from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoencoder_vae.dataset import load_run_dataset_split
from autoencoder_vae.generation import load_vae_npz, sample_from_prior


# --------------------------------------------------------------------------
# Embeddings: medimos originalidad en espacio de features, no de pixeles.
#   - latente del VAE (encode -> mu): semantico, mismo encoder para todos.
#   - PCA sobre pixeles (numpy SVD): embedding independiente del modelo,
#     para no juzgar al VAE solo con su propio espacio (evita circularidad).
# --------------------------------------------------------------------------

def latent_embeddings(model, X_flat: np.ndarray) -> np.ndarray:
    return model.encode(X_flat)


def fit_pca(X_train_flat: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train_flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(X_train_flat - mean, full_matrices=False)
    return mean, vt[:n_components]


def pca_transform(X_flat: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (X_flat - mean) @ components.T


# --------------------------------------------------------------------------
# Vecino mas cercano + calibracion estilo AuthPct.
# --------------------------------------------------------------------------

def nearest_neighbor(query: np.ndarray, ref: np.ndarray, exclude_self: bool = False, chunk: int = 512, label: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Para cada fila de query, distancia y argmin contra ref (chunked en memoria)."""
    n = query.shape[0]
    min_dist = np.empty(n, dtype=float)
    arg = np.empty(n, dtype=int)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d = cdist(query[start:end], ref)
        if exclude_self:  # valido solo cuando query es el mismo array que ref
            for row in range(end - start):
                d[row, start + row] = np.inf
        min_dist[start:end] = d.min(axis=1)
        arg[start:end] = d.argmin(axis=1)
        if label:
            print(f"\r    {label}: {end}/{n}", end="", flush=True)
    if label:
        print(flush=True)
    return min_dist, arg


def originality_in_space(emb_gen: np.ndarray, emb_train: np.ndarray, space: str = "") -> dict:
    """Distancia generado->original, calibrada contra la densidad local del dataset."""
    tag = f"[{space}] " if space else ""
    gen_dist, gen_match = nearest_neighbor(emb_gen, emb_train, label=f"{tag}generados -> originales")
    train_nn_dist, _ = nearest_neighbor(emb_train, emb_train, exclude_self=True, label=f"{tag}densidad del dataset (real->real)")
    # AuthPct: ratio = d(gen, vecino_real) / d(ese_vecino, su_propio_vecino_real)
    ratio = gen_dist / train_nn_dist[gen_match]
    # percentil de la distancia del generado dentro de las distancias reales-reales
    pct = np.array([(train_nn_dist < d).mean() * 100.0 for d in gen_dist])
    return {
        "dist": gen_dist,
        "match": gen_match,
        "ratio": ratio,
        "percentile": pct,
        "train_nn_dist": train_nn_dist,
    }


# --------------------------------------------------------------------------
# Chequeos perceptuales para el ranking (SSIM estructural + paleta de color).
# --------------------------------------------------------------------------

def ssim(img_a: np.ndarray, img_b: np.ndarray, sigma: float = 1.5) -> float:
    """SSIM con ventana gaussiana, promediado sobre canales. img en [0,1] (24,24,3)."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    scores = []
    for ch in range(img_a.shape[2]):
        a, b = img_a[..., ch], img_b[..., ch]
        mu_a = gaussian_filter(a, sigma)
        mu_b = gaussian_filter(b, sigma)
        va = gaussian_filter(a * a, sigma) - mu_a ** 2
        vb = gaussian_filter(b * b, sigma) - mu_b ** 2
        vab = gaussian_filter(a * b, sigma) - mu_a * mu_b
        s = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
        scores.append(float(s.mean()))
    return float(np.mean(scores))


def color_histogram(img: np.ndarray, bins: int = 8) -> np.ndarray:
    parts = [np.histogram(img[..., ch], bins=bins, range=(0.0, 1.0))[0] for ch in range(img.shape[2])]
    hist = np.concatenate(parts).astype(float)
    return hist / hist.sum()


def histogram_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Interseccion de histogramas de color en [0,1] (1 = misma paleta)."""
    return float(np.minimum(color_histogram(img_a), color_histogram(img_b)).sum())


# --------------------------------------------------------------------------
# Reporte y ranking
# --------------------------------------------------------------------------

def build_report(latent: dict, pca: dict, reference_indices: np.ndarray) -> list[dict]:
    rows = []
    for i in range(len(latent["dist"])):
        lat_novel = latent["ratio"][i] >= 1.0
        pca_novel = pca["ratio"][i] >= 1.0
        rows.append({
            "gen_idx": i,
            "latent_dist": round(float(latent["dist"][i]), 4),
            "latent_match": int(reference_indices[int(latent["match"][i])]),
            "latent_ratio": round(float(latent["ratio"][i]), 3),
            "latent_pctile": round(float(latent["percentile"][i]), 1),
            "pca_dist": round(float(pca["dist"][i]), 4),
            "pca_match": int(reference_indices[int(pca["match"][i])]),
            "pca_ratio": round(float(pca["ratio"][i]), 3),
            "pca_pctile": round(float(pca["percentile"][i]), 1),
            "novel": bool(lat_novel and pca_novel),
        })
    return rows


def plot_ranking(
    gen_img: np.ndarray,
    originals: np.ndarray,
    original_indices: np.ndarray,
    neighbor_idx: np.ndarray,
    dists: np.ndarray,
    gen_idx: int,
    verdict: str,
    output_path: Path,
) -> None:
    top_n = len(neighbor_idx)
    fig, axes = plt.subplots(1, top_n + 1, figsize=((top_n + 1) * 1.25, 1.9))
    axes[0].imshow(np.clip(gen_img, 0, 1))
    axes[0].set_title(f"generado #{gen_idx}\n{verdict}", fontsize=8)
    axes[0].axis("off")
    for rank, (idx, ax) in enumerate(zip(neighbor_idx, axes[1:]), start=1):
        ax.imshow(np.clip(originals[idx], 0, 1))
        ax.set_title(f"#{rank}: punk {int(original_indices[idx])}\nd={dists[rank - 1]:.2f}", fontsize=7)
        ax.axis("off")
    fig.suptitle("Originales más parecidos al generado (ranking por latente)", fontsize=9)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mide originalidad de punks generados y rankea los originales más parecidos.")
    parser.add_argument("run_dir", nargs="?", default=str(THIS_DIR / "output" / "base"), help="Run dir con model.npz.")
    parser.add_argument("--n", type=int, default=200, help="Cantidad de punks generados a evaluar.")
    parser.add_argument("--seed", type=int, default=7, help="Semilla de muestreo.")
    parser.add_argument("--top-n", type=int, default=8, help="Cuántos originales mostrar en el ranking.")
    parser.add_argument("--rankings", type=int, default=10, help="Cuántas imágenes de ranking generar (los N generados más originales).")
    parser.add_argument("--index", type=int, default=None, help="Rankear solo este generado (sobrescribe --rankings).")
    parser.add_argument("--pca-dim", type=int, default=50, help="Componentes PCA del embedding sobre píxeles.")
    parser.add_argument(
        "--reference-split",
        choices=["train", "all"],
        default=None,
        help="Qué split usar como referencia para originalidad. Por defecto usa train si existe validación, o all si no.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "originality"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] cargando modelo {run_dir.name} y dataset...", flush=True)
    model = load_vae_npz(run_dir / "model.npz")
    _, dataset_split = load_run_dataset_split(run_dir)
    if args.reference_split is None:
        reference_split = "train" if dataset_split.validation_size else "all"
    else:
        reference_split = args.reference_split

    if reference_split == "train":
        reference_images = dataset_split.train_images
        reference_flat = dataset_split.train_flat
        reference_indices = dataset_split.train_indices
    else:
        reference_images = dataset_split.images
        reference_flat = dataset_split.flat
        reference_indices = dataset_split.selected_indices
    print(
        f"      {reference_flat.shape[0]} punks originales cargados"
        f" (referencia={reference_split})",
        flush=True,
    )

    print(f"[2/6] generando {args.n} punks desde el prior (seed={args.seed})...", flush=True)
    rng = np.random.default_rng(args.seed)
    gen_images = sample_from_prior(model, args.n, rng)
    gen_flat = gen_images.reshape(args.n, -1)

    print("[3/6] calculando embeddings (latente VAE + PCA)...", flush=True)
    lat_train = latent_embeddings(model, reference_flat)
    lat_gen = latent_embeddings(model, gen_flat)
    mean, comps = fit_pca(reference_flat, args.pca_dim)
    pca_train = pca_transform(reference_flat, mean, comps)
    pca_gen = pca_transform(gen_flat, mean, comps)

    print("[4/6] originalidad en espacio LATENTE...", flush=True)
    latent = originality_in_space(lat_gen, lat_train, space="latente")
    print("[5/6] originalidad en espacio PCA...", flush=True)
    pca = originality_in_space(pca_gen, pca_train, space="PCA")
    print("[6/6] armando reporte, CSV y ranking...", flush=True)
    rows = build_report(latent, pca, reference_indices)

    # CSV por muestra
    csv_path = out_dir / "originality.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # resumen agregado
    lat_ratio = latent["ratio"]
    pca_ratio = pca["ratio"]
    novel = np.array([r["novel"] for r in rows])
    print(f"Evaluados {args.n} generados (run {run_dir.name})")
    print(f"  originalidad latente : ratio medio {lat_ratio.mean():.2f}  | originales (ratio>1): {(lat_ratio >= 1).mean() * 100:.0f}%")
    print(f"  originalidad PCA     : ratio medio {pca_ratio.mean():.2f}  | originales (ratio>1): {(pca_ratio >= 1).mean() * 100:.0f}%")
    print(f"  'nuevos' en AMBOS espacios: {novel.mean() * 100:.0f}%")

    # rankings: --index -> solo ese; si no, los N generados más originales (por ratio latente)
    if args.index is not None:
        chosen_indices = [args.index]
    else:
        chosen_indices = list(np.argsort(-lat_ratio)[: args.rankings])

    print(f"  CSV -> {csv_path}")
    for chosen in chosen_indices:
        chosen = int(chosen)
        dist_to_train = cdist(lat_gen[chosen:chosen + 1], lat_train)[0]
        order = np.argsort(dist_to_train)[: args.top_n]
        verdict = (
            f"ratio L={lat_ratio[chosen]:.2f} P={pca_ratio[chosen]:.2f}"
            f" ({'NUEVO' if rows[chosen]['novel'] else 'parecido a un real'})"
        )
        ranking_path = out_dir / f"ranking_gen_{chosen}.png"
        plot_ranking(
            gen_images[chosen],
            reference_images,
            reference_indices,
            order,
            dist_to_train[order],
            chosen,
            verdict,
            ranking_path,
        )
        print(f"  ranking del generado #{chosen} (ratio L={lat_ratio[chosen]:.2f}) -> {ranking_path}")


if __name__ == "__main__":
    main()
