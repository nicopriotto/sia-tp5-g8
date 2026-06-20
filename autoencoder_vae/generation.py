from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoencoder_vae.model import VariationalAutoencoder

IMAGE_SHAPE = (24, 24, 3)


def load_vae_npz(path: str | Path) -> VariationalAutoencoder:
    """Rebuild a trained VariationalAutoencoder from a model.npz file."""
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        model = VariationalAutoencoder(
            input_dim=int(meta["input_dim"]),
            latent_dim=int(meta["latent_dim"]),
            encoder_hidden_layers=list(meta["encoder_hidden_layers"]),
            decoder_hidden_layers=list(meta["decoder_hidden_layers"]),
            hidden_activation=str(meta["hidden_activation"]),
            output_activation=str(meta["output_activation"]),
            weight_init=str(meta["weight_init"]),
            dropout=float(meta["dropout"]),
            beta=float(meta.get("beta", 1.0)),
        )
        model.load_parameters({key: data[key] for key in data.files if key != "metadata"})
    return model


def flat_to_images(flat: np.ndarray) -> np.ndarray:
    """Reshape decoder output (N, 1728) in [0, 1] to (N, 24, 24, 3)."""
    return np.clip(flat, 0.0, 1.0).reshape(-1, *IMAGE_SHAPE)


def images_to_uint8(images: np.ndarray) -> np.ndarray:
    return (np.clip(images, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def sample_from_prior(model: VariationalAutoencoder, n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n new punks by sampling z ~ N(0, I) and decoding (the 2c deliverable)."""
    z = rng.standard_normal((n, model.latent_dim))
    return flat_to_images(model.decode(z))


def reconstruct(model: VariationalAutoencoder, X_flat: np.ndarray) -> np.ndarray:
    """Deterministic reconstruction (z = mu) of flattened punks."""
    return flat_to_images(model.reconstruct(X_flat))


def interpolate_latent(model: VariationalAutoencoder, x_a: np.ndarray, x_b: np.ndarray, steps: int) -> np.ndarray:
    """Decode a linear interpolation in latent space between two punks (encoded to mu)."""
    mu_a = model.encode(x_a.reshape(1, -1))
    mu_b = model.encode(x_b.reshape(1, -1))
    ts = np.linspace(0.0, 1.0, steps).reshape(-1, 1)
    z = (1.0 - ts) * mu_a + ts * mu_b
    return flat_to_images(model.decode(z))


def tile_grid(images: np.ndarray, rows: int, cols: int, pad: int = 1, pad_value: int = 255) -> np.ndarray:
    """Tile (N, 24, 24, 3) uint8 images into a single grid image with padding."""
    h, w, c = IMAGE_SHAPE
    grid = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, c), pad_value, dtype=np.uint8)
    for idx in range(min(len(images), rows * cols)):
        r, col = divmod(idx, cols)
        y = pad + r * (h + pad)
        x = pad + col * (w + pad)
        grid[y : y + h, x : x + w] = images[idx]
    return grid


def save_pngs(images_uint8: np.ndarray, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(len(images_uint8)):
        Image.fromarray(images_uint8[idx], mode="RGB").save(out_dir / f"{prefix}_{idx:03d}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate new punks from a trained VAE (2c).")
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=str(THIS_DIR / "output" / "base"),
        help="Run directory containing model.npz. Defaults to autoencoder_vae/output/base.",
    )
    parser.add_argument("--n", type=int, default=64, help="Number of punks to generate.")
    parser.add_argument("--seed", type=int, default=2024, help="Sampling seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    model = load_vae_npz(run_dir / "model.npz")
    rng = np.random.default_rng(args.seed)

    samples = images_to_uint8(sample_from_prior(model, args.n, rng))
    gen_dir = run_dir / "generated"
    save_pngs(samples, gen_dir, "punk")

    cols = 8
    rows = (args.n + cols - 1) // cols
    grid = tile_grid(samples, rows, cols)
    Image.fromarray(grid, mode="RGB").save(run_dir / "generated_samples.png")

    print(f"Generated {args.n} punks -> {gen_dir} (+ grid {run_dir / 'generated_samples.png'})")


if __name__ == "__main__":
    main()
