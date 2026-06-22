from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoencoder_core.dataset import flat_to_grid
from autoencoder_core.serialization import load_model_npz
from autoencoder_core.visualization import plot_binary_pattern


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode a latent point with a trained basic autoencoder and export the 5x7 pattern as PNG."
    )
    parser.add_argument(
        "model_source",
        help="Path to model.npz or to a run directory containing model.npz.",
    )
    parser.add_argument(
        "coordinates",
        nargs="+",
        type=float,
        help="Latent coordinates to decode. Count must match the model latent_dim.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path. Defaults to <run_dir>/decoded_latent_<coords>.png",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize decoder probabilities. Default: 0.5.",
    )
    return parser.parse_args()


def resolve_model_path(model_source: str | Path) -> Path:
    source = Path(model_source)
    if source.is_dir():
        model_path = source / "model.npz"
    else:
        model_path = source
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return model_path.resolve()


def default_output_path(model_path: Path, coordinates: np.ndarray) -> Path:
    coord_slug = "_".join(f"{value:g}" for value in coordinates.tolist())
    safe_slug = coord_slug.replace(".", "p")
    return model_path.parent / f"decoded_latent_{safe_slug}.png"


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_source)
    model = load_model_npz(model_path)

    coordinates = np.asarray(args.coordinates, dtype=float)
    if coordinates.shape != (model.latent_dim,):
        raise ValueError(
            f"Expected {model.latent_dim} latent coordinates for this model, got {coordinates.shape[0]}"
        )

    latent_point = coordinates.reshape(1, -1)
    decoded_probabilities = model.decode(latent_point)[0]
    pattern_binary = (decoded_probabilities >= float(args.threshold)).astype(int)

    output_path = Path(args.output) if args.output else default_output_path(model_path, coordinates)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coords_text = ", ".join(f"{value:g}" for value in coordinates.tolist())
    plot_binary_pattern(
        pattern_binary=pattern_binary,
        output_path=output_path,
        title="Generated 5x7 pattern",
        subtitle=f"Latent coordinates: ({coords_text})",
    )

    print(f"Wrote {output_path.resolve()}")
    for row in flat_to_grid(pattern_binary).astype(int):
        print("".join(str(int(value)) for value in row))


if __name__ == "__main__":
    main()
