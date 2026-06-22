from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoencoder_core.dataset import flat_to_grid, load_font_dataset
from autoencoder_core.serialization import load_model_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate linearly between two latent points of a trained basic autoencoder "
            "and export a GIF with latent-space trajectory plus decoded 5x7 pattern."
        )
    )
    parser.add_argument(
        "model_source",
        help="Path to model.npz or to a run directory containing model.npz and resolved_config.json.",
    )
    parser.add_argument(
        "--start",
        nargs="+",
        type=float,
        required=True,
        help="Latent coordinates of the starting point.",
    )
    parser.add_argument(
        "--end",
        nargs="+",
        type=float,
        required=True,
        help="Latent coordinates of the ending point.",
    )
    parser.add_argument(
        "--intermediate-samples",
        type=int,
        required=True,
        help="How many samples to generate strictly between the start and end points.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output GIF path. Defaults to <run_dir>/latent_interpolation_<...>.gif",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config path. If omitted, resolved_config.json is searched next to the model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize decoder probabilities. Default: 0.5.",
    )
    parser.add_argument(
        "--duration-ms",
        type=int,
        default=320,
        help="Frame duration in milliseconds. Default: 320.",
    )
    return parser.parse_args()


def resolve_model_path(model_source: str | Path) -> tuple[Path, Path]:
    source = Path(model_source)
    if source.is_dir():
        run_dir = source.resolve()
        model_path = run_dir / "model.npz"
    else:
        model_path = source.resolve()
        run_dir = model_path.parent
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return model_path, run_dir


def resolve_config_path(run_dir: Path, explicit_config: str | None) -> Path:
    if explicit_config is not None:
        config_path = Path(explicit_config).resolve()
    else:
        config_path = run_dir / "resolved_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "resolved_config.json not found. Pass --config or use a run directory that contains it."
        )
    return config_path


def load_resolved_config(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_path = Path(config["dataset"]["source_path"])
    if not dataset_path.is_absolute():
        dataset_path = (config_path.parent / dataset_path).resolve()
    config["dataset"]["source_path"] = str(dataset_path)
    return config


def _slug_coords(coords: np.ndarray) -> str:
    text = "_".join(f"{value:g}" for value in coords.tolist())
    return text.replace(".", "p")


def default_output_path(model_path: Path, start: np.ndarray, end: np.ndarray, intermediate_samples: int) -> Path:
    return model_path.parent / (
        f"latent_interpolation_{_slug_coords(start)}__{_slug_coords(end)}__mid_{intermediate_samples}.gif"
    )


def _validate_coords(name: str, coords: np.ndarray, latent_dim: int) -> None:
    if coords.shape != (latent_dim,):
        raise ValueError(f"{name} must contain exactly {latent_dim} coordinates, got {coords.shape[0]}")


def _interpolation_points(start: np.ndarray, end: np.ndarray, intermediate_samples: int) -> np.ndarray:
    if intermediate_samples < 0:
        raise ValueError("intermediate_samples must be >= 0")
    return np.linspace(start, end, intermediate_samples + 2)


def _latent_axis_limits(latent_codes: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    all_points = np.vstack([latent_codes, start.reshape(1, -1), end.reshape(1, -1)])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    ranges = np.maximum(maxs - mins, 1.0)
    padding = 0.12 * ranges
    x_limits = (float(mins[0] - padding[0]), float(maxs[0] + padding[0]))
    y_limits = (float(mins[1] - padding[1]), float(maxs[1] + padding[1]))
    return x_limits, y_limits


def _render_frame(
    pattern_binary: np.ndarray,
    coords: np.ndarray,
    latent_codes: np.ndarray,
    labels: list[str],
    start: np.ndarray,
    end: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    frame_idx: int,
    total_frames: int,
) -> Image.Image:
    grid = flat_to_grid(pattern_binary).astype(int)

    fig, (ax_latent, ax_pattern) = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.8),
        gridspec_kw={"width_ratios": [1.85, 1.0]},
    )
    fig.suptitle("Interpolacion lineal en el espacio latente", fontsize=16, y=0.98)

    ax_latent.scatter(latent_codes[:, 0], latent_codes[:, 1], color="#2C7FB8", s=48, alpha=0.9)
    for idx, label in enumerate(labels):
        ax_latent.annotate(
            label,
            (latent_codes[idx, 0], latent_codes[idx, 1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=10,
        )
    ax_latent.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color="#FF8C00",
        linewidth=2.2,
        linestyle="-",
        alpha=0.95,
        zorder=2,
    )
    ax_latent.scatter([start[0]], [start[1]], color="#2E8B57", s=90, label="Inicio", zorder=3)
    ax_latent.scatter([end[0]], [end[1]], color="#6A3D9A", s=90, label="Fin", zorder=3)
    ax_latent.scatter(
        [coords[0]],
        [coords[1]],
        color="#E31A1C",
        s=180,
        edgecolors="white",
        linewidths=1.2,
        label="Punto actual",
        zorder=4,
    )
    ax_latent.set_title("Espacio latente", fontsize=13)
    ax_latent.set_xlabel("Latent dimension 1")
    ax_latent.set_ylabel("Latent dimension 2")
    ax_latent.set_xlim(*x_limits)
    ax_latent.set_ylim(*y_limits)
    ax_latent.axhline(0.0, color="#cccccc", linewidth=0.8)
    ax_latent.axvline(0.0, color="#cccccc", linewidth=0.8)
    ax_latent.grid(alpha=0.28)
    ax_latent.legend(loc="best", fontsize=10)

    ax_pattern.imshow(grid, cmap="binary", vmin=0, vmax=1)
    ax_pattern.set_title("Patron decodificado 5x7", fontsize=13)
    ax_pattern.set_xticks(range(5))
    ax_pattern.set_yticks(range(7))
    ax_pattern.set_xticklabels(range(1, 6))
    ax_pattern.set_yticklabels(range(1, 8))
    ax_pattern.set_xlabel(
        f"Frame {frame_idx + 1}/{total_frames}\n"
        f"z=({coords[0]:.3f}, {coords[1]:.3f})",
        fontsize=11,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def main() -> None:
    args = parse_args()
    model_path, run_dir = resolve_model_path(args.model_source)
    config_path = resolve_config_path(run_dir, args.config)
    resolved_config = load_resolved_config(config_path)
    model = load_model_npz(model_path)

    start = np.asarray(args.start, dtype=float)
    end = np.asarray(args.end, dtype=float)
    _validate_coords("start", start, model.latent_dim)
    _validate_coords("end", end, model.latent_dim)

    dataset = load_font_dataset(
        source_path=resolved_config["dataset"]["source_path"],
        subset=resolved_config["dataset"]["subset"],
        seed=int(resolved_config["dataset"]["seed"]),
    )
    latent_codes = model.encode(dataset.X)
    x_limits, y_limits = _latent_axis_limits(latent_codes, start, end)

    points = _interpolation_points(start, end, int(args.intermediate_samples))
    decoded_probabilities = model.decode(points)
    decoded_binary = (decoded_probabilities >= float(args.threshold)).astype(int)

    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(model_path, start, end, int(args.intermediate_samples))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = [
        _render_frame(
            pattern_binary=decoded_binary[idx],
            coords=points[idx],
            latent_codes=latent_codes,
            labels=dataset.labels,
            start=start,
            end=end,
            x_limits=x_limits,
            y_limits=y_limits,
            frame_idx=idx,
            total_frames=len(points),
        )
        for idx in range(len(points))
    ]

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(args.duration_ms),
        loop=0,
    )

    print(f"Wrote GIF to {output_path.resolve()}")
    print(
        f"frames={len(points)} duration_ms={int(args.duration_ms)} "
        f"start={tuple(start.tolist())} end={tuple(end.tolist())}"
    )


if __name__ == "__main__":
    main()
