from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


GRID_PNG = Path(__file__).resolve().parent / "raw" / "punks.png"
PNG_DIR = Path(__file__).resolve().parent / "punks"
TENSOR_DIR = Path(__file__).resolve().parent / "tensors"

GRID_SIDE = 100  # 100 x 100 punks per side
PUNK_SIDE = 24   # 24 x 24 pixels per punk
NUM_PUNKS = GRID_SIDE * GRID_SIDE  # 10000

# Rec. 601 luminance weights used to collapse RGB into a single channel.
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Named background colors used to composite the transparent original punks.
# "teal" is the #638596 background OpenSea renders behind every punk, so the
# composited images look like the canonical CryptoPunks people recognize.
BG_COLORS = {
    "teal": (0x63, 0x85, 0x96),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}


def parse_bg(value: str) -> np.ndarray:
    """Resolve a background to an RGB triple from a name or a #RRGGBB hex."""
    if value in BG_COLORS:
        rgb = BG_COLORS[value]
    else:
        hex_value = value.lstrip("#")
        if len(hex_value) != 6:
            raise ValueError(f"Background must be a name {list(BG_COLORS)} or #RRGGBB, got {value!r}")
        rgb = tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))
    return np.array(rgb, dtype=np.float32)


def load_grid(grid_path: str | Path = GRID_PNG) -> np.ndarray:
    """Read the composite CryptoPunks PNG as a (2400, 2400, 4) uint8 array."""
    grid = np.asarray(Image.open(grid_path).convert("RGBA"), dtype=np.uint8)
    expected = (GRID_SIDE * PUNK_SIDE, GRID_SIDE * PUNK_SIDE, 4)
    if grid.shape != expected:
        raise ValueError(f"Expected grid shape {expected}, got {grid.shape}")
    return grid


def slice_punks(grid: np.ndarray) -> np.ndarray:
    """Cut the grid into (10000, 24, 24, 4) so that punks[i] is CryptoPunk #i.

    The grid is laid out row-major, so reshape + transpose recovers each tile
    losslessly: index i == row * 100 + col == the official punk ID.
    """
    tiles = grid.reshape(GRID_SIDE, PUNK_SIDE, GRID_SIDE, PUNK_SIDE, 4)
    tiles = tiles.transpose(0, 2, 1, 3, 4)  # (row, col, h, w, channels)
    punks = tiles.reshape(NUM_PUNKS, PUNK_SIDE, PUNK_SIDE, 4)
    return np.ascontiguousarray(punks)


def to_rgb(punks_rgba: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Composite RGBA over a solid background and drop the alpha channel.

    Each pixel becomes `fg*alpha + bg*(1-alpha)`: transparent pixels (alpha=0)
    take the background color, semi-transparent ones (alpha=128) blend, and
    opaque ones are kept as-is.
    """
    alpha = punks_rgba[..., 3:4].astype(np.float32) / 255.0
    fg = punks_rgba[..., :3].astype(np.float32)
    rgb = (fg * alpha + bg * (1.0 - alpha)).round()
    return rgb.astype(np.uint8)


def to_gray(punks_rgb: np.ndarray) -> np.ndarray:
    """Collapse RGB into a single luminance channel -> (N, 24, 24, 1)."""
    gray = punks_rgb.astype(np.float32) @ LUMA_WEIGHTS
    return gray.round().astype(np.uint8)[..., None]


def save_pngs(punks_rgb: np.ndarray, out_dir: Path = PNG_DIR) -> None:
    """Dump 10000 individual RGB PNGs named by punk ID (punk_0000.png ...).

    These are the composited images (same background as the tensors), so what
    you see in a PNG is exactly what the network is trained on.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for index in range(punks_rgb.shape[0]):
        Image.fromarray(punks_rgb[index], mode="RGB").save(
            out_dir / f"punk_{index:04d}.png"
        )


def save_tensor(array: np.ndarray, name: str, out_dir: Path = TENSOR_DIR) -> Path:
    """Persist a uint8 tensor as .npy (values 0..255; normalize at load time)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"punks_{name}.npy"
    np.save(path, array)
    return path


def build(modes: list[str], write_pngs: bool, bg: np.ndarray) -> None:
    grid = load_grid()
    punks_rgba = slice_punks(grid)
    print(f"Sliced {punks_rgba.shape[0]} punks of {PUNK_SIDE}x{PUNK_SIDE} (RGBA)")

    punks_rgb = to_rgb(punks_rgba, bg)
    print(f"Composited over background RGB {tuple(int(c) for c in bg)}")

    if write_pngs:
        save_pngs(punks_rgb)
        print(f"Wrote {NUM_PUNKS} PNGs to {PNG_DIR}")

    tensors = {
        "rgba": punks_rgba,
        "rgb": punks_rgb,
        "gray": to_gray(punks_rgb),
    }
    for mode in modes:
        path = save_tensor(tensors[mode], mode)
        print(f"Saved {mode:4s} tensor {tensors[mode].shape} uint8 -> {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice punks.png into tensors and PNGs.")
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=["rgba", "rgb", "gray", "all"],
        default=["rgb"],
        help="Which tensor(s) to generate (default: rgb).",
    )
    parser.add_argument(
        "--no-pngs",
        action="store_true",
        help="Skip writing the 10000 individual PNGs.",
    )
    parser.add_argument(
        "--bg",
        default="teal",
        help="Background for rgb/gray compositing: teal/black/white or #RRGGBB (default: teal = OpenSea #638596).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = ["rgba", "rgb", "gray"] if "all" in args.mode else list(dict.fromkeys(args.mode))
    build(modes, write_pngs=not args.no_pngs, bg=parse_bg(args.bg))


if __name__ == "__main__":
    main()
