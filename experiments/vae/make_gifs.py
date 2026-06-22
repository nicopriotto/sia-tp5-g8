"""Build animated GIFs for the presentation (ejercicio 2).

Instead of cramming 64 tiny punks into a static grid, each GIF cycles a few
LARGE punks (~1 per second) so the audience can actually see them while still
getting the overall sense of the distribution. Comparison axes (beta, latent)
render as a single synchronised GIF with one labelled panel per value.

Pixel-art stays crisp via integer nearest-neighbour upscaling (no blur).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoencoder_vae.generation import load_vae_npz, sample_from_prior, images_to_uint8

OUTPUT = REPO_ROOT / "autoencoder_vae" / "output"
GIF_DIR = OUTPUT / "gifs"

SCALE = 7          # integer upscale per punk (24 -> 168 px), nearest = crisp
PAD = 6            # white padding between punks
GRID = (2, 2)      # punks shown per panel per frame
N_FRAMES = 12      # number of distinct frames (cycled)
FRAME_MS = 1000    # ~1 fps
BG = (255, 255, 255)


def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        import matplotlib
        path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        return ImageFont.truetype(str(path), size)
    except Exception:
        return ImageFont.load_default()


def _upscale(punk_uint8: np.ndarray) -> np.ndarray:
    """(24,24,3) uint8 -> (24*SCALE, 24*SCALE, 3) nearest-neighbour."""
    return np.repeat(np.repeat(punk_uint8, SCALE, axis=0), SCALE, axis=1)


def _panel(frame_punks: np.ndarray) -> Image.Image:
    """Tile GRID punks (already uint8 24x24x3) into one padded panel image."""
    rows, cols = GRID
    cell = 24 * SCALE
    w = cols * cell + (cols + 1) * PAD
    h = rows * cell + (rows + 1) * PAD
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for idx in range(min(len(frame_punks), rows * cols)):
        r, c = divmod(idx, cols)
        y = PAD + r * (cell + PAD)
        x = PAD + c * (cell + PAD)
        canvas[y : y + cell, x : x + cell] = _upscale(frame_punks[idx])
    return Image.fromarray(canvas, "RGB")


def _sample(run_dir: Path, n: int, seed: int) -> np.ndarray:
    model = load_vae_npz(run_dir / "model.npz")
    rng = np.random.default_rng(seed)
    return images_to_uint8(sample_from_prior(model, n, rng))


def build_gif(panels: list[tuple[str, Path]], out_path: Path, seed: int = 2024,
              label_h: int = 46, gap: int = 26) -> None:
    """panels: list of (label, run_dir). One synchronised GIF, one column per panel."""
    rows, cols = GRID
    per_frame = rows * cols
    n_total = per_frame * N_FRAMES
    # sample once per panel; vary seed per panel so columns are not identical draws
    punks = [_sample(rd, n_total, seed + i * 100) for i, (_, rd) in enumerate(panels)]

    font = _font(34)
    frames: list[Image.Image] = []
    for f in range(N_FRAMES):
        panel_imgs = []
        for p in range(len(panels)):
            chunk = punks[p][f * per_frame : (f + 1) * per_frame]
            panel_imgs.append(_panel(chunk))
        pw, ph = panel_imgs[0].size
        W = len(panels) * pw + (len(panels) + 1) * gap
        H = label_h + ph + gap
        frame = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(frame)
        for p, (label, _) in enumerate(panels):
            x = gap + p * (pw + gap)
            frame.paste(panel_imgs[p], (x, label_h))
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((x + (pw - tw) / 2, 6), label, fill=(20, 20, 20), font=font)
        frames.append(frame)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=FRAME_MS, loop=0, disposal=2)
    print(f"[gif] {out_path}  ({len(frames)} frames, {len(panels)} panel(s))")


def main() -> None:
    # 1) Beta refinement: 0 (fantasmas) -> 2 (óptimo) -> 8 (colapso), ceteris paribus (latente 4)
    build_gif(
        [("β = 0", OUTPUT / "final_beta0"),
         ("β = 2", OUTPUT / "final_beta2"),
         ("β = 8", OUTPUT / "final_beta8")],
        GIF_DIR / "beta_refinement.gif",
    )
    # 2) Latent capacity: 4 -> 8 -> 16 (β = 2)
    build_gif(
        [("latente 4", OUTPUT / "final_beta2"),
         ("latente 8", OUTPUT / "final_lat8_b2"),
         ("latente 16", OUTPUT / "final_lat16_b2")],
        GIF_DIR / "latent_capacity.gif",
    )
    # 3) Generation showcase: final model (latente 8 · β2), z ~ N(0, I), τ = 1
    build_gif(
        [("z ~ N(0, I)", OUTPUT / "final_lat8_b2")],
        GIF_DIR / "generation.gif",
        label_h=46,
    )


if __name__ == "__main__":
    main()
