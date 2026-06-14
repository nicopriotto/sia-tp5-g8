from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PATTERN_RE = re.compile(
    r"\{\s*(?P<rows>[^}]+?)\s*\}\s*,?\s*//\s*0x(?P<ascii>[0-9a-fA-F]{2})\s*,\s*(?P<label>.+?)\s*$"
)


@dataclass
class FontDataset:
    X: np.ndarray
    labels: list[str]
    ascii_codes: np.ndarray
    hex_rows: np.ndarray
    grids: np.ndarray
    indices: np.ndarray
    subset_name: str
    subset_mode: str
    source_path: str
    seed: int

    def subset_manifest(self) -> dict:
        return {
            "subset_name": self.subset_name,
            "subset_mode": self.subset_mode,
            "source_path": self.source_path,
            "seed": self.seed,
            "indices": self.indices.astype(int).tolist(),
            "labels": list(self.labels),
            "ascii_codes": self.ascii_codes.astype(int).tolist(),
        }


def hex_row_to_bits(hex_value: int) -> np.ndarray:
    if not 0 <= hex_value <= 0x1F:
        raise ValueError(f"Row value out of range 0x00..0x1f: {hex_value:#04x}")
    return np.array([(hex_value >> bit) & 1 for bit in range(4, -1, -1)], dtype=int)


def bits_to_hex_row(bits: np.ndarray) -> int:
    bits = np.asarray(bits, dtype=int)
    if bits.shape != (5,):
        raise ValueError("Expected a row of shape (5,)")
    if not np.isin(bits, [0, 1]).all():
        raise ValueError("Bits must be binary")
    value = 0
    for idx, bit in enumerate(bits):
        value |= int(bit) << (4 - idx)
    return value


def hex_rows_to_grid(hex_rows: np.ndarray) -> np.ndarray:
    rows = [hex_row_to_bits(int(value)) for value in np.asarray(hex_rows)]
    if len(rows) != 7:
        raise ValueError("Each glyph must have exactly 7 rows")
    return np.vstack(rows)


def grid_to_hex_rows(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    if grid.shape != (7, 5):
        raise ValueError("Expected a grid of shape (7, 5)")
    return np.array([bits_to_hex_row(row) for row in grid], dtype=int)


def flat_to_grid(flat_35: np.ndarray) -> np.ndarray:
    flat = np.asarray(flat_35, dtype=int)
    if flat.shape != (35,):
        raise ValueError("Expected a flat vector of shape (35,)")
    return flat.reshape(7, 5)


def parse_font_h(source_path: str | Path) -> list[dict]:
    source = Path(source_path)
    lines = source.read_text(encoding="utf-8").splitlines()
    patterns: list[dict] = []

    for line in lines:
        match = PATTERN_RE.search(line)
        if match is None:
            continue
        row_tokens = [token.strip() for token in match.group("rows").split(",")]
        if len(row_tokens) != 7:
            raise ValueError("Each glyph must contain exactly 7 hexadecimal rows")
        hex_rows = np.array([int(token, 16) for token in row_tokens], dtype=int)
        grid = hex_rows_to_grid(hex_rows)
        flat = grid.reshape(-1)
        ascii_code = int(match.group("ascii"), 16)
        label = match.group("label").strip()
        patterns.append(
            {
                "label": label,
                "ascii_code": ascii_code,
                "hex_rows": hex_rows,
                "bits_5x7": grid,
                "flat_35": flat,
            }
        )

    if len(patterns) != 32:
        raise ValueError(f"Expected exactly 32 patterns in {source}, found {len(patterns)}")

    stacked = np.stack([pattern["flat_35"] for pattern in patterns], axis=0)
    if stacked.shape != (32, 35):
        raise ValueError(f"Expected all_32 matrix shape (32, 35), got {stacked.shape}")
    if not np.isin(stacked, [0, 1]).all():
        raise ValueError("Parsed dataset contains non-binary values")

    reconstructed = np.stack(
        [grid_to_hex_rows(pattern["bits_5x7"]) for pattern in patterns],
        axis=0,
    )
    original = np.stack([pattern["hex_rows"] for pattern in patterns], axis=0)
    if not np.array_equal(reconstructed, original):
        raise ValueError("Round-trip validation failed for font.h parsing")

    return patterns


def _resolve_subset_indices(subset: dict, dataset_size: int, seed: int) -> np.ndarray:
    mode = subset.get("mode", "all_32")
    indices = np.asarray(subset.get("indices", []), dtype=int)

    if mode == "all_32":
        return np.arange(dataset_size, dtype=int)

    if mode == "fixed_indices":
        if indices.size == 0:
            raise ValueError("fixed_indices subset requires a non-empty indices list")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("fixed_indices subset cannot contain repeated indices")
        if np.any(indices < 0) or np.any(indices >= dataset_size):
            raise ValueError("fixed_indices subset contains out-of-range indices")
        if not np.all(indices[:-1] < indices[1:]):
            raise ValueError("fixed_indices subset must be strictly ordered in dataset order")
        return indices

    if mode == "sample_without_replacement":
        sample_size = subset.get("sample_size")
        if sample_size is None:
            raise ValueError("sample_without_replacement requires subset.sample_size")
        sample_size = int(sample_size)
        if sample_size <= 0 or sample_size > dataset_size:
            raise ValueError("subset.sample_size must be in 1..dataset_size")
        rng = np.random.default_rng(seed)
        sampled = np.sort(rng.choice(dataset_size, size=sample_size, replace=False).astype(int))
        return sampled

    raise ValueError(f"Unsupported subset mode: {mode}")


def load_font_dataset(
    source_path: str | Path,
    subset: dict | None = None,
    seed: int = 42,
) -> FontDataset:
    subset = subset or {"mode": "all_32", "name": "all_32", "indices": []}
    patterns = parse_font_h(source_path)
    selected_indices = _resolve_subset_indices(subset, len(patterns), seed)
    selected = [patterns[int(index)] for index in selected_indices]

    X = np.stack([item["flat_35"] for item in selected], axis=0).astype(float)
    labels = [item["label"] for item in selected]
    ascii_codes = np.array([item["ascii_code"] for item in selected], dtype=int)
    hex_rows = np.stack([item["hex_rows"] for item in selected], axis=0)
    grids = np.stack([item["bits_5x7"] for item in selected], axis=0)

    if not np.isin(X, [0.0, 1.0]).all():
        raise ValueError("Dataset values must be exact 0.0/1.0")

    return FontDataset(
        X=X,
        labels=labels,
        ascii_codes=ascii_codes,
        hex_rows=hex_rows,
        grids=grids,
        indices=selected_indices.astype(int),
        subset_name=str(subset.get("name", subset.get("mode", "all_32"))),
        subset_mode=str(subset.get("mode", "all_32")),
        source_path=str(Path(source_path).resolve()),
        seed=int(seed),
    )


def write_subset_manifest(dataset: FontDataset, output_path: str | Path) -> None:
    path = Path(output_path)
    path.write_text(json.dumps(dataset.subset_manifest(), indent=2), encoding="utf-8")
