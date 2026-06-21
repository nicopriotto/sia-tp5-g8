from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
PREPROCESS_SCRIPT = REPO_ROOT / "data" / "data_punks_bundle" / "preprocess.py"


@dataclass
class VAEDatasetSplit:
    images: np.ndarray
    flat: np.ndarray
    selected_indices: np.ndarray
    train_size: int
    validation_size: int
    dataset_seed: int
    split_seed: int
    tensor_path: str
    tensor_shape: tuple[int, ...]
    limit: int | None
    validation_fraction: float

    @property
    def n_samples(self) -> int:
        return int(self.flat.shape[0])

    @property
    def train_indices(self) -> np.ndarray:
        return self.selected_indices[: self.train_size]

    @property
    def validation_indices(self) -> np.ndarray:
        return self.selected_indices[self.train_size :]

    @property
    def train_flat(self) -> np.ndarray:
        return self.flat[: self.train_size]

    @property
    def validation_flat(self) -> np.ndarray:
        return self.flat[self.train_size :]

    @property
    def train_images(self) -> np.ndarray:
        return self.images[: self.train_size]

    @property
    def validation_images(self) -> np.ndarray:
        return self.images[self.train_size :]

    def manifest(self) -> dict:
        return {
            "tensor_path": self.tensor_path,
            "tensor_shape": list(self.tensor_shape),
            "limit": self.limit,
            "dataset_seed": self.dataset_seed,
            "split_seed": self.split_seed,
            "validation_fraction": self.validation_fraction,
            "n_samples": self.n_samples,
            "n_train_samples": int(self.train_size),
            "n_validation_samples": int(self.validation_size),
            "selected_indices": self.selected_indices.astype(int).tolist(),
            "train_indices": self.train_indices.astype(int).tolist(),
            "validation_indices": self.validation_indices.astype(int).tolist(),
        }


def normalize_dataset_config(dataset_cfg: dict) -> dict:
    normalized = dict(dataset_cfg)
    if "validation_fraction" not in normalized:
        normalized["validation_fraction"] = 0.0
    if "split_seed" not in normalized or normalized["split_seed"] is None:
        normalized["split_seed"] = normalized["seed"]

    normalized["seed"] = int(normalized["seed"])
    normalized["split_seed"] = int(normalized["split_seed"])
    normalized["validation_fraction"] = float(normalized["validation_fraction"])

    if not 0.0 <= normalized["validation_fraction"] < 1.0:
        raise ValueError("dataset.validation_fraction must be in [0.0, 1.0)")

    limit = normalized.get("limit")
    if limit is None:
        normalized["limit"] = None
    else:
        normalized["limit"] = int(limit)
        if normalized["limit"] <= 0:
            raise ValueError("dataset.limit must be positive when provided")

    return normalized


def load_dataset_tensor(dataset_cfg: dict) -> np.ndarray:
    tensor_path = Path(dataset_cfg["tensor_path"])
    if not tensor_path.exists():
        raise FileNotFoundError(
            f"Tensor dataset not found at {tensor_path}. "
            f"Generate it with `python3 {PREPROCESS_SCRIPT}`."
        )

    raw = np.load(tensor_path)
    if raw.ndim < 2:
        raise ValueError(f"Expected tensor with at least 2 dimensions, got shape {raw.shape}")
    return raw.astype(np.float32) / 255.0


def _validation_size(n_samples: int, validation_fraction: float) -> int:
    if validation_fraction == 0.0:
        return 0
    if n_samples < 2:
        raise ValueError("dataset.validation_fraction requires at least 2 selected samples")

    n_validation = int(round(n_samples * validation_fraction))
    n_validation = max(1, n_validation)
    return min(n_validation, n_samples - 1)


def prepare_dataset_split(config: dict) -> VAEDatasetSplit:
    dataset_cfg = normalize_dataset_config(config["dataset"])
    raw = load_dataset_tensor(dataset_cfg)

    source_indices = np.arange(raw.shape[0], dtype=int)
    limit = dataset_cfg["limit"]
    if limit is not None:
        source_indices = source_indices[:limit]
        raw = raw[:limit]

    split_seed = int(dataset_cfg["split_seed"])
    rng = np.random.default_rng(split_seed)
    permutation = rng.permutation(raw.shape[0])

    images = raw[permutation]
    selected_indices = source_indices[permutation]
    flat = images.reshape(images.shape[0], -1)

    validation_size = _validation_size(flat.shape[0], float(dataset_cfg["validation_fraction"]))
    train_size = int(flat.shape[0] - validation_size)

    return VAEDatasetSplit(
        images=images,
        flat=flat,
        selected_indices=selected_indices.astype(int),
        train_size=train_size,
        validation_size=validation_size,
        dataset_seed=int(dataset_cfg["seed"]),
        split_seed=split_seed,
        tensor_path=str(Path(dataset_cfg["tensor_path"]).resolve()),
        tensor_shape=tuple(int(dim) for dim in raw.shape),
        limit=limit,
        validation_fraction=float(dataset_cfg["validation_fraction"]),
    )


def write_dataset_manifest(dataset_split: VAEDatasetSplit, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(dataset_split.manifest(), indent=2), encoding="utf-8")


def load_run_dataset_split(run_dir: str | Path) -> tuple[dict, VAEDatasetSplit]:
    run_path = Path(run_dir)
    resolved_config = json.loads((run_path / "resolved_config.json").read_text(encoding="utf-8"))
    resolved_config["dataset"] = normalize_dataset_config(resolved_config["dataset"])

    manifest_path = run_path / "dataset_manifest.json"
    if not manifest_path.exists():
        return resolved_config, prepare_dataset_split(resolved_config)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = load_dataset_tensor(resolved_config["dataset"])
    selected_indices = np.asarray(manifest["selected_indices"], dtype=int)
    images = raw[selected_indices]
    flat = images.reshape(images.shape[0], -1)

    return resolved_config, VAEDatasetSplit(
        images=images,
        flat=flat,
        selected_indices=selected_indices,
        train_size=int(manifest["n_train_samples"]),
        validation_size=int(manifest["n_validation_samples"]),
        dataset_seed=int(manifest.get("dataset_seed", resolved_config["dataset"]["seed"])),
        split_seed=int(manifest.get("split_seed", resolved_config["dataset"]["split_seed"])),
        tensor_path=str(Path(resolved_config["dataset"]["tensor_path"]).resolve()),
        tensor_shape=tuple(int(dim) for dim in manifest.get("tensor_shape", raw.shape)),
        limit=manifest.get("limit"),
        validation_fraction=float(
            manifest.get("validation_fraction", resolved_config["dataset"]["validation_fraction"])
        ),
    )
