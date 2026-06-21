from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from autoencoder_vae.dataset import PREPROCESS_SCRIPT, prepare_dataset_split
from autoencoder_vae.main import run_vae
from experiments.shared.vae_utils import aggregate_variant_summary, select_best_variant_summary


def _write_tensor(path: Path, shape: tuple[int, ...], seed: int = 1234) -> Path:
    rng = np.random.default_rng(seed)
    tensor = rng.integers(0, 256, size=shape, dtype=np.uint8)
    np.save(path, tensor)
    return path


def _base_config(
    tensor_path: Path,
    validation_fraction: float,
    limit: int | None = None,
    input_dim: int | None = None,
) -> dict:
    if input_dim is None:
        sample = np.load(tensor_path)
        input_dim = int(np.prod(sample.shape[1:]))
    return {
        "dataset": {
            "tensor_path": str(tensor_path),
            "limit": limit,
            "seed": 17,
            "validation_fraction": validation_fraction,
            "split_seed": None,
        },
        "model": {
            "input_dim": input_dim,
            "latent_dim": 3,
            "encoder_hidden_layers": [10],
            "decoder_hidden_layers": [10],
            "hidden_activation": "tanh",
            "output_activation": "sigmoid",
            "weight_init": "xavier_uniform",
            "dropout": 0.0,
            "loss_function": "mean_squared_error",
            "optimizer": "adam",
            "beta": 1.0,
        },
        "training": {
            "learning_rate": 0.01,
            "batch_size": 4,
            "epochs_max": 6,
            "early_stopping_patience": 3,
            "l2_weight_decay": 0.0,
            "gradient_clip_norm": 5.0,
            "progress_every": 0,
        },
        "evaluation": {},
        "denoising": {
            "train_with_clean_target": True,
            "noise_level": 0.0,
            "noise_type": "bit_flip",
            "noise_on_train_only": True,
        },
    }


def _write_config(path: Path, config: dict) -> Path:
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


class VAEDatasetSplitTests(unittest.TestCase):
    def test_split_is_reproducible_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tensor_path = _write_tensor(root / "tiny.npy", shape=(20, 2, 2, 3))
            config = _base_config(tensor_path, validation_fraction=0.25, limit=12)

            split_a = prepare_dataset_split(config)
            split_b = prepare_dataset_split(config)

            self.assertListEqual(split_a.train_indices.tolist(), split_b.train_indices.tolist())
            self.assertListEqual(split_a.validation_indices.tolist(), split_b.validation_indices.tolist())
            self.assertEqual(split_a.n_samples, 12)
            self.assertEqual(split_a.train_size + split_a.validation_size, 12)
            self.assertTrue(set(split_a.train_indices.tolist()).isdisjoint(split_a.validation_indices.tolist()))

    def test_missing_tensor_mentions_preprocess_script(self) -> None:
        config = _base_config(Path("/tmp/does-not-exist.npy"), validation_fraction=0.1, limit=10, input_dim=12)
        with self.assertRaises(FileNotFoundError) as ctx:
            prepare_dataset_split(config)
        self.assertIn(str(PREPROCESS_SCRIPT), str(ctx.exception))


class VAERunTests(unittest.TestCase):
    def test_run_vae_with_validation_tracks_train_and_validation_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tensor_path = _write_tensor(root / "tiny.npy", shape=(24, 2, 2, 3))
            config = _base_config(tensor_path, validation_fraction=0.25, limit=16)
            config_path = _write_config(root / "config.json", config)
            output_dir = root / "run_with_val"

            result = run_vae(config=config, config_path=config_path, output_dir=output_dir)

            metrics = result["metrics"]
            self.assertEqual(metrics["selection_metric"], "validation_total_loss")
            self.assertEqual(metrics["n_train_samples"] + metrics["n_validation_samples"], 16)

            with (output_dir / "training_history.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("train_total_loss", rows[0])
            self.assertIn("validation_total_loss", rows[0])

            validation_totals = [float(row["validation_total_loss"]) for row in rows]
            best_epoch = int(np.argmin(validation_totals) + 1)
            self.assertEqual(metrics["best_epoch"], best_epoch)
            self.assertAlmostEqual(metrics["total_loss"], validation_totals[best_epoch - 1], places=6)

            manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["n_train_samples"], metrics["n_train_samples"])
            self.assertEqual(manifest["n_validation_samples"], metrics["n_validation_samples"])

    def test_run_vae_without_validation_keeps_train_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tensor_path = _write_tensor(root / "tiny.npy", shape=(18, 2, 2, 3))
            config = _base_config(tensor_path, validation_fraction=0.0, limit=10)
            config_path = _write_config(root / "config.json", config)
            output_dir = root / "run_no_val"

            result = run_vae(config=config, config_path=config_path, output_dir=output_dir)

            metrics = result["metrics"]
            self.assertEqual(metrics["selection_metric"], "total_loss")
            self.assertEqual(metrics["n_validation_samples"], 0)
            self.assertEqual(metrics["n_train_samples"], 10)

            with (output_dir / "training_history.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("train_total_loss", rows[0])
            self.assertNotIn("validation_total_loss", rows[0])

            total_losses = [float(row["total_loss"]) for row in rows]
            best_epoch = int(np.argmin(total_losses) + 1)
            self.assertEqual(metrics["best_epoch"], best_epoch)
            self.assertAlmostEqual(metrics["total_loss"], total_losses[best_epoch - 1], places=6)


class VAEExperimentSummaryTests(unittest.TestCase):
    def test_summary_uses_validation_ranking_criterion(self) -> None:
        metrics_a = {
            "latent_dim": 8,
            "beta": 1.0,
            "selection_metric": "validation_total_loss",
            "n_train_samples": 1800,
            "n_validation_samples": 200,
            "reconstruction_loss": 10.2,
            "kl_loss": 0.8,
            "total_loss": 11.0,
            "train_reconstruction_loss": 9.5,
            "train_kl_loss": 0.7,
            "train_total_loss": 10.2,
            "validation_reconstruction_loss": 10.2,
            "validation_kl_loss": 0.8,
            "validation_total_loss": 11.0,
        }
        metrics_b = {
            **metrics_a,
            "validation_total_loss": 10.5,
            "validation_reconstruction_loss": 9.9,
            "total_loss": 10.5,
            "reconstruction_loss": 9.9,
        }

        summary_a = aggregate_variant_summary("beta", "beta_1.0", "quick", 2025, [1, 2], [metrics_a, metrics_a])
        summary_b = aggregate_variant_summary("beta", "beta_2.0", "quick", 2025, [1, 2], [metrics_b, metrics_b])

        self.assertIn("validation_total_loss_mean", summary_a)
        best = select_best_variant_summary([summary_a, summary_b])
        self.assertEqual(best["variant"], "beta_2.0")


if __name__ == "__main__":
    unittest.main()
