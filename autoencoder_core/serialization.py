from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .model import Autoencoder


def save_model_npz(model: Autoencoder, path: str | Path) -> None:
    output_path = Path(path)
    metadata = model.architecture_config()
    payload = {"metadata": np.array(json.dumps(metadata))}
    payload.update(model.parameters())
    np.savez(output_path, **payload)


def load_model_npz(path: str | Path) -> Autoencoder:
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        model = Autoencoder(
            input_dim=int(metadata["input_dim"]),
            latent_dim=int(metadata["latent_dim"]),
            encoder_hidden_layers=list(metadata["encoder_hidden_layers"]),
            decoder_hidden_layers=list(metadata["decoder_hidden_layers"]),
            hidden_activation=str(metadata["hidden_activation"]),
            output_activation=str(metadata["output_activation"]),
            weight_init=str(metadata["weight_init"]),
            dropout=float(metadata["dropout"]),
        )
        params = {}
        for key in data.files:
            if key == "metadata":
                continue
            params[key] = data[key]
        model.load_parameters(params)
        return model
