from __future__ import annotations

import numpy as np


MASTER_SEED = 2025
N_RUNS_FORMAL = 10
N_RUNS_QUICK = 5


def generate_formal_seeds(master_seed: int = MASTER_SEED) -> list[int]:
    rng = np.random.default_rng(master_seed)
    return [int(value) for value in rng.integers(0, 100_000, size=N_RUNS_FORMAL)]


def get_mode_seeds(mode: str, master_seed: int = MASTER_SEED) -> list[int]:
    formal = generate_formal_seeds(master_seed=master_seed)
    if mode == "formal":
        return formal
    if mode == "quick":
        return formal[:N_RUNS_QUICK]
    raise ValueError("mode must be 'formal' or 'quick'")
