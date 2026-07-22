from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

# Fixed bases for reproducible evaluation seed pools (roadmap section 6.2).
DEV_SEED_BASE = 20260722
VAL_SEED_BASE = 20260723
TEST_SEED_BASE = 20260724

DEFAULT_DEV_EPISODES = 200
DEFAULT_VAL_EPISODES = 1000


def generate_seeds(base: int, count: int) -> list[int]:
    rng = np.random.default_rng(base)
    return [int(x) for x in rng.integers(0, 2**31 - 1, size=count)]


def dev_seeds(count: int = DEFAULT_DEV_EPISODES) -> list[int]:
    return generate_seeds(DEV_SEED_BASE, count)


def val_seeds(count: int = DEFAULT_VAL_EPISODES) -> list[int]:
    return generate_seeds(VAL_SEED_BASE, count)


def test_seeds(count: int) -> list[int]:
    return generate_seeds(TEST_SEED_BASE, count)


def save_seeds(path: Path, seeds: Iterable[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(seeds), indent=2), encoding="utf-8")


def load_seeds(path: Path) -> list[int]:
    return json.loads(path.read_text(encoding="utf-8"))
