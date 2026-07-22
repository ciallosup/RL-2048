from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class WilsonInterval:
    rate: float
    lower: float
    upper: float
    n: int
    successes: int


def wilson_interval(successes: int, n: int, z: float = 1.96) -> WilsonInterval:
    """Wilson score interval for Bernoulli proportion (roadmap section 6.1)."""
    if n == 0:
        return WilsonInterval(rate=0.0, lower=0.0, upper=0.0, n=0, successes=0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return WilsonInterval(
        rate=p,
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
        n=n,
        successes=successes,
    )


def distribution_summary(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "iqr": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def tile_reach_probs(max_tiles: Iterable[int], k_range: range = range(8, 14)) -> dict[str, float]:
    tiles = np.asarray(list(max_tiles), dtype=np.int64)
    n = tiles.size
    if n == 0:
        return {f"P(>={2**k})": 0.0 for k in k_range}
    return {f"P(>={2**k})": float(np.mean(tiles >= 2**k)) for k in k_range}


def tile_reach_intervals(max_tiles: Iterable[int], k_range: range = range(8, 14)) -> dict[str, WilsonInterval]:
    tiles = list(max_tiles)
    n = len(tiles)
    return {
        f"P(>={2**k})": wilson_interval(sum(t >= 2**k for t in tiles), n)
        for k in k_range
    }
