"""Batch-1 latency, timed the same way for both detectors.

The clock starts with a resized uint8 image in host memory and stops when its boxes and scores are back in host
memory as NumPy arrays, so host-to-device copy, preprocessing, the network, NMS and post-processing are all inside.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch


def latency_ms(run: Callable[[np.ndarray], object], images: list[np.ndarray], warmup: int = 10,
               cuda: bool = True) -> np.ndarray:
    """Milliseconds per image for ``run`` over ``images`` after ``warmup`` untimed calls."""
    for im in images[:warmup]:
        run(im)
    out = []
    for im in images:
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run(im)
        if cuda:
            torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1000)
    return np.array(out)


def summary(ms: np.ndarray) -> dict[str, float]:
    """Median, 90th percentile and mean latency, and throughput from the mean."""
    return {"median_ms": float(np.median(ms)), "p90_ms": float(np.percentile(ms, 90)), "mean_ms": float(ms.mean()),
            "fps": float(1000 / ms.mean()), "images": int(len(ms))}


def n_params(model: torch.nn.Module) -> int:
    """Number of parameters."""
    return int(sum(p.numel() for p in model.parameters()))
