"""Axis-aligned box helpers (x0, y0, x1, y1 in pixels, x1/y1 exclusive)."""

from __future__ import annotations

import numpy as np

# Polyp size bins by box area as a share of the image area (roughly tertiles of the training boxes).
SIZE_BINS: dict[str, tuple[float, float]] = {"small": (0.0, 0.10), "medium": (0.10, 0.25), "large": (0.25, 1.0)}


def area(boxes: np.ndarray) -> np.ndarray:
    """Areas of (n, 4) xyxy boxes."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (n, 4) and (m, 4) xyxy boxes, shape (n, m)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def relative_area(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    """Box area as a share of the image area."""
    return area(boxes) / float(width * height)


def size_bin(rel_area: float) -> str:
    """Name of the size bin that a relative box area falls into."""
    for name, (lo, hi) in SIZE_BINS.items():
        if lo <= rel_area < hi or (name == "large" and rel_area >= hi):
            return name
    raise ValueError(f"relative area out of range: {rel_area}")


def normalize(boxes: np.ndarray, width: float, height: float, scale: float = 1000.0) -> np.ndarray:
    """Map boxes into a ``scale`` x ``scale`` frame; IoU is unchanged because x and y are scaled separately."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return boxes * np.array([scale / width, scale / height, scale / width, scale / height])
