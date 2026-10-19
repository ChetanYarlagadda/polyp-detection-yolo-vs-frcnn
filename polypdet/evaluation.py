"""One COCO-style evaluation for both detectors.

Predictions of every model are stored in the same format (per image: xyxy boxes in original pixels and scores) and
scored with pycocotools. Boxes are first mapped into a 1000 x 1000 frame per image, which leaves IoU unchanged but
makes box areas relative to the image, so the size bins (small / medium / large polyps) use the share of the image
a polyp covers rather than absolute pixels (the images range from 332 x 487 to 1920 x 1072).

``match_images`` + ``average_precision`` re-implement the pycocotools matching and 101-point AP for one class; they
reproduce pycocotools exactly (checked in the pipeline and the tests) and are fast enough for the bootstrap.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from polypdet.boxes import SIZE_BINS, area, iou_matrix, normalize

NORM = 1000.0
AREA_RANGES = {"all": (0.0, 1e10), **{k: (lo * NORM**2, hi * NORM**2 if k != "large" else 1e10)
                                      for k, (lo, hi) in SIZE_BINS.items()}}
COCO_IOUS = np.linspace(0.5, 0.95, 10)
JHA_IOUS = np.linspace(0.25, 0.75, 11)  # AP@[.25:.05:.75], the summary AP of Jha et al. (2021)
ALL_IOUS = np.round(np.arange(0.25, 0.951, 0.05), 2)
REC_THRS = np.linspace(0.0, 1.0, 101)
MAX_DET = 100

Preds = dict[str, tuple[np.ndarray, np.ndarray]]  # image id -> (boxes (n, 4) xyxy, scores (n,))


def _normalized(gt: dict[str, np.ndarray], preds: Preds, sizes: dict[str, tuple[int, int]]):
    """Ground truth and predictions in the 1000 x 1000 frame, keyed by image id."""
    g = {i: normalize(gt[i], *sizes[i], NORM) for i in gt}
    p = {i: (normalize(preds[i][0], *sizes[i], NORM), np.asarray(preds[i][1], dtype=float)) for i in gt}
    return g, p


def coco_eval(gt: dict[str, np.ndarray], preds: Preds, sizes: dict[str, tuple[int, int]],
              iou_thrs: np.ndarray = COCO_IOUS) -> dict[str, np.ndarray]:
    """pycocotools COCOeval (bbox) over the images in ``gt``.

    Returns the accumulated precision array (iou, recall, area) and recall array (iou, area), with the area
    ranges in the order of ``AREA_RANGES``.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    g, p = _normalized(gt, preds, sizes)
    ids = sorted(g)
    images = [{"id": k + 1, "file_name": i, "width": NORM, "height": NORM} for k, i in enumerate(ids)]
    anns, dets = [], []
    for k, i in enumerate(ids):
        for b in g[i]:
            anns.append({"id": len(anns) + 1, "image_id": k + 1, "category_id": 1, "iscrowd": 0,
                         "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "area": float(area(b)[0])})
        for b, s in zip(*p[i]):
            dets.append({"image_id": k + 1, "category_id": 1, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                         "score": float(s)})
    T, A = len(iou_thrs), len(AREA_RANGES)
    if not dets:
        return {"precision": np.zeros((T, len(REC_THRS), A)), "recall": np.zeros((T, A))}
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "polyp"}]}
        cg.createIndex()
        cd = cg.loadRes(dets)
        E = COCOeval(cg, cd, "bbox")
        E.params.iouThrs = np.asarray(iou_thrs, dtype=float)
        E.params.areaRng = [list(v) for v in AREA_RANGES.values()]
        E.params.areaRngLbl = list(AREA_RANGES)
        E.params.maxDets = [1, 10, MAX_DET]
        E.evaluate()
        E.accumulate()
    return {"precision": E.eval["precision"][:, :, 0, :, -1], "recall": E.eval["recall"][:, 0, :, -1]}


def summarize(ev: dict[str, np.ndarray], iou_thrs: np.ndarray = COCO_IOUS) -> dict[str, float]:
    """AP over all IoU thresholds, AP at single thresholds and AR, per area range, from ``coco_eval`` output."""
    out = {}
    prec, rec = ev["precision"], ev["recall"]
    for a, name in enumerate(AREA_RANGES):
        suffix = "" if name == "all" else f"_{name}"
        p = prec[:, :, a]
        out[f"AP{suffix}"] = float(p[p > -1].mean()) if (p > -1).any() else np.nan
        r = rec[:, a]
        out[f"AR{suffix}"] = float(r[r > -1].mean()) if (r > -1).any() else np.nan
        for t, thr in enumerate(iou_thrs):
            pt = p[t]
            out[f"AP{round(thr * 100)}{suffix}"] = float(pt[pt > -1].mean()) if (pt > -1).any() else np.nan
    return out


@dataclass
class ImageMatches:
    """Per-image matching result for one area range: detections sorted by score (at most MAX_DET)."""

    scores: np.ndarray  # (d,)
    matched: np.ndarray  # (t, d) bool: detection matched a ground-truth box at IoU threshold t
    ignored: np.ndarray  # (t, d) bool: detection ignored (matched an out-of-range box, or unmatched and out of range)
    n_gt: int  # ground-truth boxes inside the area range
    iou: np.ndarray  # (t, d) IoU with the matched box (0 if unmatched)
    gt_index: np.ndarray  # (t, d) index of the matched box in the input order (-1 if unmatched)


def match_image(gt: np.ndarray, boxes: np.ndarray, scores: np.ndarray, iou_thrs: np.ndarray,
                area_rng: tuple[float, float] = (0.0, 1e10)) -> ImageMatches:
    """Greedy COCO matching of one image's detections (highest score first) to its ground-truth boxes."""
    gt = np.asarray(gt, dtype=float).reshape(-1, 4)
    g_area = area(gt)
    g_ign = (g_area < area_rng[0]) | (g_area > area_rng[1])
    gorder = np.argsort(g_ign, kind="mergesort")
    gt, g_ign = gt[gorder], g_ign[gorder]
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")[:MAX_DET]
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)[order]
    scores = np.asarray(scores, dtype=float)[order]
    ious = iou_matrix(boxes, gt)
    T, D, G = len(iou_thrs), len(boxes), len(gt)
    matched = np.zeros((T, D), bool)
    ignored = np.zeros((T, D), bool)
    miou = np.zeros((T, D))
    gidx = np.full((T, D), -1)
    for t, thr in enumerate(iou_thrs):
        taken = np.zeros(G, bool)
        for d in range(D):
            best, m = min(thr, 1 - 1e-10), -1
            for g in range(G):
                if taken[g]:
                    continue
                if m > -1 and not g_ign[m] and g_ign[g]:
                    break
                if ious[d, g] < best:
                    continue
                best, m = ious[d, g], g
            if m == -1:
                continue
            matched[t, d], ignored[t, d], taken[m], miou[t, d] = True, g_ign[m], True, ious[d, m]
            gidx[t, d] = gorder[m]
    d_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # signed, as pycocotools (w * h)
    out_of_range = (d_area < area_rng[0]) | (d_area > area_rng[1])
    ignored |= ~matched & out_of_range[None, :]
    return ImageMatches(scores, matched, ignored, int((~g_ign).sum()), miou, gidx)


def match_images(gt: dict[str, np.ndarray], preds: Preds, sizes: dict[str, tuple[int, int]],
                 iou_thrs: np.ndarray, area_name: str = "all") -> list[ImageMatches]:
    """``match_image`` for every image in ``gt`` (sorted by id), in the normalised frame."""
    g, p = _normalized(gt, preds, sizes)
    return [match_image(g[i], p[i][0], p[i][1], iou_thrs, AREA_RANGES[area_name]) for i in sorted(g)]


def average_precision(matches: list[ImageMatches], idx: np.ndarray | None = None) -> np.ndarray:
    """101-point interpolated AP per IoU threshold (pycocotools ``accumulate``) over images ``idx`` (repeats allowed)."""
    sel = matches if idx is None else [matches[i] for i in idx]
    T = sel[0].matched.shape[0] if sel else 0
    n_gt = sum(m.n_gt for m in sel)
    if n_gt == 0:
        return np.full(T, np.nan)
    scores = np.concatenate([m.scores for m in sel])
    order = np.argsort(-scores, kind="mergesort")
    tp = np.concatenate([m.matched for m in sel], axis=1)[:, order]
    ig = np.concatenate([m.ignored for m in sel], axis=1)[:, order]
    tps = np.cumsum(tp & ~ig, axis=1).astype(float)
    fps = np.cumsum(~tp & ~ig, axis=1).astype(float)
    ap = np.zeros(T)
    for t in range(T):
        if tps.shape[1] == 0:
            continue
        rc = tps[t] / n_gt
        pr = tps[t] / (tps[t] + fps[t] + np.spacing(1))
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        q = np.zeros(len(REC_THRS))
        inds = np.searchsorted(rc, REC_THRS, side="left")
        valid = inds < len(pr)
        q[valid] = pr[inds[valid]]
        ap[t] = q.mean()
    return ap


def pr_curve(matches: list[ImageMatches], t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall and score threshold along the ranked detections at IoU threshold index ``t``."""
    n_gt = sum(m.n_gt for m in matches)
    scores = np.concatenate([m.scores for m in matches])
    order = np.argsort(-scores, kind="mergesort")
    tp = np.concatenate([m.matched[t] & ~m.ignored[t] for m in matches])[order]
    fp = np.concatenate([~m.matched[t] & ~m.ignored[t] for m in matches])[order]
    tps, fps = np.cumsum(tp), np.cumsum(fp)
    return tps / np.maximum(tps + fps, 1), tps / max(n_gt, 1), scores[order]


def best_f1_threshold(matches: list[ImageMatches], t: int) -> float:
    """Score threshold that maximises F1 at IoU threshold index ``t`` (chosen on validation images)."""
    p, r, s = pr_curve(matches, t)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(s[int(np.argmax(f1))])


def operating_point(matches: list[ImageMatches], threshold: float, t: int) -> dict[str, float]:
    """Counts and rates of the detections with score >= ``threshold`` at IoU threshold index ``t``."""
    tp = fp = 0
    ious = []
    hit_images = 0
    for m in matches:
        keep = (m.scores >= threshold) & ~m.ignored[t]
        tp_i = int((m.matched[t] & keep).sum())
        tp += tp_i
        fp += int((~m.matched[t] & keep).sum())
        ious.extend(m.iou[t][m.matched[t] & keep].tolist())
        hit_images += tp_i > 0
    n_gt = sum(m.n_gt for m in matches)
    fn = n_gt - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(n_gt, 1)
    return {"threshold": threshold, "images": len(matches), "polyps": n_gt, "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "fp_per_image": fp / max(len(matches), 1), "images_with_hit": hit_images / max(len(matches), 1),
            "mean_iou_tp": float(np.mean(ious)) if ious else np.nan}


def bootstrap_indices(n: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """(n_boot, n) image indices resampled with replacement."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ap(matches: list[ImageMatches], boot: np.ndarray) -> np.ndarray:
    """Mean AP over the IoU thresholds for every bootstrap resample, shape (n_boot,)."""
    return np.array([np.nanmean(average_precision(matches, idx)) for idx in boot])


def paired_p(diffs: np.ndarray) -> float:
    """Two-sided p-value of a paired bootstrap difference (normal approximation with the bootstrap SE)."""
    d = diffs[np.isfinite(diffs)]
    se = d.std(ddof=1)
    if se == 0:
        return 1.0
    return float(2 * norm.sf(abs(d.mean()) / se))


def holm(p: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()
