import numpy as np
import pytest

from polypdet import evaluation as ev
from polypdet.boxes import iou_matrix, normalize, size_bin


def synthetic(n_images: int = 30, seed: int = 0):
    """Ground truth with 1-2 boxes per image and noisy detections plus random false positives."""
    rng = np.random.default_rng(seed)
    gt, preds, sizes = {}, {}, {}
    for k in range(n_images):
        i = f"img{k:03d}"
        w, h = int(rng.integers(300, 900)), int(rng.integers(300, 700))
        sizes[i] = (w, h)
        boxes = []
        for _ in range(int(rng.integers(1, 3))):
            bw, bh = rng.uniform(0.05, 0.6) * w, rng.uniform(0.05, 0.6) * h
            x0, y0 = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
            boxes.append([x0, y0, x0 + bw, y0 + bh])
        g = np.array(boxes)
        gt[i] = g
        jitter = g + rng.normal(0, 0.06, g.shape) * np.repeat(g[:, 2:] - g[:, :2], 2, axis=1)
        extra = []
        for _ in range(int(rng.integers(0, 4))):
            x0, y0 = rng.uniform(0, w * 0.8), rng.uniform(0, h * 0.8)
            extra.append([x0, y0, x0 + rng.uniform(10, w * 0.2), y0 + rng.uniform(10, h * 0.2)])
        b = np.vstack([jitter] + ([np.array(extra)] if extra else []))
        b[:, [0, 2]] = np.sort(b[:, [0, 2]], axis=1)
        b[:, [1, 3]] = np.sort(b[:, [1, 3]], axis=1)
        preds[i] = (b, rng.uniform(0, 1, len(b)))
    return gt, preds, sizes


def test_iou_is_unchanged_by_the_normalised_frame():
    a = np.array([[10, 20, 110, 70], [0, 0, 50, 50]], dtype=float)
    b = np.array([[30, 30, 130, 90]], dtype=float)
    assert iou_matrix(a, a).diagonal() == pytest.approx([1.0, 1.0])
    assert iou_matrix(a[:1], b)[0, 0] == pytest.approx((80 * 40) / (100 * 50 + 100 * 60 - 80 * 40))
    assert np.allclose(iou_matrix(a, b), iou_matrix(normalize(a, 640, 480), normalize(b, 640, 480)))


@pytest.mark.parametrize("area", list(ev.AREA_RANGES))
def test_fast_ap_matches_pycocotools(area):
    gt, preds, sizes = synthetic()
    e = ev.coco_eval(gt, preds, sizes)
    a_i = list(ev.AREA_RANGES).index(area)
    ref = e["precision"][:, :, a_i]
    ref = np.array([r[r > -1].mean() if (r > -1).any() else np.nan for r in ref])
    fast = ev.average_precision(ev.match_images(gt, preds, sizes, ev.COCO_IOUS, area))
    assert np.allclose(fast, ref, equal_nan=True, atol=1e-12)


def test_perfect_and_empty_predictions():
    gt, _, sizes = synthetic(10, seed=1)
    perfect = {i: (g, np.linspace(1, 0.5, len(g))) for i, g in gt.items()}
    s = ev.summarize(ev.coco_eval(gt, perfect, sizes))
    assert s["AP"] == pytest.approx(1.0) and s["AP50"] == pytest.approx(1.0)
    empty = {i: (np.zeros((0, 4)), np.zeros(0)) for i in gt}
    assert ev.summarize(ev.coco_eval(gt, empty, sizes))["AP"] == 0.0


def test_duplicate_detection_counts_as_false_positive():
    g = np.array([[0, 0, 100, 100]], dtype=float)
    dets = np.array([[0, 0, 100, 100], [2, 2, 100, 100]], dtype=float)
    m = ev.match_image(g, dets, np.array([0.9, 0.8]), np.array([0.5]))
    assert m.matched[0].tolist() == [True, False] and m.gt_index[0].tolist() == [0, -1]
    op = ev.operating_point([m], threshold=0.5, t=0)
    assert (op["tp"], op["fp"], op["fn"]) == (1, 1, 0)
    assert ev.operating_point([m], threshold=0.85, t=0)["precision"] == 1.0


def test_bootstrap_identity_resample_reproduces_point_estimate():
    gt, preds, sizes = synthetic(20, seed=2)
    m = ev.match_images(gt, preds, sizes, ev.COCO_IOUS)
    point = np.nanmean(ev.average_precision(m))
    idx = np.arange(len(m))[None, :]
    assert ev.bootstrap_ap(m, idx)[0] == pytest.approx(point)


def test_holm_and_size_bins():
    assert ev.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert [size_bin(x) for x in (0.01, 0.0999, 0.10, 0.2, 0.25, 0.99, 1.0)] == [
        "small", "small", "medium", "medium", "large", "large", "large"]
