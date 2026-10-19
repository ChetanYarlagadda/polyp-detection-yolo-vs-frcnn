import json

import numpy as np
from PIL import Image

from polypdet import data


def make_dataset(tmp_path):
    """Two tiny images with masks and a box JSON in the Kvasir-SEG layout."""
    root = tmp_path / "Kvasir-SEG"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    meta = {}
    for k, (w, h, rects) in enumerate([(80, 60, [(10, 10, 40, 30)]), (100, 50, [(5, 5, 25, 45), (60, 10, 90, 40)])]):
        i = f"im{k}"
        Image.fromarray(np.full((h, w, 3), 120, np.uint8)).save(root / "images" / f"{i}.jpg")
        mask = np.zeros((h, w), np.uint8)
        for x0, y0, x1, y1 in rects:
            mask[y0:y1, x0:x1] = 255
        Image.fromarray(mask).save(root / "masks" / f"{i}.jpg", format="PNG")  # lossless
        # the JSON boxes start one pixel before the mask, as in Kvasir-SEG
        meta[i] = {"height": h, "width": w, "bbox": [{"label": "polyp", "xmin": x0 - 1, "ymin": y0 - 1, "xmax": x1,
                                                     "ymax": y1} for x0, y0, x1, y1 in rects]}
    meta["im1"]["bbox"].append({"label": "polyp", "xmin": 97, "ymin": 0, "xmax": 99, "ymax": 30})  # a sliver
    (root / "kavsir_bboxes.json").write_text(json.dumps(meta))
    return root


def test_boxes_slivers_and_mask_crosscheck(tmp_path):
    root = make_dataset(tmp_path)
    images, boxes = data.load_boxes(root)
    assert len(images) == 2 and len(boxes) == 4
    assert boxes.sliver.tolist() == [False, False, False, True]
    per_box, per_image = data.crosscheck_masks(root, images, boxes)
    real = per_box[~per_box.sliver]
    assert (real.best_component_iou > 0.9).all()
    assert (real.dx0 == -1).all() and (real.dx1 == 0).all()
    assert per_image.mask_pixels_in_boxes.min() > 0.99


def test_splits_are_seeded_and_disjoint():
    train = [f"t{i:03d}" for i in range(50)]
    test = [f"x{i:03d}" for i in range(10)]
    a = data.assign_splits(train, test, n_val=8, seed=0)
    b = data.assign_splits(list(reversed(train)), test, n_val=8, seed=0)
    assert a == b  # order of the list file does not matter
    assert sum(v == "val" for v in a.values()) == 8 and all(a[i] == "test" for i in test)
    assert data.assign_splits(train, test, n_val=8, seed=1) != a


def test_prepare_images_writes_yolo_labels(tmp_path):
    root = make_dataset(tmp_path)
    images, boxes = data.load_boxes(root)
    prep = data.prepare_images(root, images, boxes, {"im0": "train", "im1": "test"}, tmp_path / "ds", max_side=40)
    r = prep.set_index("image_id").loc["im1"]
    assert (r.rw, r.rh) == (40, 20)
    lines = (tmp_path / "ds" / "labels" / "test" / "im1.txt").read_text().split("\n")
    lines = [ln for ln in lines if ln]
    assert len(lines) == 2  # the sliver is left out
    cls, cx, cy, w, h = map(float, lines[0].split())
    assert cls == 0 and abs(cx - 14.5 / 100) < 1e-6 and abs(w - 21 / 100) < 1e-6
