"""Download, verify and prepare Kvasir-SEG for detection.

Kvasir-SEG (Jha et al., 2020) has 1,000 colonoscopy images with a polyp segmentation mask each and a JSON file of
polyp bounding boxes. The archive and the official 880 / 120 train / test lists (Jha et al., 2021) are pinned by
SHA-256. A validation set is carved out of the official training list with a fixed seed, so the 120 official test
images are never used for any training or tuning decision.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy import ndimage

from polypdet.boxes import iou_matrix, relative_area, size_bin

ZIP_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
ZIP_SHA256 = "03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7"
ZIP_BYTES = 46_227_172
SPLIT_COMMIT = "dd6d4215a7c9a6f4c4b5ec7fcd91e2a965a7d557"  # github.com/DebeshJha/Kvasir-SEG, Data-split/
SPLIT_URL = "https://raw.githubusercontent.com/DebeshJha/Kvasir-SEG/{commit}/Data-split/{name}.txt"
SPLIT_SHA256 = {
    "train": "49992c7c6c80abf02032050d4ae754236da3ba1041f609659ad0043ff6d1795c",
    "val": "d9b48bd3d980324edb39df97764fdbb5489b2583f088fb49c659b76f6d20876b",
}
# Published facts used for the data check (dataset page and Jha et al., 2020 / 2021).
PUBLISHED = {"images": 1000, "masks": 1000, "official_train": 880, "official_test": 120}
SLIVER_MAX_SIDE = 5  # boxes whose shorter side is below this many pixels are annotation slivers, not polyps
MASK_MIN_AREA = 50  # mask components smaller than this (pixels) are treated as JPEG specks


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, path: Path, expected: str) -> Path:
    """Download ``url`` to ``path`` once and check its SHA-256."""
    if path.exists() and sha256(path) == expected:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    got = sha256(tmp)
    if got != expected:
        tmp.unlink()
        raise ValueError(f"SHA-256 mismatch for {path.name}: {got} != {expected}")
    tmp.replace(path)
    return path


def download(raw_dir: Path) -> Path:
    """Download and extract Kvasir-SEG plus the official split lists; return the dataset folder."""
    zpath = _fetch(ZIP_URL, raw_dir / "kvasir-seg.zip", ZIP_SHA256)
    root = raw_dir / "Kvasir-SEG"
    if not (root / "kavsir_bboxes.json").exists():
        with zipfile.ZipFile(zpath) as z:
            for member in z.namelist():  # extract only the expected layout
                if member.startswith("Kvasir-SEG/") and ".." not in member:
                    z.extract(member, raw_dir)
    for name, digest in SPLIT_SHA256.items():
        _fetch(SPLIT_URL.format(commit=SPLIT_COMMIT, name=name), raw_dir / "split" / f"{name}.txt", digest)
    return root


def read_split(raw_dir: Path) -> tuple[list[str], list[str]]:
    """Official training (880) and test (120, called 'val' upstream) image ids."""
    read = lambda n: [s for s in (raw_dir / "split" / f"{n}.txt").read_text().split() if s]  # noqa: E731
    return read("train"), read("val")


def assign_splits(official_train: list[str], official_test: list[str], n_val: int, seed: int) -> dict[str, str]:
    """Map image id -> 'train' / 'val' / 'test'; ``n_val`` validation images are drawn from the official training list."""
    rng = np.random.default_rng(seed)
    train = sorted(official_train)
    val = set(np.array(train)[rng.permutation(len(train))[:n_val]].tolist())
    split = {i: ("val" if i in val else "train") for i in train}
    split.update({i: "test" for i in official_test})
    return split


def load_boxes(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Images (id, width, height) and boxes (one row per JSON box) with size, sliver flag and size bin."""
    meta = json.loads((root / "kavsir_bboxes.json").read_text())
    images, boxes = [], []
    for image_id in sorted(meta):
        v = meta[image_id]
        w, h = int(v["width"]), int(v["height"])
        images.append({"image_id": image_id, "width": w, "height": h, "n_boxes_json": len(v["bbox"])})
        for j, b in enumerate(v["bbox"]):
            xyxy = np.array([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], dtype=float)
            rel = float(relative_area(xyxy, w, h)[0])
            short = float(min(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]))
            boxes.append({"image_id": image_id, "box": j, "label": b["label"], "x0": xyxy[0], "y0": xyxy[1],
                          "x1": xyxy[2], "y1": xyxy[3], "short_side": short, "rel_area": rel,
                          "sliver": short < SLIVER_MAX_SIDE, "size": size_bin(rel)})
    return pd.DataFrame(images), pd.DataFrame(boxes)


def mask_components(mask: np.ndarray, min_area: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Bounding boxes (xyxy, exclusive) and pixel areas of the 8-connected foreground components of a binary mask."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return np.zeros((0, 4)), np.zeros(0)
    areas = ndimage.sum(mask, lab, range(1, n + 1))
    boxes = np.array([[s[1].start, s[0].start, s[1].stop, s[0].stop] for s in ndimage.find_objects(lab)], dtype=float)
    keep = areas >= min_area
    return boxes[keep], areas[keep]


def crosscheck_masks(root: Path, images: pd.DataFrame, boxes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every JSON box with the connected components of its segmentation mask.

    Returns one row per JSON box (best IoU with any mask component, offsets, mask fill) and one row per image
    (component counts, IoU of the overall extents, share of mask pixels inside the JSON boxes).
    """
    per_box, per_image = [], []
    for img in images.itertuples():
        mask = np.asarray(Image.open(root / "masks" / f"{img.image_id}.jpg").convert("L")) > 127
        size_ok = mask.shape == (img.height, img.width)
        comp_all, _ = mask_components(mask, 1)
        comp, _ = mask_components(mask, MASK_MIN_AREA)
        b = boxes[boxes.image_id == img.image_id]
        jb = b[["x0", "y0", "x1", "y1"]].to_numpy()
        ious = iou_matrix(jb, comp_all)
        inside = np.zeros_like(mask)
        for x0, y0, x1, y1 in jb.astype(int):
            inside[y0:y1, x0:x1] = True
        for k, row in enumerate(b.itertuples()):
            m = int(ious[k].argmax()) if ious.shape[1] else -1
            x0, y0, x1, y1 = (int(v) for v in jb[k])
            fill = float(mask[y0:y1, x0:x1].mean()) if (x1 > x0 and y1 > y0) else 0.0
            per_box.append({"image_id": img.image_id, "box": row.box, "sliver": row.sliver,
                            "best_component_iou": float(ious[k, m]) if m >= 0 else 0.0,
                            "dx0": float(jb[k, 0] - comp_all[m, 0]) if m >= 0 else np.nan,
                            "dy0": float(jb[k, 1] - comp_all[m, 1]) if m >= 0 else np.nan,
                            "dx1": float(jb[k, 2] - comp_all[m, 2]) if m >= 0 else np.nan,
                            "dy1": float(jb[k, 3] - comp_all[m, 3]) if m >= 0 else np.nan,
                            "mask_fill": fill})
        ys, xs = np.nonzero(mask)
        ext_mask = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]) if len(xs) else np.zeros(4)
        ext_json = np.array([jb[:, 0].min(), jb[:, 1].min(), jb[:, 2].max(), jb[:, 3].max()])
        per_image.append({"image_id": img.image_id, "mask_size_matches": size_ok, "n_json_boxes": len(jb),
                          "n_components": len(comp_all), "n_components_min_area": len(comp),
                          "extent_iou": float(iou_matrix(ext_json, ext_mask)[0, 0]),
                          "mask_pixels_in_boxes": float(inside[mask].mean()) if mask.any() else np.nan})
    return pd.DataFrame(per_box), pd.DataFrame(per_image)


def prepare_images(root: Path, images: pd.DataFrame, boxes: pd.DataFrame, split: dict[str, str], out_dir: Path,
                   max_side: int) -> pd.DataFrame:
    """Resize every image so its longer side is ``max_side`` and write a YOLO-format data set.

    Images go to ``out_dir/images/<split>/<id>.jpg`` (JPEG quality 95) and labels (class cx cy w h, normalised) to
    ``out_dir/labels/<split>/<id>.txt``. Sliver boxes are left out of the labels. Both detectors read these files.
    Returns one row per image with the resized size and file path.
    """
    rows = []
    for img in images.itertuples():
        s = split[img.image_id]
        scale = max_side / max(img.width, img.height)
        rw, rh = round(img.width * scale), round(img.height * scale)
        ip = out_dir / "images" / s / f"{img.image_id}.jpg"
        lp = out_dir / "labels" / s / f"{img.image_id}.txt"
        if not ip.exists():
            ip.parent.mkdir(parents=True, exist_ok=True)
            im = Image.open(root / "images" / f"{img.image_id}.jpg").convert("RGB")
            im.resize((rw, rh), Image.BILINEAR).save(ip, quality=95)
        b = boxes[(boxes.image_id == img.image_id) & ~boxes.sliver]
        lp.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"0 {(r.x0 + r.x1) / 2 / img.width:.6f} {(r.y0 + r.y1) / 2 / img.height:.6f} "
                 f"{(r.x1 - r.x0) / img.width:.6f} {(r.y1 - r.y0) / img.height:.6f}" for r in b.itertuples()]
        lp.write_text("\n".join(lines) + "\n")
        rows.append({"image_id": img.image_id, "split": s, "width": img.width, "height": img.height,
                     "rw": rw, "rh": rh, "path": str(ip)})
    return pd.DataFrame(rows)


def gt_boxes(boxes: pd.DataFrame, ids: list[str], keep_slivers: bool = False) -> dict[str, np.ndarray]:
    """Ground-truth boxes (xyxy, original pixels) per image id."""
    b = boxes if keep_slivers else boxes[~boxes.sliver]
    grouped = {k: g[["x0", "y0", "x1", "y1"]].to_numpy(dtype=float) for k, g in b.groupby("image_id")}
    return {i: grouped.get(i, np.zeros((0, 4))) for i in ids}


def load_resized(paths: list[str]) -> list[np.ndarray]:
    """Resized images as uint8 (h, w, 3) arrays."""
    return [np.array(Image.open(p).convert("RGB")) for p in paths]
