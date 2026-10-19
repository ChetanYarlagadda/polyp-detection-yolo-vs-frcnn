"""Ultralytics YOLO11n training and inference (AGPL-3.0 library).

Ultralytics keeps its settings, downloaded weights and run folders under ``data/cache``; usage analytics and every
logging integration are switched off. Training uses the library's own recipe (AdamW chosen by ``optimizer=auto``,
mosaic and HSV / flip / scale / translate augmentation, best epoch by validation mAP@[.5:.95]); the ablation turns
every augmentation off.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

AUG_OFF = {"mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
           "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0, "perspective": 0.0, "flipud": 0.0,
           "fliplr": 0.0, "close_mosaic": 0}
# Ultralytics defaults plus vertical flips (endoscopy frames have no up or down); mosaic off for the last 5 epochs.
AUG_ON = {"flipud": 0.5, "close_mosaic": 5}
INTEGRATIONS = ("clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard", "wandb")


@dataclass(frozen=True)
class YOLOConfig:
    """Training settings of one YOLO run."""

    weights: str = "yolo11n.pt"
    img_size: int = 416
    epochs: int = 15
    patience: int = 6
    batch_size: int = 16
    augment: bool = True
    seed: int = 0


def setup(cache: Path):
    """Import Ultralytics with its settings, weights and runs kept under ``cache`` and analytics disabled."""
    (cache / "ultralytics").mkdir(parents=True, exist_ok=True)  # must exist, or Ultralytics falls back to /tmp
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache / "ultralytics"))
    import ultralytics.data.base
    import ultralytics.data.dataset
    import ultralytics.data.loaders
    import ultralytics.utils as uu
    import ultralytics.utils.torch_utils
    from ultralytics import YOLO, settings

    for mod in (uu, ultralytics.utils.torch_utils, ultralytics.data.base, ultralytics.data.dataset,
                ultralytics.data.loaders):
        mod.NUM_THREADS = 2  # low-impact mode: image caching pools and CPU inference use at most 2 threads

    settings.update({"sync": False, "weights_dir": str(cache / "weights"), "runs_dir": str(cache / "yolo_runs"),
                     "datasets_dir": str(cache / "datasets"), **{k: False for k in INTEGRATIONS if k in settings}})
    uu.WEIGHTS_DIR = Path(settings["weights_dir"])  # the AMP check downloads its probe model here
    return YOLO


def write_data_yaml(dataset_dir: Path) -> Path:
    """Ultralytics data file for the prepared images / labels folders."""
    path = dataset_dir / "polyp.yaml"
    path.write_text(f"path: {dataset_dir.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                    "names:\n  0: polyp\n")
    return path


def train(data_yaml: Path, cfg: YOLOConfig, run_dir: Path, cache: Path) -> tuple[Path, pd.DataFrame, float]:
    """Train one model; return the best weights, the per-epoch history and the training seconds."""
    YOLO = setup(cache)
    (cache / "weights").mkdir(parents=True, exist_ok=True)
    model = YOLO(str(cache / "weights" / cfg.weights))
    aug = AUG_ON if cfg.augment else AUG_OFF
    t0 = time.perf_counter()
    model.train(data=str(data_yaml), imgsz=cfg.img_size, epochs=cfg.epochs, patience=cfg.patience,
                batch=cfg.batch_size, workers=0, cache="ram", device=0, seed=cfg.seed, deterministic=True,
                project=str(run_dir.parent), name=run_dir.name, exist_ok=True, plots=False, verbose=False,
                amp=True, **aug)
    seconds = time.perf_counter() - t0
    torch.use_deterministic_algorithms(False)  # Ultralytics switches these on process-wide
    torch.backends.cudnn.deterministic = False
    hist = pd.read_csv(run_dir / "results.csv")
    hist.columns = [c.strip() for c in hist.columns]
    return run_dir / "weights" / "best.pt", hist, seconds


def load(weights: Path, cache: Path):
    """A trained YOLO model."""
    return setup(cache)(str(weights))


def predict(model, images: list[np.ndarray], img_size: int, device: int | str = 0, batch_size: int = 16,
            half: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """Boxes (xyxy, input pixels) and scores for RGB uint8 images; conf 0.001, NMS IoU 0.7, at most 100 boxes."""
    out = []
    for i in range(0, len(images), batch_size):
        bgr = [np.ascontiguousarray(im[..., ::-1]) for im in images[i:i + batch_size]]
        for r in model.predict(bgr, imgsz=img_size, conf=0.001, iou=0.7, max_det=100, device=device, half=half,
                               verbose=False):
            out.append((r.boxes.xyxy.cpu().numpy().astype(float), r.boxes.conf.cpu().numpy().astype(float)))
    return out


def config_dict(cfg: YOLOConfig) -> dict:
    """Settings plus the augmentation overrides passed to Ultralytics."""
    return {**asdict(cfg), "aug_overrides": AUG_ON if cfg.augment else AUG_OFF}
