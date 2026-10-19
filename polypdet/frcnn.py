"""Faster R-CNN (torchvision, MobileNetV3-Large FPN backbone) fine-tuned for one class.

Images are the resized uint8 arrays shared with YOLO; augmentation runs on the GPU (flips, zoom-in crops and
colour jitter) and can be switched off for the ablation. Training uses SGD with warm-up and a cosine schedule,
fp16 autocast, and keeps the epoch with the best validation mAP@[.5:.95], the same selection rule Ultralytics uses.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


@dataclass(frozen=True)
class FRCNNConfig:
    """Training settings of one Faster R-CNN run."""

    img_size: int = 416
    epochs: int = 8
    patience: int = 4
    batch_size: int = 8
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_iters: int = 100
    augment: bool = True
    seed: int = 0
    pretrained: bool = True


def build_model(cfg: FRCNNConfig) -> torch.nn.Module:
    """COCO-pretrained Faster R-CNN with a new 2-class (background, polyp) box predictor."""
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if cfg.pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, weights_backbone=None, min_size=cfg.img_size, max_size=cfg.img_size,
        box_score_thresh=0.001, box_detections_per_img=100)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


def to_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 (h, w, 3) array -> float (3, h, w) tensor in [0, 1] on ``device``."""
    return torch.from_numpy(np.ascontiguousarray(img)).to(device).permute(2, 0, 1).float().div_(255)


def augment(img: torch.Tensor, boxes: torch.Tensor, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Random horizontal / vertical flip, zoom-in crop and colour jitter of one image and its xyxy boxes."""
    _, h, w = img.shape
    r = torch.rand(8, generator=g).tolist()
    boxes = boxes.clone()
    if r[0] < 0.5:
        img = img.flip(-1)
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    if r[1] < 0.5:
        img = img.flip(-2)
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
    if r[2] < 0.5:  # crop 60-100% of each side; keep boxes that retain >= 40% of their area
        cw, ch = int(w * (0.6 + 0.4 * r[3])), int(h * (0.6 + 0.4 * r[4]))
        x0, y0 = int((w - cw) * r[5]), int((h - ch) * r[6])
        nb = boxes - torch.tensor([x0, y0, x0, y0], dtype=boxes.dtype, device=boxes.device)
        nb[:, [0, 2]] = nb[:, [0, 2]].clamp(0, cw)
        nb[:, [1, 3]] = nb[:, [1, 3]].clamp(0, ch)
        kept = ((nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1])) >= 0.4 * (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        if kept.any():
            img, boxes = img[:, y0:y0 + ch, x0:x0 + cw], nb[kept]
    j = (0.8 + 0.4 * torch.rand(3, generator=g)).tolist()  # brightness, contrast, saturation factors
    img = img * j[0]
    img = (img - img.mean()) * j[1] + img.mean()
    grey = img.mean(0, keepdim=True)
    img = (grey + (img - grey) * j[2]).clamp_(0, 1)
    return img, boxes


def _lr_at(it: int, total: int, cfg: FRCNNConfig) -> float:
    """Linear warm-up, then cosine decay to 1% of the base learning rate."""
    if it < cfg.warmup_iters:
        return cfg.lr * (0.001 + 0.999 * it / cfg.warmup_iters)
    frac = (it - cfg.warmup_iters) / max(1, total - cfg.warmup_iters)
    return cfg.lr * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * frac)))


@torch.no_grad()
def predict(model: torch.nn.Module, images: list[np.ndarray], device: torch.device, batch_size: int = 8,
            amp: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """Boxes (xyxy, in the input image's pixels) and scores for every image (fp32 unless ``amp``)."""
    model.eval()
    out = []
    for i in range(0, len(images), batch_size):
        batch = [to_tensor(im, device) for im in images[i:i + batch_size]]
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            res = model(batch)
        out += [(r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy()) for r in res]
    return out


def train(images: list[np.ndarray], targets: list[np.ndarray], val_images: list[np.ndarray],
          val_score: Callable[[list[tuple[np.ndarray, np.ndarray]]], dict[str, float]], cfg: FRCNNConfig,
          device: torch.device, log: Callable[[str], None] = print) -> tuple[torch.nn.Module, pd.DataFrame]:
    """Fine-tune on ``images`` / ``targets`` (xyxy boxes per image) and keep the best epoch on validation.

    ``val_score`` turns validation predictions into metrics and must return the key ``AP`` (mAP@[.5:.95]).
    Returns the model with the best weights and one history row per epoch.
    """
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    model = build_model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    n = len(images)
    per_epoch = int(np.ceil(n / cfg.batch_size))
    total = per_epoch * cfg.epochs
    it, best, best_state, bad, history = 0, -1.0, None, 0, []
    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        model.train()
        order = torch.randperm(n, generator=g).tolist()
        losses: dict[str, float] = {}
        for b in range(per_epoch):
            idx = order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            imgs, tgts = [], []
            for i in idx:
                im = to_tensor(images[i], device)
                bx = torch.as_tensor(targets[i], dtype=torch.float32, device=device).reshape(-1, 4)
                if cfg.augment:
                    im, bx = augment(im, bx, g)
                imgs.append(im)
                tgts.append({"boxes": bx, "labels": torch.ones(len(bx), dtype=torch.int64, device=device)})
            for group in opt.param_groups:
                group["lr"] = _lr_at(it, total, cfg)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss_dict = model(imgs, tgts)
                loss = sum(loss_dict.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            it += 1
            for k, v in loss_dict.items():
                losses[k] = losses.get(k, 0.0) + v.item() / per_epoch
        metrics = val_score(predict(model, val_images, device))
        history.append({"epoch": epoch + 1, "lr": _lr_at(it, total, cfg), "train_loss": sum(losses.values()),
                        **{f"loss_{k.replace('loss_', '')}": v for k, v in losses.items()},
                        **{f"val_{k}": v for k, v in metrics.items()}, "seconds": time.perf_counter() - t0})
        log(f"  frcnn epoch {epoch + 1}: loss {history[-1]['train_loss']:.3f} val mAP {metrics['AP']:.3f} "
            f"AP50 {metrics.get('AP50', float('nan')):.3f} ({history[-1]['seconds']:.1f}s)")
        if metrics["AP"] > best:
            best, best_state, bad = metrics["AP"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)
