"""All report figures, drawn from the result tables in reports/results/."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

from polypdet import plotting
from polypdet.boxes import SIZE_BINS
from polypdet.plotting import DETECTOR_COLOR, GT_COLOR, INK, INK_2, MUTED, SPLIT_COLOR, model_style

MAIN = ["YOLO11n", "Faster R-CNN"]
ALL4 = ["YOLO11n", "YOLO11n (no aug.)", "Faster R-CNN", "Faster R-CNN (no aug.)"]
SIZE_LABEL = {"small": "small\n(< 10% of frame)", "medium": "medium\n(10-25%)", "large": "large\n(> 25%)"}


def _suptitle(fig: plt.Figure, text: str, gap_in: float = 0.3) -> None:
    """Left-aligned takeaway title placed ``gap_in`` inches above the top of the axes."""
    fig.suptitle(text, x=0.01, y=fig.subplotpars.top + gap_in / fig.get_figheight(), ha="left", va="bottom",
                 fontsize=12, fontweight="bold", color=INK)


def _decode(jpeg: np.ndarray) -> np.ndarray:
    """RGB array from JPEG bytes."""
    return np.asarray(Image.open(io.BytesIO(jpeg.tobytes())).convert("RGB"))


def _draw_image(ax: plt.Axes, img: np.ndarray, rows: pd.DataFrame, det_color: str | None = None) -> None:
    """Show an image with its ground-truth boxes (solid) and detections (dashed, with scores)."""
    h, w = img.shape[:2]
    ax.imshow(img)
    for r in rows.itertuples():
        x0, y0, x1, y1 = r.x0 * w, r.y0 * h, r.x1 * w, r.y1 * h
        if r.kind == "ground truth":  # white with a dark outline: visible on tissue and on the green probe marker
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=GT_COLOR, lw=2,
                                   path_effects=[pe.Stroke(linewidth=3.6, foreground=INK), pe.Normal()]))
        else:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=det_color or INK, lw=1.8, ls="--"))
            ax.text(x0 + 2, max(y0 - 3, 8), f"{r.score:.2f}", color="white", fontsize=7, va="bottom",
                    bbox={"fc": det_color or INK, "ec": "none", "pad": 0.8})
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_data(res: Path, path: Path) -> Path:
    """Example training images with boxes, box-size distribution per split, and the mask cross-check."""
    ex = np.load(res / "image_examples.npz")
    rows = pd.read_csv(res / "image_examples.csv")
    sizes = pd.read_csv(res / "box_sizes.csv")
    agree = pd.read_csv(res / "box_mask_agreement.csv").set_index("statistic").value
    split = pd.read_csv(res / "split_summary.csv")
    data_ids = list(dict.fromkeys(rows[rows.role == "data"].image_id))[:4]
    fig = plt.figure(figsize=(12, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.2], hspace=0.38, wspace=0.08)
    for j, i in enumerate(data_ids):
        ax = fig.add_subplot(gs[0, j])
        _draw_image(ax, _decode(ex[i]), rows[(rows.image_id == i) & (rows.role == "data")])
        if j == 0:
            ax.set_title("Training frames with their reference polyp boxes", loc="left")
    bottom = gs[1, :].subgridspec(1, 2, wspace=0.25)
    ax = fig.add_subplot(bottom[0, 0])
    s = sizes[~sizes.sliver]
    bins = np.logspace(np.log10(0.003), 0, 25)
    for sp in ("train", "val", "test"):
        v = s[s.split == sp].rel_area
        ax.hist(v, bins=bins, histtype="step", lw=2, color=SPLIT_COLOR[sp], weights=np.full(len(v), 1 / len(v)),
                label=f"{sp} ({len(v)} polyps)")
    for name, (lo, hi) in SIZE_BINS.items():
        if lo > 0:
            ax.axvline(lo, color=MUTED, lw=1, ls=":")
        ax.text(np.sqrt(max(lo, 0.01) * hi), 1.01, name, ha="center", va="bottom", fontsize=8.5, color=INK_2,
                transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel("Polyp box area (share of the frame, log scale)")
    ax.set_ylabel("Share of the split's polyps per bin")
    ax.set_title("Polyp size per split and the three size bins", loc="left", pad=16)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(loc="upper left", fontsize=8.5)
    ax = fig.add_subplot(bottom[0, 1])
    iou = sizes[~sizes.sliver].best_component_iou
    ax.hist(iou, bins=np.linspace(0, 1, 51), color=SPLIT_COLOR["train"], alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("IoU of each JSON box with its best mask-component box")
    ax.set_ylabel("Boxes (log scale)")
    ax.set_title("JSON boxes vs segmentation masks", loc="left", pad=16)
    test_sliver = int(split[split.split == "test"].sliver_boxes_dropped.iloc[0])
    ax.text(0.03, 0.96, f"{agree['non-sliver boxes with a mask component at IoU >= 0.9']:.1%} of boxes match a mask "
            f"component\nat IoU >= 0.9 (median {agree['median best IoU with a mask component (non-sliver)']:.3f}); "
            f"{int(agree['non-sliver boxes with best IoU < 0.9'])} boxes are\nlooser than their mask. "
            f"{int(agree['sliver boxes (shorter side < 5 px)'])} sliver boxes (< 5 px\nwide, {test_sliver} in test) "
            "are dropped.", transform=ax.transAxes, va="top", fontsize=8.5, color=INK_2)
    n = int(agree["JSON boxes"])
    _suptitle(fig, f"Kvasir-SEG: 1,000 frames and {n:,} JSON boxes, which match the segmentation masks for "
                   f"{agree['non-sliver boxes with a mask component at IoU >= 0.9']:.0%} of polyps", 0.3)
    plotting.save(fig, path)
    return path


def fig_pr(res: Path, path: Path) -> Path:
    """Precision-recall curves on the test set at IoU 0.5 and 0.75 with the validation-chosen operating points."""
    pr = pd.read_csv(res / "pr_curves.csv")
    op = pd.read_csv(res / "operating_point.csv")
    m = pd.read_csv(res / "coco_metrics.csv")
    m = m[(m.split == "test") & (m.ground_truth == "cleaned")].set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, thr in zip(axes, (0.5, 0.75)):
        for model in MAIN:
            d = pr[(pr.model == model) & (pr.iou == thr)]
            d = d[d.precision >= 0]
            ap = m.loc[model, "AP50" if thr == 0.5 else "AP75"]
            ax.plot(d.recall, d.precision, **model_style(model), label=f"{model}  (AP{round(thr * 100)} {ap:.3f})")
            if thr == 0.5:
                o = op[(op.model == model) & (op.split == "test")].iloc[0]
                ax.plot(o.recall, o.precision, "o", color=DETECTOR_COLOR[model], ms=8, mec="white", mew=1.5)
                ax.annotate(f"score >= {o.threshold:.2f} (set on validation):\n{int(o.tp)}/{int(o.polyps)} polyps found, "
                            f"{int(o.fp)} false positives", (o.recall, o.precision),
                            xytext=(0.22, 0.50 if model == MAIN[0] else 0.68), textcoords="data", ha="left",
                            fontsize=8.5, color=DETECTOR_COLOR[model],
                            arrowprops={"arrowstyle": "->", "color": DETECTOR_COLOR[model], "lw": 1})
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("Recall (share of test polyps found)")
        ax.set_title(f"IoU >= {thr}", loc="left")
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("Precision")
    _suptitle(fig, f"On 120 test frames the two detectors are statistically tied: AP50 {m.loc['YOLO11n', 'AP50']:.3f} vs "
                   f"{m.loc['Faster R-CNN', 'AP50']:.3f}, mAP@[.5:.95] {m.loc['YOLO11n', 'AP']:.3f} vs "
                   f"{m.loc['Faster R-CNN', 'AP']:.3f} (YOLO11n vs Faster R-CNN)")
    plotting.save(fig, path)
    return path


def fig_iou(res: Path, path: Path) -> Path:
    """AP against the IoU threshold for all four models, with the published Jha et al. (2021) points."""
    a = pd.read_csv(res / "ap_by_iou.csv")
    pub = pd.read_csv(res / "published_comparison.csv")
    pub = pub[pub.source.str.startswith("Jha")]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for model in ALL4:
        d = a[a.model == model]
        ax.plot(d.iou, d.AP, **model_style(model), marker="o", ms=3.5, label=model)
    markers = {"Faster R-CNN": "s", "YOLOv4": "^", "RetinaNet": "D"}
    for r in pub[pub.model.isin(markers)].itertuples():
        ax.plot([0.25, 0.5, 0.75], [r.AP25, r.AP50, r.AP75], ls="none", marker=markers[r.model], color=MUTED, ms=7,
                mfc="none", mew=1.5, label=f"{r.model} {r.backbone}, Jha et al. (2021)")
    ax.set_xlabel("IoU threshold for a correct box")
    ax.set_ylabel("Average precision (test set)")
    ax.set_xlim(0.22, 0.98)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    y9 = a[(a.iou == 0.9)].set_index("model").AP
    ax.set_title("Hollow markers: published results on the same official test split (512 px input)", loc="left",
                 fontsize=9.5, fontweight="normal", color=INK_2)
    y5 = a[(a.iou == 0.5)].set_index("model").AP
    p50 = pub[pub.model.isin(markers)].AP50
    _suptitle(fig, f"Both detectors match published results on this split (AP50 {y5['YOLO11n']:.2f} / "
                   f"{y5['Faster R-CNN']:.2f} vs {p50.min():.2f}-{p50.max():.2f}); at IoU 0.9 YOLO11n keeps "
                   f"{y9['YOLO11n']:.2f} AP, Faster R-CNN {y9['Faster R-CNN']:.2f}")
    plotting.save(fig, path)
    return path


def fig_size(res: Path, path: Path) -> Path:
    """mAP and recall at the operating point by polyp size bin."""
    s = pd.read_csv(res / "size_breakdown.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(SIZE_BINS))
    wdt = 0.36
    for k, model in enumerate(MAIN):
        d = s[s.model == model].set_index("size").loc[list(SIZE_BINS)]
        off = (k - 0.5) * wdt
        axes[0].bar(x + off, d.AP, wdt, color=DETECTOR_COLOR[model], label=model)
        axes[0].errorbar(x + off, d.AP, yerr=[d.AP - d.AP_ci_low, d.AP_ci_high - d.AP], fmt="none", ecolor=INK,
                         elinewidth=1, capsize=3)
        axes[1].bar(x + off, d.recall, wdt, color=DETECTOR_COLOR[model], label=model)
        for xi, (r, tp, n) in enumerate(zip(d.recall, d.tp, d.polyps)):
            axes[1].text(xi + off, r + 0.01, f"{int(tp)}/{int(n)}", ha="center", va="bottom", fontsize=8, color=INK_2)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([SIZE_LABEL[k] for k in SIZE_BINS], fontsize=9)
        ax.set_xlabel("Polyp size (box area as share of the frame)")
    axes[0].set_ylabel("mAP@[.5:.95] (95% bootstrap CI)")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Box quality by polyp size", loc="left")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[1].set_ylabel("Recall at the operating point (IoU 0.5)")
    axes[1].set_ylim(0, 1.12)
    axes[1].set_title("Polyps found at the validation-chosen threshold", loc="left")
    d = s.pivot_table(index="size", columns="model", values="AP")
    r = s.set_index(["model", "size"])
    _suptitle(fig, f"Large polyps, not small ones, score lowest (mAP {d.loc['large', 'YOLO11n']:.2f} for both; CIs overlap), "
                   f"and YOLO11n misses {int(r.loc[('YOLO11n', 'large'), 'fn'])} of "
                   f"{int(r.loc[('YOLO11n', 'large'), 'polyps'])} of them at its operating point")
    plotting.save(fig, path)
    return path


def fig_augmentation(res: Path, path: Path) -> Path:
    """Validation mAP per epoch and test AP with and without augmentation."""
    h = pd.read_csv(res / "training_history.csv")
    c = pd.read_csv(res / "metric_cis.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.1, 1]})
    for model in ALL4:
        d = h[h.model == model]
        axes[0].plot(d.epoch, d.val_AP, **model_style(model), marker="o", ms=3, label=model)
        b = d.loc[d.val_AP.idxmax()]
        axes[0].plot(b.epoch, b.val_AP, "*", color=model_style(model)["color"], ms=11, mec="white")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation mAP@[.5:.95]")
    axes[0].set_title("Validation curves (star: kept epoch; YOLO scored by Ultralytics)", loc="left")
    axes[0].legend(loc="lower right", fontsize=8.5)
    metrics = ["AP", "AP50", "AP75"]
    x = np.arange(len(metrics))
    wdt = 0.2
    for k, model in enumerate(ALL4):
        d = c[(c.model == model) & c.metric.isin(metrics)].set_index("metric").loc[metrics]
        st = model_style(model)
        off = (k - 1.5) * wdt
        axes[1].bar(x + off, d.value, wdt, color=st["color"], alpha=1.0 if st["ls"] == "-" else 0.45,
                    hatch=None if st["ls"] == "-" else "//", ec="white", label=model)
        axes[1].errorbar(x + off, d.value, yerr=[d.value - d.ci_low, d.ci_high - d.value], fmt="none", ecolor=INK,
                         elinewidth=1, capsize=2.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["mAP@[.5:.95]", "AP50", "AP75"])
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Test AP (95% bootstrap CI)")
    axes[1].set_title("Test set, with and without augmentation", loc="left")
    axes[1].legend(loc="upper center", fontsize=8, ncol=2)
    v = c[c.metric == "AP"].set_index("model").value
    _suptitle(fig, f"Augmentation is worth {v['YOLO11n'] - v['YOLO11n (no aug.)']:+.3f} test mAP to YOLO11n but only "
                   f"{v['Faster R-CNN'] - v['Faster R-CNN (no aug.)']:+.3f} (not significant) to Faster R-CNN; all four "
                   "were still improving")
    plotting.save(fig, path)
    return path


def fig_speed(res: Path, path: Path) -> Path:
    """Batch-1 latency per detector, device and precision."""
    sp = pd.read_csv(res / "speed.csv")
    sp["setting"] = sp.device.str.split(" ").str[0] + " " + sp.precision
    settings = list(dict.fromkeys(sp.setting))
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    y = np.arange(len(settings))
    hgt = 0.36
    for k, model in enumerate(MAIN):
        d = sp[sp.model == model].set_index("setting").reindex(settings)
        off = (0.5 - k) * hgt
        ax.barh(y + off, d.median_ms, hgt, color=DETECTOR_COLOR[model], label=model)
        for yi, (ms, mean) in enumerate(zip(d.median_ms, d.mean_ms)):
            if ms == ms:
                ax.text(ms, yi + off, f"  median {ms:.1f} ms (mean {mean:.1f})", va="center", fontsize=8.5,
                        color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(settings)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(right=sp.median_ms.max() * 6)
    ax.set_xlabel("Median latency per image, batch 1 (ms, log scale; RTX 4050 Laptop GPU / 2 CPU threads)")
    ax.legend(loc="lower right", fontsize=9)
    g = sp[sp.setting == "GPU fp32"].set_index("model").median_ms
    cpu = sp[sp.setting.str.startswith("CPU")].set_index("model").median_ms
    _suptitle(fig, f"YOLO11n is {g['Faster R-CNN'] / g['YOLO11n']:.1f}x faster on the GPU ({g['YOLO11n']:.1f} vs "
                   f"{g['Faster R-CNN']:.1f} ms) and {cpu['Faster R-CNN'] / cpu['YOLO11n']:.1f}x faster on 2 CPU "
                   "threads; fp16 does not help at batch 1")
    plotting.save(fig, path)
    return path


def fig_errors(res: Path, path: Path) -> Path:
    """Missed polyps and false positives of both detectors at their operating points."""
    ex = np.load(res / "image_examples.npz")
    rows = pd.read_csv(res / "image_examples.csv")
    g = rows[rows.role != "data"]
    order = [(m, k) for m in ("yolo11n-aug", "frcnn-aug") for k in ("missed polyp", "false positive")]
    tiles = {key: list(dict.fromkeys(g[g.role == f"{key[0]} | {key[1]}"].image_id)) for key in order}
    ncol = max([len(v) for v in tiles.values()] + [1])
    fig, axes = plt.subplots(len(order), ncol, figsize=(2.6 * ncol, 2.45 * len(order)), squeeze=False)
    label = {"yolo11n-aug": "YOLO11n", "frcnn-aug": "Faster R-CNN"}
    for r_i, key in enumerate(order):
        model = label[key[0]]
        for c_i in range(ncol):
            ax = axes[r_i, c_i]
            if c_i >= len(tiles[key]):
                ax.axis("off")
                continue
            i = tiles[key][c_i]
            sub = g[(g.role == f"{key[0]} | {key[1]}") & (g.image_id == i)]
            _draw_image(ax, _decode(ex[i]), sub, DETECTOR_COLOR[model])
            if c_i == 0:
                ax.set_ylabel(f"{model}\n{key[1]}s", fontsize=9.5, color=INK, fontweight="bold")
    handles = [Line2D([], [], color=GT_COLOR, lw=2, label="reference box (Kvasir-SEG)",
                      path_effects=[pe.Stroke(linewidth=3.6, foreground=INK), pe.Normal()]),
               Line2D([], [], color=DETECTOR_COLOR["YOLO11n"], lw=1.8, ls="--", label="YOLO11n detection (score)"),
               Line2D([], [], color=DETECTOR_COLOR["Faster R-CNN"], lw=1.8, ls="--",
                      label="Faster R-CNN detection (score)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(hspace=0.08, wspace=0.04, bottom=0.06)
    err = pd.read_csv(res / "error_cases.csv")
    fp = err[(err.kind == "false positive") & err.key.isin(["yolo11n-aug", "frcnn-aug"])]
    on_polyp = fp.groupby("key").best_iou.apply(lambda x: int((x >= 0.1).sum()))
    n_fp = fp.groupby("key").size()
    _suptitle(fig, f"Most false positives are boxes on a real polyp with the wrong extent ({on_polyp['frcnn-aug']} of "
                   f"{n_fp['frcnn-aug']} for Faster R-CNN, {on_polyp['yolo11n-aug']} of {n_fp['yolo11n-aug']} for "
                   "YOLO11n)\nRows: the smallest missed polyps and the highest-scoring false positives at each "
                   "detector's operating point", 0.25)
    plotting.save(fig, path)
    return path


def make_all(results: Path, figures: Path) -> list[Path]:
    """Draw every figure from the CSVs in ``results`` into ``figures``."""
    plotting.apply_style()
    return [
        fig_data(results, figures / "00_data.png"),
        fig_pr(results, figures / "01_pr_curves.png"),
        fig_iou(results, figures / "02_ap_by_iou.png"),
        fig_size(results, figures / "03_size.png"),
        fig_augmentation(results, figures / "04_augmentation.png"),
        fig_speed(results, figures / "05_speed.png"),
        fig_errors(results, figures / "06_errors.png"),
    ]
