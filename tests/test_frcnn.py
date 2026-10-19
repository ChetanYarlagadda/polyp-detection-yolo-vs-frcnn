import numpy as np
import torch

from polypdet import frcnn


def test_augment_keeps_boxes_inside_the_image_and_flips_consistently():
    g = torch.Generator().manual_seed(0)
    img = torch.rand(3, 60, 80)
    boxes = torch.tensor([[10.0, 5.0, 30.0, 25.0], [50.0, 30.0, 75.0, 55.0]])
    for _ in range(50):
        out, b = frcnn.augment(img, boxes, g)
        _, h, w = out.shape
        assert len(b) >= 1 and out.min() >= 0 and out.max() <= 1
        assert (b[:, 0] >= 0).all() and (b[:, 2] <= w).all() and (b[:, 1] >= 0).all() and (b[:, 3] <= h).all()
        assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()
    # a horizontal flip maps x to w - x
    flipped = img.flip(-1)
    b = boxes.clone()
    b[:, [0, 2]] = 80 - boxes[:, [2, 0]]
    assert torch.equal(flipped[:, 5:25, int(b[0, 0]):int(b[0, 2])], img[:, 5:25, 10:30].flip(-1))


def test_model_has_two_classes_and_predicts_in_input_pixels():
    torch.manual_seed(0)
    cfg = frcnn.FRCNNConfig(img_size=128, pretrained=False)
    model = frcnn.build_model(cfg)
    assert model.roi_heads.box_predictor.cls_score.out_features == 2
    imgs = [np.random.default_rng(0).integers(0, 255, (96, 128, 3), dtype=np.uint8)]
    (boxes, scores), = frcnn.predict(model, imgs, torch.device("cpu"))
    assert boxes.shape[1] == 4 and len(boxes) == len(scores) <= 100
    if len(boxes):
        assert boxes[:, [0, 2]].max() <= 128 + 1e-3 and boxes[:, [1, 3]].max() <= 96 + 1e-3


def test_learning_rate_schedule_warms_up_then_decays():
    cfg = frcnn.FRCNNConfig(lr=0.01, warmup_iters=10)
    lrs = [frcnn._lr_at(i, 100, cfg) for i in range(100)]
    assert lrs[0] < 0.001 and abs(lrs[10] - 0.01) < 1e-9
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:])) and lrs[-1] > 0.01 * 0.01 - 1e-12
