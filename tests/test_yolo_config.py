import importlib.util
from pathlib import Path

import yaml

from polypdet import yolo


def ultralytics_defaults() -> dict:
    """Ultralytics' default.yaml, read without importing the package (importing writes a settings file)."""
    pkg = Path(importlib.util.find_spec("ultralytics").submodule_search_locations[0])
    return yaml.safe_load((pkg / "cfg" / "default.yaml").read_text(encoding="utf-8"))


def test_augmentation_overrides_are_valid_ultralytics_arguments():
    defaults = ultralytics_defaults()
    for key in {**yolo.AUG_ON, **yolo.AUG_OFF}:
        assert key in defaults, key


def test_no_augmentation_switches_off_every_image_augmentation_that_is_on_by_default():
    defaults = ultralytics_defaults()
    aug_keys = ["mosaic", "mixup", "cutmix", "copy_paste", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
                "shear", "perspective", "flipud", "fliplr"]
    on_by_default = [k for k in aug_keys if defaults[k]]
    assert on_by_default  # the defaults do augment
    assert all(yolo.AUG_OFF[k] == 0 for k in aug_keys)
    cfg = yolo.config_dict(yolo.YOLOConfig(augment=False))
    assert cfg["aug_overrides"] is yolo.AUG_OFF and cfg["img_size"] == 416
