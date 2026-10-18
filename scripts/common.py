"""Shared helpers used by the pipeline scripts."""
import re
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT = Path(__file__).resolve().parent.parent

# Pixel size buckets used everywhere (sqrt of box area, in pixels of the ORIGINAL image)
SIZE_BUCKETS = [("tiny <16px", 0, 16), ("small 16-32px", 16, 32),
                ("medium 32-96px", 32, 96), ("large >96px", 96, 1e9)]


def bucket_of(px):
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= px < hi:
            return name
    return SIZE_BUCKETS[-1][0]


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def class_names_from_data_yaml(dataset_dir):
    """Read class names from a Roboflow/Ultralytics data.yaml (list or dict form)."""
    p = Path(dataset_dir) / "data.yaml"
    if not p.exists():
        return []
    names = load_yaml(p).get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return list(names)


def label_path_for(img_path):
    """images/xxx.jpg -> labels/xxx.txt (standard YOLO layout)."""
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img_path.with_suffix(".txt")


def read_yolo_labels(txt_path):
    """Return list of (cls, xc, yc, w, h). Polygon (segmentation) lines are converted to boxes."""
    out = []
    if not txt_path or not Path(txt_path).exists():
        return out
    for line in Path(txt_path).read_text().splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        c = int(float(v[0]))
        nums = [float(x) for x in v[1:]]
        if len(nums) == 4:
            out.append((c, *nums))
        elif len(nums) >= 6 and len(nums) % 2 == 0:  # polygon
            xs, ys = nums[0::2], nums[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            out.append((c, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return out


_RF = re.compile(r"(_(jpg|jpeg|png|webp|bmp))?\.rf\.[0-9a-f]+$", re.I)
_FRAME = re.compile(r"([_\-. ]?(frame|img|image|f))?[_\-. ]?\d+$", re.I)


def group_key(stem, mode="image"):
    """Images that share a group must stay in the same split.
    mode='image': only Roboflow augmented copies of the same original are grouped
                  ('photo12_jpg.rf.abc' and 'photo12_jpg.rf.def' -> 'photo12').
    mode='video': also strips the trailing frame number, so every frame of a video is one group
                  ('video3_frame_0042_jpg.rf.abc' -> 'video3')."""
    s = _RF.sub("", stem)
    if mode == "video":
        s = _FRAME.sub("", s) or s
    return s.lower()
