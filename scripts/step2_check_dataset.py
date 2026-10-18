"""STEP 2 - Look at your data before training.

  python scripts/step2_check_dataset.py

Prints, for each split: images per class, how many background (no-target) images,
and how SMALL the targets are. If most Shaheds are 'tiny' or 'small', the
P2 head and bigger image size (step 3) matter a lot for you.
"""
import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

from common import IMG_EXTS, ROOT, SIZE_BUCKETS, bucket_of, load_yaml, read_yolo_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/merged/data.yaml"))
    args = ap.parse_args()
    d = load_yaml(args.data)
    names = d["names"]
    names = [names[k] for k in sorted(names, key=int)] if isinstance(names, dict) else names
    base = Path(d["path"])

    for split in ("train", "val", "test"):
        idir = base / d[split]
        if not idir.exists():
            continue
        imgs = [p for p in idir.iterdir() if p.suffix.lower() in IMG_EXTS]
        per_cls, sizes, empty = Counter(), {n: Counter() for n in names}, 0
        for p in imgs:
            boxes = read_yolo_labels(p.parent.parent / "labels" / (p.stem + ".txt"))
            if not boxes:
                empty += 1
                continue
            W, H = Image.open(p).size
            for c, _, _, w, h in boxes:
                per_cls[names[c]] += 1
                sizes[names[c]][bucket_of(((w * W) * (h * H)) ** 0.5)] += 1
        print(f"\n=== {split}: {len(imgs)} images, {empty} background (no target) ===")
        for n in names:
            tot = per_cls[n]
            if not tot:
                print(f"  {n:<16} 0 boxes")
                continue
            dist = "  ".join(f"{b}: {100*sizes[n][b]/tot:4.1f}%" for b, _, _ in SIZE_BUCKETS)
            print(f"  {n:<16} {tot:>6} boxes | {dist}")

    print("\nRules of thumb:\n"
          " - background images: aim for roughly 5-15% of train (paper planes, kites, empty sky...)\n"
          " - if test has < ~200 shahed boxes, your accuracy numbers will be noisy\n"
          " - if >30% of shahed boxes are tiny/small, try --p2 and --imgsz 960 in step 3")


if __name__ == "__main__":
    main()
