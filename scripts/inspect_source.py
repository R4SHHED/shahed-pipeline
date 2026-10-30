"""Look inside a downloaded dataset that mixes several sources.

  python scripts/inspect_source.py data/raw/shahed-e-yjnj4

Groups images into "families" by their file-name pattern (e.g. V_AIRPLANE_###_mp4-#,
UAV_#-###-, <random letters>####), counts the labels of each class per family, and saves a
picture sheet per family (12 random images with boxes drawn) so you can SEE which
families are real photos and which are renders.

Output: printed table + report/inspect/<dataset>/<family>.jpg
"""
import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import IMG_EXTS, ROOT, class_names_from_data_yaml, label_path_for, read_yolo_labels

_RF = re.compile(r"(_(jpg|jpeg|png|webp|bmp))?\.rf\.[0-9a-f]+$", re.I)
COLORS = [(0, 200, 255), (255, 0, 200), (0, 255, 0), (255, 128, 0), (128, 128, 255)]


def family_of(stem):
    s = _RF.sub("", stem)
    s = re.sub(r"[A-Za-z]{12,}", "*", s)   # long random letter strings
    s = re.sub(r"\d+", "#", s)             # any number
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="folder with data.yaml, e.g. data/raw/shahed-e-yjnj4")
    ap.add_argument("--per-sheet", type=int, default=12)
    ap.add_argument("--min-images", type=int, default=10, help="only make sheets for families this big")
    args = ap.parse_args()

    d = Path(args.dataset)
    names = class_names_from_data_yaml(d)
    imgs = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXTS)
    fam_imgs = defaultdict(list)
    fam_cls = defaultdict(Counter)
    for p in imgs:
        f = family_of(p.stem)
        fam_imgs[f].append(p)
        for c, *_ in read_yolo_labels(label_path_for(p)):
            fam_cls[f][names[c] if c < len(names) else str(c)] += 1

    out = ROOT / "report" / "inspect" / d.name
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(fam_imgs, key=lambda f: -len(fam_imgs[f]))
    print(f"{len(imgs)} images, classes {names}\n")
    print(f"{'#':>3}  {'family (name pattern)':<34} {'images':>6}  " + "  ".join(f"{n:>8}" for n in names))
    small = Counter()
    for i, f in enumerate(rows, 1):
        n = len(fam_imgs[f])
        if n < args.min_images:
            small["images"] += n
            for k, v in fam_cls[f].items():
                small[k] += v
            continue
        print(f"{i:>3}  {f[:34]:<34} {n:>6}  " + "  ".join(f"{fam_cls[f][c]:>8}" for c in names))
        # picture sheet
        rng = random.Random(0)
        pick = rng.sample(fam_imgs[f], min(args.per_sheet, n))
        tiles = []
        for p in pick:
            im = cv2.imread(str(p))
            if im is None:
                continue
            H, W = im.shape[:2]
            for c, xc, yc, w, h in read_yolo_labels(label_path_for(p)):
                x0, y0 = int((xc - w / 2) * W), int((yc - h / 2) * H)
                x1, y1 = int((xc + w / 2) * W), int((yc + h / 2) * H)
                col = COLORS[c % len(COLORS)]
                cv2.rectangle(im, (x0, y0), (x1, y1), col, max(2, W // 300))
                cv2.putText(im, names[c] if c < len(names) else str(c), (x0, max(12, y0 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.5, W / 900), col, max(1, W // 500))
            tiles.append(cv2.resize(im, (400, 300)))
        while len(tiles) % 4:
            tiles.append(np.zeros((300, 400, 3), np.uint8))
        sheet = np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)])
        cv2.putText(sheet, f"family {i}: {f}  ({n} images)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
        cv2.imwrite(str(out / f"{i:02d}.jpg"), sheet)
    if small["images"]:
        print(f"{'':>3}  {'(small families, combined)':<34} {small['images']:>6}  "
              + "  ".join(f"{small[c]:>8}" for c in names))
    print(f"\nPicture sheets saved in {out}  (file NN.jpg = family number NN in the table)")
    print("Colours: " + ", ".join(f"{n}={['orange', 'pink', 'green', 'blue', 'purple'][i % 5]}"
                                  for i, n in enumerate(names)))


if __name__ == "__main__":
    main()
