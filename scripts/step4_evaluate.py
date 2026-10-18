"""STEP 4 - The evaluation that actually matters for this project.

  python scripts/step4_evaluate.py --weights runs/baseline/weights/best.pt

mAP alone hides two things your team will care about:
  1. RECALL BY SIZE  - does it still catch the Shahed when it is far away (tiny)?
  2. FALSE ALARMS    - how often does it shout "Shahed" at a bird, paper plane or empty sky?
This script prints both, on the real-only test set, and saves the worst mistakes as images
so you can SEE what confuses the model.
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from common import IMG_EXTS, ROOT, SIZE_BUCKETS, bucket_of, load_yaml, read_yolo_labels


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/merged/data.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=None, help="default: same as training")
    ap.add_argument("--conf", type=float, default=0.25, help="the threshold you will use on the Pi")
    ap.add_argument("--iou", type=float, default=0.3, help="match threshold (low on purpose: tiny boxes)")
    ap.add_argument("--target", type=int, default=0, help="class id of the target (shahed)")
    ap.add_argument("--save-errors", type=int, default=40)
    args = ap.parse_args()

    d = load_yaml(args.data)
    names = d["names"]
    names = [names[k] for k in sorted(names, key=int)] if isinstance(names, dict) else names
    idir = Path(d["path"]) / d[args.split]
    imgs = sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXTS)
    model = YOLO(args.weights)
    imgsz = args.imgsz
    if imgsz is None:
        ckpt = getattr(model, "ckpt", None) or {}
        imgsz = (ckpt.get("train_args") or {}).get("imgsz", 640)

    gt_by_bucket, hit_by_bucket = Counter(), Counter()
    fp_total, fp_imgs, n_no_target, fp_on_no_target = 0, 0, 0, 0
    fp_on_class = Counter()   # false alarms that landed on a labelled bird / aircraft
    errors = []

    for p in imgs:
        H, W = cv2.imread(str(p)).shape[:2]
        gts = []
        for c, xc, yc, w, h in read_yolo_labels(p.parent.parent / "labels" / (p.stem + ".txt")):
            box = [(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H]
            gts.append((c, box))
        tgt = [b for c, b in gts if c == args.target]
        others = [(c, b) for c, b in gts if c != args.target]

        r = model.predict(str(p), imgsz=imgsz, conf=args.conf, verbose=False)[0]
        dets = [(float(s), b.tolist()) for s, b, c in zip(r.boxes.conf, r.boxes.xyxy, r.boxes.cls)
                if int(c) == args.target]
        dets.sort(key=lambda x: -x[0])

        used, img_fp, img_miss = set(), 0, 0
        for s, db in dets:
            best, bj = 0, -1
            for j, gb in enumerate(tgt):
                if j not in used and (v := iou(db, gb)) > best:
                    best, bj = v, j
            if best >= args.iou:
                used.add(bj)
            else:
                img_fp += 1
                for c, ob in others:
                    if iou(db, ob) >= args.iou:
                        fp_on_class[names[c]] += 1
                        break
        for j, gb in enumerate(tgt):
            b = bucket_of(((gb[2] - gb[0]) * (gb[3] - gb[1])) ** 0.5)
            gt_by_bucket[b] += 1
            if j in used:
                hit_by_bucket[b] += 1
            else:
                img_miss += 1
        fp_total += img_fp
        fp_imgs += img_fp > 0
        if not tgt:
            n_no_target += 1
            fp_on_no_target += img_fp > 0
        if img_fp or img_miss:
            errors.append((img_fp + img_miss, p, r, tgt))

    print(f"\nEvaluated {len(imgs)} images from '{args.split}' at imgsz={imgsz}, conf={args.conf}")
    print("\nRECALL BY SIZE (how many real Shaheds were found)")
    tot_g = tot_h = 0
    for b, _, _ in SIZE_BUCKETS:
        g, h = gt_by_bucket[b], hit_by_bucket[b]
        tot_g += g
        tot_h += h
        print(f"  {b:<16} {h:>5}/{g:<5} = {100*h/g:5.1f}%" if g else f"  {b:<16}     no examples")
    print(f"  {'ALL':<16} {tot_h:>5}/{tot_g:<5} = {100*tot_h/max(tot_g,1):5.1f}%")

    print("\nFALSE ALARMS (said 'Shahed' when it wasn't)")
    print(f"  total false boxes: {fp_total}  on {fp_imgs} images")
    if n_no_target:
        print(f"  images with NO Shahed that still raised an alarm: {fp_on_no_target}/{n_no_target}"
              f" = {100*fp_on_no_target/n_no_target:.1f}%")
    if fp_on_class:
        print(f"  false alarms that hit a labelled object: {dict(fp_on_class)}")

    out = Path(args.weights).resolve().parent.parent / f"errors_{args.split}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for _, p, r, tgt in sorted(errors, key=lambda e: -e[0])[: args.save_errors]:
        im = r.plot()  # predictions
        for gb in tgt:  # ground truth in green
            cv2.rectangle(im, (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])), (0, 255, 0), 2)
        cv2.imwrite(str(out / p.name), im)
    summary = {"recall_by_size": {b: [hit_by_bucket[b], gt_by_bucket[b]] for b, _, _ in SIZE_BUCKETS},
               "false_boxes": fp_total, "no_target_images_with_alarm": [fp_on_no_target, n_no_target]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWorst {min(len(errors), args.save_errors)} mistakes saved to {out}  (green = truth, colored = model)")


if __name__ == "__main__":
    main()
