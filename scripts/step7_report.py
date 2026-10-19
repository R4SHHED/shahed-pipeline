"""STEP 7 - One results table for the team.

  python scripts/step7_report.py --weights runs/baseline/weights/best.pt --sizes 640 480 320 ^
         --pi-fps 640=2.75 480=4.92 320=10.05

For every image size it measures, on the REAL-ONLY test set:
  precision, recall, F1 (at the confidence you will use on the Pi, IoU >= 0.5)
  each with a 95% confidence range (Wilson) - important with a small test set
  mAP50 and mAP50-95 (standard benchmark numbers)
  false boxes per image, and false alarms on images with NO drone (if the test set has any)
  laptop CPU time per image, and the Pi FPS you measured
It also reports the render validation score once, for context.

Writes report/results.md (paste into a document / message) and report/results.csv.
"""
import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from common import IMG_EXTS, ROOT, load_yaml, read_yolo_labels


def wilson(k, n, z=1.96):
    """95% confidence range for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def match_eval(model, imgs, imgsz, conf, iou_thr, target):
    tp = fp = fn = 0
    empty_imgs = empty_alarm = 0
    times = []
    for i, p in enumerate(imgs):
        im = cv2.imread(str(p))
        H, W = im.shape[:2]
        gts = [[(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H]
               for c, xc, yc, w, h in read_yolo_labels(p.parent.parent / "labels" / (p.stem + ".txt"))
               if c == target]
        t0 = time.perf_counter()
        r = model.predict(im, imgsz=imgsz, conf=conf, verbose=False)[0]
        if i > 0:  # skip warm-up
            times.append(time.perf_counter() - t0)
        dets = sorted(((float(s), b.tolist()) for s, b, c in zip(r.boxes.conf, r.boxes.xyxy, r.boxes.cls)
                       if int(c) == target), key=lambda x: -x[0])
        used = set()
        for _, db in dets:
            best, bj = 0, -1
            for j, gb in enumerate(gts):
                if j not in used and (v := iou(db, gb)) > best:
                    best, bj = v, j
            if best >= iou_thr:
                used.add(bj)
                tp += 1
            else:
                fp += 1
        fn += len(gts) - len(used)
        if not gts:
            empty_imgs += 1
            empty_alarm += bool(dets)
    ms = 1000 * sum(times) / len(times) if times else float("nan")
    return tp, fp, fn, empty_imgs, empty_alarm, ms


def pct(x):
    return "n/a" if x != x else f"{100 * x:.1f}%"


def rng(lo_hi):
    lo, hi = lo_hi
    return "n/a" if lo != lo else f"{100 * lo:.0f}–{100 * hi:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="the .pt model (results match its NCNN export)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[640])
    ap.add_argument("--pi-fps", nargs="*", default=[], help="measured Pi FPS per size, e.g. 640=2.75 480=4.92")
    ap.add_argument("--data", default=str(ROOT / "data/merged/data.yaml"))
    ap.add_argument("--conf", type=float, default=0.3, help="the confidence threshold used on the Pi")
    ap.add_argument("--iou", type=float, default=0.5, help="box overlap needed to count as a hit (0.5 = standard)")
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "report"))
    args = ap.parse_args()

    pi = {}
    for kv in args.pi_fps:
        k, v = kv.split("=")
        pi[int(k)] = v

    d = load_yaml(args.data)
    base = Path(d["path"])
    imgs = sorted(p for p in (base / d["test"]).iterdir() if p.suffix.lower() in IMG_EXTS)
    n_drones = sum(1 for p in imgs for c, *_ in read_yolo_labels(p.parent.parent / "labels" / (p.stem + ".txt"))
                   if c == args.target)
    model = YOLO(args.weights)

    # context: score on the (render) validation set at the first size
    v = model.val(data=args.data, split="val", imgsz=args.sizes[0], plots=False, verbose=False)
    val_map50 = float(v.box.map50)

    rows = []
    for s in args.sizes:
        tp, fp, fn, n_empty, empty_alarm, ms = match_eval(model, imgs, s, args.conf, args.iou, args.target)
        P = tp / (tp + fp) if tp + fp else float("nan")
        R = tp / (tp + fn) if tp + fn else float("nan")
        F1 = 2 * P * R / (P + R) if P == P and R == R and P + R else float("nan")
        m = model.val(data=args.data, split="test", imgsz=s, plots=False, verbose=False)
        rows.append({
            "image_size": s,
            "drones_found": f"{tp}/{tp + fn}",
            "recall": pct(R), "recall_95%_range": rng(wilson(tp, tp + fn)),
            "precision": pct(P), "precision_95%_range": rng(wilson(tp, tp + fp)),
            "F1": f"{F1:.3f}" if F1 == F1 else "n/a",
            "mAP50": f"{float(m.box.map50):.3f}", "mAP50-95": f"{float(m.box.map):.3f}",
            "false_boxes": fp, "false_boxes_per_image": f"{fp / len(imgs):.2f}",
            "empty_images_with_alarm": f"{empty_alarm}/{n_empty}" if n_empty else "not tested (0 empty images)",
            "laptop_ms_per_image": f"{ms:.0f}",
            "pi4_fps": pi.get(s, "not measured"),
        })
        print(f"size {s}: found {tp}/{tp + fn}, P={pct(P)}, R={pct(R)}, F1={rows[-1]['F1']}, "
              f"mAP50={rows[-1]['mAP50']}, false boxes={fp}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cols = ["image_size", "drones_found", "recall", "recall_95%_range", "precision", "precision_95%_range",
            "F1", "mAP50", "mAP50-95", "false_boxes", "empty_images_with_alarm", "pi4_fps"]
    md = [f"# Shahed detector – test results ({datetime.now():%Y-%m-%d})", "",
          f"- Model: `{Path(args.weights).as_posix()}` (YOLO11n), deployed as NCNN on Raspberry Pi 4",
          f"- Test set: **{len(imgs)} real images, {n_drones} drones** (real footage only; no renders)",
          f"- Settings: confidence ≥ {args.conf}, a hit needs box overlap (IoU) ≥ {args.iou}",
          f"- For context: mAP50 on the render validation set = {val_map50:.3f} (renders are much easier)", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    md += ["", "**How to read this**",
           "- *Recall* = share of real drones found. *Precision* = share of alarms that were real drones. "
           "*F1* balances both (1.0 = perfect).",
           "- The *95% range* is where the true value most likely lies. It is wide because the test set is small.",
           "- *Empty images with alarm* = false-alarm rate on sky with no drone. "
           "Not measurable until empty/negative test images are added.",
           "- *mAP50* is the standard benchmark score used in research papers.", "",
           "**Limits of these numbers**",
           f"- Only {len(imgs)} real test images, probably from one or two videos.",
           "- No small/far-away drones and no birds/aircraft in the test set yet.",
           "- The model was trained almost entirely on 3D renders."]
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {out / 'results.md'} and {out / 'results.csv'}")


if __name__ == "__main__":
    main()
