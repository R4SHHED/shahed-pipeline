"""STEP 3 - Train one experiment and log it.

Run the BASELINE first, then change ONE thing at a time:

  python scripts/step3_train.py --name baseline
  python scripts/step3_train.py --name imgsz960   --imgsz 960
  python scripts/step3_train.py --name p2         --p2
  python scripts/step3_train.py --name p2_960     --p2 --imgsz 960

Every run is evaluated on the REAL-ONLY test set and appended to experiments.csv.
That file becomes your "ablation table" for the team.
"""
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

from common import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="experiment name, e.g. baseline")
    ap.add_argument("--data", default=str(ROOT / "data/merged/data.yaml"))
    ap.add_argument("--model", default="yolo11n.pt",
                    help="pretrained weights. Swap for a newer nano model (e.g. yolo26n.pt) if your ultralytics version has it")
    ap.add_argument("--p2", action="store_true", help="add a high-resolution P2 head for tiny targets (YOLOv8n-P2)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=25, help="stop early if no improvement")
    ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU (auto if empty)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.p2:
        # P2 config + load the matching pretrained weights wherever layers fit
        model = YOLO("yolov8n-p2.yaml").load("yolov8n.pt")
        model_desc = "yolov8n-p2"
    else:
        model = YOLO(args.model)
        model_desc = Path(args.model).stem

    t0 = time.time()
    model.train(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        patience=args.patience, device=args.device, seed=args.seed, deterministic=True,
        project=str(ROOT / "runs"), name=args.name, exist_ok=True,
        # small-object friendly augmentation: keep mosaic, add mild scale + blur-ish variety
        mosaic=1.0, scale=0.5, degrees=5, fliplr=0.5, hsv_v=0.4,
        close_mosaic=10,
    )
    train_min = (time.time() - t0) / 60

    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=args.data, split="test", imgsz=args.imgsz, device=args.device,
                            project=str(ROOT / "runs"), name=f"{args.name}_test", exist_ok=True)

    # per-class numbers for the shahed class (class id 0). class_result() is indexed by position
    # in ap_class_index, not by class id, so look the position up.
    idx = list(m.ap_class_index).index(0) if 0 in list(m.ap_class_index) else None
    p, r, ap50, ap = (float(v) for v in m.box.class_result(idx)) if idx is not None else (0.0,) * 4
    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "name": args.name, "model": model_desc,
        "imgsz": args.imgsz, "epochs": args.epochs,
        "test_mAP50_all": round(float(m.box.map50), 4), "test_mAP50-95_all": round(float(m.box.map), 4),
        "shahed_P": round(p, 4), "shahed_R": round(r, 4),
        "shahed_AP50": round(ap50, 4), "shahed_AP50-95": round(ap, 4),
        "train_minutes": round(train_min, 1), "weights": str(best.relative_to(ROOT)),
        "pi_fps": "", "notes": "",
    }
    log = ROOT / "experiments.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"\nLogged to {log}:\n{row}")
    print("Next: python scripts/step4_evaluate.py --weights", row["weights"])


if __name__ == "__main__":
    main()
