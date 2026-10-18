"""STEP 5 - Make the model small and fast for the Raspberry Pi 4.

  python scripts/step5_export.py --weights runs/baseline/weights/best.pt            # NCNN (recommended for Pi)
  python scripts/step5_export.py --weights runs/baseline/weights/best.pt --int8     # also TFLite INT8

Then ALWAYS re-check accuracy of the exported model (quantization can hurt tiny targets):
  python scripts/step4_evaluate.py --weights runs/baseline/weights/best_ncnn_model --imgsz 640
"""
import argparse

from ultralytics import YOLO

from common import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640, help="Pi speed: try 640, 480, 320 and compare")
    ap.add_argument("--int8", action="store_true", help="also export TFLite INT8 (needs --data for calibration)")
    ap.add_argument("--data", default=str(ROOT / "data/merged/data.yaml"))
    args = ap.parse_args()

    m = YOLO(args.weights)
    p = m.export(format="ncnn", imgsz=args.imgsz)
    print(f"NCNN model: {p}  -> copy this whole folder to the Pi")
    if args.int8:
        p2 = YOLO(args.weights).export(format="tflite", imgsz=args.imgsz, int8=True, data=args.data)
        print(f"TFLite INT8 model: {p2}")


if __name__ == "__main__":
    main()
