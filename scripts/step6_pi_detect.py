"""STEP 6 - Run on the Raspberry Pi 4 with a camera (or a video file).

  # live camera, with a window
  python step6_pi_detect.py --model best_ncnn_model
  # no screen attached (SSH): saves an annotated video instead
  python step6_pi_detect.py --model best_ncnn_model --headless --save out.mp4
  # FPS benchmark on a recorded video (do this on DAY 1 with the plain pretrained model!)
  python step6_pi_detect.py --model yolo11n_ncnn_model --source test.mp4 --headless --max-frames 300

Tracking + confirmation: a Shahed is only CONFIRMED after it is tracked for
--confirm frames. One-frame blips (a bird, a glint) never raise the alarm.
Every confirmed detection is written to detections.csv.
"""
import argparse
import csv
import time
from collections import defaultdict, deque
from datetime import datetime

import cv2
from ultralytics import YOLO


def open_source(src, width, height):
    if src == "picam":
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            cam.configure(cam.create_video_configuration(main={"size": (width, height), "format": "RGB888"}))
            cam.start()

            class PiCam:
                def read(self):
                    return True, cam.capture_array()  # RGB888 in picamera2 is BGR order for OpenCV

                def release(self):
                    cam.stop()
            return PiCam()
        except ImportError:
            print("picamera2 not found, falling back to OpenCV camera 0")
            src = "0"
    cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="best_ncnn_model folder (or .pt / .tflite)")
    ap.add_argument("--source", default="picam", help="picam, 0 (USB cam), or a video file")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--target", type=int, default=0, help="class id of shahed")
    ap.add_argument("--confirm", type=int, default=3, help="frames a track must survive to be confirmed")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--save", default=None, help="save annotated video to this file")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--log", default="detections.csv")
    args = ap.parse_args()

    model = YOLO(args.model, task="detect")
    names = model.names
    cap = open_source(args.source, args.width, args.height)
    writer = None
    seen = defaultdict(int)          # track id -> frames seen
    confirmed = set()
    times = deque(maxlen=30)
    all_times = []
    import os
    new_log = not os.path.exists(args.log)
    log = open(args.log, "a", newline="")
    lw = csv.writer(log)
    if new_log:
        lw.writerow(["time", "frame", "track_id", "class", "conf", "x0", "y0", "x1", "y1"])
    n = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            r = model.track(frame, imgsz=args.imgsz, conf=args.conf, persist=True,
                            tracker="bytetrack.yaml", verbose=False)[0]
            dt = time.perf_counter() - t0
            times.append(dt)
            all_times.append(dt)
            fps = len(times) / sum(times)

            alarm = False
            if r.boxes is not None and r.boxes.id is not None:
                for box, cls, conf, tid in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf, r.boxes.id):
                    tid, cls = int(tid), int(cls)
                    x0, y0, x1, y1 = map(int, box.tolist())
                    if cls == args.target:
                        seen[tid] += 1
                        if seen[tid] >= args.confirm:
                            alarm = True
                            if tid not in confirmed:
                                confirmed.add(tid)
                                print(f"[ALARM] shahed track {tid} conf={float(conf):.2f}")
                            lw.writerow([datetime.now().isoformat(timespec="milliseconds"), n, tid,
                                         names[cls], f"{float(conf):.3f}", x0, y0, x1, y1])
                            color = (0, 0, 255)
                        else:
                            color = (0, 200, 255)   # seen but not confirmed yet
                    else:
                        color = (200, 200, 200)     # bird / other aircraft
                    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                    cv2.putText(frame, f"{names[cls]} {float(conf):.2f} #{tid}", (x0, max(15, y0 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.putText(frame, f"FPS {fps:4.1f}" + ("  SHAHED!" if alarm else ""), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255) if alarm else (0, 255, 0), 2)
            if args.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                writer.write(frame)
            if not args.headless:
                cv2.imshow("shahed detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        log.close()
        cv2.destroyAllWindows()

    if all_times:
        warm = all_times[5:] or all_times
        print(f"\n{n} frames | average inference FPS (after warm-up): {len(warm)/sum(warm):.2f} "
              f"| imgsz={args.imgsz} | confirmed tracks: {len(confirmed)}")
        print("Put this FPS number in the pi_fps column of experiments.csv")


if __name__ == "__main__":
    main()
