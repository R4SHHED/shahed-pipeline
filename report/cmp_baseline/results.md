# Shahed detector – test results (2026-10-31)

- Model: `runs/baseline/weights/best.pt` (YOLO11n), deployed as NCNN on Raspberry Pi 4
- Test set: **1094 real images**: 49 Shahed drones, and 1045 images with NO Shahed (birds / sky) to measure false alarms. Real footage only; no renders.
- Settings: confidence ≥ 0.3, a hit needs box overlap (IoU) ≥ 0.5
- For context: mAP50 on the render validation set = 0.964 (renders are much easier)

| image_size | drones_found | recall | recall_95%_range | precision | precision_95%_range | F1 | mAP50 | mAP50-95 | false_boxes | empty_images_with_alarm | pi4_fps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 640 | 49/49 | 100.0% | 93–100% | 2.4% | 2–3% | 0.047 | 0.086 | 0.025 | 1990 | 866/1045 (82.9%) | 2.75 |

**How to read this**
- *Recall* = share of real drones found. *Precision* = share of alarms that were real drones. *F1* balances both (1.0 = perfect).
- The *95% range* is where the true value most likely lies. It is wide because the test set is small.
- *Empty images with alarm* = images with NO Shahed (e.g. only birds) where the model still raised a Shahed alarm, i.e. the false-alarm rate.
- mAP50 / mAP50-95 here are for the **Shahed class only**.
- *mAP50* is the standard benchmark score used in research papers.

**Limits of these numbers**
- Only 1094 real test images, probably from one or two videos.
- No small/far-away Shaheds and no aircraft in the test set yet.
- The model was trained almost entirely on 3D renders.