# Shahed detector – test results (2026-10-30)

- Model: `runs/birds_v1/weights/best.pt` (YOLO11n), deployed as NCNN on Raspberry Pi 4
- Test set: **313 real images**: 49 Shahed drones, and 264 images with NO Shahed (birds / sky) to measure false alarms. Real footage only; no renders.
- Settings: confidence ≥ 0.3, a hit needs box overlap (IoU) ≥ 0.5
- For context: mAP50 on the render validation set = 0.992 (renders are much easier)

| image_size | drones_found | recall | recall_95%_range | precision | precision_95%_range | F1 | mAP50 | mAP50-95 | false_boxes | empty_images_with_alarm | pi4_fps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 640 | 12/49 | 24.5% | 15–38% | 85.7% | 60–96% | 0.381 | 0.779 | 0.252 | 2 | 2/264 (0.8%) | 2.75 |
| 480 | 14/49 | 28.6% | 18–42% | 93.3% | 70–99% | 0.437 | 0.764 | 0.257 | 1 | 1/264 (0.4%) | 4.92 |

**How to read this**
- *Recall* = share of real drones found. *Precision* = share of alarms that were real drones. *F1* balances both (1.0 = perfect).
- The *95% range* is where the true value most likely lies. It is wide because the test set is small.
- *Empty images with alarm* = images with NO Shahed (e.g. only birds) where the model still raised a Shahed alarm, i.e. the false-alarm rate.
- mAP50 / mAP50-95 here are for the **Shahed class only**.
- *mAP50* is the standard benchmark score used in research papers.

**Limits of these numbers**
- Only 313 real test images, probably from one or two videos.
- No small/far-away Shaheds and no aircraft in the test set yet.
- The model was trained almost entirely on 3D renders.