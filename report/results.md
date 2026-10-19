# Shahed detector – test results (2026-10-20)

- Model: `runs/baseline/weights/best.pt` (YOLO11n), deployed as NCNN on Raspberry Pi 4
- Test set: **49 real images, 49 drones** (real footage only; no renders)
- Settings: confidence ≥ 0.3, a hit needs box overlap (IoU) ≥ 0.5
- For context: mAP50 on the render validation set = 0.992 (renders are much easier)

| image_size | drones_found | recall | recall_95%_range | precision | precision_95%_range | F1 | mAP50 | mAP50-95 | false_boxes | empty_images_with_alarm | pi4_fps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 640 | 49/49 | 100.0% | 93–100% | 100.0% | 93–100% | 1.000 | 0.995 | 0.321 | 0 | not tested (0 empty images) | 2.75 |
| 480 | 47/49 | 95.9% | 86–99% | 94.0% | 84–98% | 0.949 | 0.987 | 0.325 | 3 | not tested (0 empty images) | 4.92 |
| 320 | 39/49 | 79.6% | 66–89% | 86.7% | 74–94% | 0.830 | 0.821 | 0.263 | 6 | not tested (0 empty images) | 10.05 |

**How to read this**
- *Recall* = share of real drones found. *Precision* = share of alarms that were real drones. *F1* balances both (1.0 = perfect).
- The *95% range* is where the true value most likely lies. It is wide because the test set is small.
- *Empty images with alarm* = false-alarm rate on sky with no drone. Not measurable until empty/negative test images are added.
- *mAP50* is the standard benchmark score used in research papers.

**Limits of these numbers**
- Only 49 real test images, probably from one or two videos.
- No small/far-away drones and no birds/aircraft in the test set yet.
- The model was trained almost entirely on 3D renders.