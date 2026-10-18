# Shahed-136 Detector → Raspberry Pi 4

A ground camera detects a Shahed-136 in the sky, using YOLO trained on your laptop/GPU and run on a Raspberry Pi 4.

## What this pipeline does better than "just train YOLO"

| Improvement | Why it matters |
|---|---|
| **Leak-free split** (by video / original image) | Frames from one video look almost identical. A random split puts them in both train and test, so accuracy looks fake-high. |
| **Real-only test set** | Synthetic renders are never in test, so your number reflects real cameras. |
| **One unified class list** | Merges several Roboflow datasets with different class names into `shahed / bird / other_aircraft`. |
| **Hard negatives folder** | Paper planes, kites, empty sky: images with no labels teach "this is NOT a Shahed". |
| **Recall by size + false-alarm rate** | mAP hides whether far-away (tiny) Shaheds are missed and how often birds trigger alarms. |
| **Experiment log (`experiments.csv`)** | Every run is logged automatically and becomes your ablation table for the team. |
| **Pi FPS check from day 1** | You find out early if the Pi is too slow, not at the end. |
| **Tracking + confirmation on the Pi** | An alarm fires only after N frames, so one-frame blips are ignored. |

## Setup (laptop / GPU machine)

```bash
pip install -r requirements.txt
```

Put each downloaded dataset (Roboflow → export **YOLOv8** format → unzip) in `data/raw/<name>/`, then edit **`sources.yaml`**:
- `class_map`: map each dataset's class names to yours. Anything not listed is dropped.
- `domain: synthetic` for 3D renders and `real` for real footage.
- `data/raw/negatives/`: drop paper-plane, kite, plane and empty-sky images here. **No labels.**

## Run order

```bash
# 1. merge + leak-free split
python scripts/step1_merge_datasets.py

# 2. look at the data (how small are the targets?)
python scripts/step2_check_dataset.py

# 3. baseline FIRST, then change one thing at a time
python scripts/step3_train.py --name baseline
python scripts/step3_train.py --name imgsz960 --imgsz 960
python scripts/step3_train.py --name p2       --p2
python scripts/step3_train.py --name p2_960   --p2 --imgsz 960

# 4. the evaluation that matters (recall by size, false alarms, error images)
python scripts/step4_evaluate.py --weights runs/baseline/weights/best.pt

# 5. export for the Pi, then RE-EVALUATE the exported model
python scripts/step5_export.py   --weights runs/p2/weights/best.pt --imgsz 640
python scripts/step4_evaluate.py --weights runs/p2/weights/best_ncnn_model --imgsz 640
```

## On the Raspberry Pi 4 (64-bit Raspberry Pi OS)

```bash
sudo apt install -y python3-picamera2
pip install ultralytics ncnn lap --break-system-packages
# copy step6_pi_detect.py and the best_ncnn_model folder to the Pi
python step6_pi_detect.py --model best_ncnn_model                     # with screen
python step6_pi_detect.py --model best_ncnn_model --headless --save out.mp4
```

**Do this on day 1:** export the plain pretrained `yolo11n.pt` to NCNN and run step 6 with `--max-frames 300` at `--imgsz 640`, `480` and `320`. That tells you the real FPS budget before you spend time training.

## The table your team wants (`experiments.csv` + a few columns)

| run | change | shahed recall (tiny) | shahed recall (all) | false-alarm rate | mAP50 | Pi FPS |
|---|---|---|---|---|---|---|
| baseline | — | | | | | |
| imgsz960 | bigger input | | | | | |
| p2 | P2 head | | | | | |
| p2 + NCNN | exported | | | | | |

Change **one thing per run**. Give every run the same number of epochs. The P2 model needs a full run, because its new head starts untrained and scores near 0 for the first few epochs. If a change doesn't beat the baseline on the real test set, drop it.

## Later / optional steps
- **Knowledge distillation:** train a bigger teacher (`yolo11s/m`), then distil it into the nano model. This isn't built into Ultralytics, so see Setyanto et al. 2025 and Jobaer et al. 2025.
- **Add your own footage:** record your actual camera at the deployment site, label about 100–300 frames, and add them as a `real` source. This often helps more than any model change.
- **Hard-negative mining:** look at `runs/<name>/errors_test/`. Whatever the model confuses most (birds? planes?), collect more of it.
