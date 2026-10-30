# Shahed-136 Detector → Raspberry Pi 4

A ground camera looks at the sky and detects the Shahed-136 drone, telling it apart from birds, planes
and other drones. The model is YOLO11n, trained on a laptop CPU and running on a Raspberry Pi 4 (NCNN).

## Current results (model `composites_v1`, image size 480)

Measured on a **real-footage-only test set of 1,094 images**: 49 real Shahed images, plus 1,045 images
of birds, planes and quadcopters with **no** Shahed. No renders and no synthetic images in the test.

| Metric | Result |
|---|---|
| Real Shaheds found (recall) | **46 / 49 = 94%**: 45/49 with the stricter standard scoring (IoU ≥ 0.5); likely true range ≈ 81–97% |
| False-alarm rate | **14 of 1,045 non-Shahed images = 1.3%** (likely range ≈ 0.8–2.2%) |
| F1 (confidence 0.3, IoU 0.5) | **0.82** |
| Shahed mAP50 | **0.906** |
| Speed on Raspberry Pi 4 | **4.89 FPS** (≈ 0.2 s per frame), NCNN, CPU only |

On the Pi, an alarm is only raised after the drone is tracked for 3 frames, which removes most single-frame false boxes.

### How we got there: same 1,094 test images, confidence 0.3

| Model | What changed | Shaheds found | False boxes | Precision | F1 | Shahed mAP50 |
|---|---|---|---|---|---|---|
| baseline | trained on 3D renders only | 49/49 | **1,990** | 2.4% | 0.05 | 0.086 |
| birds_v1 | + real bird video frames | **12/49** | 94 | 11.3% | 0.16 | 0.198 |
| **composites_v1 (480)** | + real planes/drones + **render Shaheds pasted onto real skies** | **45/49** | **16** | 73.8% | **0.82** | **0.906** |

**Key lessons**
1. A model trained only on renders finds Shaheds but **calls every bird and plane a Shahed**.
2. Adding real birds fixed the false alarms but caused **shortcut learning**: every real-camera object in training
   was "not a Shahed", so the model learned *"real-looking = bird"* and labelled 22 of 49 real Shaheds as birds.
3. **Sim-to-real composites** (`make_composites.py`) break the shortcut: render Shaheds are cut out, darkened into
   silhouettes, shrunk to 8–48 px, blurred and pasted onto **real sky video frames from the training set only**.
4. Image size **480** is the best trade-off on the Pi: it's as accurate as 640 or better, and about 1.8× faster.

### Limitations (please read before quoting the numbers)
- Only **49 real Shahed test images**, probably from one or two videos, so the recall range is wide.
- No **real far-away (tiny) Shaheds** in the test set yet; the composites train for them, but we can't measure it.
- **No real Shahed footage in training**, only renders and composites.
- The Pi speed test used a video file; the **live camera test** is still to do.

## Datasets (credit the authors; check each license)

| Source (Roboflow Universe) | Use | License |
|---|---|---|
| shahed136-detect, by shahed136 | Shahed **renders** → train/val | Public domain |
| Target Detection – Shahed, by **Mariam Aljaberi** | renders → train/val; **49 real frames → test** | CC BY 4.0 |
| Birds, by **EvilSUmrak** | real flying-bird video frames | CC BY 4.0 |
| shahed, by **e-yjnj4** | real planes, quadcopters, birds (its render part is removed) | CC BY 4.0 |
| composites (made here) | render Shaheds on real training skies → train only | — |

Datasets and model weights are **not** in Git (too big). Download them from Roboflow in **YOLOv8** format into
`data/raw/<folder>` as listed in `sources.yaml`.

## What the pipeline does

| Feature | Why it matters |
|---|---|
| **Leak-free split** by video / original image, kept consistent **across datasets** | Near-identical frames never end up in both train and test |
| **Real-only test set** | Renders and composites never reach test, so the score reflects real cameras |
| `include` / `exclude` / `skip_images_with` per source | One downloaded folder can be split into real and render parts |
| **Recall by size + false-alarm rate + 95% ranges** | mAP alone hides missed far-away drones and bird false alarms |
| `inspect_source.py` | Picture sheets per file-name family: see what a dataset really contains |
| `make_composites.py` | Sim-to-real Shaheds on real skies, with a preview sheet for checking |
| Experiment log (`experiments.csv`) + `step7_report.py` | Every run logged; a ready-to-share results table |
| Tracking + 3-frame confirmation on the Pi | One-frame blips don't raise an alarm |

## Run order (laptop)

```bash
pip install -r requirements.txt

python scripts/step1_merge_datasets.py             # merge + leak-free split (reads sources.yaml)
python scripts/step2_check_dataset.py              # class counts and object sizes per split
python scripts/make_composites.py --n 3000         # sim-to-real Shaheds (then check report/inspect/composites_preview.jpg)
python scripts/step1_merge_datasets.py             # merge again, now including the composites

python scripts/step3_train.py --name composites_v1 --model runs/birds_v1/weights/best.pt --imgsz 640 --epochs 6 --device cpu
python scripts/step3_train.py --name composites_v1 --resume --device cpu        # if it was interrupted

python scripts/step7_report.py --weights runs/composites_v1/weights/best.pt --sizes 640 480 --out report/cmp_composites
python scripts/step5_export.py --weights runs/composites_v1/weights/best.pt --imgsz 480   # -> best_ncnn_model
python scripts/step4_evaluate.py --weights runs/composites_v1/weights/best_480_ncnn_model --imgsz 480 --conf 0.3
```

Helpful extras: `python scripts/inspect_source.py data/raw/<dataset>` shows what a new dataset contains.

## On the Raspberry Pi 4 (64-bit Raspberry Pi OS)

```bash
sudo apt install -y python3-venv python3-picamera2
python3 -m venv --system-site-packages ~/shahed-venv && source ~/shahed-venv/bin/activate
mkdir -p ~/tmp && export TMPDIR=~/tmp                                         # pip needs disk space, not RAM
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch!
pip install --no-cache-dir ultralytics ncnn lap
```

From the laptop: `scp -r <model>_ncnn_model scripts/step6_pi_detect.py r4shhed@raspberrypi4.local:~/shahed/`
(use the **name** `raspberrypi4.local`, because the Pi's IP address can change).

```bash
python step6_pi_detect.py --model composites_480_ncnn_model --imgsz 480 --conf 0.3                  # live camera
python step6_pi_detect.py --model composites_480_ncnn_model --imgsz 480 --conf 0.3 \
       --source test_video.mp4 --headless --max-frames 150                                         # speed test
```

## Next steps
1. **Live camera test** on the Pi.
2. **More real Shahed footage**, especially distant ones, for training and a bigger test set.
3. More hard negatives (kites, paper planes, gliders), then re-measure the false-alarm rate.
4. Try a faster Pi setup (Pi 5 or an AI accelerator) if more than about 5 FPS is needed.
