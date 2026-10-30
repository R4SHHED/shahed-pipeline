"""Make render Shaheds look like REAL camera footage ("sim-to-real" composites).

  python scripts/make_composites.py --n 3000

Why: every Shahed the model has seen is a clean 3D render, and every real-camera image it has
seen is a bird / plane / quadcopter. So it learns "real-looking = not a Shahed". Here we cut the
Shahed out of the renders and paste it - small, DARK (a real Shahed against a bright sky is a
dark silhouette), hazy, blurred, noisy, JPEG-compressed - onto REAL sky VIDEO frames from the
TRAINING split only (the test set is never touched).

Quality rules (from a manual review of the first version):
  * backgrounds: only real sky video frames (file names with mp4 / video / UAV_), mostly open sky,
    no close-ups, not crowded with birds
  * cut-outs: wing-shaped only (no bars / blobs)
  * pasted drone: dark silhouette, on plain sky, in the upper part, >= 20 px from edges and objects
Writes report/inspect/composites_preview.jpg (zoomed drones + whole images) - LOOK AT IT.

Output: data/raw/composites/ (YOLO format), used by the 'composites' source in sources.yaml.
"""
import argparse
import math
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from common import IMG_EXTS, ROOT, class_names_from_data_yaml, label_path_for, read_yolo_labels


def cut_object(img, box, pad=0.15):
    """Cut the drone out of a render: pixels that differ from the plain render sky = drone."""
    H, W = img.shape[:2]
    xc, yc, w, h = box
    x0, x1 = int((xc - w / 2 * (1 + pad)) * W), int((xc + w / 2 * (1 + pad)) * W)
    y0, y1 = int((yc - h / 2 * (1 + pad)) * H), int((yc + h / 2 * (1 + pad)) * H)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    crop = img[y0:y1, x0:x1].astype(np.float32)
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    bg = np.median(border, axis=0)
    if np.linalg.norm(border - bg, axis=1).mean() > 20:   # busy render background -> unreliable cut
        return None
    diff = np.linalg.norm(crop - bg, axis=2)
    thr = max(18.0, np.percentile(np.linalg.norm(border - bg, axis=1), 95) * 1.5)
    mask = cv2.morphologyEx((diff > thr).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lab == big).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    x0m, x1m, y0m, y1m = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    m = mask[y0m:y1m, x0m:x1m]
    bw, bh = x1m - x0m, y1m - y0m
    # shape checks: a wing shape is neither a solid bar/blob (high solidity) nor a sliver
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    solidity = cv2.contourArea(c) / hull if hull > 0 else 1
    fill = m.mean()
    full_box = (crop.shape[0] * crop.shape[1]) / (1 + pad) ** 2
    if not (0.15 < fill < 0.75) or not (0.3 < bw / bh < 3.5) or not (0.3 < solidity < 0.9) \
            or bw * bh < 0.35 * full_box:
        return None
    return crop[y0m:y1m, x0m:x1m].astype(np.uint8), m.astype(np.float32)


def sky_score(img):
    """Share of the upper 60% of the image that is smooth, bright (sky-like) and not black border."""
    h = int(img.shape[0] * 0.6)
    g = cv2.cvtColor(img[:h], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (320, max(1, int(320 * h / img.shape[1]))))
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F))
    return ((lap < 5) & (g > 60)).mean()


def degrade(img, rng):
    """Camera effects on the WHOLE image so the pasted drone and the sky look alike."""
    if rng.random() < 0.5:
        f = rng.uniform(0.55, 0.9)
        h, w = img.shape[:2]
        img = cv2.resize(cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA), (w, h))
    if rng.random() < 0.3:
        k = 3
        ker = np.zeros((k, k), np.float32)
        if rng.random() < 0.5:
            ker[1, :] = 1
        else:
            ker[:, 1] = 1
        img = cv2.filter2D(img, -1, ker / ker.sum())
    if rng.random() < 0.5:
        img = np.clip(img.astype(np.float32) + np.random.normal(0, rng.uniform(1, 4), img.shape), 0, 255).astype(np.uint8)
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(45, 90)])[1], cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", nargs="+", default=["data/raw/shahed136-detect"])
    ap.add_argument("--render-class", nargs="+", default=["1", "shahed"])
    ap.add_argument("--backgrounds", default="data/merged/train/images", help="TRAINING images only")
    ap.add_argument("--bg-prefix", nargs="+", default=["birds_evilsumrak__", "mixed_eyjnj4__"])
    ap.add_argument("--bg-name", default=r"(mp4|video|UAV_)", help="only real sky VIDEO frames")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--min-size", type=int, default=8)
    ap.add_argument("--max-size", type=int, default=48)
    ap.add_argument("--min-sky", type=float, default=0.7)
    ap.add_argument("--max-objects", type=int, default=6, help="skip backgrounds with more labelled objects (flocks)")
    ap.add_argument("--margin", type=int, default=20)
    ap.add_argument("--out", default="data/raw/composites")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # 1. cut out drones
    objects, tried = [], 0
    for rdir in args.renders:
        rdir = ROOT / rdir
        names = class_names_from_data_yaml(rdir)
        ids = {i for i, n in enumerate(names) if n in args.render_class}
        files = sorted(p for p in rdir.rglob("*") if p.suffix.lower() in IMG_EXTS)
        rng.shuffle(files)
        for p in files[:2500]:
            boxes = [b for b in read_yolo_labels(label_path_for(p)) if b[0] in ids]
            if not boxes:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            tried += 1
            o = cut_object(img, boxes[0][1:])
            if o is not None:
                objects.append(o)
    print(f"cut-outs: {len(objects)} good of {tried} tried (bad shapes rejected)")
    if not objects:
        raise SystemExit("No drones could be cut out - check --renders / --render-class")

    # 2. backgrounds: real sky VIDEO frames from the TRAINING split only
    bdir = ROOT / args.backgrounds
    cand = [p for p in sorted(bdir.iterdir()) if p.suffix.lower() in IMG_EXTS
            and any(p.name.startswith(pre) for pre in args.bg_prefix) and re.search(args.bg_name, p.name, re.I)]
    bgs = []
    for p in cand:
        labs = read_yolo_labels(p.parent.parent / "labels" / (p.stem + ".txt"))
        if len(labs) > args.max_objects or any(w * h > 0.02 for _, _, _, w, h in labs):
            continue
        im = cv2.imread(str(p))
        if im is None or min(im.shape[:2]) < 300:
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        speckle = (np.abs(g.astype(np.int16) - cv2.medianBlur(g, 3).astype(np.int16)) > 35).mean()
        if speckle > 0.002:                                  # salt-and-pepper noise added by the dataset
            continue
        if sky_score(im) >= args.min_sky:
            bgs.append(p)
    print(f"backgrounds: {len(bgs)} open-sky video frames (of {len(cand)} video frames in train)")
    if len(bgs) < 20:
        raise SystemExit("Too few backgrounds - try --min-sky 0.6")

    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "train" / "images").mkdir(parents=True)
    (out / "train" / "labels").mkdir(parents=True)
    merged_names = class_names_from_data_yaml(bdir.parent.parent)
    (out / "data.yaml").write_text(yaml.safe_dump({"names": merged_names}))
    shahed_id = merged_names.index("shahed")

    made, sizes, zooms, wholes = 0, [], [], []
    for i in range(args.n):
        bp = rng.choice(bgs)
        bg = cv2.imread(str(bp))
        H, W = bg.shape[:2]
        if W > 1280:
            bg = cv2.resize(bg, (1280, int(H * 1280 / W)), interpolation=cv2.INTER_AREA)
            H, W = bg.shape[:2]
        keep = read_yolo_labels(bp.parent.parent / "labels" / (bp.stem + ".txt"))
        M = args.margin
        taken = [((xc - w / 2) * W - M, (yc - h / 2) * H - M, (xc + w / 2) * W + M, (yc + h / 2) * H + M)
                 for _, xc, yc, w, h in keep]
        new = []
        for _ in range(1 if rng.random() < 0.85 else 2):
            obj, m = rng.choice(objects)
            size = int(math.exp(rng.uniform(math.log(args.min_size), math.log(args.max_size))))
            s = size / max(obj.shape[:2])
            ow, oh = max(4, int(obj.shape[1] * s)), max(4, int(obj.shape[0] * s))
            o = cv2.resize(obj, (ow, oh), interpolation=cv2.INTER_AREA)
            mm = cv2.resize(m, (ow, oh), interpolation=cv2.INTER_AREA)
            if rng.random() < 0.5:
                o, mm = o[:, ::-1].copy(), mm[:, ::-1].copy()
            for _try in range(50):        # plain sky, upper 55%, away from edges and other objects
                x = rng.randint(M, W - ow - M)
                y = rng.randint(M, max(M + 1, int(H * 0.55) - oh))
                b = (x, y, x + ow, y + oh)
                if any(not (b[2] < t[0] or b[0] > t[2] or b[3] < t[1] or b[1] > t[3]) for t in taken):
                    continue
                region = bg[y - M // 2:y + oh + M // 2, x - M // 2:x + ow + M // 2]
                g = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
                b_, g_, r_ = region.reshape(-1, 3).mean(0)
                if g.mean() > 70 and g.std() < 10 and b_ >= r_ - 10:   # sky is blue or grey, not red/orange
                    break
            else:
                continue
            patch = bg[y:y + oh, x:x + ow].astype(np.float32)
            local = patch.reshape(-1, 3).mean(0)
            # DARK silhouette: keep the drone's shading, but scale it far below the sky brightness,
            # nearly colourless, with a touch of the sky colour (haze) for small/far drones
            gray = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY).astype(np.float32)
            inside = mm > 0.5
            ref = gray[inside].mean() if inside.any() else gray.mean()
            rel = gray / max(ref, 1.0)                        # shading relative to the drone's own average (~1.0)
            rel = 1.0 + (rel - 1.0) * rng.uniform(0.15, 0.4)  # keep only a little of the shading
            target = local.mean() * rng.uniform(0.2, 0.55)    # drone = 20-55% as bright as the sky around it
            of = np.clip(rel * target, 0, 255)[..., None] * np.ones(3, np.float32)
            haze = rng.uniform(0.0, 0.35) * min(1.0, 20 / size)
            of = of * (1 - haze) + local * haze
            if of[inside].mean() >= local.mean() * 0.8 if inside.any() else True:
                continue                                      # never paste a bright drone
            sigma = rng.uniform(0.3, 0.9)
            a = cv2.GaussianBlur(mm, (0, 0), sigma)[..., None]
            of = cv2.GaussianBlur(of.astype(np.float32), (0, 0), sigma)
            bg[y:y + oh, x:x + ow] = np.clip(patch * (1 - a) + of * a, 0, 255).astype(np.uint8)
            ys, xs = np.nonzero(a[..., 0] > 0.3)
            if len(xs) == 0:
                continue
            bx0, bx1, by0, by1 = x + xs.min(), x + xs.max() + 1, y + ys.min(), y + ys.max() + 1
            taken.append((bx0 - M, by0 - M, bx1 + M, by1 + M))
            new.append((shahed_id, (bx0 + bx1) / 2 / W, (by0 + by1) / 2 / H, (bx1 - bx0) / W, (by1 - by0) / H))
            sizes.append(max(bx1 - bx0, by1 - by0))
        if not new:
            continue
        img = degrade(bg, rng)
        name = f"comp_{i:05d}"
        cv2.imwrite(str(out / "train" / "images" / f"{name}.jpg"), img)
        lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in list(keep) + new]
        (out / "train" / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
        made += 1
        # preview material
        if len(zooms) < 24:
            _, xc, yc, w, h = new[0]
            cx, cy, r = int(xc * W), int(yc * H), int(max(w * W, h * H) * 1.5) + 12
            z = img[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
            zooms.append(cv2.resize(z, (160, 160), interpolation=cv2.INTER_NEAREST))
        if len(wholes) < 8:
            t = img.copy()
            for _, xc, yc, w, h in new:
                cv2.circle(t, (int(xc * W), int(yc * H)), int(max(w * W, h * H)) + 25, (0, 0, 255), 3)
            wholes.append(cv2.resize(t, (320, 180)))

    sizes = np.array(sizes)
    print(f"made {made} composite images with {len(sizes)} Shaheds "
          f"(size px: median {np.median(sizes):.0f}, <16px {100 * (sizes < 16).mean():.0f}%, "
          f"16-32px {100 * ((sizes >= 16) & (sizes < 32)).mean():.0f}%)")
    prev = ROOT / "report" / "inspect"
    prev.mkdir(parents=True, exist_ok=True)
    row = lambda tiles, n: np.hstack(tiles[:n] + [np.zeros_like(tiles[0])] * (n - len(tiles[:n])))
    zsheet = np.vstack([row(zooms[r:r + 8], 8) for r in range(0, 24, 8)])
    wsheet = np.vstack([row(wholes[r:r + 4], 4) for r in range(0, 8, 4)])
    wsheet = cv2.resize(wsheet, (zsheet.shape[1], int(wsheet.shape[0] * zsheet.shape[1] / wsheet.shape[1])))
    cv2.putText(zsheet, "zoomed drones", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(prev / "composites_preview.jpg"), np.vstack([zsheet, wsheet]))
    print(f"saved to {out}\nPREVIEW (look at it!): {prev / 'composites_preview.jpg'}")


if __name__ == "__main__":
    main()
