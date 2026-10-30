"""STEP 1 - Merge all datasets into one, with a leak-free split.

  python scripts/step1_merge_datasets.py            # uses sources.yaml
  python scripts/step1_merge_datasets.py --out data/merged

Why: frames from the same video look almost identical. If some go to train and some to test,
your accuracy looks great on paper but fails on real cameras. Here every video/original image
stays in ONE split, and the test set contains REAL images only.
"""
import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from common import IMG_EXTS, ROOT, class_names_from_data_yaml, group_key, label_path_for, load_yaml, read_yolo_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(ROOT / "sources.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data/merged"))
    args = ap.parse_args()

    cfg = load_yaml(args.sources)
    classes = cfg["classes"]
    cls_id = {n: i for i, n in enumerate(classes)}
    sp = cfg.get("split", {})
    val_f, test_f, seed = sp.get("val", 0.15), sp.get("test", 0.15), sp.get("seed", 42)
    rng = random.Random(seed)

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val", "test"):
        (out / s / "images").mkdir(parents=True)
        (out / s / "labels").mkdir(parents=True)

    report = []
    global_assign = {}   # group key -> split, shared by ALL sources (same video never in two splits)
    for src in cfg["sources"]:
        sdir = (ROOT / src["path"]) if not Path(src["path"]).is_absolute() else Path(src["path"])
        if not sdir.exists():
            print(f"[skip] {src['name']}: folder not found -> {sdir}")
            continue
        src_names = class_names_from_data_yaml(sdir)
        cmap = src.get("class_map") or {}
        # source class id -> unified id (None = drop)
        remap = {}
        for i, n in enumerate(src_names):
            tgt = cmap.get(n)
            if tgt is not None and tgt not in cls_id:
                raise SystemExit(f"{src['name']}: '{n}' maps to '{tgt}', which is not in classes {classes}")
            remap[i] = cls_id.get(tgt) if tgt else None
        dropped = [n for i, n in enumerate(src_names) if remap.get(i) is None]
        if dropped:
            print(f"[info] {src['name']}: dropping classes not in class_map: {dropped}")

        imgs = sorted(p for p in sdir.rglob("*") if p.suffix.lower() in IMG_EXTS)
        # optional filters on the file name, so ONE folder can be split into two sources
        # (e.g. real frames named 0001_... vs renders with random names)
        if src.get("include"):
            imgs = [p for p in imgs if re.search(src["include"], p.name)]
        if src.get("exclude"):
            imgs = [p for p in imgs if not re.search(src["exclude"], p.name)]
        # optionally drop whole images that contain a given source class (e.g. render Shaheds
        # mixed into an otherwise real dataset) - better than keeping them with the label removed
        skip_names = set(src.get("skip_images_with") or [])
        if skip_names:
            skip_ids = {i for i, n in enumerate(src_names) if n in skip_names}
            before = len(imgs)
            imgs = [p for p in imgs if not any(c in skip_ids for c, *_ in read_yolo_labels(label_path_for(p)))]
            print(f"[info] {src['name']}: skipped {before - len(imgs)} images containing {sorted(skip_names)}")
        mode = src.get("group_by", "image")
        groups = defaultdict(list)
        for p in imgs:
            groups[group_key(p.stem, mode)].append(p)
        keys = sorted(groups)
        if len(imgs) >= 20 and len(keys) <= 3 and src.get("test", test_f) < 1.0:
            print(f"[warn] {src['name']}: {len(imgs)} images collapsed into only {len(keys)} group(s). "
                  f"Check file names, or use group_by: image for this source.")
        if mode == "video" and len(keys) == len(imgs) and len(imgs) > 20:
            print(f"[warn] {src['name']}: group_by: video found no frame numbers in the file names, "
                  f"so frames of one video may leak across splits. Look at a few file names.")
        rng.shuffle(keys)

        real = src.get("domain", "real") == "real"
        n = len(keys)
        s_test = src.get("test", test_f)   # per-source override, e.g. test: 1.0 = all to test
        s_val = src.get("val", val_f)
        n_test = int(round(n * s_test)) if real else 0
        n_val = min(n - n_test, max(1 if n > 1 else 0, int(round(n * s_val))))
        assign = {}
        for i, k in enumerate(keys):
            assign[k] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        # same group already placed by an earlier source (e.g. the same bird video in two datasets)
        # -> reuse that split, so nothing leaks between train and test across datasets
        reused = 0
        for k in keys:
            if k in global_assign:
                g = global_assign[k]
                assign[k] = "val" if (g == "test" and not real) else g
                reused += 1
            else:
                global_assign[k] = assign[k]
        if reused:
            print(f"[info] {src['name']}: {reused} group(s) also appear in an earlier source -> kept in the same split")

        counts = defaultdict(int)
        for k, plist in groups.items():
            split = assign[k]
            for p in plist:
                boxes = read_yolo_labels(label_path_for(p))
                lines = []
                for c, xc, yc, w, h in boxes:
                    u = remap.get(c)
                    if u is None:
                        continue
                    lines.append(f"{u} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                # unique name: source + relative path (avoids collisions)
                rel = "_".join(p.relative_to(sdir).with_suffix("").parts)
                name = f"{src['name']}__{rel}"
                shutil.copy2(p, out / split / "images" / f"{name}{p.suffix.lower()}")
                (out / split / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                counts[split] += 1
        report.append((src["name"], src.get("domain", "real"), len(imgs), n, dict(counts)))

    data_yaml = {"path": str(out.resolve()), "train": "train/images", "val": "val/images",
                 "test": "test/images", "names": {i: n for i, n in enumerate(classes)}}
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    print("\nsource           domain     images  groups  train/val/test")
    for name, dom, ni, ng, c in report:
        print(f"{name:<16} {dom:<10} {ni:>6}  {ng:>6}  {c.get('train',0)}/{c.get('val',0)}/{c.get('test',0)}")
    print(f"\nWrote {out/'data.yaml'}")
    if not any(r[4].get("test") for r in report):
        print("WARNING: test set is empty. Add at least one source with domain: real.")


if __name__ == "__main__":
    main()
