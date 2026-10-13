"""
Fuse human boxes, teacher boxes and synthetic walls into one YOLO dataset (training/data/yolo).

Rules (all counted in the printed report):
- window/door: human boxes always kept. A teacher window/door box is added when its score is high and it
  does not overlap a human box. This adds recall where annotators skipped objects.
- ac_unit, meter_panel, pipe, grill: teacher boxes only, per-class score threshold, size and shape limits.
  A window or door mistaken for an AC or a meter is dropped when it overlaps a human box.
- A photo with uncertain teacher proposals (score in the grey zone just under the threshold) for a rare
  class is left out of training, so the model is not taught that an object is background.
- Test photos use the same rules, so the rare-class numbers measured on them are numbers against the
  teacher, not against people. The report keeps those separate.

    training/gpu/.venv/Scripts/python.exe training/gpu/fuse.py --synth 1200
"""

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from wall_synth import generate_wall, to_yolo_lines  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
POOL = os.path.join(ROOT, "training", "data", "pool")
OUT = os.path.join(ROOT, "training", "data", "yolo")
NAMES = ["window", "door", "ac_unit", "meter_panel", "pipe", "grill"]

ADD_WD = 0.30  # teacher window/door boxes added at or above this score
THRESH = {2: 0.30, 3: 0.30, 4: 0.30, 5: 0.30}
GREY = {2: 0.24, 3: 0.24, 4: 0.24, 5: 0.24}
AREA = {2: (0.002, 0.08), 3: (0.001, 0.05), 4: (0.0008, 0.20), 5: (0.002, 0.25)}  # share of the image
MAX_PER_CLASS = 8  # more than this in one photo is a misfire pattern (balcony rows, for example)


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(items, thresh=0.5):
    items = sorted(items, key=lambda t: -t[1])
    keep = []
    for item in items:
        if all(iou(item[2:], k[2:]) < thresh for k in keep):
            keep.append(item)
    return keep


def read_human(path):
    boxes = []
    for line in open(path):
        cls, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        boxes.append((int(cls), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def fuse_image(human, proposals):
    """Returns (boxes [(cls, x1, y1, x2, y2)], usable, stats)."""
    stats = {"teacher_wd_added": 0, "teacher_rare_added": 0, "dropped_misfire": 0}
    boxes = list(human)
    human_boxes = [b[1:] for b in human]
    by_class = {c: [p for p in proposals if p[0] == c] for c in range(6)}

    for cls in (0, 1):
        for _, score, x1, y1, x2, y2 in nms(by_class[cls]):
            area = (x2 - x1) * (y2 - y1)
            if score < ADD_WD or area > 0.6 or area < 0.0008:
                continue
            if any(iou((x1, y1, x2, y2), hb) > 0.3 for hb in human_boxes):
                continue
            boxes.append((cls, x1, y1, x2, y2))
            stats["teacher_wd_added"] += 1

    usable = True
    window_door = [b[1:] for b in boxes if b[0] in (0, 1)]
    human_classes = {b[0] for b in human}
    for cls in (2, 3, 4, 5):
        if cls in human_classes:
            continue  # people labelled this class in this photo: the teacher adds nothing and cannot veto the photo
        cands = nms(by_class[cls])
        strong = []
        for _, score, x1, y1, x2, y2 in cands:
            area = (x2 - x1) * (y2 - y1)
            lo, hi = AREA[cls]
            if area < lo or area > hi:
                continue
            w, h = x2 - x1, y2 - y1
            if cls == 4 and max(w, h) / max(min(w, h), 1e-6) < 3.5:
                continue  # a pipe is long and thin
            if cls in (2, 3) and any(iou((x1, y1, x2, y2), wb) > 0.5 for wb in window_door):
                continue
            if score >= THRESH[cls]:
                strong.append((cls, x1, y1, x2, y2))
            elif score >= GREY[cls]:
                usable = False  # unsure: leave the photo out of training instead of calling it background
        if len(strong) > MAX_PER_CLASS:
            stats["dropped_misfire"] += len(strong)
            usable = False
            continue
        boxes.extend(strong)
        stats["teacher_rare_added"] += len(strong)
    return boxes, usable, stats


def write(split, image_id, image_path, boxes):
    folder_i = os.path.join(OUT, split, "images")
    folder_l = os.path.join(OUT, split, "labels")
    os.makedirs(folder_i, exist_ok=True)
    os.makedirs(folder_l, exist_ok=True)
    shutil.copyfile(image_path, os.path.join(folder_i, image_id + ".jpg"))
    with open(os.path.join(folder_l, image_id + ".txt"), "w") as f:
        for cls, x1, y1, x2, y2 in boxes:
            f.write(f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}\n")


def _write_synth(job):
    split, i, seed = job
    image, boxes = generate_wall(seed)
    h, w = image.shape[:2]
    folder_i = os.path.join(OUT, split, "images")
    folder_l = os.path.join(OUT, split, "labels")
    os.makedirs(folder_i, exist_ok=True)
    os.makedirs(folder_l, exist_ok=True)
    cv2.imwrite(os.path.join(folder_i, f"synth_{i:05d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(os.path.join(folder_l, f"synth_{i:05d}.txt"), "w") as f:
        f.write("\n".join(to_yolo_lines(boxes, w, h)) + ("\n" if boxes else ""))
    counts = {}
    for b in boxes:
        counts[NAMES[b[0]]] = counts.get(NAMES[b[0]], 0) + 1
    return split, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synth", type=int, default=1200, help="synthetic training walls")
    parser.add_argument("--synth-val", type=int, default=0, help="synthetic walls in the validation set; 0 so the best epoch is chosen on real photos only")
    parser.add_argument("--synth-test", type=int, default=250)
    args = parser.parse_args(argv)

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    index = json.load(open(os.path.join(POOL, "splits.json")))
    teacher = json.load(open(os.path.join(POOL, "teacher.json")))
    report = {"real": {}, "left_out": {}}
    totals = {}
    for image_id, rec in sorted(index.items()):
        if image_id not in teacher:
            continue
        split = rec["split"]
        human = read_human(os.path.join(POOL, "labels_human", image_id + ".txt"))
        boxes, usable, stats = fuse_image(human, teacher[image_id])
        if not usable and split == "train":
            report["left_out"][split] = report["left_out"].get(split, 0) + 1
            continue
        write(split, image_id, os.path.join(POOL, "images", image_id + ".jpg"), boxes)
        for box in boxes:
            key = (split, NAMES[box[0]], "human" if box in human else "teacher")
            totals[key] = totals.get(key, 0) + 1
        report["real"][split] = report["real"].get(split, 0) + 1

    jobs = [("train", i, 100000 + i) for i in range(args.synth)]
    jobs += [("val", i, 200000 + i) for i in range(args.synth_val)]
    jobs += [("synth_test", i, 300000 + i) for i in range(args.synth_test)]
    synth_counts = {}
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) - 2)) as pool:
        for split, counts in pool.map(_write_synth, jobs, chunksize=16):
            for name, n in counts.items():
                synth_counts[(split, name)] = synth_counts.get((split, name), 0) + n

    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {OUT.replace(chr(92), '/')}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n")
        for i, name in enumerate(NAMES):
            f.write(f"  {i}: {name}\n")

    print("real photos kept per split:", report["real"])
    print("real training photos left out (uncertain or misfiring teacher boxes):", report["left_out"].get("train", 0))
    print("real boxes (split, class, origin):")
    for key in sorted(totals):
        print("  ", key, totals[key])
    print("synthetic boxes:", dict(sorted(synth_counts.items())))
    json.dump({"real": report, "boxes": {"|".join(k): v for k, v in totals.items()},
               "synthetic": {"|".join(k): v for k, v in synth_counts.items()}},
              open(os.path.join(OUT, "report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
