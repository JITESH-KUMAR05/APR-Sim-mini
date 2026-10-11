"""
Build one pool of real photos with human-made window and door boxes from the staged downloads
(Open Images, CMP Facade, ADE20K). Every image is resized to at most 960 px on its long side and gets a
split (train 80, val 10, test 10) from a hash of its id, so a photo is on one side only.

    training/gpu/.venv/Scripts/python.exe training/gpu/pool.py

Output: training/data/pool/images/<id>.jpg, labels_human/<id>.txt (YOLO, class 0 window, 1 door), splits.json
"""

import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "training", "data")
POOL = os.path.join(DATA, "pool")
MAX_SIDE = 960


def split_of(image_id):
    bucket = int(hashlib.sha1(image_id.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def save(image_id, image_bgr, boxes, source):
    """boxes: [(cls, cx, cy, w, h)] normalised. Returns True if saved."""
    h, w = image_bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(POOL, "images", f"{image_id}.jpg"), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(os.path.join(POOL, "labels_human", f"{image_id}.txt"), "w") as f:
        for cls, cx, cy, bw, bh in boxes:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return True


def from_open_images(index):
    base = os.path.join(DATA, "staging", "oi")
    count = 0
    for split in ("validation", "test"):
        folder = os.path.join(base, "images", split)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            image_id = name[:-4]
            image = cv2.imread(os.path.join(folder, name))
            if image is None:
                continue
            boxes = []
            with open(os.path.join(base, "labels", split, image_id + ".txt")) as f:
                for line in f:
                    cls, cx, cy, w, h = line.split()
                    boxes.append((int(cls), float(cx), float(cy), float(w), float(h)))
            if not boxes:
                continue
            new_id = f"oi_{image_id}"
            save(new_id, image, boxes, "oi")
            index[new_id] = {"source": "oi", "split": split_of(new_id)}
            count += 1
    return count


def from_cmp(index):
    folder = os.path.join(DATA, "staging", "cmp", "base")
    count = 0
    if not os.path.isdir(folder):
        return 0
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".jpg"):
            continue
        stem = name[:-4]
        image = cv2.imread(os.path.join(folder, name))
        if image is None:
            continue
        root = ET.fromstring("<r>" + open(os.path.join(folder, stem + ".xml"), encoding="utf-8").read() + "</r>")
        boxes = []
        for obj in root.findall("object"):
            label = obj.findtext("labelname")
            cls = {"window": 0, "door": 1}.get(label)
            if cls is None:
                continue
            xs = [float(v.text) for v in obj.find("points").findall("x")]
            ys = [float(v.text) for v in obj.find("points").findall("y")]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            if x2 - x1 < 0.005 or y2 - y1 < 0.005:
                continue
            boxes.append((cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        new_id = f"cmp_{stem}"
        save(new_id, image, boxes, "cmp")
        index[new_id] = {"source": "cmp", "split": split_of(new_id)}
        count += 1
    return count


# ADE20K SceneParsing class ids (1-based in the annotation PNGs): building 2, windowpane 9, door 15
ADE_BUILDING, ADE_WINDOW, ADE_DOOR = 2, 9, 15


def from_ade(index):
    path = os.path.join(DATA, "raw", "ade20k.zip")
    if not os.path.exists(path):
        return 0
    count = 0
    with zipfile.ZipFile(path) as z:
        names = {n for n in z.namelist()}
        for name in sorted(n for n in names if n.startswith("ADEChallengeData2016/images/") and n.endswith(".jpg")):
            ann_name = name.replace("/images/", "/annotations/").replace(".jpg", ".png")
            if ann_name not in names:
                continue
            ann = cv2.imdecode(np.frombuffer(z.read(ann_name), np.uint8), cv2.IMREAD_UNCHANGED)
            if ann is None or (ann == ADE_BUILDING).mean() < 0.25:
                continue  # only photos where a building fills a good part of the frame
            image = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            h, w = ann.shape[:2]
            boxes = []
            for cls, value in ((0, ADE_WINDOW), (1, ADE_DOOR)):
                mask = (ann == value).astype(np.uint8)
                n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                for i in range(1, n):
                    x, y, bw, bh, area = stats[i]
                    if area < 0.0004 * w * h or bw < 0.02 * w or bh < 0.02 * h:
                        continue
                    if area / float(bw * bh) < 0.55:
                        continue  # a ragged blob, not one window
                    boxes.append((cls, (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
            if not boxes:
                continue
            new_id = "ade_" + re.sub(r"\W", "_", os.path.basename(name)[:-4])
            save(new_id, image, boxes, "ade")
            index[new_id] = {"source": "ade", "split": split_of(new_id)}
            count += 1
    return count


def main(argv=None):
    sources = argv if argv else ["oi", "cmp", "ade"]
    os.makedirs(os.path.join(POOL, "images"), exist_ok=True)
    os.makedirs(os.path.join(POOL, "labels_human"), exist_ok=True)
    index = {}
    index_path = os.path.join(POOL, "splits.json")
    if os.path.exists(index_path):
        index = json.load(open(index_path))
    if "oi" in sources:
        print("open images", from_open_images(index), flush=True)
    if "cmp" in sources:
        print("cmp", from_cmp(index), flush=True)
    if "ade" in sources:
        print("ade", from_ade(index), flush=True)
    json.dump(index, open(index_path, "w"))
    per = {}
    for rec in index.values():
        per[(rec["source"], rec["split"])] = per.get((rec["source"], rec["split"]), 0) + 1
    print(per)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
