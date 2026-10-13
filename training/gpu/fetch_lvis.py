"""
Take real photos of air conditioners and pipes with human boxes from LVIS (annotations CC BY 4.0, images are
COCO photos from Flickr with their own licences, see training/DATA_SOURCES.md).

    training/gpu/.venv/Scripts/python.exe training/gpu/fetch_lvis.py

Output: training/data/staging/lvis/{images,labels}/<id>.{jpg,txt}. Class ids follow obstacle_detector.CLASS_NAMES:
air_conditioner becomes ac_unit (2), pipe becomes pipe (4) but only long thin pipes (aspect ratio of at least 3),
because most LVIS pipes are plumbing parts. Other objects in these photos are not labelled here (LVIS has no
door or window class), so fuse.py adds window and door boxes from the teacher where it is confident.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import ijson
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "training", "data", "raw")
OUT = os.path.join(ROOT, "training", "data", "staging", "lvis")
WANT = {"air_conditioner": 2, "pipe": 4}
MIN_PIPE_ASPECT = 3.0
MIN_SIDE_FRACTION = 0.01


def collect(path):
    with open(path, "rb") as f:
        categories = {c["id"]: c["name"] for c in ijson.items(f, "categories.item", use_float=True)}
    ids = {cid: WANT[name] for cid, name in categories.items() if name in WANT}
    boxes = {}
    with open(path, "rb") as f:
        for a in ijson.items(f, "annotations.item", use_float=True):
            cls = ids.get(a["category_id"])
            if cls is None:
                continue
            boxes.setdefault(a["image_id"], []).append((cls, a["bbox"]))  # x, y, w, h in pixels
    wanted = {}
    with open(path, "rb") as f:
        for im in ijson.items(f, "images.item", use_float=True):
            if im["id"] in boxes:
                wanted[im["id"]] = (im["coco_url"], im["width"], im["height"])
    return boxes, wanted


def convert(boxes, width, height):
    lines = []
    for cls, (x, y, w, h) in boxes:
        if w < MIN_SIDE_FRACTION * width or h < MIN_SIDE_FRACTION * height:
            continue
        if cls == 4 and max(w, h) / max(min(w, h), 1e-6) < MIN_PIPE_ASPECT:
            continue
        lines.append(f"{cls} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}")
    return lines


def download(job):
    image_id, url, lines = job
    image_path = os.path.join(OUT, "images", f"lvis_{image_id}.jpg")
    label_path = os.path.join(OUT, "labels", f"lvis_{image_id}.txt")
    if os.path.exists(image_path) and os.path.exists(label_path):
        return True
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200 or len(response.content) < 2000:
            return False
    except requests.RequestException:
        return False
    with open(image_path, "wb") as f:
        f.write(response.content)
    with open(label_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return True


def main():
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels"), exist_ok=True)
    jobs = []
    for name in ("lvis_v1_val.json", "lvis_v1_train.json"):
        path = os.path.join(RAW, name)
        if not os.path.exists(path):
            print("missing", path)
            continue
        boxes, wanted = collect(path)
        for image_id, (url, w, h) in wanted.items():
            lines = convert(boxes[image_id], w, h)
            if lines:
                jobs.append((image_id, url, lines))
        print(name, len(wanted), "photos with the wanted classes,", len(jobs), "usable so far", flush=True)
    with ThreadPoolExecutor(16) as pool:
        done = sum(pool.map(download, jobs))
    print(f"{done} of {len(jobs)} photos downloaded")
    counts = {2: 0, 4: 0}
    for name in os.listdir(os.path.join(OUT, "labels")):
        for line in open(os.path.join(OUT, "labels", name)):
            counts[int(line.split()[0])] += 1
    print("boxes:", {"ac_unit": counts[2], "pipe": counts[4]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
