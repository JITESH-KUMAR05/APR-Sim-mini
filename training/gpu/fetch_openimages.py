"""
Download Open Images photos that contain windows or doors, with their human boxes, as YOLO labels.

    training/gpu/.venv/Scripts/python.exe training/gpu/fetch_openimages.py

Output: training/data/staging/oi/{images,labels}/<split>/<id>.{jpg,txt}. Class ids follow
obstacle_detector.CLASS_NAMES (window 0, door 1). Open Images annotations are CC BY 4.0, the
images carry CC BY 2.0 licences (credit the authors when publishing); see training/DATA_SOURCES.md.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "training", "data", "raw")
OUT = os.path.join(ROOT, "training", "data", "staging", "oi")
LABELS = {"/m/0d4v4": 0, "/m/02dgv": 1}  # Window, Door
BASE = "https://storage.googleapis.com/openimages/v5/{split}-annotations-bbox.csv"
IMAGE_URL = "https://s3.amazonaws.com/open-images-dataset/{split}/{image_id}.jpg"


def ensure_csv(split):
    path = os.path.join(RAW, f"oi_{split}_bbox.csv")
    if not os.path.exists(path):
        os.makedirs(RAW, exist_ok=True)
        with requests.get(BASE.format(split=split), stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(path, "wb") as f:
                for chunk in response.iter_content(1 << 20):
                    f.write(chunk)
    return path


BUILDING_LABELS = {"/m/0cgh4", "/m/03jm5", "/m/079cl", "/m/021sj1"}  # Building, House, Skyscraper, Office building


def building_images(split):
    """Image ids that annotators confirmed show a building. Car and bus windows are not wall obstacles."""
    path = os.path.join(RAW, f"oi_{split}_imagelabels.csv")
    if not os.path.exists(path):
        with requests.get(f"https://storage.googleapis.com/openimages/v5/{split}-annotations-human-imagelabels-boxable.csv",
                          timeout=120) as response:
            response.raise_for_status()
            with open(path, "wb") as f:
                f.write(response.content)
    ids = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["LabelName"] in BUILDING_LABELS and row["Confidence"] == "1":
                ids.add(row["ImageID"])
    return ids


def collect(split, limit_groups):
    boxes = defaultdict(list)
    buildings = building_images(split)
    with open(ensure_csv(split), newline="") as f:
        for row in csv.DictReader(f):
            cls = LABELS.get(row["LabelName"])
            if cls is None or row["IsGroupOf"] == "1" or row["ImageID"] not in buildings:
                continue
            x1, x2, y1, y2 = (float(row[k]) for k in ("XMin", "XMax", "YMin", "YMax"))
            boxes[row["ImageID"]].append((cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return boxes


def download(split, image_id, annotations):
    image_path = os.path.join(OUT, "images", split, f"{image_id}.jpg")
    label_path = os.path.join(OUT, "labels", split, f"{image_id}.txt")
    if os.path.exists(image_path) and os.path.exists(label_path):
        return True
    try:
        response = requests.get(IMAGE_URL.format(split=split, image_id=image_id), timeout=30)
        if response.status_code != 200 or len(response.content) < 2000:
            return False
    except requests.RequestException:
        return False
    with open(image_path, "wb") as f:
        f.write(response.content)
    with open(label_path, "w") as f:
        for cls, cx, cy, w, h in annotations:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=0, help="per split, 0 means all")
    args = parser.parse_args(argv)

    for split in args.splits:
        os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)
        boxes = collect(split, 0)
        ids = sorted(boxes)
        if args.max_images:
            ids = ids[: args.max_images]
        done = 0
        with ThreadPoolExecutor(args.workers) as pool:
            for ok in pool.map(lambda i: download(split, i, boxes[i]), ids):
                done += ok
        print(f"{split}: {done} of {len(ids)} images downloaded", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
