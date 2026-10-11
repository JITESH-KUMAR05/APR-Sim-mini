"""
Run the OWLv2 teacher over every pool image and keep its raw proposals (score >= 0.10) in
training/data/pool/teacher.json, normalised xyxy. Resumable: images already done are skipped.

    training/gpu/.venv/Scripts/python.exe training/gpu/run_teacher.py
"""

import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_owl import OwlTeacher  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
POOL = os.path.join(ROOT, "training", "data", "pool")


def main():
    path = os.path.join(POOL, "teacher.json")
    done = json.load(open(path)) if os.path.exists(path) else {}
    names = sorted(n for n in os.listdir(os.path.join(POOL, "images")) if n.endswith(".jpg"))
    todo = [n for n in names if n[:-4] not in done]
    print(f"{len(done)} done, {len(todo)} to do", flush=True)
    teacher = OwlTeacher()
    start = time.time()
    for i, name in enumerate(todo, 1):
        image = cv2.imread(os.path.join(POOL, "images", name))
        h, w = image.shape[:2]
        proposals = teacher.propose(image, threshold=0.10)
        done[name[:-4]] = [[c, round(s, 4), round(x1 / w, 5), round(y1 / h, 5), round(x2 / w, 5), round(y2 / h, 5)]
                           for c, s, x1, y1, x2, y2 in proposals]
        if i % 100 == 0:
            json.dump(done, open(path, "w"))
            print(f"{i}/{len(todo)} {(time.time() - start) / i:.2f}s per image", flush=True)
    json.dump(done, open(path, "w"))
    print("finished", len(done))


if __name__ == "__main__":
    main()
