"""
Compare the trained model with the classical fallback on the held-out real photos, in the terms that matter for
painting: is a human-labelled window or door covered by a keep-out zone (of any class, including "unverified"),
and how much of the photo does the detector mark as keep-out. Both go through the app's own GeometryEngine.

    .venv/Scripts/python.exe training/compare_detectors.py

Needs training/data/eval/real_human (written by training/eval_report.py).
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rover" / "tools"))
sys.path.insert(0, str(ROOT / "training"))

from demo_random_wall import _mask, coverage  # noqa: E402
from evaluate import read_yolo_labels  # noqa: E402
from geometry_engine import GeometryEngine  # noqa: E402
from obstacle_detector import ObstacleDetector  # noqa: E402

SCALE = 10.0  # one photo pixel is 10 mm: the grid in demo_random_wall then has one cell per pixel


def measure(engine, images, labels):
    found = planted = found_big = planted_big = 0
    share = []
    for path in sorted(images.glob("*.jpg")):
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        wall_w, wall_h = w * SCALE, h * SCALE
        truth = []
        for cls, x1, y1, x2, y2 in read_yolo_labels(labels / (path.stem + ".txt"), w, h):
            truth.append({"x": x1 * SCALE, "y": wall_h - y2 * SCALE, "w": (x2 - x1) * SCALE, "h": (y2 - y1) * SCALE,
                          "big": (x2 - x1) * (y2 - y1) / (w * h) >= 0.03})
        zones = engine.detect(image, wall_w, wall_h, 0.5)["obstacles"]
        everything = coverage(truth, zones, wall_w, wall_h)
        big = coverage([t for t in truth if t["big"]], zones, wall_w, wall_h)
        found, planted = found + everything["found"], planted + everything["planted"]
        found_big, planted_big = found_big + big["found"], planted_big + big["planted"]
        share.append(float(_mask(zones, wall_w, wall_h).mean()))
    return found / planted, found_big / planted_big, sum(share) / len(share), planted, planted_big


def main():
    images = ROOT / "training" / "data" / "eval" / "real_human" / "images"
    labels = ROOT / "training" / "data" / "eval" / "real_human" / "labels"
    model = GeometryEngine(detector=ObstacleDetector(model_path=str(ROOT / "models" / "apr_obstacles.onnx")))
    classical = GeometryEngine(detector=None)
    print("| detector | window/door boxes covered | boxes over 3% of the photo covered | share of the photo marked keep-out |")
    print("|---|---|---|---|")
    for name, engine in (("trained model v1", model), ("classical fallback", classical)):
        covered, covered_big, share, planted, planted_big = measure(engine, images, labels)
        print(f"| {name} | {covered:.2f} of {planted} | {covered_big:.2f} of {planted_big} | {share * 100:.0f}% |")


if __name__ == "__main__":
    main()
