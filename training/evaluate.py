"""
Evaluate the exported ONNX detector on held-out YOLO-format images, using the same
thresholds the app uses at its default sensitivity.

    python training/evaluate.py --model models/apr_obstacles.onnx --images training/data/merged/test/images

Reports per class recall and precision (IoU >= 0.5, class must match) plus, for every class,
the count of ground-truth objects that got no box at all (neither classified nor unverified).
Ultralytics' per-class mAP50 for the same split is printed by training/train.py.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from obstacle_detector import CLASS_NAMES, ObstacleDetector, thresholds_for_sensitivity  # noqa: E402

IOU_MATCH = 0.5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGETS = {"window": (0.90, 0.85), "door": (0.90, 0.85)}  # class -> (min recall, min precision)

Box = Tuple[float, float, float, float]


def read_yolo_labels(path, img_w: int, img_h: int) -> List[Tuple[int, float, float, float, float]]:
    path = Path(path)
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:])
        boxes.append((cls, (cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h))
    return boxes


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(gt_boxes, det_boxes, det_scores, iou_threshold: float = IOU_MATCH) -> List[Tuple[int, int]]:
    """Match detections, highest score first, each to the best still-unmatched ground truth."""
    matches, used = [], set()
    for d in sorted(range(len(det_boxes)), key=lambda i: -det_scores[i]):
        best, best_iou = None, 0.0
        for g in range(len(gt_boxes)):
            if g in used:
                continue
            value = iou(gt_boxes[g], det_boxes[d])
            if value >= iou_threshold and value > best_iou:
                best, best_iou = g, value
        if best is not None:
            used.add(best)
            matches.append((best, d))
    return matches


def evaluate(detector, images_dir, labels_dir, class_names, min_score: float) -> Dict[str, Dict[str, int]]:
    stats = {name: {"gt": 0, "tp": 0, "fp": 0, "fn": 0, "found_any": 0} for name in class_names}
    for image_path in sorted(Path(images_dir).iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        truths = read_yolo_labels(Path(labels_dir) / f"{image_path.stem}.txt", width, height)
        dets = detector.detect(image, min_score)
        det_boxes = [(d.x1, d.y1, d.x2, d.y2) for d in dets]
        det_scores = [d.confidence for d in dets]

        found = {g for g, _ in greedy_match([t[1:] for t in truths], det_boxes, det_scores)}
        for class_id, name in enumerate(class_names):
            gt_idx = [i for i, t in enumerate(truths) if t[0] == class_id]
            det_idx = [i for i, d in enumerate(dets) if d.class_name == name]
            matches = greedy_match(
                [truths[i][1:] for i in gt_idx], [det_boxes[i] for i in det_idx], [det_scores[i] for i in det_idx]
            )
            entry = stats[name]
            entry["gt"] += len(gt_idx)
            entry["tp"] += len(matches)
            entry["fn"] += len(gt_idx) - len(matches)
            entry["fp"] += len(det_idx) - len(matches)
            entry["found_any"] += sum(1 for i in gt_idx if i in found)
    return stats


def summarize(stats) -> Dict[str, Dict[str, Optional[float]]]:
    summary = {}
    for name, s in stats.items():
        summary[name] = {
            "recall": s["tp"] / s["gt"] if s["gt"] else None,
            "precision": s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else None,
            "agnostic_recall": s["found_any"] / s["gt"] if s["gt"] else None,
        }
    return summary


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/apr_obstacles.onnx")
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", help="defaults to the sibling labels folder of --images")
    parser.add_argument("--sensitivity", type=float, default=0.5)
    args = parser.parse_args()

    labels = args.labels or str(Path(args.images).parent / "labels")
    review_threshold, _ = thresholds_for_sensitivity(args.sensitivity)
    stats = evaluate(ObstacleDetector(model_path=args.model), args.images, labels, CLASS_NAMES, review_threshold)
    summary = summarize(stats)

    print(f"{'class':<12}{'gt':>5}{'recall':>9}{'precision':>11}{'no box at all':>15}")
    for name in CLASS_NAMES:
        s, m = stats[name], summary[name]
        print(f"{name:<12}{s['gt']:>5}{_fmt(m['recall']):>9}{_fmt(m['precision']):>11}{s['gt'] - s['found_any']:>15}")

    failed = False
    for name, (min_recall, min_precision) in TARGETS.items():
        m = summary[name]
        if m["recall"] is None:
            print(f"{name}: no ground truth in this set, target not checked")
        elif m["recall"] < min_recall or (m["precision"] or 0.0) < min_precision:
            failed = True
            print(f"{name}: below target (recall >= {min_recall}, precision >= {min_precision})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
