"""
Open-vocabulary "teacher" that proposes boxes for the classes no public dataset labels well:
ac_unit, meter_panel, pipe, grill (and window / door, only used to cross-check human labels).

The teacher is YOLO-World (ultralytics). Its boxes are machine labels, not human labels: every report
that uses them says so.

    training/gpu/.venv/Scripts/python.exe training/gpu/teacher.py --images samples --out training/data/teacher_test
"""

import argparse
import os
import sys

import cv2

# prompt -> app class id (obstacle_detector.CLASS_NAMES order)
PROMPTS = {
    "window": 0,
    "door": 1,
    "air conditioner": 2,
    "outdoor air conditioning unit": 2,
    "electric meter box": 3,
    "electricity meter": 3,
    "pipe": 4,
    "drain pipe": 4,
    "window grill": 5,
    "metal grille": 5,
}
CLASS_NAMES = ["window", "door", "ac_unit", "meter_panel", "pipe", "grill"]


def load(weights="yolov8l-worldv2.pt"):
    from ultralytics import YOLOWorld

    model = YOLOWorld(weights)
    model.set_classes(list(PROMPTS))
    return model


def propose(model, image_bgr, conf=0.05):
    """Returns [(class_id, score, x1, y1, x2, y2)] in pixels."""
    result = model.predict(image_bgr, conf=conf, imgsz=640, verbose=False)[0]
    prompts = list(PROMPTS.values())
    out = []
    for box, score, idx in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
        out.append((prompts[int(idx)], float(score), *box))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--conf", type=float, default=0.1)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    model = load()
    colors = [(0, 200, 0), (200, 0, 0), (0, 0, 255), (0, 200, 200), (200, 0, 200), (0, 128, 255)]
    for name in sorted(os.listdir(args.images)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image = cv2.imread(os.path.join(args.images, name))
        if image is None:
            continue
        boxes = propose(model, image, args.conf)
        for cls, score, x1, y1, x2, y2 in boxes:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[cls], 2)
            cv2.putText(image, f"{CLASS_NAMES[cls]} {score:.2f}", (int(x1), max(12, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], 1)
        cv2.imwrite(os.path.join(args.out, name), image)
        counts = {n: sum(1 for b in boxes if b[0] == i) for i, n in enumerate(CLASS_NAMES)}
        print(name, counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
