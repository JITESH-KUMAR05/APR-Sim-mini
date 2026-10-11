"""
Open-vocabulary teacher based on OWLv2 (transformers). Same job as teacher.py, stronger and slower.

    training/gpu/.venv/Scripts/python.exe training/gpu/teacher_owl.py --images DIR --out DIR
"""

import argparse
import os
import sys

import cv2
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

CLASS_NAMES = ["window", "door", "ac_unit", "meter_panel", "pipe", "grill"]
# text query -> class id
QUERIES = [
    ("a window", 0),
    ("a door", 1),
    ("an air conditioner", 2),
    ("an outdoor air conditioning unit on a wall", 2),
    ("an electricity meter box", 3),
    ("a drain pipe", 4),
    ("a metal window grill", 5),
]
MODEL = "google/owlv2-base-patch16-ensemble"


class OwlTeacher:
    def __init__(self, device="cuda"):
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL).to(device).eval().half()
        self.texts = [[q for q, _ in QUERIES]]

    @torch.no_grad()
    def propose(self, image_bgr, threshold=0.1):
        """[(class_id, score, x1, y1, x2, y2)] in pixels of the given image."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self.processor(text=self.texts, images=pil, return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].half()
        outputs = self.model(**inputs)
        h, w = rgb.shape[:2]
        side = max(h, w)  # OWLv2 pads the image to a square before resizing
        results = self.processor.image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=torch.tensor([[side, side]], device=self.device)
        )[0]
        out = []
        for box, score, label in zip(results["boxes"].tolist(), results["scores"].tolist(), results["labels"].tolist()):
            x1, y1, x2, y2 = box
            x1, x2 = max(0.0, min(w, x1)), max(0.0, min(w, x2))
            y1, y2 = max(0.0, min(h, y1)), max(0.0, min(h, y2))
            if x2 - x1 > 2 and y2 - y1 > 2:
                out.append((QUERIES[int(label)][1], float(score), x1, y1, x2, y2))
        return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    teacher = OwlTeacher()
    colors = [(0, 200, 0), (200, 0, 0), (0, 0, 255), (0, 200, 200), (200, 0, 200), (0, 128, 255)]
    for name in sorted(os.listdir(args.images)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image = cv2.imread(os.path.join(args.images, name))
        if image is None:
            continue
        boxes = teacher.propose(image, args.threshold)
        for cls, score, x1, y1, x2, y2 in boxes:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[cls], 2)
            cv2.putText(image, f"{CLASS_NAMES[cls]} {score:.2f}", (int(x1), max(12, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], 1)
        cv2.imwrite(os.path.join(args.out, name), image)
        print(name, {n: sum(1 for b in boxes if b[0] == i) for i, n in enumerate(CLASS_NAMES)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
