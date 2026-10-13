"""
Turn what the operator confirmed or dismissed in the review panel into training data.

Every time the operator changes the detector's obstacles for a photo, the photo and the corrected boxes are saved
as one more labelled sample in an ordinary YOLO folder (train/images, train/labels, data.yaml). That folder can be
fed straight to the training pipeline:

    training/gpu/.venv/Scripts/python.exe training/gpu/pool.py yolo corrections exports/corrections \
        "window=window,door=door,ac_unit=ac_unit,meter_panel=meter_panel,pipe=pipe,grill=grill" --keep-empty

Only confirmed types become labels. An `unverified` box that the operator did not confirm is left out, which
teaches the model that it was not an obstacle. The photos stay on the machine that ran the app.
"""

import hashlib
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from obstacle_detector import CLASS_NAMES

MAX_SAMPLES = 1000  # a local folder, not an archive: stop collecting rather than fill the disk


def yolo_lines(obstacles: List[Dict[str, Any]], wall_w_mm: float, wall_h_mm: float) -> List[str]:
    """YOLO label lines for the confirmed obstacles. The photo is the wall, y is flipped to image coordinates."""
    lines = []
    for obs in obstacles:
        if obs.get("type") not in CLASS_NAMES:
            continue
        x, y, w, h = (float(obs[key]) for key in ("x", "y", "w", "h"))
        cx = (x + w / 2) / wall_w_mm
        cy = 1.0 - (y + h / 2) / wall_h_mm
        lines.append(f"{CLASS_NAMES.index(obs['type'])} {cx:.6f} {cy:.6f} {w / wall_w_mm:.6f} {h / wall_h_mm:.6f}")
    return lines


def _signature(obstacles: List[Dict[str, Any]]):
    return sorted((o.get("type"), round(float(o["x"])), round(float(o["y"])), round(float(o["w"])), round(float(o["h"])))
                  for o in obstacles)


def same_obstacles(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> bool:
    """True if both lists hold the same boxes of the same types, ignoring order, ids and sub-millimetre noise."""
    return _signature(a) == _signature(b)


def save_correction(
    folder: str,
    image_bgr: np.ndarray,
    obstacles: List[Dict[str, Any]],
    wall_w_mm: float,
    wall_h_mm: float,
) -> Optional[str]:
    """
    Write the photo and its corrected labels. The same photo always maps to the same sample, so reviewing it
    again updates it. Returns the sample id, or None if the folder is full and this is a new photo.
    """
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return None
    sample_id = hashlib.sha1(encoded.tobytes()).hexdigest()[:12]
    images = os.path.join(folder, "train", "images")
    labels = os.path.join(folder, "train", "labels")
    os.makedirs(images, exist_ok=True)
    os.makedirs(labels, exist_ok=True)
    existing = os.path.exists(os.path.join(images, sample_id + ".jpg"))
    if not existing and len(os.listdir(images)) >= MAX_SAMPLES:
        return None
    with open(os.path.join(images, sample_id + ".jpg"), "wb") as f:
        f.write(encoded.tobytes())
    with open(os.path.join(labels, sample_id + ".txt"), "w", encoding="utf-8") as f:
        lines = yolo_lines(obstacles, wall_w_mm, wall_h_mm)
        f.write("\n".join(lines) + ("\n" if lines else ""))
    with open(os.path.join(folder, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("path: .\ntrain: train/images\nnames: [" + ", ".join(f"'{name}'" for name in CLASS_NAMES) + "]\n")
    return sample_id
