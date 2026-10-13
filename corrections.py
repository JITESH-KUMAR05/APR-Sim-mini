"""
Turn what the operator changed in the review panel into training data.

When the operator changes the detector's obstacles for an uploaded photo, the photo and the corrected boxes are
saved as one more labelled sample in an ordinary YOLO folder (train/images, train/labels, data.yaml). That folder
can be fed straight to the training pipeline:

    training/gpu/.venv/Scripts/python.exe training/gpu/pool.py yolo corrections exports/corrections \
        "window=window,door=door,ac_unit=ac_unit,meter_panel=meter_panel,pipe=pipe,grill=grill" --keep-empty

Read this before you do:

- Only confirmed types become labels (`unverified` and legacy types are left out). Every confident detection the
  operator did not touch is saved too, as a label, so the labels are the model's own output plus the operator's
  changes. Look at a contact sheet of a batch before training on it.
- The photos can show people's homes, faces and number plates. They stay on the machine that ran the app, are
  never served back, and the app only saves when the operator leaves the "keep my corrections" box ticked.
  Review them before they are shared or trained on.
"""

import hashlib
import os
import threading
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from obstacle_detector import CLASS_NAMES

MAX_SAMPLES = 1000  # a local folder, not an archive: stop collecting rather than fill the disk
_lock = threading.Lock()


def yolo_lines(obstacles: List[Dict[str, Any]], wall_w_mm: float, wall_h_mm: float) -> List[str]:
    """YOLO label lines for the confirmed obstacles. The photo is the wall, y is flipped to image coordinates."""
    lines = []
    for obs in obstacles:
        if obs.get("type") not in CLASS_NAMES:
            continue
        x, y, w, h = (float(obs[key]) for key in ("x", "y", "w", "h"))
        x1, x2 = max(0.0, x), min(wall_w_mm, x + w)
        y1, y2 = max(0.0, y), min(wall_h_mm, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2 / wall_w_mm
        cy = 1.0 - (y1 + y2) / 2 / wall_h_mm
        lines.append(f"{CLASS_NAMES.index(obs['type'])} {cx:.6f} {cy:.6f} {(x2 - x1) / wall_w_mm:.6f} {(y2 - y1) / wall_h_mm:.6f}")
    return lines


def _signature(obstacles: List[Dict[str, Any]]):
    return sorted((o.get("type"), round(float(o["x"])), round(float(o["y"])), round(float(o["w"])), round(float(o["h"])))
                  for o in obstacles)


def same_obstacles(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> bool:
    """True if both lists hold the same boxes of the same types, ignoring order, ids and sub-millimetre noise."""
    return _signature(a) == _signature(b)


def _write_atomically(path: str, data: bytes) -> None:
    temporary = path + ".tmp"
    with open(temporary, "wb") as f:
        f.write(data)
    os.replace(temporary, path)


def save_correction(
    folder: str,
    image_bgr: np.ndarray,
    obstacles: List[Dict[str, Any]],
    wall_w_mm: float,
    wall_h_mm: float,
) -> Optional[str]:
    """
    Write the photo and its corrected labels. The same photo always maps to the same sample, so reviewing it
    again updates it. Returns the sample id, or None if the folder is full and this is a new photo. The label is
    written before the photo, each through a temporary file, so a crash never leaves a photo without its label.
    """
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return None
    sample_id = hashlib.sha1(encoded.tobytes()).hexdigest()[:12]
    images = os.path.join(folder, "train", "images")
    labels = os.path.join(folder, "train", "labels")
    with _lock:
        os.makedirs(images, exist_ok=True)
        os.makedirs(labels, exist_ok=True)
        image_path = os.path.join(images, sample_id + ".jpg")
        if not os.path.exists(image_path) and len([n for n in os.listdir(images) if n.endswith(".jpg")]) >= MAX_SAMPLES:
            return None
        lines = yolo_lines(obstacles, wall_w_mm, wall_h_mm)
        _write_atomically(os.path.join(labels, sample_id + ".txt"), ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))
        _write_atomically(image_path, encoded.tobytes())
        yaml_path = os.path.join(folder, "data.yaml")
        if not os.path.exists(yaml_path):
            text = "path: .\ntrain: train/images\nnames: [" + ", ".join(f"'{name}'" for name in CLASS_NAMES) + "]\n"
            _write_atomically(yaml_path, text.encode("utf-8"))
    return sample_id
