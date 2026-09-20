"""
Learned obstacle detector: YOLO ONNX pre/post-processing and inference wrapper.
Kept free of Flask and GeometryEngine so it can be tested without a model file.
"""

from typing import List, Sequence, Tuple

import cv2
import numpy as np

CLASS_NAMES = ["window", "door", "ac_unit", "meter_panel", "pipe", "grill"]

INPUT_SIZE = 640
NMS_IOU = 0.5
MIN_BOX_FRACTION = 0.005  # drop boxes under 0.5% of the frame in either dimension


def letterbox(image_bgr: np.ndarray, size: int = INPUT_SIZE) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize keeping aspect ratio and pad to a square. Returns (canvas, scale, (pad_x, pad_y))."""
    h, w = image_bgr.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def to_input_tensor(canvas_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC -> RGB float32 NCHW in 0..1."""
    rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])


def decode(
    output: np.ndarray,
    scale: float,
    pad: Tuple[int, int],
    image_size: Tuple[int, int],
    min_score: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Turn raw YOLO output (1, 4+classes, N) or (4+classes, N) into boxes in original
    image pixels. image_size is (width, height). Boxes are clipped to the image.
    """
    preds = np.asarray(output)
    if preds.ndim == 3:
        preds = preds[0]
    preds = preds.T  # (N, 4 + classes)

    class_scores = preds[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]
    keep = scores >= min_score
    cx, cy, w, h = preds[keep, :4].T
    scores, class_ids = scores[keep], class_ids[keep]

    img_w, img_h = image_size
    x1 = np.clip((cx - w / 2 - pad[0]) / scale, 0, img_w)
    y1 = np.clip((cy - h / 2 - pad[1]) / scale, 0, img_h)
    x2 = np.clip((cx + w / 2 - pad[0]) / scale, 0, img_w)
    y2 = np.clip((cy + h / 2 - pad[1]) / scale, 0, img_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    return boxes, scores.astype(np.float32), class_ids


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area + areas - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = NMS_IOU,
) -> List[int]:
    """Class-aware non-maximum suppression. Returns kept indices, highest score first."""
    keep: List[int] = []
    order = np.argsort(-scores)
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        suppress = (class_ids[rest] == class_ids[i]) & (_iou(boxes[i], boxes[rest]) > iou_threshold)
        order = rest[~suppress]
    return keep


def thresholds_for_sensitivity(sensitivity: float) -> Tuple[float, float]:
    """Map the UI sensitivity slider to (review_threshold, accept_threshold)."""
    review = min(0.40, max(0.10, 0.40 - 0.30 * sensitivity))
    return review, review + 0.25
