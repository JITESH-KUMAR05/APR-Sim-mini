# Real-World Obstacle Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the classical edge/contour obstacle detector with a trained YOLO11n ONNX detector, add an operator review step for uncertain detections, and add four-corner wall rectification, so the photo-to-mission pipeline works on real walls.

**Architecture:** A new `obstacle_detector.py` (pure numpy/cv2 helpers plus an `onnxruntime` wrapper) feeds `GeometryEngine.detect`, which returns the same obstacle dicts the planner, exporters and viewers already consume. Uncertain detections become `type: "unverified"` obstacles that the planner still avoids; a new `/api/replan` endpoint re-plans after the operator confirms or dismisses them. Training happens offline (Colab) via scripts in `training/`; only the exported ONNX file ships.

**Tech Stack:** Python 3.12, Flask, OpenCV (headless), numpy, onnxruntime, unittest, vanilla JS. Training only: ultralytics (never a runtime dependency).

## Global Constraints

- Never add PyTorch or ultralytics to `requirements.txt` or `pyproject.toml`. Inference uses `onnxruntime` only.
- Dependencies are pinned with exact `==` versions. `requirements.txt` and `pyproject.toml` must list identical pins (Vercel prefers `pyproject.toml` when both exist; an empty one already caused a production outage).
- Obstacle output contract is unchanged: `{id, type, label, x, y, w, h, depth_mm, norm{x,y,w,h}}`, millimetres, origin bottom-left. New optional fields: `confidence` (0 to 1) and, for unverified boxes only, `guess` (the model's best class).
- Classes, in this order (this is the model's class index order): `window`, `door`, `ac_unit`, `meter_panel`, `pipe`, `grill`. Plus the inference-time type `unverified`.
- Default `depth_mm`: window 120, door 80, ac_unit 400, meter_panel 100, pipe 80, grill 60, unverified 50. They live in one dict in `geometry_engine.py`.
- Thresholds from the `sensitivity` slider (0 to 1, default 0.5): `review = clamp(0.40 - 0.30 * sensitivity, 0.10, 0.40)`, `accept = review + 0.25`. At or above `accept`: keep class. Between: `unverified` (still an obstacle for the planner). Below `review`: discard.
- Boxes smaller than 0.5% of the frame in either dimension are dropped. The old 1.2% area / 5% size filters apply only to the classical fallback.
- Aggregate confidence is the mean confidence of all kept detections as a percentage, or `null` when there are none.
- `detector` field values: `"yolo11n-onnx"` or `"classical-fallback"`.
- Model file: `models/apr_obstacles.onnx`, overridable with env var `APR_OBSTACLE_MODEL`. Missing or unloadable model means fallback, never a crash.
- Model input 640x640, NMS IoU 0.5, NMS done in our code, not baked into the export.
- Tests run with `uv run python -m unittest discover -s tests -p "<file>" -v`. Baseline before this plan: 7 tests pass.
- Commit trailer on every commit: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `obstacle_detector.py` | Create | Letterbox, decode, NMS, thresholds, `Detection`, `ObstacleDetector`, `load_default_detector` |
| `geometry_engine.py` | Modify | `detect()`, class info dicts, `_obstacles_from_detections`, `normalize_obstacles`, `parse_corners`, `rectify_wall`; old detector renamed `_detect_classical` |
| `cad_exporter.py` | Modify | Treat `meter_panel` like `switchboard` |
| `app.py` | Modify | Load detector, `build_mission` helper, `/api/analyze` additions, `/api/replan` |
| `static/viewer3d.js` | Modify | `meter_panel` renders as fixture |
| `static/app.js` | Modify | Review panel, banner, null confidence, 2D unverified styling, corner flow |
| `static/corner_picker.js` | Create | Four-corner picker widget |
| `static/style.css`, `templates/index.html` | Modify | Review panel, banner, corner modal, corrected labels |
| `tests/fakes.py` | Create | `FakeDetector` shared by tests |
| `tests/test_obstacle_detector.py`, `test_detection_integration.py`, `test_api.py`, `test_dependencies.py`, `test_training_tools.py` | Create | New tests |
| `training/build_dataset.py`, `evaluate.py`, `train.py`, `class_map.json`, `README.md` | Create | Offline training and evaluation |
| `models/` | Create (later, by training) | Holds the exported ONNX file |

---

### Task 1: Detector core functions (letterbox, decode, NMS, thresholds)

**Files:**
- Create: `obstacle_detector.py`
- Test: `tests/test_obstacle_detector.py`

**Interfaces:**
- Produces: `CLASS_NAMES: list[str]`, `INPUT_SIZE = 640`, `NMS_IOU = 0.5`, `MIN_BOX_FRACTION = 0.005`; `letterbox(image_bgr, size=INPUT_SIZE) -> (canvas, scale, (pad_x, pad_y))`; `to_input_tensor(canvas_bgr) -> float32 ndarray (1,3,H,W)`; `decode(output, scale, pad, image_size, min_score) -> (boxes (K,4) xyxy in original pixels, scores (K,), class_ids (K,))` where `image_size = (width, height)`; `nms(boxes, scores, class_ids, iou_threshold=NMS_IOU) -> list[int]`; `thresholds_for_sensitivity(sensitivity) -> (review, accept)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_obstacle_detector.py`:

```python
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from obstacle_detector import (
    CLASS_NAMES,
    INPUT_SIZE,
    decode,
    letterbox,
    nms,
    thresholds_for_sensitivity,
    to_input_tensor,
)

NUM_CLASSES = len(CLASS_NAMES)


def make_output(rows):
    """rows: (cx, cy, w, h, class_id, score) in letterboxed 640px space.
    Returns an array shaped like a YOLO ONNX export: (1, 4 + classes, N)."""
    out = np.zeros((1, 4 + NUM_CLASSES, len(rows)), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(rows):
        out[0, 0:4, i] = (cx, cy, w, h)
        out[0, 4 + cls, i] = score
    return out


class TestLetterbox(unittest.TestCase):
    def test_wide_image_is_padded_top_and_bottom(self):
        canvas, scale, pad = letterbox(np.zeros((600, 1200, 3), dtype=np.uint8))
        self.assertEqual(canvas.shape, (INPUT_SIZE, INPUT_SIZE, 3))
        self.assertAlmostEqual(scale, 640 / 1200)
        self.assertEqual(pad, (0, 160))

    def test_tall_image_is_padded_left_and_right(self):
        _, scale, pad = letterbox(np.zeros((1200, 600, 3), dtype=np.uint8))
        self.assertAlmostEqual(scale, 640 / 1200)
        self.assertEqual(pad, (160, 0))

    def test_padding_is_grey_114(self):
        canvas, _, _ = letterbox(np.zeros((600, 1200, 3), dtype=np.uint8))
        self.assertTrue((canvas[0] == 114).all())


class TestToInputTensor(unittest.TestCase):
    def test_shape_dtype_and_rgb_order(self):
        canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        canvas[..., 0] = 255  # blue channel in BGR
        tensor = to_input_tensor(canvas)
        self.assertEqual(tensor.shape, (1, 3, INPUT_SIZE, INPUT_SIZE))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue((tensor[0, 2] == 1.0).all())  # blue is the last RGB channel
        self.assertTrue((tensor[0, 0] == 0.0).all())


class TestDecode(unittest.TestCase):
    SCALE = 640 / 1200

    def test_wide_image_round_trip(self):
        out = make_output([(320, 320, 64, 64, 0, 0.9)])
        boxes, scores, class_ids = decode(out, self.SCALE, (0, 160), (1200, 600), 0.25)
        self.assertEqual(class_ids.tolist(), [0])
        self.assertAlmostEqual(float(scores[0]), 0.9, places=5)
        np.testing.assert_allclose(boxes[0], [540, 240, 660, 360], atol=0.01)

    def test_tall_image_round_trip(self):
        out = make_output([(320, 320, 64, 64, 0, 0.9)])
        boxes, _, _ = decode(out, self.SCALE, (160, 0), (600, 1200), 0.25)
        np.testing.assert_allclose(boxes[0], [240, 540, 360, 660], atol=0.01)

    def test_boxes_are_clipped_to_the_image(self):
        out = make_output([(10, 320, 100, 100, 1, 0.8)])
        boxes, _, _ = decode(out, self.SCALE, (0, 160), (1200, 600), 0.25)
        self.assertEqual(float(boxes[0][0]), 0.0)

    def test_low_scores_are_dropped(self):
        out = make_output([(320, 320, 64, 64, 0, 0.10), (100, 100, 64, 64, 2, 0.60)])
        _, _, class_ids = decode(out, self.SCALE, (0, 160), (1200, 600), 0.25)
        self.assertEqual(class_ids.tolist(), [2])

    def test_empty_result_has_consistent_shapes(self):
        out = make_output([(320, 320, 64, 64, 0, 0.10)])
        boxes, scores, class_ids = decode(out, self.SCALE, (0, 160), (1200, 600), 0.5)
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(scores.shape, (0,))
        self.assertEqual(class_ids.shape, (0,))

    def test_accepts_output_without_batch_dimension(self):
        out = make_output([(320, 320, 64, 64, 0, 0.9)])[0]
        boxes, _, _ = decode(out, self.SCALE, (0, 160), (1200, 600), 0.25)
        self.assertEqual(boxes.shape, (1, 4))


class TestNms(unittest.TestCase):
    def test_overlapping_same_class_keeps_highest_score(self):
        boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [300, 300, 400, 400]], dtype=np.float32)
        scores = np.array([0.6, 0.9, 0.7], dtype=np.float32)
        keep = nms(boxes, scores, np.array([0, 0, 0]), iou_threshold=0.5)
        self.assertEqual(keep, [1, 2])

    def test_overlapping_different_classes_are_both_kept(self):
        boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105]], dtype=np.float32)
        keep = nms(boxes, np.array([0.6, 0.9], dtype=np.float32), np.array([0, 1]))
        self.assertEqual(sorted(keep), [0, 1])

    def test_empty_input(self):
        self.assertEqual(nms(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)), [])


class TestThresholds(unittest.TestCase):
    def test_default_sensitivity(self):
        review, accept = thresholds_for_sensitivity(0.5)
        self.assertAlmostEqual(review, 0.25)
        self.assertAlmostEqual(accept, 0.50)

    def test_higher_sensitivity_lowers_thresholds(self):
        low_sens_review, _ = thresholds_for_sensitivity(0.1)
        high_sens_review, _ = thresholds_for_sensitivity(1.0)
        self.assertGreater(low_sens_review, high_sens_review)

    def test_review_threshold_is_clamped(self):
        self.assertAlmostEqual(thresholds_for_sensitivity(5.0)[0], 0.10)
        self.assertAlmostEqual(thresholds_for_sensitivity(-5.0)[0], 0.40)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest discover -s tests -p "test_obstacle_detector.py" -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'obstacle_detector'`.

- [ ] **Step 3: Write the implementation**

Create `obstacle_detector.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m unittest discover -s tests -p "test_obstacle_detector.py" -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add obstacle_detector.py tests/test_obstacle_detector.py
git commit -m "Add YOLO ONNX pre/post-processing helpers" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: ObstacleDetector class, model loader, onnxruntime dependency

**Files:**
- Modify: `obstacle_detector.py`, `pyproject.toml`, `requirements.txt`, `uv.lock` (via `uv lock`)
- Test: `tests/test_obstacle_detector.py`, `tests/test_dependencies.py`

**Interfaces:**
- Consumes: everything from Task 1.
- Produces: `Detection` (frozen dataclass: `class_name: str`, `confidence: float`, `x1, y1, x2, y2: float`, original-image pixels); `ObstacleDetector(model_path=None, session=None, class_names=CLASS_NAMES)` with `detect(image_bgr, min_score) -> List[Detection]`; `load_default_detector() -> Optional[ObstacleDetector]`; `DEFAULT_MODEL_PATH`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_obstacle_detector.py`, change the imports to:

```python
import tempfile
from unittest import mock

from obstacle_detector import (
    CLASS_NAMES,
    INPUT_SIZE,
    ObstacleDetector,
    decode,
    letterbox,
    load_default_detector,
    nms,
    thresholds_for_sensitivity,
    to_input_tensor,
)
```

Insert before `if __name__ == "__main__":`:

```python
class FakeInput:
    name = "images"


class FakeSession:
    def __init__(self, output):
        self.output = output
        self.feeds = []

    def get_inputs(self):
        return [FakeInput()]

    def run(self, output_names, feeds):
        self.feeds.append(feeds)
        return [self.output]


class TestObstacleDetector(unittest.TestCase):
    IMAGE = np.zeros((600, 1200, 3), dtype=np.uint8)

    def test_detect_maps_boxes_back_to_the_original_image(self):
        session = FakeSession(make_output([(320, 320, 64, 64, 0, 0.9)]))
        detections = ObstacleDetector(session=session).detect(self.IMAGE, min_score=0.25)
        self.assertEqual(len(detections), 1)
        det = detections[0]
        self.assertEqual(det.class_name, "window")
        self.assertAlmostEqual(det.confidence, 0.9, places=5)
        self.assertAlmostEqual(det.x1, 540, delta=0.01)
        self.assertAlmostEqual(det.y2, 360, delta=0.01)
        self.assertEqual(session.feeds[0]["images"].shape, (1, 3, 640, 640))

    def test_boxes_under_half_a_percent_of_the_frame_are_dropped(self):
        # 1.6 px in letterbox space is 3 px (0.25%) of a 1200 px wide image
        session = FakeSession(make_output([(320, 320, 1.6, 64, 0, 0.9), (100, 300, 64, 64, 1, 0.8)]))
        detections = ObstacleDetector(session=session).detect(self.IMAGE, min_score=0.25)
        self.assertEqual([d.class_name for d in detections], ["door"])

    def test_duplicate_detections_are_suppressed(self):
        session = FakeSession(make_output([(320, 320, 64, 64, 0, 0.9), (322, 322, 64, 64, 0, 0.7)]))
        self.assertEqual(len(ObstacleDetector(session=session).detect(self.IMAGE, 0.25)), 1)

    def test_class_count_mismatch_raises(self):
        bad = np.zeros((1, 4 + NUM_CLASSES + 1, 5), dtype=np.float32)
        with self.assertRaises(ValueError):
            ObstacleDetector(session=FakeSession(bad)).detect(self.IMAGE, 0.25)


class TestLoadDefaultDetector(unittest.TestCase):
    def test_missing_model_returns_none(self):
        missing = os.path.join(os.path.dirname(__file__), "no_such_model.onnx")
        with mock.patch.dict(os.environ, {"APR_OBSTACLE_MODEL": missing}):
            self.assertIsNone(load_default_detector())

    def test_unreadable_model_returns_none(self):
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as handle:
            handle.write(b"this is not an onnx model")
        try:
            with mock.patch.dict(os.environ, {"APR_OBSTACLE_MODEL": handle.name}):
                self.assertIsNone(load_default_detector())
        finally:
            os.remove(handle.name)
```

Create `tests/test_dependencies.py`:

```python
import os
import re
import tomllib
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _normalise(spec):
    name, version = spec.split("==")
    return name.strip().lower().replace("_", "-"), version.strip()


class TestDependencyPins(unittest.TestCase):
    def test_requirements_and_pyproject_match_and_are_exact(self):
        with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as handle:
            requirements = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
            declared = tomllib.load(handle)["project"]["dependencies"]

        for spec in requirements + declared:
            self.assertRegex(spec, r"^[A-Za-z0-9_.\-]+==[0-9][A-Za-z0-9.\-]*$", f"{spec} is not an exact pin")
        self.assertEqual(sorted(map(_normalise, requirements)), sorted(map(_normalise, declared)))

    def test_no_training_frameworks_in_runtime_dependencies(self):
        with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as handle:
            text = handle.read().lower()
        for banned in ("torch", "ultralytics"):
            self.assertIsNone(re.search(rf"^{banned}\b", text, re.M), f"{banned} must not be a runtime dependency")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest discover -s tests -p "test_obstacle_detector.py" -v`
Expected: ERROR `ImportError: cannot import name 'ObstacleDetector'`.

- [ ] **Step 3: Add the dependency**

Edit `requirements.txt` to add a line `onnxruntime==1.20.1`, and add `"onnxruntime==1.20.1",` to the `dependencies` list in `pyproject.toml`. Then:

Run: `uv lock && uv sync`
Expected: resolves and installs `onnxruntime`. If `1.20.1` fails to resolve for Python 3.12 with `numpy==1.26.4`, run `uv add onnxruntime` once to see which version resolves, then pin that exact version in both files.

- [ ] **Step 4: Implement the class and loader**

Change the top of `obstacle_detector.py` imports and add the new code at the end. Replace `from typing import List, Sequence, Tuple` with:

```python
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
```

Append:

```python
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "apr_obstacles.onnx")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class ObstacleDetector:
    """Runs a YOLO ONNX model. Pass `session` to inject a fake in tests."""

    def __init__(self, model_path: Optional[str] = None, session=None, class_names: Sequence[str] = CLASS_NAMES):
        if session is None:
            import onnxruntime as ort  # imported lazily so the fallback path never needs it

            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self.class_names = list(class_names)

    def detect(self, image_bgr: np.ndarray, min_score: float) -> List[Detection]:
        img_h, img_w = image_bgr.shape[:2]
        canvas, scale, pad = letterbox(image_bgr)
        output = self._session.run(None, {self._input_name: to_input_tensor(canvas)})[0]
        model_classes = output.shape[-2] - 4
        if model_classes != len(self.class_names):
            raise ValueError(
                f"Model has {model_classes} classes but {len(self.class_names)} are configured"
            )

        boxes, scores, class_ids = decode(output, scale, pad, (img_w, img_h), min_score)
        big_enough = ((boxes[:, 2] - boxes[:, 0]) >= MIN_BOX_FRACTION * img_w) & (
            (boxes[:, 3] - boxes[:, 1]) >= MIN_BOX_FRACTION * img_h
        )
        boxes, scores, class_ids = boxes[big_enough], scores[big_enough], class_ids[big_enough]

        detections = []
        for i in nms(boxes, scores, class_ids):
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            detections.append(Detection(self.class_names[int(class_ids[i])], float(scores[i]), x1, y1, x2, y2))
        return detections


def load_default_detector() -> Optional[ObstacleDetector]:
    """Load the shipped model, or return None so callers fall back to classical detection."""
    path = os.environ.get("APR_OBSTACLE_MODEL", DEFAULT_MODEL_PATH)
    if not os.path.exists(path):
        return None
    try:
        return ObstacleDetector(model_path=path)
    except Exception as exc:  # a corrupt or wrong-format model must not take the app down
        log.warning("Could not load obstacle model %s: %s", path, exc)
        return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m unittest discover -s tests -p "test_*.py" -v`
Expected: all tests PASS (7 existing plus new ones).

- [ ] **Step 6: Commit**

```bash
git add obstacle_detector.py tests/test_obstacle_detector.py tests/test_dependencies.py pyproject.toml requirements.txt uv.lock
git commit -m "Add ONNX obstacle detector wrapper and onnxruntime dependency" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: GeometryEngine uses the detector, with classical fallback

**Files:**
- Modify: `geometry_engine.py` (imports and class header near lines 8-15; the `detect_obstacles_from_image` def at line 31)
- Create: `tests/fakes.py`
- Test: `tests/test_detection_integration.py`

**Interfaces:**
- Consumes: `ObstacleDetector.detect(image_bgr, min_score) -> List[Detection]`, `thresholds_for_sensitivity`.
- Produces: `GeometryEngine(detector: Optional[ObstacleDetector] = None)`; `GeometryEngine.detect(image_bgr, wall_w_mm, wall_h_mm, sensitivity=0.5) -> {"obstacles": list, "confidence": float | None, "detector": str}`; `detect_obstacles_from_image(...)` still returns `(obstacles, confidence)`; module constants `OBSTACLE_CLASS_INFO`, `UNVERIFIED_INFO`, `LEGACY_OBSTACLE_INFO` (each maps type to `(base_label, depth_mm)`, except `UNVERIFIED_INFO` which is one tuple).

- [ ] **Step 1: Write the failing tests**

Create `tests/fakes.py`:

```python
class FakeDetector:
    """Stands in for ObstacleDetector: returns preset detections at or above min_score."""

    def __init__(self, detections):
        self.detections = list(detections)
        self.min_scores = []

    def detect(self, image_bgr, min_score):
        self.min_scores.append(min_score)
        return [d for d in self.detections if d.confidence >= min_score]
```

Create `tests/test_detection_integration.py`:

```python
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fakes import FakeDetector
from geometry_engine import GeometryEngine
from obstacle_detector import Detection, load_default_detector

IMAGE = np.zeros((500, 1000, 3), dtype=np.uint8)
WALL_W, WALL_H = 4000.0, 2000.0


def engine_with(*detections):
    detector = FakeDetector(detections)
    return GeometryEngine(detector=detector), detector


class TestModelDetection(unittest.TestCase):
    def test_confident_detection_keeps_class_and_maps_to_wall_mm(self):
        engine, _ = engine_with(Detection("window", 0.9, 100, 50, 300, 250))
        obstacle = engine.detect(IMAGE, WALL_W, WALL_H)["obstacles"][0]
        self.assertEqual(obstacle["type"], "window")
        self.assertEqual(obstacle["label"], "Glazed Window #1")
        self.assertEqual((obstacle["x"], obstacle["y"], obstacle["w"], obstacle["h"]), (400.0, 1000.0, 800.0, 800.0))
        self.assertEqual(obstacle["depth_mm"], 120.0)
        self.assertEqual(obstacle["confidence"], 0.9)
        self.assertEqual(obstacle["norm"]["x"], 0.1)

    def test_low_confidence_becomes_unverified_but_keeps_the_guess(self):
        engine, _ = engine_with(Detection("ac_unit", 0.35, 100, 50, 300, 250))
        obstacle = engine.detect(IMAGE, WALL_W, WALL_H)["obstacles"][0]
        self.assertEqual(obstacle["type"], "unverified")
        self.assertEqual(obstacle["label"], "Unverified object #1")
        self.assertEqual(obstacle["depth_mm"], 50.0)
        self.assertEqual(obstacle["guess"], "ac_unit")

    def test_below_review_threshold_is_discarded(self):
        engine, _ = engine_with(Detection("window", 0.20, 100, 50, 300, 250))
        self.assertEqual(engine.detect(IMAGE, WALL_W, WALL_H)["obstacles"], [])

    def test_detector_is_asked_for_the_review_threshold(self):
        engine, detector = engine_with()
        engine.detect(IMAGE, WALL_W, WALL_H, sensitivity=0.5)
        self.assertAlmostEqual(detector.min_scores[0], 0.25)

    def test_confidence_is_the_mean_of_kept_detections_in_percent(self):
        engine, _ = engine_with(
            Detection("window", 0.9, 100, 50, 300, 250), Detection("door", 0.3, 500, 50, 700, 250)
        )
        result = engine.detect(IMAGE, WALL_W, WALL_H)
        self.assertEqual(result["confidence"], 60.0)
        self.assertEqual(result["detector"], "yolo11n-onnx")

    def test_no_detections_gives_null_confidence(self):
        engine, _ = engine_with()
        result = engine.detect(IMAGE, WALL_W, WALL_H)
        self.assertEqual(result["obstacles"], [])
        self.assertIsNone(result["confidence"])

    def test_ids_and_labels_run_left_to_right(self):
        engine, _ = engine_with(
            Detection("door", 0.9, 600, 50, 800, 250), Detection("window", 0.9, 100, 50, 300, 250)
        )
        obstacles = engine.detect(IMAGE, WALL_W, WALL_H)["obstacles"]
        self.assertEqual([o["id"] for o in obstacles], ["obs_1", "obs_2"])
        self.assertEqual([o["type"] for o in obstacles], ["window", "door"])
        self.assertEqual(obstacles[1]["label"], "Door Opening #2")


class TestFallback(unittest.TestCase):
    def test_no_detector_uses_classical_fallback(self):
        engine = GeometryEngine()
        image = engine.generate_benchmark_wall_image(1200, 840)
        result = engine.detect(image, 4000.0, 2800.0)
        self.assertEqual(result["detector"], "classical-fallback")
        self.assertGreaterEqual(len(result["obstacles"]), 1)
        self.assertGreaterEqual(result["confidence"], 75.0)

    def test_tuple_contract_is_preserved(self):
        engine, _ = engine_with(Detection("window", 0.9, 100, 50, 300, 250))
        obstacles, confidence = engine.detect_obstacles_from_image(IMAGE, WALL_W, WALL_H)
        self.assertEqual(len(obstacles), 1)
        self.assertEqual(confidence, 90.0)


def spray_segments(waypoints):
    for start, end in zip(waypoints, waypoints[1:]):
        if start["type"] == "PASS_START" and end["type"] == "PASS_END":
            yield start, end


class TestPlannerAvoidsEveryKeptBox(unittest.TestCase):
    def test_no_spray_pass_crosses_an_accepted_or_unverified_box_plus_buffer(self):
        engine, _ = engine_with(
            Detection("window", 0.9, 100, 50, 300, 250), Detection("pipe", 0.35, 600, 100, 800, 300)
        )
        obstacles = engine.detect(IMAGE, WALL_W, WALL_H)["obstacles"]
        self.assertEqual({o["type"] for o in obstacles}, {"window", "unverified"})

        buffer_mm = 60.0
        waypoints, _ = engine.plan_coverage_path(WALL_W, WALL_H, 250.0, 12.0, buffer_mm, obstacles)
        for start, end in spray_segments(waypoints):
            for obs in obstacles:
                box_x1, box_x2 = max(0.0, obs["x"] - buffer_mm), min(WALL_W, obs["x"] + obs["w"] + buffer_mm)
                box_y1, box_y2 = max(0.0, obs["y"] - buffer_mm), min(WALL_H, obs["y"] + obs["h"] + buffer_mm)
                if box_y1 <= start["y"] <= box_y2:
                    low, high = sorted((start["x"], end["x"]))
                    self.assertFalse(low < box_x2 - 0.5 and high > box_x1 + 0.5, f"pass crosses {obs['id']}")


@unittest.skipUnless(load_default_detector() is not None, "models/apr_obstacles.onnx not present")
class TestRealModel(unittest.TestCase):
    def test_real_model_returns_in_bounds_obstacles(self):
        import cv2

        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        image = cv2.imread(os.path.join(root, "samples", "benchmark_residential_facade.jpg"))
        result = GeometryEngine(detector=load_default_detector()).detect(image, 4000.0, 2800.0)
        self.assertEqual(result["detector"], "yolo11n-onnx")
        for obs in result["obstacles"]:
            self.assertLessEqual(obs["x"] + obs["w"], 4000.0 + 1.0)
            self.assertLessEqual(obs["y"] + obs["h"], 2800.0 + 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m unittest discover -s tests -p "test_detection_integration.py" -v`
Expected: ERROR `TypeError: GeometryEngine() takes no arguments` / missing `detect`.

- [ ] **Step 3: Update imports, class info and constructor**

In `geometry_engine.py`, replace:

```python
from typing import Dict, List, Tuple, Any, Optional


class GeometryEngine:
    def __init__(self):
```

with:

```python
from typing import Dict, List, Tuple, Any, Optional

from obstacle_detector import ObstacleDetector, thresholds_for_sensitivity

# type -> (label prefix, depth_mm). depth_mm only drives the 3D export.
OBSTACLE_CLASS_INFO = {
    "window": ("Glazed Window", 120.0),
    "door": ("Door Opening", 80.0),
    "ac_unit": ("AC Unit", 400.0),
    "meter_panel": ("Meter Panel", 100.0),
    "pipe": ("Pipe", 80.0),
    "grill": ("Grill", 60.0),
}
UNVERIFIED_INFO = ("Unverified object", 50.0)
LEGACY_OBSTACLE_INFO = {  # types still emitted by the classical fallback
    "switchboard": ("Utility Panel", 35.0),
    "fixture": ("Architectural Opening", 50.0),
}


class GeometryEngine:
    def __init__(self, detector: Optional[ObstacleDetector] = None):
        self.detector = detector
```

- [ ] **Step 4: Add `detect` and rename the classical detector**

In `geometry_engine.py`, replace this block:

```python
    def detect_obstacles_from_image(
        self,
        image_bgr: np.ndarray,
        wall_w_mm: float,
        wall_h_mm: float,
        sensitivity: float = 0.5
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Segment openings and fixtures (windows, doors, vents, electrical boxes)
        from wall photograph using adaptive edge and contour hierarchy analysis.
        Returns detected obstacles in wall millimeter coordinates and a confidence score.
        """
```

with:

```python
    def detect(
        self,
        image_bgr: np.ndarray,
        wall_w_mm: float,
        wall_h_mm: float,
        sensitivity: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect obstacles in wall millimetre coordinates (origin bottom-left).
        Uses the trained detector when loaded, otherwise the classical fallback.
        Returns {"obstacles", "confidence", "detector"}; confidence is a percentage
        or None when the model kept no detections.
        """
        if self.detector is None:
            obstacles, confidence = self._detect_classical(image_bgr, wall_w_mm, wall_h_mm, sensitivity)
            return {"obstacles": obstacles, "confidence": confidence, "detector": "classical-fallback"}

        img_h, img_w = image_bgr.shape[:2]
        review_threshold, accept_threshold = thresholds_for_sensitivity(sensitivity)
        detections = self.detector.detect(image_bgr, min_score=review_threshold)
        obstacles = self._obstacles_from_detections(
            detections, img_w, img_h, wall_w_mm, wall_h_mm, accept_threshold
        )
        confidence = None
        if detections:
            confidence = round(float(np.mean([d.confidence for d in detections])) * 100.0, 1)
        return {"obstacles": obstacles, "confidence": confidence, "detector": "yolo11n-onnx"}

    def detect_obstacles_from_image(
        self,
        image_bgr: np.ndarray,
        wall_w_mm: float,
        wall_h_mm: float,
        sensitivity: float = 0.5
    ) -> Tuple[List[Dict[str, Any]], Optional[float]]:
        result = self.detect(image_bgr, wall_w_mm, wall_h_mm, sensitivity)
        return result["obstacles"], result["confidence"]

    def _obstacles_from_detections(
        self,
        detections: List[Any],
        img_w: int,
        img_h: int,
        wall_w_mm: float,
        wall_h_mm: float,
        accept_threshold: float
    ) -> List[Dict[str, Any]]:
        obstacles: List[Dict[str, Any]] = []
        for det in sorted(detections, key=lambda d: d.x1):
            norm_x = det.x1 / img_w
            norm_y = (img_h - det.y2) / img_h   # bottom-left origin
            norm_w = (det.x2 - det.x1) / img_w
            norm_h = (det.y2 - det.y1) / img_h

            if det.confidence >= accept_threshold:
                obs_type = det.class_name
                base_label, depth_mm = OBSTACLE_CLASS_INFO[det.class_name]
            else:
                obs_type = "unverified"
                base_label, depth_mm = UNVERIFIED_INFO

            number = len(obstacles) + 1
            obstacle = {
                "id": f"obs_{number}",
                "type": obs_type,
                "label": f"{base_label} #{number}",
                "x": round(norm_x * wall_w_mm, 1),
                "y": round(norm_y * wall_h_mm, 1),
                "w": round(norm_w * wall_w_mm, 1),
                "h": round(norm_h * wall_h_mm, 1),
                "depth_mm": depth_mm,
                "confidence": round(det.confidence, 3),
                "norm": {
                    "x": round(norm_x, 4),
                    "y": round(norm_y, 4),
                    "w": round(norm_w, 4),
                    "h": round(norm_h, 4)
                }
            }
            if obs_type == "unverified":
                obstacle["guess"] = det.class_name
            obstacles.append(obstacle)
        return obstacles

    def _detect_classical(
        self,
        image_bgr: np.ndarray,
        wall_w_mm: float,
        wall_h_mm: float,
        sensitivity: float = 0.5
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Fallback used when no trained model is loaded: segment openings and fixtures
        with adaptive edge and contour hierarchy analysis. Returns obstacles in wall
        millimetre coordinates and a heuristic confidence score.
        """
```

- [ ] **Step 5: Run the whole suite**

Run: `uv run python -m unittest discover -s tests -p "test_*.py" -v`
Expected: all PASS, `TestRealModel` skipped.

- [ ] **Step 6: Commit**

```bash
git add geometry_engine.py tests/fakes.py tests/test_detection_integration.py
git commit -m "Route obstacle detection through the trained model with classical fallback" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: New classes render in DXF, OBJ and the 3D viewer

**Files:**
- Modify: `cad_exporter.py:247`, `static/viewer3d.js:248`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: obstacle `type` strings `meter_panel`, `ac_unit`, `pipe`, `grill`, `unverified`.
- Produces: `meter_panel` gets the same electrical-fixture geometry `switchboard` has; every other new type renders as the generic red keep-out box without error.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py` inside `TestPaintPilotPipeline`, after `test_07_flask_endpoints`:

```python
    def test_08_new_obstacle_classes_export_without_error(self):
        """New detector classes must not break the DXF/OBJ exporters."""
        waypoints = [
            {"seq": 1, "x": 0, "y": 125, "z": 250, "spray_active": True, "row": 1},
            {"seq": 2, "x": 4000, "y": 125, "z": 250, "spray_active": True, "row": 1},
        ]
        for obstacle_type in ("meter_panel", "ac_unit", "pipe", "grill", "unverified"):
            obstacles = [{"id": "obs_1", "type": obstacle_type, "x": 500, "y": 500, "w": 500, "h": 500, "depth_mm": 100}]
            bundle = self.exporter.export_obj_and_mtl(4000.0, 2800.0, obstacles, waypoints)
            self.assertIn("f ", bundle["obj"], obstacle_type)
            self.assertIn("NO_PAINT_OBSTACLES", self.exporter.export_dxf(4000.0, 2800.0, obstacles, waypoints))

        meter = [{"id": "obs_1", "type": "meter_panel", "x": 500, "y": 500, "w": 500, "h": 500, "depth_mm": 100}]
        obj = self.exporter.export_obj_and_mtl(4000.0, 2800.0, meter, waypoints)["obj"]
        self.assertIn("usemtl Electrical_Fixture_Material", obj)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m unittest discover -s tests -p "test_pipeline.py" -v`
Expected: `test_08` FAILS on `usemtl Electrical_Fixture_Material` (meter_panel currently gets no fixture geometry).

- [ ] **Step 3: Implement**

In `cad_exporter.py` replace `elif obs["type"] == "switchboard":` with:

```python
            elif obs["type"] in ("switchboard", "meter_panel"):
```

In `static/viewer3d.js` replace `} else if (obs.type === 'switchboard') {` with:

```javascript
            } else if (obs.type === 'switchboard' || obs.type === 'meter_panel') {
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m unittest discover -s tests -p "test_pipeline.py" -v`
Expected: all 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add cad_exporter.py static/viewer3d.js tests/test_pipeline.py
git commit -m "Render meter_panel as an electrical fixture and cover new classes in export tests" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: API returns detector, review count and honest confidence

**Files:**
- Modify: `app.py` (imports at lines 17-18, engine creation at line 36, `analyze_wall` at lines 86-228)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `GeometryEngine(detector=...)`, `GeometryEngine.detect(...)`, `load_default_detector()`.
- Produces: `_sanitize_wall_params(w, h, spray, overlap, buffer) -> 5-tuple`; `build_mission(wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles) -> (waypoints, path_stats, metrics)` (plans, computes metrics, writes every export file, refreshes `current_session` except `confidence`); `count_unverified(obstacles) -> int`; `/api/analyze` JSON gains `detector` (str), `needs_review` (int) and `confidence` may be `null`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api.py`:

```python
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module
from fakes import FakeDetector
from obstacle_detector import Detection

FORM = {
    "width_mm": "4000", "height_mm": "2800", "spray_width_mm": "250",
    "overlap_pct": "12", "safety_buffer_mm": "60", "sensitivity": "0.5",
    "sample_id": "residential",
}


class TestAnalyzeResponse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app_module.app.test_client()

    def test_reports_detector_and_review_count(self):
        data = json.loads(self.client.post("/api/analyze", data=FORM).data)
        self.assertIn(data["detector"], ("yolo11n-onnx", "classical-fallback"))
        unverified = sum(1 for o in data["obstacles"] if o["type"] == "unverified")
        self.assertEqual(data["needs_review"], unverified)

    def test_model_path_reports_unverified_and_null_confidence(self):
        detector = FakeDetector([Detection("window", 0.35, 100, 100, 400, 400)])
        with mock.patch.object(app_module.geometry_engine, "detector", detector):
            data = json.loads(self.client.post("/api/analyze", data=FORM).data)
        self.assertEqual(data["detector"], "yolo11n-onnx")
        self.assertEqual(data["needs_review"], 1)
        self.assertEqual(data["obstacles"][0]["type"], "unverified")

        with mock.patch.object(app_module.geometry_engine, "detector", FakeDetector([])):
            data = json.loads(self.client.post("/api/analyze", data=FORM).data)
        self.assertEqual(data["obstacles"], [])
        self.assertIsNone(data["confidence"])
        self.assertEqual(data["needs_review"], 0)

    def test_bad_number_is_a_400_not_a_500(self):
        res = self.client.post("/api/analyze", data={**FORM, "width_mm": "wide"})
        self.assertEqual(res.status_code, 400)
        self.assertFalse(json.loads(res.data)["success"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m unittest discover -s tests -p "test_api.py" -v`
Expected: FAIL (`KeyError: 'detector'` and the bad-number test gets 500).

- [ ] **Step 3: Update imports and engine creation**

In `app.py` replace:

```python
from geometry_engine import GeometryEngine
from cad_exporter import CADExporter
```

with:

```python
from geometry_engine import GeometryEngine
from cad_exporter import CADExporter
from obstacle_detector import load_default_detector
```

and replace `geometry_engine = GeometryEngine()` with:

```python
geometry_engine = GeometryEngine(detector=load_default_detector())
```

- [ ] **Step 4: Replace `analyze_wall` with helpers plus the new endpoint**

In `app.py`, replace everything from the line `@app.route("/api/analyze", methods=["POST"])` down to (not including) `@app.route("/api/download/<file_type>", methods=["GET"])` with:

```python
def _sanitize_wall_params(wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm):
    return (
        max(1000.0, min(50000.0, wall_w_mm)),
        max(1000.0, min(30000.0, wall_h_mm)),
        max(150.0, min(500.0, spray_width_mm)),
        max(0.0, min(40.0, overlap_pct)),
        max(10.0, min(300.0, safety_buffer_mm)),
    )


def count_unverified(obstacles):
    return sum(1 for obs in obstacles if obs.get("type") == "unverified")


def build_mission(wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles):
    """Plan coverage, compute metrics, write every export file and refresh the session cache."""
    waypoints, path_stats = geometry_engine.plan_coverage_path(
        wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles
    )
    metrics = geometry_engine.compute_metrics(wall_w_mm, wall_h_mm, obstacles, path_stats)

    dxf_content = cad_exporter.export_dxf(wall_w_mm, wall_h_mm, obstacles, waypoints)
    dxf_path = os.path.join(EXPORTS_DIR, "paintpilot_wall.dxf")
    with open(dxf_path, "w", encoding="utf-8") as f:
        f.write(dxf_content)

    obj_bundle = cad_exporter.export_obj_and_mtl(wall_w_mm, wall_h_mm, obstacles, waypoints)
    obj_path = os.path.join(EXPORTS_DIR, "paintpilot_wall.obj")
    mtl_path = os.path.join(EXPORTS_DIR, "paintpilot_wall.mtl")
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(obj_bundle["obj"])
    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write(obj_bundle["mtl"])

    # Package 3D bundle (OBJ + MTL) into a single zip archive for 1-click import
    zip_path = os.path.join(EXPORTS_DIR, "paintpilot_wall_3d_bundle.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(obj_path, arcname="paintpilot_wall.obj")
        zf.write(mtl_path, arcname="paintpilot_wall.mtl")

    mission_json = cad_exporter.export_mission_json(wall_w_mm, wall_h_mm, obstacles, waypoints, metrics)
    json_path = os.path.join(EXPORTS_DIR, "apr_mission_manifest.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(mission_json)

    motion_csv = cad_exporter.export_motion_csv(waypoints)
    csv_path = os.path.join(EXPORTS_DIR, "apr_waypoints.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(motion_csv)

    current_session.update({
        "wall_w_mm": wall_w_mm,
        "wall_h_mm": wall_h_mm,
        "obstacles": obstacles,
        "waypoints": waypoints,
        "metrics": metrics,
        "stats": path_stats,
    })
    return waypoints, path_stats, metrics


@app.route("/api/analyze", methods=["POST"])
def analyze_wall():
    """
    Core API endpoint:
    Detects obstacle zones in a wall photo, generates the 3D CAD mesh and Boustrophedon
    rover toolpath, and prepares exports. Uncertain detections come back as
    type "unverified" for the operator to review.
    """
    try:
        wall_w_mm = float(request.form.get("width_mm", 4000.0))
        wall_h_mm = float(request.form.get("height_mm", 2800.0))
        spray_width_mm = float(request.form.get("spray_width_mm", 250.0))
        overlap_pct = float(request.form.get("overlap_pct", 12.0))
        safety_buffer_mm = float(request.form.get("safety_buffer_mm", 60.0))
        sensitivity = float(request.form.get("sensitivity", 0.5))
        sample_id = request.form.get("sample_id", None)

        wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm = _sanitize_wall_params(
            wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm
        )

        # 1. Load image (from uploaded file or pre-packaged sample)
        img_bgr = None
        source_name = "Uploaded Image"

        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            source_name = file.filename
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        elif sample_id == "commercial":
            sample_path = os.path.join(SAMPLES_DIR, "benchmark_commercial_facade.jpg")
            if os.path.exists(sample_path):
                img_bgr = cv2.imread(sample_path)
                source_name = "Commercial Facade Benchmark"
        else:
            # Default or residential sample
            sample_path = os.path.join(SAMPLES_DIR, "benchmark_residential_facade.jpg")
            if os.path.exists(sample_path):
                img_bgr = cv2.imread(sample_path)
                source_name = "Residential Facade Benchmark"
            else:
                img_bgr = geometry_engine.generate_benchmark_wall_image()
                source_name = "Synthetic Calibrated Wall"

        if img_bgr is None:
            img_bgr = geometry_engine.generate_benchmark_wall_image()
            source_name = "Synthetic Calibrated Wall"

        # 2. Detect obstacles
        detection = geometry_engine.detect(img_bgr, wall_w_mm, wall_h_mm, sensitivity)
        obstacles = detection["obstacles"]
        confidence = detection["confidence"]

        # 3. Plan coverage, compute metrics, write exports
        waypoints, path_stats, metrics = build_mission(
            wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles
        )
        current_session["confidence"] = confidence

        # Encode image to base64 for 2D canvas overlay
        # Resize to max 1200px wide for snappy UI responsiveness
        h, w = img_bgr.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img_disp = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_disp = img_bgr

        _, buffer = cv2.imencode(".jpg", img_disp, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "success": True,
            "source_name": source_name,
            "detector": detection["detector"],
            "needs_review": count_unverified(obstacles),
            "wall": {
                "width_mm": wall_w_mm,
                "height_mm": wall_h_mm,
                "spray_width_mm": spray_width_mm,
                "overlap_pct": overlap_pct,
                "safety_buffer_mm": safety_buffer_mm
            },
            "obstacles": obstacles,
            "waypoints": waypoints,
            "stats": path_stats,
            "metrics": metrics,
            "confidence": confidence,
            "image_data_url": img_b64,
            "timestamp": time.strftime("%H:%M:%S")
        })

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


```

- [ ] **Step 5: Run the whole suite**

Run: `uv run python -m unittest discover -s tests -p "test_*.py" -v`
Expected: all PASS (including `test_07_flask_endpoints`).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_api.py
git commit -m "Report detector and review count from /api/analyze, share mission building" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: `/api/replan` and obstacle normalisation

**Files:**
- Modify: `geometry_engine.py` (add `import math` is already present; add method after `_obstacles_from_detections`), `app.py` (new route before `/api/download/<file_type>`)
- Test: `tests/test_api.py`, `tests/test_detection_integration.py`

**Interfaces:**
- Consumes: `build_mission`, `_sanitize_wall_params`, `count_unverified` (Task 5); `OBSTACLE_CLASS_INFO`, `UNVERIFIED_INFO`, `LEGACY_OBSTACLE_INFO` (Task 3).
- Produces: `GeometryEngine.normalize_obstacles(obstacles) -> list[dict]` (raises `ValueError`); `POST /api/replan` with JSON `{width_mm, height_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles}` returning `{success, obstacles, waypoints, stats, metrics, needs_review}` or 400 `{success: false, error}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_detection_integration.py` before `if __name__ == "__main__":`:

```python
class TestNormalizeObstacles(unittest.TestCase):
    engine = GeometryEngine()

    def box(self, **overrides):
        base = {"type": "unverified", "x": 100, "y": 200, "w": 300, "h": 400}
        base.update(overrides)
        return base

    def test_confirmed_class_refreshes_label_depth_and_id(self):
        result = self.engine.normalize_obstacles([self.box(type="ac_unit", label="stale", depth_mm=1)])
        self.assertEqual(result[0]["id"], "obs_1")
        self.assertEqual(result[0]["label"], "AC Unit #1")
        self.assertEqual(result[0]["depth_mm"], 400.0)

    def test_unverified_keeps_guess(self):
        result = self.engine.normalize_obstacles([self.box(guess="pipe")])
        self.assertEqual(result[0]["type"], "unverified")
        self.assertEqual(result[0]["guess"], "pipe")

    def test_legacy_types_are_accepted(self):
        self.assertEqual(self.engine.normalize_obstacles([self.box(type="switchboard")])[0]["depth_mm"], 35.0)

    def test_rejects_bad_input(self):
        for bad in (
            "nope",
            [5],
            [{"type": "window", "x": 1, "y": 1, "w": 10}],
            [self.box(w=-5)],
            [self.box(w="abc")],
            [self.box(type="spaceship")],
            [self.box(x=float("nan"))],
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.engine.normalize_obstacles(bad)
```

Append to `tests/test_api.py` before `if __name__ == "__main__":`:

```python
WALL = {"width_mm": 4000, "height_mm": 2800, "spray_width_mm": 250, "overlap_pct": 12, "safety_buffer_mm": 60}
BOX = {"type": "unverified", "x": 1500, "y": 1000, "w": 1000, "h": 800}


class TestReplan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app_module.app.test_client()

    def replan(self, obstacles, **wall):
        res = self.client.post("/api/replan", json={**WALL, **wall, "obstacles": obstacles})
        return res, json.loads(res.data)

    def test_dismissing_an_obstacle_changes_the_plan(self):
        _, with_box = self.replan([BOX])
        _, without = self.replan([])
        self.assertTrue(with_box["success"])
        self.assertGreater(len(with_box["waypoints"]), len(without["waypoints"]))
        self.assertEqual(with_box["needs_review"], 1)
        self.assertEqual(without["needs_review"], 0)

    def test_confirming_sets_class_label_and_depth_without_changing_the_route(self):
        _, unverified = self.replan([BOX])
        _, confirmed = self.replan([{**BOX, "type": "ac_unit"}])
        self.assertEqual(confirmed["obstacles"][0]["label"], "AC Unit #1")
        self.assertEqual(confirmed["obstacles"][0]["depth_mm"], 400.0)
        self.assertEqual(confirmed["needs_review"], 0)
        self.assertEqual(len(confirmed["waypoints"]), len(unverified["waypoints"]))

    def test_replan_refreshes_the_downloadable_exports(self):
        self.replan([BOX])
        with_box = self.client.get("/api/download/csv").data.count(b"\n")
        self.replan([])
        without = self.client.get("/api/download/csv").data.count(b"\n")
        self.assertGreater(with_box, without)

    def test_bad_requests_are_400(self):
        self.assertEqual(self.client.post("/api/replan", data="not json").status_code, 400)
        missing = {k: v for k, v in WALL.items() if k != "width_mm"}
        self.assertEqual(self.client.post("/api/replan", json={**missing, "obstacles": []}).status_code, 400)
        self.assertEqual(self.replan([{"type": "window", "x": 1, "y": 1, "w": 10}])[0].status_code, 400)
        self.assertEqual(self.replan([{**BOX, "type": "spaceship"}])[0].status_code, 400)
        self.assertEqual(self.replan([{**BOX, "w": -5}])[0].status_code, 400)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run python -m unittest discover -s tests -p "test_*.py" -v`
Expected: FAIL/ERROR (`normalize_obstacles` missing, `/api/replan` returns 404).

- [ ] **Step 3: Implement `normalize_obstacles`**

In `geometry_engine.py`, add this method directly before `def _detect_classical(`:

```python
    def normalize_obstacles(self, obstacles: Any) -> List[Dict[str, Any]]:
        """
        Validate client-supplied obstacles and refresh id, label and depth from the
        obstacle type, so the server stays the single source of truth for those.
        Raises ValueError with a message safe to show to the operator.
        """
        if not isinstance(obstacles, list):
            raise ValueError("obstacles must be a list")

        cleaned: List[Dict[str, Any]] = []
        for number, obs in enumerate(obstacles, start=1):
            if not isinstance(obs, dict):
                raise ValueError(f"obstacle {number} must be an object")
            try:
                x, y, w, h = (float(obs[key]) for key in ("x", "y", "w", "h"))
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"obstacle {number} needs numeric x, y, w and h")
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                raise ValueError(f"obstacle {number} has an invalid size or position")

            obs_type = obs.get("type", "unverified")
            if obs_type in OBSTACLE_CLASS_INFO:
                base_label, depth_mm = OBSTACLE_CLASS_INFO[obs_type]
            elif obs_type in LEGACY_OBSTACLE_INFO:
                base_label, depth_mm = LEGACY_OBSTACLE_INFO[obs_type]
            elif obs_type == "unverified":
                base_label, depth_mm = UNVERIFIED_INFO
            else:
                raise ValueError(f"obstacle {number} has unknown type '{obs_type}'")

            cleaned_obs = {
                "id": f"obs_{number}",
                "type": obs_type,
                "label": f"{base_label} #{number}",
                "x": round(x, 1),
                "y": round(y, 1),
                "w": round(w, 1),
                "h": round(h, 1),
                "depth_mm": depth_mm,
            }
            for optional in ("confidence", "norm"):
                if optional in obs:
                    cleaned_obs[optional] = obs[optional]
            if obs_type == "unverified" and obs.get("guess") in OBSTACLE_CLASS_INFO:
                cleaned_obs["guess"] = obs["guess"]
            cleaned.append(cleaned_obs)
        return cleaned

```

- [ ] **Step 4: Add the endpoint**

In `app.py`, insert directly before `@app.route("/api/download/<file_type>", methods=["GET"])`:

```python
@app.route("/api/replan", methods=["POST"])
def replan():
    """Re-plan the mission after the operator confirms or dismisses uncertain obstacles."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"success": False, "error": "Expected a JSON body"}), 400

    try:
        wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm = _sanitize_wall_params(
            float(payload["width_mm"]),
            float(payload["height_mm"]),
            float(payload["spray_width_mm"]),
            float(payload["overlap_pct"]),
            float(payload["safety_buffer_mm"]),
        )
        obstacles = geometry_engine.normalize_obstacles(payload.get("obstacles"))
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"success": False, "error": f"Invalid request: {exc}"}), 400

    waypoints, path_stats, metrics = build_mission(
        wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles
    )
    return jsonify({
        "success": True,
        "obstacles": obstacles,
        "waypoints": waypoints,
        "stats": path_stats,
        "metrics": metrics,
        "needs_review": count_unverified(obstacles),
    })


```

- [ ] **Step 5: Run the whole suite**

Run: `uv run python -m unittest discover -s tests -p "test_*.py" -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add app.py geometry_engine.py tests/test_api.py tests/test_detection_integration.py
git commit -m "Add /api/replan and server-side obstacle normalisation" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: Review UI, fallback banner and null confidence

Frontend only. There is no JS test harness in this repo, so verification is a syntax check plus a scripted manual pass with a demo detector.

**Files:**
- Modify: `templates/index.html`, `static/style.css` (append), `static/app.js`

**Interfaces:**
- Consumes: `/api/analyze` fields `detector`, `needs_review`, obstacle `type/confidence/guess/id`; `POST /api/replan` (Task 6).
- Produces: `refreshViews(data)`, `updateReviewUI(data)`, `resolveObstacle(id, newType | null)`, `replan(obstacles)` inside the `DOMContentLoaded` closure of `static/app.js`.

- [ ] **Step 1: Add the panel markup**

In `templates/index.html`, `index.html` is tab-indented and the edits below use single-line anchors so indentation does not matter. Replace the line `<!-- Multi-Format CAD & Robotics Mission Exports -->` with:

```html
<div class="detector-banner" id="detectorBanner" hidden>
						Trained model not loaded. Using basic edge detection, which is less reliable on real walls.
					</div>

					<div class="review-panel" id="reviewPanel" hidden>
						<div class="k">Obstacles to review <b id="reviewCount">0</b></div>
						<p class="review-hint">The detector was unsure about these. Paint keeps clear of them until you confirm a class or dismiss them.</p>
						<div id="reviewList"></div>
					</div>

					<!-- Multi-Format CAD & Robotics Mission Exports -->
```

Replace the line `<div class="k" style="margin-bottom:2px;">ENGINEERING EXPORTS</div>` with:

```html
<div class="k" style="margin-bottom:2px;">ENGINEERING EXPORTS</div>
						<div class="export-caution" id="exportCaution" hidden></div>
```

- [ ] **Step 2: Add the styles**

Append to the end of `static/style.css`:

```css
/* Detector status and obstacle review */
[hidden] {
	display: none !important;
}

.detector-banner {
	margin: 0 16px 12px;
	padding: 10px 12px;
	border: 1px solid #684f22;
	border-radius: 5px;
	background: #2a2110;
	color: var(--amber);
	font-size: 11.5px;
	line-height: 1.5;
}

.review-panel {
	padding: 14px 16px;
	border-top: 1px solid var(--line);
}

.review-panel .k {
	color: var(--muted);
	font: 10px 'DM Mono', monospace;
	text-transform: uppercase;
	letter-spacing: 0.8px;
}

.review-panel .k b {
	float: right;
	color: var(--amber);
}

.review-hint {
	margin: 8px 0 10px;
	color: var(--muted);
	font-size: 11.5px;
	line-height: 1.5;
}

.review-row {
	border: 1px solid var(--line);
	border-radius: 5px;
	padding: 8px 10px;
	margin-bottom: 8px;
	background: #0e2027;
}

.review-info {
	font: 11px 'DM Mono', monospace;
	color: var(--text);
	margin-bottom: 6px;
}

.review-info span {
	color: var(--muted);
}

.review-actions {
	display: flex;
	gap: 6px;
}

.review-select {
	flex: 1;
	min-width: 0;
	background: #071016;
	color: var(--text);
	border: 1px solid var(--line-light);
	border-radius: 4px;
	padding: 5px 6px;
	font: 11px 'DM Mono', monospace;
}

.export-caution {
	color: var(--amber);
	font-size: 11px;
	line-height: 1.5;
}
```

- [ ] **Step 3: Wire the DOM references and state**

In `static/app.js`, replace the line `const activeObstacleCount = $('activeObstacleCount');` with:

```javascript
    const activeObstacleCount = $('activeObstacleCount');
    const detectorBanner = $('detectorBanner');
    const reviewPanel = $('reviewPanel');
    const reviewCount = $('reviewCount');
    const reviewList = $('reviewList');
    const exportCaution = $('exportCaution');
```

- [ ] **Step 4: Route analysis results through `refreshViews`**

In `static/app.js`, replace:

```javascript
            currentData = data;
            updateMetrics(data);
            updateTelemetry(data);

            // Update 3D Viewer
            if (viewer3D) {
                viewer3D.loadWallData(data);
            }

            // Update Autonomous Simulation Engine
            if (roverSim) {
                roverSim.init(data);
            }
```

with:

```javascript
            currentData = data;
            refreshViews(data);
```

Replace the line `appendLog(`Model generated: ${data.obstacles.length} keep-out zones detected, ${data.waypoints.length} waypoints planned`, 'success');` with:

```javascript
            const reviewNote = data.needs_review ? `, ${data.needs_review} need review` : '';
            appendLog(`Model generated: ${data.obstacles.length} keep-out zones detected${reviewNote}, ${data.waypoints.length} waypoints planned`, 'success');
```

- [ ] **Step 5: Handle null confidence**

In `static/app.js`, replace:

```javascript
        confidenceValue.textContent = `${confidence}%`;
        confidenceBar.style.width = `${Math.min(100, Math.max(10, confidence))}%`;
```

with:

```javascript
        const hasConfidence = confidence !== null && confidence !== undefined;
        confidenceValue.textContent = hasConfidence ? `${confidence}%` : 'n/a';
        confidenceBar.style.width = hasConfidence ? `${Math.min(100, Math.max(10, confidence))}%` : '0%';
```

- [ ] **Step 6: Add the review logic**

In `static/app.js`, insert directly before the line `// 2D High-Resolution Canvas Rendering`:

```javascript
    function refreshViews(data) {
        updateMetrics(data);
        updateTelemetry(data);
        updateReviewUI(data);
        if (viewer3D) viewer3D.loadWallData(data);
        if (roverSim) roverSim.init(data);
    }

    const CLASS_OPTIONS = [
        ['window', 'Window'],
        ['door', 'Door'],
        ['ac_unit', 'AC unit'],
        ['meter_panel', 'Meter panel'],
        ['pipe', 'Pipe'],
        ['grill', 'Grill']
    ];

    function updateReviewUI(data) {
        const pending = data.obstacles.filter((obs) => obs.type === 'unverified');

        detectorBanner.hidden = data.detector !== 'classical-fallback';
        reviewPanel.hidden = pending.length === 0;
        exportCaution.hidden = pending.length === 0;
        reviewCount.textContent = pending.length;
        exportCaution.textContent =
            `${pending.length} obstacle${pending.length === 1 ? '' : 's'} not reviewed. ` +
            'The plan keeps clear of them, so exports are safe to use but may leave paintable wall unpainted.';

        reviewList.innerHTML = '';
        pending.forEach((obs) => {
            const row = document.createElement('div');
            row.className = 'review-row';

            const info = document.createElement('div');
            info.className = 'review-info';
            const pct = Math.round((obs.confidence || 0) * 100);
            info.textContent = obs.label + ' ';
            const meta = document.createElement('span');
            meta.textContent = `${pct}% sure, ${obs.w}×${obs.h} mm`;
            info.appendChild(meta);

            const select = document.createElement('select');
            select.className = 'review-select';
            select.setAttribute('aria-label', `Class for ${obs.label}`);
            CLASS_OPTIONS.forEach(([value, text]) => {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = text;
                if (value === obs.guess) option.selected = true;
                select.appendChild(option);
            });

            const confirmBtn = document.createElement('button');
            confirmBtn.type = 'button';
            confirmBtn.className = 'export-btn';
            confirmBtn.textContent = 'Confirm';
            confirmBtn.addEventListener('click', () => resolveObstacle(obs.id, select.value));

            const dismissBtn = document.createElement('button');
            dismissBtn.type = 'button';
            dismissBtn.className = 'export-btn';
            dismissBtn.textContent = 'Dismiss';
            dismissBtn.addEventListener('click', () => resolveObstacle(obs.id, null));

            const actions = document.createElement('div');
            actions.className = 'review-actions';
            actions.append(select, confirmBtn, dismissBtn);

            row.append(info, actions);
            reviewList.appendChild(row);
        });
    }

    function resolveObstacle(id, newType) {
        if (!currentData) return;
        const obstacles = currentData.obstacles
            .map((obs) => (obs.id === id ? (newType ? { ...obs, type: newType } : null) : obs))
            .filter(Boolean);
        replan(obstacles);
    }

    function replan(obstacles) {
        const wall = currentData.wall;
        appendLog('Re-planning mission with reviewed obstacles...', 'info');

        fetch('/api/replan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                width_mm: wall.width_mm,
                height_mm: wall.height_mm,
                spray_width_mm: wall.spray_width_mm,
                overlap_pct: wall.overlap_pct,
                safety_buffer_mm: wall.safety_buffer_mm,
                obstacles
            })
        })
        .then((res) => res.json().then((body) => ({ ok: res.ok, body })))
        .then(({ ok, body }) => {
            if (!ok || !body.success) throw new Error(body.error || 'Re-plan failed');
            currentData = {
                ...currentData,
                obstacles: body.obstacles,
                waypoints: body.waypoints,
                stats: body.stats,
                metrics: body.metrics,
                needs_review: body.needs_review
            };
            refreshViews(currentData);
            render2DCanvas();
            appendLog(`Mission re-planned: ${body.obstacles.length} keep-out zones, ${body.waypoints.length} waypoints`, 'success');
        })
        .catch((err) => appendLog(`Re-plan error: ${err.message}`, 'warn'));
    }

```

- [ ] **Step 7: Style unverified obstacles on the 2D map**

In `static/app.js`, replace:

```javascript
            // Obstacle fill & border (Red/Coral)
            ctx2d.fillStyle = 'rgba(250, 116, 110, 0.25)';
            ctx2d.fillRect(ox, oy, ow, oh);

            ctx2d.strokeStyle = '#fa746e';
            ctx2d.lineWidth = 2;
            ctx2d.strokeRect(ox, oy, ow, oh);
```

with:

```javascript
            // Obstacle fill & border: red for confirmed classes, dashed amber while unverified
            const unverified = obs.type === 'unverified';
            ctx2d.fillStyle = unverified ? 'rgba(244, 184, 96, 0.22)' : 'rgba(250, 116, 110, 0.25)';
            ctx2d.fillRect(ox, oy, ow, oh);

            ctx2d.save();
            if (unverified) ctx2d.setLineDash([6, 4]);
            ctx2d.strokeStyle = unverified ? '#f4b860' : '#fa746e';
            ctx2d.lineWidth = 2;
            ctx2d.strokeRect(ox, oy, ow, oh);
            ctx2d.restore();
```

and replace:

```javascript
            ctx2d.fillStyle = '#fa746e';
            ctx2d.font = '10px "DM Mono", monospace';
            ctx2d.fillText(obs.label, ox + 6, oy + 16);
```

with:

```javascript
            ctx2d.fillStyle = unverified ? '#f4b860' : '#fa746e';
            ctx2d.font = '10px "DM Mono", monospace';
            ctx2d.fillText(obs.label, ox + 6, oy + 16);
```

- [ ] **Step 8: Syntax check**

Run: `node --check static/app.js` (skip if Node is not installed; the manual pass below then covers it).
Expected: no output.

- [ ] **Step 9: Manual pass with a demo detector**

Run this to start the app with a stand-in detector that returns one confident and two uncertain boxes:

```bash
uv run python -c "
import app as m
from obstacle_detector import Detection
class Demo:
    def detect(self, image_bgr, min_score):
        h, w = image_bgr.shape[:2]
        return [Detection('window', 0.9, w*0.10, h*0.20, w*0.35, h*0.55),
                Detection('door', 0.35, w*0.60, h*0.40, w*0.80, h*0.95),
                Detection('pipe', 0.30, w*0.90, h*0.10, w*0.93, h*0.90)]
m.geometry_engine.detector = Demo()
m.app.run(port=5000)
"
```

Open `http://127.0.0.1:5000` and check:
1. The "Obstacles to review" panel lists 2 rows with the guessed class preselected; no fallback banner; exports show the "not reviewed" caution.
2. On the 2D PLAN MAP tab the two uncertain boxes are dashed amber, the window is red.
3. Confirm one row: it disappears from the list, the caution says 1 obstacle, the count and waypoint log update, the log shows "Mission re-planned".
4. Dismiss the other: panel and caution disappear, waypoint count drops.
5. Stop the server, run `uv run python app.py` (no demo detector): the amber fallback banner shows and confidence renders as a percentage.

- [ ] **Step 10: Commit**

```bash
git add templates/index.html static/style.css static/app.js
git commit -m "Add obstacle review panel, fallback banner and null-safe confidence" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 8: Four-corner wall rectification (backend)

This task and Task 9 are cuttable as a pair. If cut, the UI keeps treating the whole photo as the wall and the README states that photos must be shot square-on.

**Files:**
- Modify: `geometry_engine.py` (imports at top; new methods before `_detect_classical`), `app.py` (`analyze_wall`)
- Test: `tests/test_detection_integration.py`, `tests/test_api.py`

**Interfaces:**
- Produces: `GeometryEngine.parse_corners(raw: str) -> list[tuple[float, float]]` (static; raises `ValueError`); `GeometryEngine.rectify_wall(image_bgr, corners_norm, wall_w_mm, wall_h_mm, max_dim=1200) -> ndarray` sized to the wall's aspect ratio; `/api/analyze` accepts an optional `corners` form field (JSON `[[x,y],[x,y],[x,y],[x,y]]`, fractions 0 to 1, ordered top-left, top-right, bottom-right, bottom-left) and returns the rectified image as `image_data_url`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_detection_integration.py` before `if __name__ == "__main__":`:

```python
class TestRectification(unittest.TestCase):
    engine = GeometryEngine()
    VALID = "[[0.2,0.2],[0.8,0.25],[0.75,0.85],[0.25,0.8]]"

    def test_parse_accepts_a_valid_quad(self):
        points = self.engine.parse_corners(self.VALID)
        self.assertEqual(len(points), 4)
        self.assertEqual(points[0], (0.2, 0.2))

    def test_parse_rejects_bad_corners(self):
        for bad in (
            "not json",
            "[[0.1,0.1],[0.9,0.1],[0.9,0.9]]",                      # three points
            "[[0.1,0.1],[0.9,0.1],[0.9,1.5],[0.1,0.9]]",            # out of range
            "[[0.1,0.1],[0.1,0.9],[0.9,0.9],[0.9,0.1]]",            # wrong winding
            "[[0.1,0.1],[0.9,0.9],[0.9,0.1],[0.1,0.9]]",            # bow-tie
            "[[0.1,0.1],[0.11,0.1],[0.11,0.11],[0.1,0.11]]",        # too small
            None,
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.engine.parse_corners(bad)

    def test_rectify_straightens_a_skewed_wall(self):
        import cv2

        image = np.zeros((800, 1000, 3), dtype=np.uint8)
        quad = np.array([[200, 160], [800, 200], [750, 680], [250, 640]], dtype=np.int32)
        cv2.fillPoly(image, [quad], (255, 255, 255))
        corners = [(x / 1000, y / 800) for x, y in quad]

        out = self.engine.rectify_wall(image, corners, 4000.0, 2000.0)
        self.assertEqual(out.shape, (600, 1200, 3))
        self.assertGreater(float(out.mean()), 240.0)

    def test_rectify_tall_wall_uses_max_dim_for_height(self):
        image = np.zeros((800, 1000, 3), dtype=np.uint8)
        out = self.engine.rectify_wall(image, self.engine.parse_corners(self.VALID), 2000.0, 4000.0)
        self.assertEqual(out.shape, (1200, 600, 3))
```

Append to `tests/test_api.py` before `if __name__ == "__main__":`:

```python
class TestAnalyzeCorners(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app_module.app.test_client()

    def upload(self, corners=None):
        import io
        import cv2

        image = app_module.geometry_engine.generate_benchmark_wall_image(1000, 700)
        _, png = cv2.imencode(".png", image)
        data = {**FORM, "file": (io.BytesIO(png.tobytes()), "wall.png")}
        data.pop("sample_id")
        if corners is not None:
            data["corners"] = corners
        return self.client.post("/api/analyze", data=data, content_type="multipart/form-data")

    def test_corners_rectify_the_returned_image_to_the_wall_aspect(self):
        import base64
        import cv2
        import numpy as np

        res = self.upload("[[0.05,0.05],[0.95,0.08],[0.93,0.95],[0.07,0.92]]")
        self.assertEqual(res.status_code, 200)
        url = json.loads(res.data)["image_data_url"]
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[:2], (840, 1200))  # 4000 x 2800 wall at max_dim 1200

    def test_bad_corners_are_a_400(self):
        self.assertEqual(self.upload("[[0,0],[1,0]]").status_code, 400)

    def test_no_corners_keeps_the_original_behaviour(self):
        self.assertEqual(self.upload().status_code, 200)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run python -m unittest discover -s tests -p "test_*.py" -v`
Expected: ERROR `AttributeError: ... has no attribute 'parse_corners'`; the API corner tests return 200 for bad corners.

- [ ] **Step 3: Implement**

In `geometry_engine.py`, add `import json` next to `import math` at the top of the file. Then add these methods directly before `def normalize_obstacles(`:

```python
    @staticmethod
    def parse_corners(raw: Any) -> List[Tuple[float, float]]:
        """
        Parse a JSON list of four [x, y] image fractions (0 to 1), ordered
        top-left, top-right, bottom-right, bottom-left. Raises ValueError.
        """
        try:
            points = [(float(x), float(y)) for x, y in json.loads(raw)]
        except (TypeError, ValueError):
            raise ValueError("corners must be a JSON list of four [x, y] pairs")
        if len(points) != 4:
            raise ValueError("corners must contain exactly four points")
        if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for point in points for v in point):
            raise ValueError("corner coordinates must be between 0 and 1")

        # Shoelace area is positive for top-left, top-right, bottom-right, bottom-left in image coordinates
        signed_area = sum(
            x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        ) / 2.0
        if signed_area < 0.01:
            raise ValueError(
                "corners must go top-left, top-right, bottom-right, bottom-left and enclose the wall"
            )
        if not cv2.isContourConvex(np.float32(points).reshape(-1, 1, 2)):
            raise ValueError("corners must form a convex four-sided shape")
        return points

    def rectify_wall(
        self,
        image_bgr: np.ndarray,
        corners_norm: List[Tuple[float, float]],
        wall_w_mm: float,
        wall_h_mm: float,
        max_dim: int = 1200
    ) -> np.ndarray:
        """Perspective-correct the photo so the marked wall fills the frame at the wall's aspect ratio."""
        img_h, img_w = image_bgr.shape[:2]
        src = np.float32([[x * img_w, y * img_h] for x, y in corners_norm])

        aspect = wall_w_mm / wall_h_mm
        if aspect >= 1.0:
            out_w, out_h = max_dim, max(1, int(round(max_dim / aspect)))
        else:
            out_w, out_h = max(1, int(round(max_dim * aspect))), max_dim
        dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])

        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image_bgr, matrix, (out_w, out_h))

```

In `app.py`, inside `analyze_wall`, replace the line `# 2. Detect obstacles` with:

```python
        # 1b. Straighten the photo if the operator marked the wall corners
        corners_raw = request.form.get("corners")
        if corners_raw:
            corners = geometry_engine.parse_corners(corners_raw)
            img_bgr = geometry_engine.rectify_wall(img_bgr, corners, wall_w_mm, wall_h_mm)

        # 2. Detect obstacles
```

- [ ] **Step 4: Run the whole suite**

Run: `uv run python -m unittest discover -s tests -p "test_*.py" -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add geometry_engine.py app.py tests/test_detection_integration.py tests/test_api.py
git commit -m "Add four-corner wall rectification to /api/analyze" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 9: Corner picker UI and honest labels

Frontend only; verification is a syntax check plus a manual pass.

**Files:**
- Create: `static/corner_picker.js`
- Modify: `templates/index.html`, `static/style.css` (append), `static/app.js`

**Interfaces:**
- Consumes: `/api/analyze` optional `corners` field (Task 8).
- Produces: `window.CornerPicker(canvas, onChange)` with `setImage(img)`, `undo()`, `isComplete()`, `getCorners()` (array of four `[x, y]` fractions or `null`) and `points`.

- [ ] **Step 1: Create the picker widget**

Create `static/corner_picker.js`:

```javascript
/**
 * CornerPicker - lets the operator mark the four wall corners on a photo.
 * Points are 0-1 fractions of the image, ordered top-left, top-right, bottom-right, bottom-left.
 */
(function () {
    class CornerPicker {
        constructor(canvas, onChange) {
            this.canvas = canvas;
            this.ctx = canvas.getContext('2d');
            this.onChange = onChange || function () {};
            this.image = null;
            this.points = [];
            this.frame = { x: 0, y: 0, w: 0, h: 0 };
            canvas.addEventListener('click', (event) => this.handleClick(event));
        }

        setImage(image) {
            this.image = image;
            this.points = [];
            const scale = Math.min(this.canvas.width / image.naturalWidth, this.canvas.height / image.naturalHeight);
            const w = image.naturalWidth * scale;
            const h = image.naturalHeight * scale;
            this.frame = { x: (this.canvas.width - w) / 2, y: (this.canvas.height - h) / 2, w, h };
            this.render();
            this.onChange();
        }

        handleClick(event) {
            if (!this.image || this.points.length >= 4) return;
            const box = this.canvas.getBoundingClientRect();
            const px = (event.clientX - box.left) * (this.canvas.width / box.width);
            const py = (event.clientY - box.top) * (this.canvas.height / box.height);
            const nx = (px - this.frame.x) / this.frame.w;
            const ny = (py - this.frame.y) / this.frame.h;
            if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return;
            this.points.push([nx, ny]);
            this.render();
            this.onChange();
        }

        undo() {
            this.points.pop();
            this.render();
            this.onChange();
        }

        isComplete() {
            return this.points.length === 4;
        }

        getCorners() {
            return this.isComplete() ? this.points.map((p) => [p[0], p[1]]) : null;
        }

        render() {
            const { ctx, canvas, frame } = this;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (!this.image) return;
            ctx.drawImage(this.image, frame.x, frame.y, frame.w, frame.h);

            const toCanvas = (p) => [frame.x + p[0] * frame.w, frame.y + p[1] * frame.h];
            ctx.lineWidth = 2;
            ctx.strokeStyle = '#5ce1d2';
            ctx.beginPath();
            this.points.forEach((p, i) => {
                const [x, y] = toCanvas(p);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            if (this.isComplete()) ctx.closePath();
            ctx.stroke();

            ctx.font = '14px "DM Mono", monospace';
            this.points.forEach((p, i) => {
                const [x, y] = toCanvas(p);
                ctx.fillStyle = '#f4b860';
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = '#e7f3f1';
                ctx.fillText(String(i + 1), x + 10, y - 8);
            });
        }
    }

    window.CornerPicker = CornerPicker;
})();
```

- [ ] **Step 2: Add markup and correct the misleading labels**

In `templates/index.html`:

Replace `Auto-rectification & edge segmentation` with `Shoot square-on, or mark the wall corners after upload`.

Replace `<span class="step-node complete">1. PHOTO RECTIFY</span>` with `<span class="step-node complete">1. WALL PHOTO</span>`.

Replace `<small>Edge Segment</small>` with `<small>Detect Obstacles</small>`.

Replace `<!-- Physical Wall Dimension Inputs -->` with:

```html
<div class="corners-row" id="cornersRow" hidden>
							<button type="button" class="pill-btn" id="cornersBtn">Mark wall corners</button>
							<small id="cornersStatus" style="color:var(--muted);">Whole photo is treated as the wall.</small>
						</div>

						<!-- Physical Wall Dimension Inputs -->
```

Replace `<script src="/static/app.js"></script>` with:

```html
<script src="/static/corner_picker.js"></script>
	<script src="/static/app.js"></script>
```

Replace `<!-- Full-Screen Autonomous Rover Simulation Modal Deck -->` with:

```html
<!-- Wall corner picker -->
	<section class="corner-modal" id="cornerModal" style="display:none;" role="dialog" aria-modal="true" aria-labelledby="cornerTitle">
		<div class="corner-dialog">
			<h3 id="cornerTitle">Mark the wall corners</h3>
			<p id="cornerHint">Click the wall's top-left corner.</p>
			<canvas id="cornerCanvas" width="900" height="600"></canvas>
			<div class="corner-actions">
				<button type="button" class="pill-btn" id="cornerUndoBtn">Undo last point</button>
				<button type="button" class="pill-btn" id="cornerClearBtn">Use whole photo</button>
				<button type="button" class="pill-btn" id="cornerCancelBtn">Cancel</button>
				<button type="button" class="primary-btn" id="cornerApplyBtn" disabled>Apply corners</button>
			</div>
		</div>
	</section>

	<!-- Full-Screen Autonomous Rover Simulation Modal Deck -->
```

- [ ] **Step 3: Add the styles**

Append to `static/style.css`:

```css
/* Wall corner picker */
.corners-row {
	display: flex;
	flex-direction: column;
	gap: 6px;
	margin-bottom: 10px;
}

.corner-modal {
	position: fixed;
	inset: 0;
	z-index: 60;
	align-items: center;
	justify-content: center;
	background: rgba(4, 10, 13, 0.86);
	padding: 20px;
}

.corner-dialog {
	background: var(--panel);
	border: 1px solid var(--line-light);
	border-radius: 8px;
	padding: 18px;
	max-width: 940px;
	width: 100%;
}

.corner-dialog h3 {
	margin: 0 0 4px;
	font-size: 15px;
}

.corner-dialog p {
	margin: 0 0 12px;
	color: var(--cyan);
	font-size: 12.5px;
}

.corner-dialog canvas {
	display: block;
	width: 100%;
	height: auto;
	background: var(--bg);
	border: 1px solid var(--line);
	border-radius: 5px;
	cursor: crosshair;
}

.corner-actions {
	display: flex;
	flex-wrap: wrap;
	gap: 8px;
	margin-top: 12px;
}

.corner-actions .primary-btn {
	width: auto;
	padding: 9px 16px;
	margin-left: auto;
}
```

- [ ] **Step 4: Wire the behaviour in `static/app.js`**

Replace the line `const exportCaution = $('exportCaution');` with:

```javascript
    const exportCaution = $('exportCaution');

    // Wall corner picker
    const cornersRow = $('cornersRow');
    const cornersBtn = $('cornersBtn');
    const cornersStatus = $('cornersStatus');
    const cornerModal = $('cornerModal');
    const cornerCanvas = $('cornerCanvas');
    const cornerHint = $('cornerHint');
    const cornerUndoBtn = $('cornerUndoBtn');
    const cornerClearBtn = $('cornerClearBtn');
    const cornerCancelBtn = $('cornerCancelBtn');
    const cornerApplyBtn = $('cornerApplyBtn');
```

Replace the line `let loadedImage2D = null;` with:

```javascript
    let loadedImage2D = null;
    let wallCorners = null;
    let cornerPreviewUrl = null;
```

Reset corners whenever the image source changes. Replace:

```javascript
        selectedFile = null;
        activeSampleId = 'residential';
```

with:

```javascript
        selectedFile = null;
        activeSampleId = 'residential';
        resetCorners(false);
```

Replace:

```javascript
        selectedFile = null;
        activeSampleId = 'commercial';
```

with:

```javascript
        selectedFile = null;
        activeSampleId = 'commercial';
        resetCorners(false);
```

Replace:

```javascript
            selectedFile = e.target.files[0];
            activeSampleId = null;
```

with:

```javascript
            selectedFile = e.target.files[0];
            activeSampleId = null;
            resetCorners(true);
```

Replace:

```javascript
            selectedFile = e.dataTransfer.files[0];
            activeSampleId = null;
```

with:

```javascript
            selectedFile = e.dataTransfer.files[0];
            activeSampleId = null;
            resetCorners(true);
```

Send the corners with the analysis request. Replace:

```javascript
        if (selectedFile) {
            formData.append('file', selectedFile);
        } else if (activeSampleId) {
```

with:

```javascript
        if (selectedFile) {
            formData.append('file', selectedFile);
            if (wallCorners) formData.append('corners', JSON.stringify(wallCorners));
        } else if (activeSampleId) {
```

Insert directly before the line `// Core Analysis Execution`:

```javascript
    // Wall corner picker
    const CORNER_PROMPTS = [
        "Click the wall's top-left corner.",
        'Click the top-right corner.',
        'Click the bottom-right corner.',
        'Click the bottom-left corner.',
        'All four corners set. Apply to straighten the photo.'
    ];

    const cornerPicker = new window.CornerPicker(cornerCanvas, updateCornerHint);

    function updateCornerHint() {
        const count = cornerPicker.points.length;
        cornerHint.textContent = CORNER_PROMPTS[count];
        cornerApplyBtn.disabled = count !== 4;
    }

    function updateCornersStatus() {
        cornersStatus.textContent = wallCorners
            ? 'Wall corners set. The photo is straightened before detection.'
            : 'Whole photo is treated as the wall.';
    }

    function resetCorners(hasUpload) {
        wallCorners = null;
        cornersRow.hidden = !hasUpload;
        updateCornersStatus();
    }

    function closeCornerModal() {
        cornerModal.style.display = 'none';
    }

    cornersBtn.addEventListener('click', () => {
        if (!selectedFile) return;
        if (cornerPreviewUrl) URL.revokeObjectURL(cornerPreviewUrl);
        cornerPreviewUrl = URL.createObjectURL(selectedFile);
        const image = new Image();
        image.onload = () => {
            cornerPicker.setImage(image);
            cornerModal.style.display = 'flex';
        };
        image.src = cornerPreviewUrl;
    });

    cornerUndoBtn.addEventListener('click', () => cornerPicker.undo());
    cornerCancelBtn.addEventListener('click', closeCornerModal);
    cornerClearBtn.addEventListener('click', () => {
        wallCorners = null;
        updateCornersStatus();
        closeCornerModal();
        runAnalysis();
    });
    cornerApplyBtn.addEventListener('click', () => {
        wallCorners = cornerPicker.getCorners();
        updateCornersStatus();
        closeCornerModal();
        runAnalysis();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && cornerModal.style.display !== 'none') closeCornerModal();
    });

```

- [ ] **Step 5: Syntax check**

Run: `node --check static/app.js && node --check static/corner_picker.js` (skip if Node is not installed).
Expected: no output.

- [ ] **Step 6: Manual pass**

Run `uv run python app.py`, open `http://127.0.0.1:5000`, and check:
1. Benchmark presets: the "Mark wall corners" row is hidden.
2. Upload a photo taken at an angle (or any photo): the row appears; the upload hint reads "Shoot square-on, or mark the wall corners after upload".
3. Click "Mark wall corners": the modal opens; prompts step through top-left to bottom-left; the polygon closes after four points; "Apply corners" is disabled until four points exist; "Undo last point" removes the last; Escape closes.
4. Apply: analysis reruns, the 2D plan map shows the straightened photo filling the wall, and the status text reads "Wall corners set...".
5. "Use whole photo" resets the status text and reruns.
6. Choosing a preset again hides the row.

- [ ] **Step 7: Commit**

```bash
git add static/corner_picker.js static/app.js static/style.css templates/index.html
git commit -m "Add wall corner picker and correct rectification labels" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 10: Dataset builder for training

**Files:**
- Create: `training/build_dataset.py`, `training/class_map.json`
- Test: `tests/test_training_tools.py`

**Interfaces:**
- Consumes: `CLASS_NAMES` from `obstacle_detector`.
- Produces: `parse_names(yaml_text) -> list[str]`; `remap_label_lines(lines, source_names, mapping, class_names) -> list[str]`; `merge_sources(sources_dir, class_map, out_dir, class_names, heldout="team") -> {"train": int, "val": int, "test": int}`; `write_dataset_yaml(out_dir, class_names)`. CLI: `python training/build_dataset.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_training_tools.py`:

```python
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

from build_dataset import merge_sources, parse_names, remap_label_lines, write_dataset_yaml
from obstacle_detector import CLASS_NAMES


class TestParseNames(unittest.TestCase):
    def test_inline_list(self):
        self.assertEqual(parse_names("train: x\nnames: ['door', 'window']\nnc: 2"), ["door", "window"])

    def test_index_mapping(self):
        self.assertEqual(parse_names("names:\n  1: window\n  0: door\nnc: 2"), ["door", "window"])

    def test_block_list(self):
        self.assertEqual(parse_names("names:\n- door\n- window\nnc: 2"), ["door", "window"])

    def test_missing_names_raises(self):
        with self.assertRaises(ValueError):
            parse_names("train: x")


class TestRemapLabelLines(unittest.TestCase):
    NAMES = ["door", "window", "wall"]
    MAPPING = {"door": "door", "window": "window"}

    def remap(self, lines):
        return remap_label_lines(lines, self.NAMES, self.MAPPING, CLASS_NAMES)

    def test_box_is_remapped_to_apr_class_ids(self):
        self.assertEqual(self.remap(["1 0.5 0.5 0.2 0.2"]), ["0 0.500000 0.500000 0.200000 0.200000"])

    def test_unmapped_out_of_range_blank_and_malformed_lines_are_dropped(self):
        self.assertEqual(self.remap(["2 0.5 0.5 0.2 0.2", "9 0.5 0.5 0.2 0.2", "", "0 0.5 0.5 0.2"]), [])

    def test_oriented_box_becomes_its_axis_aligned_envelope(self):
        self.assertEqual(
            self.remap(["0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4"]),
            ["1 0.200000 0.300000 0.200000 0.200000"],
        )


def write(path, text=b""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text if isinstance(text, bytes) else text.encode())


class TestMergeSources(unittest.TestCase):
    def test_splits_remaps_and_keeps_team_test_held_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources, out = root / "sources", root / "merged"

            write(sources / "pub" / "data.yaml", "names: ['door', 'window', 'wall']")
            write(sources / "pub" / "train" / "images" / "a.jpg", b"x")
            write(sources / "pub" / "train" / "labels" / "a.txt", "0 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.1 0.1\n")
            write(sources / "pub" / "test" / "images" / "b.jpg", b"x")

            write(sources / "team" / "data.yaml", "names: ['window']")
            write(sources / "team" / "valid" / "images" / "c.jpg", b"x")  # no label file: a clean wall
            write(sources / "team" / "test" / "images" / "d.jpg", b"x")
            write(sources / "team" / "test" / "labels" / "d.txt", "0 0.5 0.5 0.3 0.3\n")

            write(sources / "other" / "data.yaml", "names: ['x']")  # not in the class map

            class_map = {"pub": {"door": "door", "window": "window"}, "team": {"window": "window"}}
            counts = merge_sources(sources, class_map, out, CLASS_NAMES, heldout="team")

            self.assertEqual(counts, {"train": 2, "val": 1, "test": 1})
            self.assertEqual((out / "train" / "labels" / "pub_a.txt").read_text(), "1 0.500000 0.500000 0.200000 0.200000\n")
            self.assertTrue((out / "train" / "images" / "pub_b.jpg").exists())  # public test feeds training
            self.assertFalse((out / "test" / "images" / "pub_b.jpg").exists())
            self.assertEqual((out / "val" / "labels" / "team_c.txt").read_text(), "")
            self.assertTrue((out / "test" / "images" / "team_d.jpg").exists())


class TestWriteDatasetYaml(unittest.TestCase):
    def test_lists_classes_in_index_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_dataset_yaml(Path(tmp), CLASS_NAMES)
            text = (Path(tmp) / "dataset.yaml").read_text()
        self.assertIn("names:\n  0: window\n  1: door", text)
        self.assertIn("test: test/images", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m unittest discover -s tests -p "test_training_tools.py" -v`
Expected: ERROR `ModuleNotFoundError: No module named 'build_dataset'`.

- [ ] **Step 3: Implement**

Create `training/build_dataset.py`:

```python
"""
Merge YOLO-format source datasets into one training set using APR's class list.

Layout expected under --sources:
    <sources>/<name>/{train,valid,test}/{images,labels}/    (a Roboflow "YOLO" export)
    <sources>/<name>/data.yaml                               (only its `names` entry is read)
--class-map is JSON: {"<name>": {"<source class>": "<APR class>", ...}, ...}
Source classes not listed are dropped; sources missing from the map are skipped.

Split rule: the source named --heldout (default "team") keeps its test split as the merged
test set. Every other split of every source feeds train or val. Held-out photos never train.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from obstacle_detector import CLASS_NAMES  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_names(yaml_text: str) -> List[str]:
    """Read the `names` entry of a data.yaml (inline list, index mapping or block list)."""
    lines = yaml_text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("names:"):
            continue
        rest = line[len("names:"):].strip()
        if rest.startswith("["):
            return [n.strip().strip("'\"") for n in rest.strip("[]").split(",") if n.strip()]

        indexed: Dict[int, str] = {}
        listed: List[str] = []
        for follow in lines[i + 1:]:
            stripped = follow.strip()
            if not stripped or not (follow.startswith((" ", "\t")) or stripped.startswith("- ")):
                break
            if stripped.startswith("- "):
                listed.append(stripped[2:].strip().strip("'\""))
            elif ":" in stripped:
                key, value = stripped.split(":", 1)
                indexed[int(key)] = value.strip().strip("'\"")
        if indexed:
            return [indexed[k] for k in sorted(indexed)]
        if listed:
            return listed
        raise ValueError("could not read class names from data.yaml")
    raise ValueError("data.yaml has no names entry")


def remap_label_lines(
    lines: List[str], source_names: List[str], mapping: Dict[str, str], class_names: List[str]
) -> List[str]:
    """
    Convert YOLO label lines to APR class ids. Oriented boxes and polygons become their
    axis-aligned envelope because the planner only handles axis-aligned rectangles.
    """
    out = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            target = mapping.get(source_names[int(parts[0])])
            coords = [float(v) for v in parts[1:]]
        except (IndexError, ValueError):
            continue
        if target is None:
            continue

        if len(coords) == 4:
            cx, cy, w, h = coords
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs, ys = coords[0::2], coords[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        else:
            continue
        out.append(f"{class_names.index(target)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def _split_targets(is_heldout: bool) -> Dict[str, str]:
    return {"train": "train", "valid": "val", "test": "test" if is_heldout else "train"}


def merge_sources(sources_dir, class_map, out_dir, class_names, heldout: str = "team") -> Dict[str, int]:
    out = Path(out_dir)
    counts = {"train": 0, "val": 0, "test": 0}

    for source in sorted(p for p in Path(sources_dir).iterdir() if p.is_dir()):
        mapping = class_map.get(source.name)
        if mapping is None:
            print(f"skip {source.name}: no entry in the class map")
            continue
        names = parse_names((source / "data.yaml").read_text(encoding="utf-8"))
        unmapped = sorted(set(names) - set(mapping))
        if unmapped:
            print(f"{source.name}: dropping source classes {unmapped}")

        for split, target in _split_targets(source.name == heldout).items():
            images_dir = source / split / "images"
            if not images_dir.is_dir():
                continue
            (out / target / "images").mkdir(parents=True, exist_ok=True)
            (out / target / "labels").mkdir(parents=True, exist_ok=True)
            for image in sorted(images_dir.iterdir()):
                if image.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                label_file = source / split / "labels" / f"{image.stem}.txt"
                raw = label_file.read_text(encoding="utf-8").splitlines() if label_file.exists() else []
                lines = remap_label_lines(raw, names, mapping, list(class_names))

                stem = f"{source.name}_{image.stem}"
                shutil.copyfile(image, out / target / "images" / f"{stem}{image.suffix}")
                text = "\n".join(lines) + "\n" if lines else ""
                (out / target / "labels" / f"{stem}.txt").write_text(text, encoding="utf-8")
                counts[target] += 1
    return counts


def write_dataset_yaml(out_dir, class_names) -> None:
    out = Path(out_dir)
    lines = [
        f"path: {out.resolve().as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(class_names)]
    (out / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", default=str(root / "data" / "sources"))
    parser.add_argument("--class-map", default=str(root / "class_map.json"))
    parser.add_argument("--out", default=str(root / "data" / "merged"))
    parser.add_argument("--heldout", default="team")
    args = parser.parse_args()

    class_map = json.loads(Path(args.class_map).read_text(encoding="utf-8"))
    counts = merge_sources(args.sources, class_map, args.out, CLASS_NAMES, args.heldout)
    write_dataset_yaml(args.out, CLASS_NAMES)
    print(f"merged images: {counts}")
    if counts["test"] == 0:
        print("WARNING: no held-out test images. Add a test split to the team source before trusting any metric.")


if __name__ == "__main__":
    main()
```

Create `training/class_map.json`:

```json
{
  "team": {
    "window": "window",
    "door": "door",
    "ac_unit": "ac_unit",
    "meter_panel": "meter_panel",
    "pipe": "pipe",
    "grill": "grill"
  },
  "door_window_detection": {
    "door": "door",
    "window": "window"
  },
  "wall_door_window_obb": {
    "door": "door",
    "window": "window"
  }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m unittest discover -s tests -p "test_training_tools.py" -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add training/build_dataset.py training/class_map.json tests/test_training_tools.py
git commit -m "Add training dataset builder with class remapping and held-out split" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 11: Held-out evaluation script

**Files:**
- Create: `training/evaluate.py`
- Test: `tests/test_training_tools.py`

**Interfaces:**
- Consumes: `ObstacleDetector.detect(image_bgr, min_score)`, `thresholds_for_sensitivity`, `CLASS_NAMES`.
- Produces: `read_yolo_labels(path, img_w, img_h) -> list[(class_id, x1, y1, x2, y2)]`; `iou(a, b)`; `greedy_match(gt_boxes, det_boxes, det_scores, iou_threshold=0.5) -> list[(gt_idx, det_idx)]`; `evaluate(detector, images_dir, labels_dir, class_names, min_score) -> {class: {"gt","tp","fp","fn","found_any"}}`; `summarize(stats) -> {class: {"recall","precision","agnostic_recall"}}` (values `None` when undefined). CLI: `python training/evaluate.py --model models/apr_obstacles.onnx --images training/data/merged/test/images`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_training_tools.py`, add these imports below the existing imports:

```python
import cv2
import numpy as np

from evaluate import evaluate, greedy_match, iou, read_yolo_labels, summarize
from fakes import FakeDetector
from obstacle_detector import Detection
```

Insert before `if __name__ == "__main__":`:

```python
class TestMatching(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)

    def test_greedy_match_pairs_highest_scoring_detection_first(self):
        gts = [(0, 0, 10, 10)]
        dets = [(0, 0, 10, 10), (1, 1, 10, 10)]
        self.assertEqual(greedy_match(gts, dets, [0.6, 0.9]), [(0, 1)])

    def test_greedy_match_ignores_low_overlap(self):
        self.assertEqual(greedy_match([(0, 0, 10, 10)], [(50, 50, 60, 60)], [0.9]), [])


class TestEvaluate(unittest.TestCase):
    def make_split(self, tmp):
        images, labels = Path(tmp) / "images", Path(tmp) / "labels"
        images.mkdir()
        labels.mkdir()
        cv2.imwrite(str(images / "wall.png"), np.zeros((100, 100, 3), dtype=np.uint8))
        (labels / "wall.txt").write_text("0 0.5 0.5 0.4 0.4\n")  # a window at (30,30)-(70,70)
        return images, labels

    def test_read_yolo_labels_returns_pixel_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, labels = self.make_split(tmp)
            self.assertEqual(read_yolo_labels(labels / "wall.txt", 100, 100), [(0, 30.0, 30.0, 70.0, 70.0)])
            self.assertEqual(read_yolo_labels(labels / "missing.txt", 100, 100), [])

    def test_correct_detection_counts_as_true_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels = self.make_split(tmp)
            stats = evaluate(FakeDetector([Detection("window", 0.9, 30, 30, 70, 70)]), images, labels, CLASS_NAMES, 0.25)
            summary = summarize(stats)
        self.assertEqual(stats["window"]["tp"], 1)
        self.assertEqual(summary["window"]["recall"], 1.0)
        self.assertEqual(summary["window"]["precision"], 1.0)
        self.assertIsNone(summary["door"]["recall"])

    def test_wrong_class_is_a_miss_for_the_class_but_still_found_by_any_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels = self.make_split(tmp)
            stats = evaluate(FakeDetector([Detection("door", 0.9, 30, 30, 70, 70)]), images, labels, CLASS_NAMES, 0.25)
        self.assertEqual(stats["window"]["fn"], 1)
        self.assertEqual(stats["door"]["fp"], 1)
        self.assertEqual(stats["window"]["found_any"], 1)

    def test_no_detection_is_a_miss_and_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels = self.make_split(tmp)
            stats = evaluate(FakeDetector([]), images, labels, CLASS_NAMES, 0.25)
        self.assertEqual(stats["window"]["fn"], 1)
        self.assertEqual(stats["window"]["found_any"], 0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m unittest discover -s tests -p "test_training_tools.py" -v`
Expected: ERROR `ModuleNotFoundError: No module named 'evaluate'`.

- [ ] **Step 3: Implement**

Create `training/evaluate.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m unittest discover -s tests -p "test_*.py" -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add training/evaluate.py tests/test_training_tools.py
git commit -m "Add held-out evaluation script for the obstacle detector" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 12: Training script, README and repo hygiene

**Files:**
- Create: `training/train.py`, `training/README.md`
- Modify: `.gitignore`

Not unit tested (it drives ultralytics on a GPU in Colab); verified by a compile check and by following the README end to end when the first dataset exists.

- [ ] **Step 1: Create the training script**

Create `training/train.py`:

```python
"""
Fine-tune YOLO11n on the merged dataset and export it as ONNX for the app.

Run in Colab (GPU runtime) after `pip install ultralytics`:
    python training/train.py                                   # stage 1: from pretrained yolo11n.pt
    python training/train.py --weights training/runs/apr_obstacles/weights/best.pt --name apr_stage2

Ultralytics is training-only and must never be added to requirements.txt or pyproject.toml.
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from obstacle_detector import CLASS_NAMES  # noqa: E402  (dataset.yaml is generated from the same list)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "training" / "data" / "merged" / "dataset.yaml"))
    parser.add_argument("--weights", default="yolo11n.pt", help="start point; use the previous stage's best.pt for stage 2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="apr_obstacles")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=20,
        project=str(ROOT / "training" / "runs"),
        name=args.name,
        exist_ok=True,
    )

    best = Path(model.trainer.best)
    tuned = YOLO(str(best))

    print("Held-out test split, per-class mAP50:")
    metrics = tuned.val(data=args.data, split="test", imgsz=args.imgsz)
    for class_index, ap50 in zip(metrics.box.ap_class_index, metrics.box.ap50):
        print(f"  {CLASS_NAMES[int(class_index)]}: {ap50:.3f}")

    onnx_path = Path(tuned.export(format="onnx", imgsz=args.imgsz, simplify=True))
    target = ROOT / "models" / "apr_obstacles.onnx"
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(onnx_path, target)
    print(f"Exported {target}. Run training/evaluate.py next, then commit the file.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create the README**

Create `training/README.md`:

````markdown
# Training the obstacle detector

The app runs a YOLO11n model exported to ONNX (`models/apr_obstacles.onnx`). Training happens here, offline, in Google Colab. Only the exported `.onnx` file ships; PyTorch and ultralytics never go into `requirements.txt` or `pyproject.toml`.

Classes, in model order: `window`, `door`, `ac_unit`, `meter_panel`, `pipe`, `grill` (defined once in `obstacle_detector.py`).

## 1. Collect and label your own photos

Public datasets cover windows and doors. Nothing public covers AC condenser units, meter boxes, exterior pipes or grills, so those classes depend on your photos.

- At least 40 to 80 labelled instances per class. More is better.
- Vary lighting (morning, noon, overcast, shade), distance, angle, wall colour and texture. Include clutter such as wires, plants, signs and shadows.
- Include walls with no obstacles at all. The model needs to learn what a clean wall looks like.
- Label in [Roboflow](https://roboflow.com) (free tier) or CVAT, using exactly the class names above.
- Export as **YOLO** format with a train/valid/test split of about 70/10/20. The test split (20%) is your held-out set: it is never trained on and never used to tune thresholds.

Put the export at `training/data/sources/team/` so it contains `data.yaml`, `train/`, `valid/` and `test/`.

## 2. Download the public datasets

Check each dataset's licence on its page before using it.

- [Door and Window Detection](https://universe.roboflow.com/construction-plan/door-and-window-detection-fmw85) (5,382 images) → `training/data/sources/door_window_detection/`
- [yolo-obb-1 wall/door/window](https://universe.roboflow.com/test-v0q9r/yolo-obb-1) (997 images, oriented boxes) → `training/data/sources/wall_door_window_obb/`

Export each in YOLO format. Oriented boxes and polygons are converted to axis-aligned rectangles automatically.

## 3. Check the class map

Open each `data.yaml` and confirm its class names match the keys in `training/class_map.json`. Edit the map if a dataset names a class differently. `build_dataset.py` prints any source classes it drops so mistakes are visible.

## 4. Build the merged dataset

```bash
python training/build_dataset.py
```

Public datasets feed train and val (their test splits go to train). The `team` source keeps its test split as the merged held-out test set. The script warns if there is no test set.

## 5. Train

In Colab (Runtime → Change runtime type → T4 GPU):

```bash
git clone <this repo> && cd <repo>
pip install ultralytics onnx
python training/build_dataset.py
python training/train.py                       # stage 1, from pretrained yolo11n.pt
python training/train.py --weights training/runs/apr_obstacles/weights/best.pt --name apr_stage2   # stage 2
```

Stage 1 can run with only the public sources (leave out `team`), stage 2 adds your photos. The script prints per-class mAP50 on the held-out test split and writes `models/apr_obstacles.onnx`.

## 6. Evaluate with the app's own pipeline

```bash
python training/evaluate.py --model models/apr_obstacles.onnx --images training/data/merged/test/images
```

Prints recall and precision per class at the app's default thresholds, and how many ground-truth objects got no box at all. Targets, set before training:

- `window` and `door`: recall at least 0.90, precision at least 0.85. The script exits non-zero if missed.
- `ac_unit`, `meter_panel`, `pipe`, `grill`: mAP50 at least 0.70 (from the `train.py` output).
- Nothing in a test photo should get no box at all for windows and doors.

Report misses honestly and collect more photos for any class that falls short.

## 7. Ship the model

Copy `models/apr_obstacles.onnx` into the repo, commit it, and push. If the file is missing in production the app silently uses the weaker classical detector and shows a banner, so check the banner after deploying.

## Licences

Ultralytics and weights trained with it are AGPL-3.0. That is acceptable while testing. Before any commercial use, buy an Ultralytics enterprise licence or retrain with a permissively licensed detector. Roboflow datasets carry their own licences, so check each one.

## Known limits

- Wall width and height are entered by hand. Estimating scale from a photo needs depth sensing or a reference marker.
- Photos should be shot square-on, or the four wall corners marked in the UI so the photo is straightened.
- A missed detection is worse than a false one. The review threshold is deliberately low so doubtful objects surface as "unverified" instead of vanishing.
````

- [ ] **Step 3: Ignore training data and weights**

Append to `.gitignore`:

```
training/data/
training/runs/
*.pt
```

- [ ] **Step 4: Verify**

Run: `uv run python -m py_compile training/train.py training/build_dataset.py training/evaluate.py`
Expected: no output.

Run: `git status --short`
Expected: only `.gitignore`, `training/train.py`, `training/README.md` listed (no `models/` yet, no `training/data`).

- [ ] **Step 5: Commit**

```bash
git add .gitignore training/train.py training/README.md
git commit -m "Add YOLO training script, README and ignore rules for training artefacts" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 13: Final verification and hand-off

**Files:** none modified (fix anything this step finds in the task that owns it).

- [ ] **Step 1: Full test run**

Run: `uv run python -m unittest discover -s tests -p "test_*.py" -v`
Expected: all PASS; `TestRealModel` skipped until a model is committed.

- [ ] **Step 2: Lockfile and pins agree**

Run: `uv lock --check`
Expected: exits 0. `tests/test_dependencies.py` already asserts that `requirements.txt` and `pyproject.toml` match.

- [ ] **Step 3: App boots and both pages serve**

Run:

```bash
uv run python -c "
import app
c = app.app.test_client()
print([c.get(p).status_code for p in ('/', '/about')], app.geometry_engine.detector)
"
```

Expected: `[200, 200] None` (no model committed yet, so the fallback is active).

- [ ] **Step 4: Confirm nothing heavy leaked into runtime dependencies**

Run: `grep -inE "torch|ultralytics" requirements.txt pyproject.toml`
Expected: no output.

- [ ] **Step 5: Report to the user, then stop before pushing**

Summarise what shipped and what still needs a human: collect and label photos, run training in Colab, commit `models/apr_obstacles.onnx`. Pushing triggers a Vercel redeploy that installs `onnxruntime`; ask the user before pushing, and after the deploy check the build log and confirm the fallback banner disappears once a model is committed.
