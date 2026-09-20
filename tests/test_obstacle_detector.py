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
