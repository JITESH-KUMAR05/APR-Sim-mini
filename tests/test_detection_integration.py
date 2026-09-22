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


class RaisingDetector:
    """A detector whose inference always fails (e.g. a model with the wrong class count)."""

    def detect(self, image_bgr, min_score):
        raise ValueError("model output has an unexpected class count")


class TestFallback(unittest.TestCase):
    def test_no_detector_uses_classical_fallback(self):
        engine = GeometryEngine()
        image = engine.generate_benchmark_wall_image(1200, 840)
        result = engine.detect(image, 4000.0, 2800.0)
        self.assertEqual(result["detector"], "classical-fallback")
        self.assertGreaterEqual(len(result["obstacles"]), 1)
        self.assertGreaterEqual(result["confidence"], 75.0)

    def test_detector_that_raises_falls_back_to_classical(self):
        engine = GeometryEngine(detector=RaisingDetector())
        image = engine.generate_benchmark_wall_image(1200, 840)
        result = engine.detect(image, 4000.0, 2800.0)
        self.assertEqual(result["detector"], "classical-fallback")
        self.assertGreaterEqual(len(result["obstacles"]), 1)

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


if __name__ == "__main__":
    unittest.main()
