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


if __name__ == "__main__":
    unittest.main()
