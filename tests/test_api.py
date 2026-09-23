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

    def test_analyze_then_replan_round_trip_preserves_obstacle_geometry(self):
        """The exact obstacles array /api/analyze returns must be accepted verbatim by
        /api/replan, with obstacle mm geometry preserved between the two responses."""
        detector = FakeDetector([
            Detection("window", 0.9, 100, 100, 400, 400),     # confident -> accepted
            Detection("ac_unit", 0.35, 600, 100, 900, 400),   # below accept threshold -> unverified
        ])
        with mock.patch.object(app_module.geometry_engine, "detector", detector):
            analyze_res = self.client.post("/api/analyze", data=FORM)
        self.assertEqual(analyze_res.status_code, 200)
        analyze_data = json.loads(analyze_res.data)
        self.assertTrue(analyze_data["success"])

        obstacles = analyze_data["obstacles"]
        self.assertEqual(len(obstacles), 2)
        self.assertEqual({o["type"] for o in obstacles}, {"window", "unverified"})

        # Take the analyze response's obstacles array verbatim and replan with it.
        replan_res, replan_data = self.replan(obstacles)
        self.assertEqual(replan_res.status_code, 200)
        self.assertTrue(replan_data["success"])

        analyze_by_id = {o["id"]: o for o in obstacles}
        self.assertEqual(len(replan_data["obstacles"]), len(obstacles))
        for obs in replan_data["obstacles"]:
            original = analyze_by_id[obs["id"]]
            self.assertEqual(obs["type"], original["type"])
            self.assertEqual(obs["x"], original["x"])
            self.assertEqual(obs["y"], original["y"])
            self.assertEqual(obs["w"], original["w"])
            self.assertEqual(obs["h"], original["h"])


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


class TestRequestTooLarge(unittest.TestCase):
    """An oversized request must still get a JSON error, not Werkzeug's HTML 413 page."""

    @classmethod
    def setUpClass(cls):
        cls.client = app_module.app.test_client()

    def setUp(self):
        self.original_limit = app_module.app.config["MAX_CONTENT_LENGTH"]
        app_module.app.config["MAX_CONTENT_LENGTH"] = 10

    def tearDown(self):
        app_module.app.config["MAX_CONTENT_LENGTH"] = self.original_limit

    def test_oversized_analyze_upload_is_a_json_413(self):
        import io

        data = {**FORM, "file": (io.BytesIO(b"x" * 100), "wall.png")}
        data.pop("sample_id")
        res = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 413)
        self.assertFalse(json.loads(res.data)["success"])

    def test_oversized_replan_payload_is_a_json_413(self):
        res = self.client.post(
            "/api/replan",
            data=json.dumps({"obstacles": [], "width_mm": 4000, "height_mm": 2800, "padding": "x" * 100}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 413)
        self.assertFalse(json.loads(res.data)["success"])


if __name__ == "__main__":
    unittest.main()
