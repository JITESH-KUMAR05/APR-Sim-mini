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
