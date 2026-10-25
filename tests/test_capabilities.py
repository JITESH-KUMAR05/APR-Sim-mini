import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module


class TestCapabilities(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def get(self):
        res = self.client.get("/api/capabilities")
        return res, json.loads(res.data)

    def test_reports_the_detector_and_the_simulator(self):
        res, data = self.get()
        self.assertEqual(res.status_code, 200)
        self.assertIn(data["detector"], ("yolo11n-onnx", "classical-fallback"))
        self.assertIsInstance(data["simulator"], bool)

    def test_the_detector_follows_what_is_loaded(self):
        with mock.patch.object(app_module.geometry_engine, "detector", None):
            self.assertEqual(self.get()[1]["detector"], "classical-fallback")
        with mock.patch.object(app_module.geometry_engine, "detector", object()):
            self.assertEqual(self.get()[1]["detector"], "yolo11n-onnx")

    def test_no_simulator_on_a_deployment_or_without_gpp(self):
        with mock.patch.dict(os.environ, {"VERCEL": "1"}):
            self.assertFalse(self.get()[1]["simulator"])
        with mock.patch("shutil.which", return_value=None):
            self.assertFalse(self.get()[1]["simulator"])


if __name__ == "__main__":
    unittest.main()
