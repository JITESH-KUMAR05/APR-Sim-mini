import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module
import corrections
from fakes import FakeDetector
from obstacle_detector import Detection

FORM = {
    "width_mm": "4000", "height_mm": "2800", "spray_width_mm": "250", "overlap_pct": "12",
    "safety_buffer_mm": "60", "sensitivity": "0.5", "sample_id": "residential",
}
WALL = {"width_mm": 4000, "height_mm": 2800, "spray_width_mm": 250, "overlap_pct": 12, "safety_buffer_mm": 60}


def image(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (60, 90, 3), dtype=np.uint8)


class TestHelpers(unittest.TestCase):
    def test_only_confirmed_obstacles_become_labels_and_y_is_flipped_to_image_coordinates(self):
        obstacles = [
            {"type": "window", "x": 1000.0, "y": 1400.0, "w": 800.0, "h": 1000.0},   # top of the wall
            {"type": "unverified", "x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0},      # not confirmed: no label
            {"type": "fixture", "x": 0.0, "y": 0.0, "w": 100.0, "h": 100.0},         # a legacy type: no label
            {"type": "ac_unit", "x": 3000.0, "y": 0.0, "w": 1000.0, "h": 280.0},
        ]
        lines = corrections.yolo_lines(obstacles, 4000.0, 2800.0)
        self.assertEqual(len(lines), 2)
        cls, cx, cy, w, h = lines[0].split()
        self.assertEqual(cls, "0")
        self.assertAlmostEqual(float(cx), 0.35, places=4)
        self.assertAlmostEqual(float(cy), 1 - (1400 + 500) / 2800, places=4)
        self.assertAlmostEqual(float(w), 0.2, places=4)
        self.assertAlmostEqual(float(h), 1000 / 2800, places=4)
        self.assertEqual(lines[1].split()[0], "2")

    def test_same_obstacles_ignores_order_ids_and_sub_millimetre_noise(self):
        a = [{"type": "window", "x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0, "id": "a"}, {"type": "door", "x": 5.0, "y": 6.0, "w": 7.0, "h": 8.0}]
        b = [{"type": "door", "x": 5.2, "y": 6.0, "w": 7.0, "h": 8.0, "id": "z"}, {"type": "window", "x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0}]
        self.assertTrue(corrections.same_obstacles(a, b))
        b[0]["type"] = "window"
        self.assertFalse(corrections.same_obstacles(a, b))
        self.assertFalse(corrections.same_obstacles(a, a[:1]))

    def test_a_sample_is_an_ordinary_yolo_dataset_the_pool_importer_can_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = corrections.save_correction(tmp, image(), [{"type": "door", "x": 0.0, "y": 0.0, "w": 400.0, "h": 1400.0}], 4000.0, 2800.0)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "train", "images", path + ".jpg")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "train", "labels", path + ".txt")))
            names = open(os.path.join(tmp, "data.yaml")).read()
            for name in ("window", "door", "ac_unit", "meter_panel", "pipe", "grill"):
                self.assertIn(name, names)
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training", "gpu"))
            import pool
            self.assertEqual(pool.read_names(os.path.join(tmp, "data.yaml")), ["window", "door", "ac_unit", "meter_panel", "pipe", "grill"])

    def test_saving_the_same_photo_again_updates_one_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = corrections.save_correction(tmp, image(1), [], 4000.0, 2800.0)
            second = corrections.save_correction(tmp, image(1), [{"type": "door", "x": 0.0, "y": 0.0, "w": 400.0, "h": 1400.0}], 4000.0, 2800.0)
            self.assertEqual(first, second)
            self.assertEqual(len(os.listdir(os.path.join(tmp, "train", "images"))), 1)
            self.assertTrue(open(os.path.join(tmp, "train", "labels", first + ".txt")).read().startswith("1 "))

    def test_a_box_that_runs_past_the_wall_is_clamped_and_one_outside_is_dropped(self):
        lines = corrections.yolo_lines([
            {"type": "door", "x": 3500.0, "y": 0.0, "w": 1000.0, "h": 1000.0},
            {"type": "window", "x": 4500.0, "y": 0.0, "w": 100.0, "h": 100.0},
        ], 4000.0, 2800.0)
        self.assertEqual(len(lines), 1)
        cls, cx, cy, w, h = (float(v) for v in lines[0].split())
        self.assertAlmostEqual(cx + w / 2, 1.0, places=5)

    def test_the_label_is_written_before_the_photo_and_no_temporary_files_remain(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = corrections.save_correction(tmp, image(4), [], 4000.0, 2800.0)
            names = sorted(os.listdir(os.path.join(tmp, "train", "images")) + os.listdir(os.path.join(tmp, "train", "labels")))
            self.assertEqual(names, [sample + ".jpg", sample + ".txt"])

    def test_the_folder_has_a_size_limit(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(corrections, "MAX_SAMPLES", 2):
            self.assertTrue(corrections.save_correction(tmp, image(1), [], 4000.0, 2800.0))
            self.assertTrue(corrections.save_correction(tmp, image(2), [], 4000.0, 2800.0))
            self.assertIsNone(corrections.save_correction(tmp, image(3), [], 4000.0, 2800.0))
            self.assertTrue(corrections.save_correction(tmp, image(1), [], 4000.0, 2800.0))  # updating an old one is fine


class TestApi(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        patcher = mock.patch.object(app_module, "CORRECTIONS_DIR", self.folder.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("VERCEL", None)
        os.environ.pop("APR_SAVE_CORRECTIONS", None)
        self.addCleanup(app_module.current_session.pop, "review", None)

    def photo(self, name="wall.jpg"):
        self.photos = getattr(self, "photos", 0) + 1
        pixels = np.zeros((500, 800, 3), np.uint8)
        pixels[:, :, 0] = np.linspace(0, 255, 800, dtype=np.uint8)[None, :]
        pixels[:, :, 1] = (self.photos * 40) % 256  # every call gives a different photo
        ok, encoded = cv2.imencode(".jpg", pixels)
        return (io.BytesIO(encoded.tobytes()), name)

    def analyze(self, upload=True, name="wall.jpg"):
        detector = FakeDetector([Detection("window", 0.35, 100, 100, 400, 400)])
        data = dict(FORM)
        if upload:
            data["file"] = self.photo(name)
            data.pop("sample_id")
        with mock.patch.object(app_module.geometry_engine, "detector", detector):
            res = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        return json.loads(res.data)

    def replan(self, obstacles, token, save=True, **extra):
        res = self.client.post("/api/replan", json={**WALL, "obstacles": obstacles, "review_token": token,
                                                    "save_corrections": save, **extra})
        return res, json.loads(res.data)

    def saved(self):
        folder = os.path.join(self.folder.name, "train", "labels")
        return sorted(f for f in os.listdir(folder) if f.endswith(".txt")) if os.path.isdir(folder) else []

    def test_an_unchanged_review_saves_nothing(self):
        data = self.analyze()
        res, body = self.replan(data["obstacles"], data["review_token"])
        self.assertEqual(res.status_code, 200)
        self.assertFalse(body["correction_saved"])
        self.assertEqual(self.saved(), [])

    def test_a_change_made_by_the_operator_is_saved_as_a_labelled_photo(self):
        data = self.analyze()
        confirmed = [dict(o, type="window") for o in data["obstacles"]]
        res, body = self.replan(confirmed, data["review_token"])
        self.assertTrue(body["correction_saved"])
        self.assertEqual(len(self.saved()), 1)
        label = open(os.path.join(self.folder.name, "train", "labels", self.saved()[0])).read().split()
        self.assertEqual(label[0], "0")

    def test_dismissing_everything_saves_a_clean_wall(self):
        data = self.analyze()
        _, body = self.replan([], data["review_token"])
        self.assertTrue(body["correction_saved"])
        self.assertEqual(open(os.path.join(self.folder.name, "train", "labels", self.saved()[0])).read(), "")

    def test_a_review_from_another_analysis_is_never_saved_against_the_latest_photo(self):
        first = self.analyze()
        self.analyze()  # another tab or user analyses a different photo with the same wall size
        _, body = self.replan([dict(o, type="door") for o in first["obstacles"]], first["review_token"])
        self.assertFalse(body["correction_saved"])
        self.assertEqual(self.saved(), [])

    def test_a_missing_or_invented_token_saves_nothing(self):
        data = self.analyze()
        for token in (None, "", "0" * 32, 5):
            with self.subTest(token=token):
                self.assertFalse(self.replan([], token)[1]["correction_saved"])
        self.assertEqual(self.saved(), [])

    def test_the_built_in_samples_and_random_walls_are_not_kept(self):
        data = self.analyze(upload=False)
        self.assertFalse(self.replan([], data["review_token"])[1]["correction_saved"])
        data = self.analyze(name="random_wall_77.jpg")
        self.assertFalse(self.replan([], data["review_token"])[1]["correction_saved"])
        self.assertEqual(self.saved(), [])

    def test_the_operator_can_switch_it_off_and_a_deployment_never_saves(self):
        data = self.analyze()
        self.assertFalse(self.replan([], data["review_token"], save=False)[1]["correction_saved"])
        with mock.patch.dict(os.environ, {"VERCEL": "1"}):
            self.assertFalse(self.replan([], data["review_token"])[1]["correction_saved"])
        with mock.patch.dict(os.environ, {"APR_SAVE_CORRECTIONS": "0"}):
            self.assertFalse(self.replan([], data["review_token"])[1]["correction_saved"])
        self.assertEqual(self.saved(), [])

    def test_a_replan_for_another_wall_size_is_not_saved(self):
        data = self.analyze()
        _, body = self.replan([], data["review_token"], width_mm=5000)
        self.assertFalse(body["correction_saved"])

    def test_changing_only_the_layout_saves_nothing(self):
        data = self.analyze()
        _, body = self.replan(data["obstacles"], data["review_token"], start_corner="top_left", direction="vertical")
        self.assertFalse(body["correction_saved"])

    def test_a_failure_while_saving_never_fails_the_replan(self):
        data = self.analyze()
        for error in (OSError("disk full"), ValueError("bad"), MemoryError()):
            with self.subTest(error=type(error).__name__), mock.patch.object(corrections, "save_correction", side_effect=error):
                res, body = self.replan([], data["review_token"])
                self.assertEqual(res.status_code, 200)
                self.assertFalse(body["correction_saved"])
                self.assertTrue(body["success"])

    def test_the_other_endpoint_tests_cannot_write_here(self):
        # a replan without a token (every older caller) must not save, so the suite cannot fill exports/corrections
        data = self.analyze()
        res = self.client.post("/api/replan", json={**WALL, "obstacles": []})
        self.assertFalse(json.loads(res.data)["correction_saved"])


if __name__ == "__main__":
    unittest.main()
