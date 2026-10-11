import base64
import io
import json
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module
from wall_synth import CLASS_NAMES, generate_wall, to_yolo_lines


def decode(data_url):
    header, payload = data_url.split(",", 1)
    assert header == "data:image/jpeg;base64"
    return cv2.imdecode(np.frombuffer(base64.b64decode(payload), np.uint8), cv2.IMREAD_COLOR)


class TestWallSynth(unittest.TestCase):
    def test_same_seed_same_wall_different_seed_different_wall(self):
        a, boxes_a = generate_wall(11)
        b, boxes_b = generate_wall(11)
        c, _ = generate_wall(12)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(boxes_a, boxes_b)
        self.assertFalse(a.shape == c.shape and np.array_equal(a, c))

    def test_boxes_are_inside_the_image_and_use_the_app_class_order(self):
        from obstacle_detector import CLASS_NAMES as APP_CLASSES
        self.assertEqual(CLASS_NAMES, APP_CLASSES)
        for seed in range(10):
            image, boxes = generate_wall(seed)
            h, w = image.shape[:2]
            for cls, x1, y1, x2, y2 in boxes:
                self.assertIn(cls, range(6))
                self.assertTrue(0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, (seed, cls, x1, y1, x2, y2, w, h))

    def test_forced_counts_are_honoured_when_there_is_room(self):
        _, boxes = generate_wall(3, counts={"window": 3, "door": 1, "ac_unit": 1, "meter_panel": 1, "pipe": 1})
        names = [CLASS_NAMES[b[0]] for b in boxes]
        for name, n in (("window", 3), ("door", 1), ("ac_unit", 1), ("meter_panel", 1), ("pipe", 1)):
            self.assertEqual(names.count(name), n, (name, names))

    def test_every_class_shows_up_across_a_set_of_walls(self):
        seen = set()
        for seed in range(16):
            seen.update(b[0] for b in generate_wall(seed)[1])
        self.assertEqual(seen, set(range(6)))

    def test_yolo_lines_are_normalised(self):
        image, boxes = generate_wall(5)
        for line in to_yolo_lines(boxes, image.shape[1], image.shape[0]):
            cls, *values = line.split()
            self.assertTrue(all(0.0 <= float(v) <= 1.0 for v in values), line)

    def test_a_wall_with_no_fixtures_is_possible(self):
        _, boxes = generate_wall(1, counts={})
        self.assertEqual(boxes, [])


class TestRandomWallEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app_module.app.test_client()

    def post(self, body):
        res = self.client.post("/api/random_wall", json=body)
        return res, json.loads(res.data)

    def test_returns_a_jpeg_wall_size_and_what_was_planted(self):
        res, data = self.post({"seed": 5})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(data["success"])
        self.assertEqual(data["seed"], 5)
        image = decode(data["image_data_url"])
        self.assertEqual(image.shape[1], 1200)
        self.assertTrue(3000 <= data["width_mm"] <= 6000)
        self.assertAlmostEqual(data["height_mm"] / data["width_mm"], image.shape[0] / image.shape[1], delta=0.03)
        self.assertEqual(set(data["planted"]), set(CLASS_NAMES))
        self.assertEqual(len(data["truth"]), sum(data["planted"].values()))

    def test_the_same_seed_gives_the_same_wall(self):
        self.assertEqual(self.post({"seed": 9})[1], self.post({"seed": 9})[1])
        self.assertNotEqual(self.post({"seed": 9})[1]["image_data_url"], self.post({"seed": 10})[1]["image_data_url"])

    def test_without_a_seed_one_is_chosen_and_reported(self):
        res, data = self.post({})
        self.assertEqual(res.status_code, 200)
        self.assertIsInstance(data["seed"], int)
        self.assertEqual(self.post({"seed": data["seed"]})[1]["image_data_url"], data["image_data_url"])

    def test_truth_boxes_are_in_wall_millimetres(self):
        _, data = self.post({"seed": 21})
        for box in data["truth"]:
            self.assertIn(box["type"], CLASS_NAMES)
            self.assertTrue(0 <= box["x"] and box["x"] + box["w"] <= data["width_mm"] + 1)
            self.assertTrue(0 <= box["y"] and box["y"] + box["h"] <= data["height_mm"] + 1)

    def test_bad_input_is_a_400(self):
        for body in ({"seed": -1}, {"seed": "x"}, {"seed": 1.5}, {"density": 9}, {"density": "lots"}):
            with self.subTest(body=body):
                self.assertEqual(self.post(body)[0].status_code, 400)
        self.assertEqual(self.client.post("/api/random_wall", data="nope").status_code, 400)

    def test_the_generated_photo_goes_through_the_normal_analysis(self):
        _, wall = self.post({"seed": 2})
        header, payload = wall["image_data_url"].split(",", 1)
        res = self.client.post("/api/analyze", data={
            "width_mm": str(wall["width_mm"]), "height_mm": str(wall["height_mm"]), "spray_width_mm": "250",
            "overlap_pct": "12", "safety_buffer_mm": "60", "sensitivity": "0.5",
            "start_corner": "auto", "direction": "auto",
            "file": (io.BytesIO(base64.b64decode(payload)), "random_wall_2.jpg"),
        }, content_type="multipart/form-data")
        data = json.loads(res.data)
        self.assertEqual(res.status_code, 200, data)
        self.assertTrue(data["success"])
        self.assertEqual(data["layout"]["mode"], "auto")


if __name__ == "__main__":
    unittest.main()
