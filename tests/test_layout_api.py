import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module
from rover_mission import replay_rover_mission, verify_checksum

FORM = {
    "width_mm": "4000", "height_mm": "2800", "spray_width_mm": "250",
    "overlap_pct": "12", "safety_buffer_mm": "60", "sensitivity": "0.5", "sample_id": "residential",
}
WALL = {"width_mm": 4000, "height_mm": 2800, "spray_width_mm": 250, "overlap_pct": 12, "safety_buffer_mm": 60}
PIPE = {"type": "pipe", "x": 600, "y": 0, "w": 120, "h": 2800}


class TestAnalyzeLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app_module.app.test_client()

    def analyze(self, **extra):
        res = self.client.post("/api/analyze", data={**FORM, **extra})
        return res, json.loads(res.data)

    def test_without_the_new_fields_the_old_behaviour_stays(self):
        res, data = self.analyze()
        self.assertEqual(res.status_code, 200)
        self.assertEqual(data["layout"]["mode"], "manual")
        self.assertEqual((data["wall"]["start_corner"], data["wall"]["direction"]), ("bottom_left", "horizontal"))

    def test_auto_returns_the_scored_options_and_a_reason(self):
        res, data = self.analyze(start_corner="auto", direction="auto")
        self.assertEqual(res.status_code, 200)
        layout = data["layout"]
        self.assertEqual(layout["mode"], "auto")
        self.assertEqual(len(layout["options"]), 8)
        self.assertEqual((layout["start_corner"], layout["direction"]), (data["wall"]["start_corner"], data["wall"]["direction"]))
        self.assertTrue(layout["reason"])

    def test_the_exported_rover_mission_matches_the_chosen_layout(self):
        _, data = self.analyze(start_corner="top_right", direction="vertical")
        self.assertEqual((data["wall"]["start_corner"], data["wall"]["direction"]), ("top_right", "vertical"))
        mission = json.loads(self.client.get("/api/download/rover").data)
        self.assertTrue(verify_checksum(mission))
        self.assertEqual(mission["wall"]["start_corner"], "top_right")
        self.assertIn(mission["start_pose"]["heading_deg"], (90.0, -90.0))  # vertical stripes start facing up or down
        sprayed = sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b, spray in replay_rover_mission(mission)["moves"] if spray)
        self.assertAlmostEqual(sprayed / 1000.0, data["stats"]["spray_distance_m"], delta=0.1)

    def test_bad_values_are_a_400_not_a_crash(self):
        self.assertEqual(self.analyze(direction="diagonal")[0].status_code, 400)
        self.assertEqual(self.analyze(start_corner="middle")[0].status_code, 400)


class TestReplanLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app_module.app.test_client()

    def replan(self, obstacles, **extra):
        res = self.client.post("/api/replan", json={**WALL, **extra, "obstacles": obstacles})
        return res, json.loads(res.data)

    def test_auto_follows_the_obstacles_a_floor_to_ceiling_pipe_starts_on_the_bigger_side(self):
        res, data = self.replan([PIPE], start_corner="auto", direction="auto")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(data["layout"]["start_corner"].endswith("right"))
        self.assertIn("cut off", data["layout"]["reason"])
        self.assertGreater(data["stats"]["unreached_area_m2"], 0)
        res, clear = self.replan([], start_corner="auto", direction="auto")
        self.assertEqual(clear["layout"]["mode"], "auto")

    def test_a_fixed_choice_is_kept(self):
        _, data = self.replan([PIPE], start_corner="bottom_right", direction="horizontal")
        self.assertEqual((data["layout"]["start_corner"], data["layout"]["direction"]), ("bottom_right", "horizontal"))
        self.assertEqual(data["layout"]["mode"], "manual")

    def test_bad_values_are_a_400(self):
        res, _ = self.replan([], direction="spiral")
        self.assertEqual(res.status_code, 400)


if __name__ == "__main__":
    unittest.main()
