import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "rover", "tools"))
sys.path.insert(0, ROOT)

from demo_random_wall import coverage, paint_on_obstacles

BOX = {"x": 1000.0, "y": 500.0, "w": 400.0, "h": 600.0}


def wp(x, y, spray):
    return {"x": x, "y": y, "spray_active": spray}


class TestCoverage(unittest.TestCase):
    def test_perfect_detection(self):
        result = coverage([BOX], [dict(BOX)], 4000, 3000)
        self.assertAlmostEqual(result["coverage"], 1.0, places=2)
        self.assertAlmostEqual(result["false_zone"], 0.0, places=2)
        self.assertEqual((result["planted"], result["found"]), (1, 1))

    def test_a_missed_obstacle(self):
        result = coverage([BOX], [], 4000, 3000)
        self.assertEqual(result["coverage"], 0.0)
        self.assertEqual(result["false_zone"], 0.0)
        self.assertEqual((result["planted"], result["found"]), (1, 0))

    def test_a_false_zone_and_a_half_covered_obstacle(self):
        half = {"x": 1000.0, "y": 500.0, "w": 200.0, "h": 600.0}
        far = {"x": 3000.0, "y": 2000.0, "w": 300.0, "h": 300.0}
        result = coverage([BOX], [half, far], 4000, 3000)
        self.assertAlmostEqual(result["coverage"], 0.5, delta=0.06)
        self.assertGreater(result["false_zone"], 0.4)
        self.assertEqual(result["found"], 1)  # exactly half is enough to count as found

    def test_nothing_planted_and_nothing_detected_is_a_clean_result(self):
        result = coverage([], [], 4000, 3000)
        self.assertEqual((result["coverage"], result["false_zone"], result["planted"]), (1.0, 0.0, 0))


class TestPaintOnObstacles(unittest.TestCase):
    def test_spray_across_the_box_counts_its_length_inside(self):
        over = paint_on_obstacles([wp(0, 800, True), wp(3000, 800, True)], [BOX])
        self.assertAlmostEqual(over, 400.0, delta=10.0)

    def test_transit_with_spray_off_does_not_count(self):
        self.assertEqual(paint_on_obstacles([wp(0, 800, False), wp(3000, 800, False)], [BOX]), 0.0)

    def test_spray_outside_the_box_does_not_count(self):
        self.assertEqual(paint_on_obstacles([wp(0, 100, True), wp(3000, 100, True)], [BOX]), 0.0)


class TestTruth(unittest.TestCase):
    def test_truth_matches_the_wall_the_button_makes(self):
        import app as app_module
        import json

        client = app_module.app.test_client()
        data = json.loads(client.post("/api/random_wall", json={"seed": 33}).data)
        from wall_synth import random_scene
        _, width_mm, height_mm, truth = random_scene(33)
        self.assertEqual((width_mm, height_mm), (data["width_mm"], data["height_mm"]))
        self.assertEqual(len(truth), len(data["truth"]))
        for mine, theirs in zip(truth, data["truth"]):
            self.assertEqual(mine["type"], theirs["type"])
            for key in ("x", "y", "w", "h"):
                self.assertAlmostEqual(mine[key], theirs[key], delta=0.11)


if __name__ == "__main__":
    unittest.main()
