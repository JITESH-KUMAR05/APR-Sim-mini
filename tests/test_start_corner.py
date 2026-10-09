import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry_engine import DEFAULT_START_CORNER, START_CORNERS, GeometryEngine

WALL_W, WALL_H = 4000.0, 2800.0
# Deliberately not symmetric about the wall centre, so a wrong mirror would put it somewhere else.
BOX = {"id": "obs_1", "type": "window", "label": "Window #1", "x": 500.0, "y": 300.0, "w": 700.0, "h": 500.0}


def plan(corner, obstacles=()):
    return GeometryEngine().plan_coverage_path(WALL_W, WALL_H, 250.0, 12.0, 60.0, list(obstacles), corner)


class TestStartCorner(unittest.TestCase):
    def test_default_is_bottom_left_with_the_first_row_near_the_floor(self):
        self.assertEqual(DEFAULT_START_CORNER, "bottom_left")
        waypoints, _ = GeometryEngine().plan_coverage_path(WALL_W, WALL_H, 250.0, 12.0, 60.0, [])
        first = waypoints[0]
        self.assertEqual((first["x"], first["y"]), (0.0, 125.0))
        self.assertEqual(first["type"], "PASS_START")

    def test_each_corner_starts_in_that_corner_and_the_first_pass_crosses_the_wall(self):
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                waypoints, _ = plan(corner)
                vertical, horizontal = corner.split("_")
                start_x = 0.0 if horizontal == "left" else WALL_W
                start_y = 125.0 if vertical == "bottom" else WALL_H - 125.0
                self.assertEqual((waypoints[0]["x"], waypoints[0]["y"]), (start_x, start_y))
                self.assertEqual(waypoints[1]["x"], WALL_W - start_x)

    def test_rows_progress_away_from_the_start_side(self):
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                waypoints, stats = plan(corner)
                row_y = [next(w["y"] for w in waypoints if w["row"] == r) for r in range(1, stats["num_rows"] + 1)]
                steps = [b - a for a, b in zip(row_y, row_y[1:])]
                if corner.startswith("bottom"):
                    self.assertTrue(all(s > 0 for s in steps), row_y)
                else:
                    self.assertTrue(all(s < 0 for s in steps), row_y)

    def test_distances_do_not_depend_on_the_corner_for_a_clear_wall(self):
        reference = plan("top_left")[1]
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                self.assertEqual(plan(corner)[1], reference)

    def test_spray_passes_avoid_the_box_at_its_real_position_from_every_corner(self):
        buffer_mm = 60.0
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                waypoints, _ = plan(corner, [BOX])
                rows_aligned_with_the_box = 0
                for start, end in zip(waypoints, waypoints[1:]):
                    if start["type"] == "PASS_START" and end["type"] == "PASS_END":
                        if BOX["y"] - buffer_mm <= start["y"] <= BOX["y"] + BOX["h"] + buffer_mm:
                            rows_aligned_with_the_box += 1
                            low, high = sorted((start["x"], end["x"]))
                            crosses = low < BOX["x"] + BOX["w"] + buffer_mm - 0.5 and high > BOX["x"] - buffer_mm + 0.5
                            self.assertFalse(crosses, f"{corner}: pass {low}..{high} crosses the box")
                self.assertGreater(rows_aligned_with_the_box, 0)

    def test_the_obstacle_dicts_passed_in_are_not_modified(self):
        before = dict(BOX)
        plan("top_right", [BOX])
        self.assertEqual(BOX, before)

    def test_parse_start_corner(self):
        self.assertEqual(GeometryEngine.parse_start_corner(None), "bottom_left")
        self.assertEqual(GeometryEngine.parse_start_corner(""), "bottom_left")
        self.assertEqual(GeometryEngine.parse_start_corner("top_right"), "top_right")
        for bad in ("middle", "TOP_RIGHT", 5, ["top_left"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                GeometryEngine.parse_start_corner(bad)

    def test_planner_rejects_an_unknown_corner(self):
        with self.assertRaises(ValueError):
            plan("middle")


if __name__ == "__main__":
    unittest.main()
