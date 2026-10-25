import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry_engine import DIRECTIONS, START_CORNERS, GeometryEngine
from region_planner import Router
from rover_mission import compile_rover_mission, load_profile
from wall_synth import random_scene

SPRAY, OVERLAP, BUFFER = 250.0, 12.0, 60.0
W, H = 4000.0, 2800.0
WINDOW = {"id": "w", "type": "window", "x": 1500.0, "y": 900.0, "w": 900.0, "h": 1100.0}
DOOR = {"id": "d", "type": "door", "x": 3300.0, "y": 0.0, "w": 700.0, "h": 2000.0}
TALL_PIPE = {"id": "p", "type": "pipe", "x": 600.0, "y": 0.0, "w": 120.0, "h": 2800.0}


def plan(obstacles, corner="top_left", direction="horizontal", w=W, h=H):
    return GeometryEngine().plan_coverage_path(w, h, SPRAY, OVERLAP, BUFFER, [dict(o) for o in obstacles], corner, direction)


def grown(obstacles):
    return [(o["x"] - BUFFER, o["y"] - BUFFER, o["x"] + o["w"] + BUFFER, o["y"] + o["h"] + BUFFER) for o in obstacles]


def enters(p, q, box, tol=0.5):
    """True if the straight move p->q passes through the inside of the box (not just along its edge)."""
    x1, y1, x2, y2 = box
    lo_x, hi_x = sorted((p["x"], q["x"]))
    lo_y, hi_y = sorted((p["y"], q["y"]))
    return lo_x < x2 - tol and hi_x > x1 + tol and lo_y < y2 - tol and hi_y > y1 + tol


def scenes():
    yield "window and door", W, H, [WINDOW, DOOR]
    for seed in (3, 11, 42, 77, 128, 901):
        _, w, h, truth = random_scene(seed, 1.2, width_px=600)
        yield f"random wall {seed}", w, h, [dict(t, id=f"o{i}") for i, t in enumerate(truth)]


class TestTheRoverStaysOffKeepOuts(unittest.TestCase):
    def test_no_move_of_any_layout_goes_through_a_keep_out_and_every_move_is_straight(self):
        for name, w, h, obstacles in scenes():
            boxes = grown(obstacles)
            for corner in START_CORNERS:
                for direction in DIRECTIONS:
                    with self.subTest(wall=name, corner=corner, direction=direction):
                        waypoints, _ = plan(obstacles, corner, direction, w, h)
                        for p, q in zip(waypoints, waypoints[1:]):
                            self.assertTrue(abs(p["x"] - q["x"]) < 0.05 or abs(p["y"] - q["y"]) < 0.05, (p, q))
                            for box in boxes:
                                self.assertFalse(enters(p, q, box), f"{p} -> {q} goes through {box}")

    def test_every_layout_still_compiles_to_a_rover_mission(self):
        profile = load_profile()
        for name, w, h, obstacles in scenes():
            for corner in START_CORNERS:
                with self.subTest(wall=name, corner=corner):
                    waypoints, stats = plan(obstacles, corner, "horizontal", w, h)
                    mission = compile_rover_mission(waypoints, w, h, SPRAY, stats["effective_step_mm"], corner, profile)
                    self.assertTrue(mission["steps"])


class TestCoverage(unittest.TestCase):
    def free_length(self, obstacles, rows):
        boxes = grown(obstacles)
        total = 0.0
        for y in rows:
            cuts = sorted((max(0.0, b[0]), min(W, b[2])) for b in boxes if b[1] <= y <= b[3])
            x, free = 0.0, 0.0
            for a, b in cuts:
                free += max(0.0, a - x)
                x = max(x, b)
            total += free + max(0.0, W - x)
        return total

    def test_a_clear_wall_is_one_zigzag_like_before(self):
        waypoints, stats = plan([])
        self.assertEqual((stats["num_regions"], stats["num_passes"], stats["unreached_regions"]), (1, stats["num_rows"], 0))
        self.assertEqual((waypoints[0]["x"], waypoints[0]["y"], waypoints[1]["x"]), (0.0, H - 125.0, W))
        self.assertEqual([w["type"] for w in waypoints[:4]], ["PASS_START", "PASS_END", "ROW_TRANSITION", "PASS_START"])

    def test_every_free_piece_of_every_row_is_painted_once(self):
        for obstacles in ([WINDOW], [WINDOW, DOOR]):
            with self.subTest(obstacles=[o["id"] for o in obstacles]):
                waypoints, stats = plan(obstacles)
                rows = sorted({wp["y"] for wp in waypoints if wp["type"] == "PASS_START"}, reverse=True)
                self.assertEqual(stats["unreached_regions"], 0)
                passes = sorted({wp["pass"] for wp in waypoints if "pass" in wp})
                self.assertEqual(passes, list(range(1, stats["num_passes"] + 1)))
                self.assertAlmostEqual(stats["spray_distance_m"] * 1000.0, self.free_length(obstacles, rows), delta=5.0)

    def test_regions_above_a_window_are_painted_before_the_region_below_it(self):
        waypoints, stats = plan([WINDOW])
        bottom = WINDOW["y"] - BUFFER
        order = [wp["y"] for wp in waypoints if wp["type"] == "PASS_START"]
        last_above = max(i for i, y in enumerate(order) if y > bottom)
        first_below = min(i for i, y in enumerate(order) if y < bottom)
        self.assertLess(last_above, first_below)
        self.assertGreater(stats["num_regions"], 2)  # above, left, right, below
        self.assertTrue(any(wp["type"] == "REGION_MOVE" for wp in waypoints))

    def test_a_floor_to_ceiling_pipe_leaves_the_far_side_out_and_says_how_much(self):
        waypoints, stats = plan([TALL_PIPE], "bottom_left")
        self.assertEqual(stats["unreached_regions"], 1)
        far_side = (W - (TALL_PIPE["x"] + TALL_PIPE["w"] + BUFFER)) * H / 1e6
        self.assertAlmostEqual(stats["unreached_area_m2"], far_side, delta=0.4)
        self.assertTrue(all(wp["x"] <= TALL_PIPE["x"] - BUFFER + 0.5 for wp in waypoints))


class TestRouter(unittest.TestCase):
    def test_a_move_blocked_by_a_box_goes_around_it(self):
        router = Router([(1000.0, 500.0, 2000.0, 1500.0)], 0.0, 4000.0, 0.0, 2800.0)
        route = router.route((500.0, 1000.0), (2500.0, 1000.0))
        self.assertIsNotNone(route)
        self.assertGreater(len(route), 2)
        for p, q in zip(route, route[1:]):
            self.assertTrue(router.free(p, q))

    def test_a_point_walled_off_cannot_be_reached(self):
        router = Router([(1000.0, -100.0, 1200.0, 2900.0)], 0.0, 4000.0, 0.0, 2800.0)
        self.assertIsNone(router.route((500.0, 1000.0), (2500.0, 1000.0)))


class TestSpeed(unittest.TestCase):
    def test_a_busy_wall_plans_all_eight_layouts_quickly(self):
        _, w, h, truth = random_scene(5, 1.5, width_px=600)
        obstacles = [dict(t, id=f"o{i}") for i, t in enumerate(truth)]
        started = time.perf_counter()
        for corner in START_CORNERS:
            for direction in DIRECTIONS:
                plan(obstacles, corner, direction, w, h)
        self.assertLess(time.perf_counter() - started, 3.0)


if __name__ == "__main__":
    unittest.main()
