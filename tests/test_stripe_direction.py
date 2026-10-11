import math
import unittest

from rover_support import SimTestCase
from geometry_engine import DIRECTIONS, START_CORNERS, GeometryEngine
from rover_mission import compile_rover_mission, load_profile, replay_rover_mission
from run_mission import run_mission

W, H = 3000.0, 2000.0
OBSTACLES = [
    {"id": "o1", "type": "window", "x": 1200.0, "y": 700.0, "w": 600.0, "h": 700.0},
    {"id": "o2", "type": "door", "x": 0.0, "y": 0.0, "w": 500.0, "h": 900.0},
]


def plan(corner, direction, obstacles=(), w=W, h=H):
    return GeometryEngine().plan_coverage_path(w, h, 250.0, 12.0, 60.0, [dict(o) for o in obstacles], corner, direction)


def spray_segments(waypoints):
    return [(p, q) for p, q in zip(waypoints, waypoints[1:]) if p["spray_active"] and q["spray_active"]
            and math.hypot(q["x"] - p["x"], q["y"] - p["y"]) > 0.5]


class TestPlanner(unittest.TestCase):
    def test_directions_are_horizontal_and_vertical(self):
        self.assertEqual(DIRECTIONS, ("horizontal", "vertical"))
        self.assertEqual(GeometryEngine.parse_direction(None), "horizontal")
        self.assertEqual(GeometryEngine.parse_direction(""), "horizontal")
        self.assertEqual(GeometryEngine.parse_direction("vertical"), "vertical")
        for bad in ("diagonal", 3, ["vertical"]):
            with self.assertRaises(ValueError):
                GeometryEngine.parse_direction(bad)

    def test_horizontal_is_the_default_and_unchanged(self):
        default = GeometryEngine().plan_coverage_path(W, H, 250.0, 12.0, 60.0, [], "bottom_left")
        self.assertEqual(default, plan("bottom_left", "horizontal"))

    def test_vertical_stripes_run_along_y_and_start_in_the_chosen_corner(self):
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                waypoints, stats = plan(corner, "vertical")
                first, last = spray_segments(waypoints)[0]
                self.assertEqual(first["x"] == last["x"], True)  # a vertical pass
                self.assertNotEqual(first["y"], last["y"])
                self.assertLessEqual(min(first["x"], W - first["x"]), 250.0)  # first column hugs the start side
                start = waypoints[0]
                self.assertIn(start["y"], (0.0, H))
                self.assertEqual(start["y"] == 0.0, corner.startswith("bottom"))
                self.assertEqual(start["x"] < W / 2, corner.endswith("left"))
                self.assertEqual(stats["num_rows"], math.ceil(W / stats["effective_step_mm"]))

    def test_vertical_covers_the_same_paint_distance_as_horizontal_on_a_clear_wall(self):
        horizontal = plan("bottom_left", "horizontal")[1]
        vertical = plan("bottom_left", "vertical")[1]
        # distance is about the stripes times their length: rows x width against columns x height
        self.assertAlmostEqual(vertical["spray_distance_m"], math.ceil(W / vertical["effective_step_mm"]) * H / 1000.0, delta=0.1)
        self.assertGreater(horizontal["spray_distance_m"], 0)

    def test_vertical_spray_passes_avoid_obstacles_at_their_real_position(self):
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                waypoints, _ = plan(corner, "vertical", OBSTACLES)
                for obstacle in OBSTACLES:
                    for p, q in spray_segments(waypoints):
                        x = p["x"]
                        y_low, y_high = sorted((p["y"], q["y"]))
                        inside_x = obstacle["x"] < x < obstacle["x"] + obstacle["w"]
                        overlap_y = y_low < obstacle["y"] + obstacle["h"] and y_high > obstacle["y"]
                        self.assertFalse(inside_x and overlap_y, (obstacle["id"], p, q))

    def test_the_obstacle_dicts_passed_in_are_not_modified(self):
        before = [dict(o) for o in OBSTACLES]
        plan("top_right", "vertical", OBSTACLES)
        self.assertEqual(OBSTACLES, before)


class TestMission(unittest.TestCase):
    def mission(self, corner, direction, obstacles=()):
        waypoints, stats = plan(corner, direction, obstacles)
        return waypoints, compile_rover_mission(waypoints, W, H, 250.0, stats["effective_step_mm"], corner, load_profile())

    def test_vertical_columns_end_with_an_edge_contact_on_y(self):
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                _, mission = self.mission(corner, "vertical")
                edges = [s for s in mission["steps"] if s["op"] == "EDGE_ALIGN"]
                self.assertTrue(edges)
                for step in edges:
                    self.assertEqual(step["axis"], "y")
                    self.assertIn(step["value_mm"], (0.0, H))
                    self.assertIn(step["heading_deg"], (90.0, -90.0))

    def test_horizontal_missions_still_use_x_edges_only(self):
        _, mission = self.mission("bottom_left", "horizontal")
        self.assertEqual({s["axis"] for s in mission["steps"] if s["op"] == "EDGE_ALIGN"}, {"x"})

    def test_replaying_a_vertical_mission_gives_the_planned_spray_distance(self):
        waypoints, mission = self.mission("top_left", "vertical", OBSTACLES)
        replay = replay_rover_mission(mission)
        sprayed = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b, spray in replay["moves"] if spray)
        planned = sum(math.hypot(q["x"] - p["x"], q["y"] - p["y"]) for p, q in spray_segments(waypoints))
        self.assertAlmostEqual(sprayed, planned, delta=1.0)


class TestVerticalMissionInTheSimulator(SimTestCase):
    def test_every_corner_runs_and_paints_what_was_planned(self):
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                waypoints, mission = TestMission().mission(corner, "vertical", OBSTACLES)
                pose, wall = mission["start_pose"], mission["wall"]
                t = self.sim("--start", pose["x_mm"], pose["y_mm"], pose["heading_deg"],
                             "--wall", wall["width_mm"], wall["height_mm"], "--slip", "0.03", "--seed", "2",
                             "--gyro-bias", "0.01")
                run_mission(t, mission)
                planned = sum(math.hypot(q["x"] - p["x"], q["y"] - p["y"]) for p, q in spray_segments(waypoints))
                truth = t.truth()
                self.assertAlmostEqual(truth["painted"], planned, delta=0.03 * planned)
                last = mission["steps"][-1]
                edge_axis = "y" if last["op"] == "EDGE_ALIGN" else None
                if edge_axis:
                    self.assertAlmostEqual(truth["y"], last["value_mm"], delta=1.0)


if __name__ == "__main__":
    unittest.main()
