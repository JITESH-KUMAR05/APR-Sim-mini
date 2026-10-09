import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fakes import planner_moves
from geometry_engine import START_CORNERS, GeometryEngine
from rover_mission import SCHEMA, compile_rover_mission, load_profile, replay_rover_mission, verify_checksum

WALL = 2438.4  # 8 ft square
OBSTACLES = [
    {"id": "obs_1", "type": "window", "x": 1000.0, "y": 600.0, "w": 700.0, "h": 800.0},
    {"id": "obs_2", "type": "door", "x": 0.0, "y": 0.0, "w": 500.0, "h": 900.0},
]


def plan_and_compile(corner, obstacles=()):
    waypoints, stats = GeometryEngine().plan_coverage_path(WALL, WALL, 250.0, 12.0, 60.0, list(obstacles), corner)
    mission = compile_rover_mission(waypoints, WALL, WALL, 250.0, stats["effective_step_mm"], corner, load_profile())
    return waypoints, stats, mission


def write_profile(tmp, **overrides):
    profile = {"schema": "apr-rover-profile/1", "wheel_diameter_mm": 80.0, "max_speed_mps": 0.25}
    profile.update(overrides)
    path = os.path.join(tmp, "profile.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f)
    return path


class TestProfile(unittest.TestCase):
    def test_the_shipped_profile_loads(self):
        profile = load_profile()
        self.assertEqual(profile["schema"], "apr-rover-profile/1")
        self.assertEqual(profile["max_speed_mps"], 0.25)

    def test_bad_profiles_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for overrides in ({"schema": "other/1"}, {"max_speed_mps": 0}, {"max_speed_mps": 0.9}, {"wheel_diameter_mm": -1}, {"max_speed_mps": True}):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    load_profile(write_profile(tmp, **overrides))


class TestCompile(unittest.TestCase):
    def test_header(self):
        _, stats, mission = plan_and_compile("bottom_left")
        self.assertEqual(mission["schema"], SCHEMA)
        self.assertEqual(mission["wall"], {"width_mm": WALL, "height_mm": WALL, "start_corner": "bottom_left"})
        self.assertEqual(mission["spray"], {"width_mm": 250.0, "row_pitch_mm": 220.0})
        self.assertEqual(mission["speed_mps"], 0.25)
        self.assertEqual(mission["profile"], load_profile())
        self.assertEqual(stats["effective_step_mm"], 220.0)

    def test_start_pose_for_each_corner(self):
        expected = {
            "bottom_left": (0.0, 125.0, 0.0),
            "bottom_right": (WALL, 125.0, 180.0),
            "top_left": (0.0, 2313.4, 0.0),
            "top_right": (WALL, 2313.4, 180.0),
        }
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                pose = plan_and_compile(corner)[2]["start_pose"]
                self.assertEqual((pose["x_mm"], pose["y_mm"], pose["heading_deg"]), expected[corner])

    def test_serpentine_shape_on_a_clear_wall(self):
        _, _, mission = plan_and_compile("bottom_left")
        steps = mission["steps"]
        self.assertEqual(
            [s["op"] for s in steps[:11]],
            ["MOVE", "EDGE_ALIGN", "TURN", "MOVE", "TURN", "MOVE", "EDGE_ALIGN", "TURN", "MOVE", "TURN", "MOVE"],
        )
        self.assertEqual([steps[i]["spray"] for i in (0, 3, 5, 8, 10)], [1, 0, 1, 0, 1])
        self.assertEqual([steps[i]["deg"] for i in (2, 4, 7, 9)], [90.0, 90.0, -90.0, -90.0])
        self.assertEqual([s["i"] for s in steps], list(range(1, len(steps) + 1)))

    def test_replaying_the_steps_reproduces_the_planner_path_from_every_corner(self):
        for corner in START_CORNERS:
            for obstacles in ((), OBSTACLES):
                with self.subTest(corner=corner, obstacles=len(obstacles)):
                    waypoints, _, mission = plan_and_compile(corner, obstacles)
                    expected = planner_moves(waypoints)
                    actual = replay_rover_mission(mission)["moves"]
                    self.assertEqual(len(actual), len(expected))
                    for (e_start, e_end, e_spray), (a_start, a_end, a_spray) in zip(expected, actual):
                        self.assertEqual(e_spray, a_spray)
                        for e, a in ((e_start, a_start), (e_end, a_end)):
                            self.assertAlmostEqual(e[0], a[0], delta=0.01)
                            self.assertAlmostEqual(e[1], a[1], delta=0.01)

    def test_stored_expected_pose_matches_the_replay_after_every_step(self):
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                _, _, mission = plan_and_compile(corner, OBSTACLES)
                poses = replay_rover_mission(mission)["poses"]
                self.assertEqual(len(poses), len(mission["steps"]))
                for step, (x, y, heading) in zip(mission["steps"], poses):
                    self.assertAlmostEqual(step["x_mm"], x, delta=0.01)
                    self.assertAlmostEqual(step["y_mm"], y, delta=0.01)
                    self.assertAlmostEqual(step["heading_deg"], heading, delta=0.01)

    def test_one_edge_align_per_full_row_pass_on_a_clear_wall(self):
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                _, stats, mission = plan_and_compile(corner)
                steps = mission["steps"]
                aligns = [i for i, s in enumerate(steps) if s["op"] == "EDGE_ALIGN"]
                self.assertEqual(len(aligns), stats["num_rows"])
                for i in aligns:
                    self.assertEqual(steps[i - 1]["op"], "MOVE")
                    self.assertIn(steps[i]["value_mm"], (0.0, WALL))
                    self.assertEqual(steps[i]["axis"], "x")
                    self.assertEqual(steps[i]["x_mm"], steps[i]["value_mm"])

    def test_checksum_detects_tampering(self):
        _, _, mission = plan_and_compile("bottom_left", OBSTACLES)
        self.assertTrue(verify_checksum(mission))
        tampered = copy.deepcopy(mission)
        tampered["steps"][0]["mm"] += 1.0
        self.assertFalse(verify_checksum(tampered))

    def test_mission_survives_a_json_round_trip(self):
        _, _, mission = plan_and_compile("top_right", OBSTACLES)
        again = json.loads(json.dumps(mission))
        self.assertEqual(again, mission)
        self.assertTrue(verify_checksum(again))

    def test_diagonal_waypoints_are_rejected(self):
        diagonal = [
            {"x": 0.0, "y": 0.0, "spray_active": True},
            {"x": 100.0, "y": 100.0, "spray_active": True},
        ]
        with self.assertRaises(ValueError):
            compile_rover_mission(diagonal, WALL, WALL, 250.0, 220.0, "bottom_left", load_profile())

    def test_a_mission_with_no_motion_is_rejected(self):
        still = [{"x": 5.0, "y": 5.0, "spray_active": True}, {"x": 5.0, "y": 5.2, "spray_active": True}]
        with self.assertRaises(ValueError):
            compile_rover_mission(still, WALL, WALL, 250.0, 220.0, "bottom_left", load_profile())

    def test_replay_rejects_an_unknown_step(self):
        _, _, mission = plan_and_compile("bottom_left")
        mission["steps"].append({"i": 99, "op": "JUMP"})
        with self.assertRaises(ValueError):
            replay_rover_mission(mission)


if __name__ == "__main__":
    unittest.main()
