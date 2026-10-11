import json
import math
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "rover", "tools"))
sys.path.insert(0, ROOT)

import app as app_module
from geometry_engine import GeometryEngine
from rover_mission import compile_rover_mission, load_profile, replay_rover_mission
from rover_support import simulator_binary
from sim_replay import replay

OBSTACLES = [{"id": "o1", "type": "window", "x": 1000.0, "y": 600.0, "w": 700.0, "h": 800.0}]


def mission(corner="bottom_left", direction="horizontal", wall=2438.4):
    waypoints, stats = GeometryEngine().plan_coverage_path(wall, wall, 250.0, 12.0, 60.0, [dict(o) for o in OBSTACLES], corner, direction)
    return compile_rover_mission(waypoints, wall, wall, 250.0, stats["effective_step_mm"], corner, load_profile())


class TestReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        simulator_binary()  # skips the whole class if the simulator cannot be built or launched here

    def test_a_mission_is_replayed_and_the_paint_matches_the_plan(self):
        m = mission()
        result = replay(m)
        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual(len(result["steps"]), len(m["steps"]))
        planned = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b, spray in replay_rover_mission(m)["moves"] if spray)
        self.assertAlmostEqual(result["summary"]["planned_spray_mm"], planned, delta=1.0)
        self.assertAlmostEqual(result["summary"]["painted_mm"], planned, delta=0.03 * planned)
        for step in result["steps"]:
            for key in ("expected", "reported", "truth"):
                self.assertEqual(set(step[key]), {"x", "y", "heading"})

    def test_the_path_starts_at_the_start_pose(self):
        m = mission("top_right", "vertical")
        result = replay(m)
        self.assertTrue(result["ok"], result.get("error"))
        start = result["start"]
        self.assertAlmostEqual(start["x"], m["start_pose"]["x_mm"], delta=0.5)
        self.assertAlmostEqual(start["y"], m["start_pose"]["y_mm"], delta=0.5)

    def test_slip_makes_the_true_path_differ_from_the_reported_one_but_edges_keep_x_exact(self):
        m = mission()
        result = replay(m, slip=0.05, gyro_bias=0.02, seed=4)
        self.assertTrue(result["ok"], result.get("error"))
        self.assertGreater(result["summary"]["max_truth_error_mm"]["y"], 1.0)
        edge_steps = [s for s, ms in zip(result["steps"], m["steps"]) if ms["op"] == "EDGE_ALIGN"]
        self.assertTrue(edge_steps)
        for step, ms in zip(result["steps"], m["steps"]):
            if ms["op"] == "EDGE_ALIGN":
                self.assertAlmostEqual(step["truth"]["x"], ms["value_mm"], delta=1.0)

    def test_a_fault_is_reported_with_the_steps_done_so_far(self):
        result = replay(mission(), extra_options=["--motor-scale", "0.1"])
        self.assertFalse(result["ok"])
        self.assertIn("STEP_TIMEOUT", result["error"])
        self.assertLess(len(result["steps"]), len(mission()["steps"]))

    def test_bad_numbers_are_refused_before_anything_runs(self):
        for kwargs in ({"slip": -0.1}, {"slip": 0.5}, {"gyro_bias": 5.0}, {"seed": -1}, {"slip": float("nan")}, {"seed": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replay(mission(), **kwargs)

    def test_a_tampered_mission_is_refused(self):
        m = mission()
        m["steps"][0]["mm"] += 5.0
        result = replay(m)
        self.assertFalse(result["ok"])
        self.assertIn("checksum", result["error"])


class TestEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        simulator_binary()
        cls.client = app_module.app.test_client()

    def generate(self, **extra):
        form = {"width_mm": "4000", "height_mm": "2800", "spray_width_mm": "250", "overlap_pct": "12",
                "safety_buffer_mm": "60", "sensitivity": "0.5", "sample_id": "residential"}
        form.update(extra)
        return json.loads(self.client.post("/api/analyze", data=form).data)

    def test_the_endpoint_replays_the_latest_mission(self):
        data = self.generate(start_corner="top_left", direction="vertical")
        res = self.client.post("/api/simulate_rover", json={"slip": 0.03, "gyro_bias": 0.01, "seed": 2})
        body = json.loads(res.data)
        self.assertEqual(res.status_code, 200, body)
        self.assertTrue(body["ok"], body.get("error"))
        self.assertAlmostEqual(body["summary"]["painted_mm"] / 1000.0, data["stats"]["spray_distance_m"],
                               delta=0.05 * data["stats"]["spray_distance_m"])

    def test_bad_numbers_are_a_400(self):
        self.generate()
        self.assertEqual(self.client.post("/api/simulate_rover", json={"slip": 9}).status_code, 400)
        self.assertEqual(self.client.post("/api/simulate_rover", data="nope").status_code, 400)

    def test_defaults_are_used_when_the_body_is_empty(self):
        self.generate()
        res = self.client.post("/api/simulate_rover", json={})
        self.assertEqual(res.status_code, 200)


class TestEndpointWithoutTheSimulator(unittest.TestCase):
    def test_a_deployment_without_the_rover_sources_answers_503_and_still_serves_the_page(self):
        client = app_module.app.test_client()
        with mock.patch.dict(sys.modules, {"sim_replay": None}):  # makes `import sim_replay` raise ImportError
            res = client.post("/api/simulate_rover", json={})
        self.assertEqual(res.status_code, 503)
        self.assertIn("local app", json.loads(res.data)["error"])
        self.assertEqual(client.get("/").status_code, 200)


if __name__ == "__main__":
    unittest.main()
