import contextlib
import io
import json
import os
import tempfile
import unittest

from rover_support import SimTestCase
from geometry_engine import START_CORNERS, GeometryEngine
from rover_link import frame
from rover_mission import _checksum, compile_rover_mission, load_profile
from run_mission import MissionError, main, pose_errors, run_mission

WALL = 2438.4  # 8 ft square
OBSTACLES = [
    {"id": "obs_1", "type": "window", "x": 1000.0, "y": 600.0, "w": 700.0, "h": 800.0},
    {"id": "obs_2", "type": "door", "x": 0.0, "y": 0.0, "w": 500.0, "h": 900.0},
]


def compiled(corner="bottom_left", obstacles=(), profile=None, wall=WALL):
    waypoints, stats = GeometryEngine().plan_coverage_path(wall, wall, 250.0, 12.0, 60.0, list(obstacles), corner)
    return compile_rover_mission(waypoints, wall, wall, 250.0, stats["effective_step_mm"], corner, profile or load_profile())


class RunMissionTestCase(SimTestCase):
    def start(self, mission, *extra):
        pose = mission["start_pose"]
        wall = mission["wall"]
        return self.sim("--start", pose["x_mm"], pose["y_mm"], pose["heading_deg"],
                        "--wall", wall["width_mm"], wall["height_mm"], *extra)


class TestRunMission(RunMissionTestCase):
    def test_a_compiled_mission_runs_to_the_end_and_matches_its_expected_poses(self):
        mission = compiled()
        t = self.start(mission)
        records = run_mission(t, mission)
        self.assertEqual(len(records), len(mission["steps"]))
        errors = pose_errors(records)
        self.assertLess(errors["x"], 5.0)
        self.assertLess(errors["y"], 10.0)
        self.assertLess(errors["heading"], 1.0)
        last = mission["steps"][-1]
        truth = t.truth()
        self.assertAlmostEqual(truth["x"], last["x_mm"], delta=1.0)  # the last step is an edge contact
        self.assertAlmostEqual(truth["y"], last["y_mm"], delta=15.0)

    def test_every_start_corner_runs(self):
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                mission = compiled(corner)
                t = self.start(mission)
                run_mission(t, mission)
                last = mission["steps"][-1]
                truth = t.truth()
                self.assertAlmostEqual(truth["x"], last["x_mm"], delta=1.0)
                self.assertAlmostEqual(truth["y"], last["y_mm"], delta=15.0)

    def test_the_stripes_painted_match_the_spray_distance_of_the_mission(self):
        for obstacles in ((), OBSTACLES):
            with self.subTest(obstacles=len(obstacles)):
                mission = compiled("bottom_left", obstacles)
                t = self.start(mission)
                run_mission(t, mission)
                sprayed = sum(step["mm"] for step in mission["steps"] if step["op"] == "MOVE" and step["spray"])
                self.assertGreater(sprayed, 10000.0)
                self.assertAlmostEqual(t.truth()["painted"], sprayed, delta=0.005 * sprayed)
                truth = t.truth()
                self.assertEqual((truth["pump"], truth["valve"]), (0, 0))

    def test_accuracy_with_three_percent_wheel_slip_and_gyro_drift(self):
        # What the edge contacts fix and what they cannot: x is re-zeroed at every row end and the
        # heading is squared, so rows stay evenly spaced. Nothing re-references y, so the absolute
        # y error grows with the slip times the vertical distance driven (about 65 mm here).
        mission = compiled()
        t = self.start(mission, "--slip", "0.03", "--seed", "3", "--gyro-bias", "0.01")
        row_ends = []

        def on_step(step, reported):
            if step["op"] == "EDGE_ALIGN":
                truth = t.truth()
                row_ends.append((truth["x"] - step["value_mm"], truth["y"] - step["y_mm"]))

        run_mission(t, mission, on_step=on_step)
        self.assertEqual(len(row_ends), 12)
        for x_error, _ in row_ends:
            self.assertAlmostEqual(x_error, 0.0, delta=1.0)
        y_errors = [y for _, y in row_ends]
        spacing_errors = [abs(b - a) for a, b in zip(y_errors, y_errors[1:])]
        self.assertLess(max(spacing_errors), 30.0, spacing_errors)
        vertical_travel = mission["steps"][-1]["y_mm"] - mission["start_pose"]["y_mm"]
        self.assertLess(abs(y_errors[-1]), 0.04 * vertical_travel, y_errors)

    def test_without_slip_every_row_lands_within_fifteen_millimetres(self):
        mission = compiled()
        t = self.start(mission, "--gyro-bias", "0.01")
        errors = []
        run_mission(t, mission, on_step=lambda step, reported: errors.append(t.truth()["y"] - step["y_mm"]) if step["op"] == "EDGE_ALIGN" else None)
        self.assertLess(max(abs(e) for e in errors), 15.0, errors)


class TestRefusals(RunMissionTestCase):
    def test_a_mission_whose_checksum_does_not_match_is_refused_before_anything_is_sent(self):
        mission = compiled()
        mission["steps"][0]["mm"] += 1.0
        t = self.start(mission)
        sent_before = list(t.log)
        with self.assertRaisesRegex(MissionError, "checksum"):
            run_mission(t, mission)
        self.assertEqual(t.log, sent_before)

    def test_a_mission_faster_than_the_firmware_allows_is_refused(self):
        mission = compiled(profile={**load_profile(), "max_speed_mps": 0.4})
        with self.assertRaisesRegex(MissionError, "above the rover"):
            run_mission(self.start(mission), mission)

    def test_a_profile_that_allows_a_longer_pump_run_than_the_firmware_is_refused(self):
        mission = compiled(profile={**load_profile(), "pump_max_continuous_s": 60.0})
        with self.assertRaisesRegex(MissionError, "pump"):
            run_mission(self.start(mission), mission)

    def test_a_nak_from_the_rover_stops_the_run_with_its_reason(self):
        waypoints, stats = GeometryEngine().plan_coverage_path(4000.0, 4000.0, 250.0, 12.0, 60.0, [], "bottom_left")
        mission = compile_rover_mission(waypoints, 2000.0, 2000.0, 250.0, stats["effective_step_mm"], "bottom_left", load_profile())
        with self.assertRaisesRegex(MissionError, "NAK 1 TOO_LONG"):
            run_mission(self.start(mission), mission)

    def test_a_rover_fault_stops_the_run_with_its_code(self):
        mission = compiled()
        t = self.start(mission, "--motor-scale", "0.1")
        with self.assertRaisesRegex(MissionError, "STEP_TIMEOUT"):
            run_mission(t, mission)


class TestCommandLine(RunMissionTestCase):
    def write_mission(self, mission):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        with handle:
            json.dump(mission, handle)
        self.addCleanup(os.remove, handle.name)
        return handle.name

    def test_the_command_line_runs_a_mission_file(self):
        mission = compiled()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([self.write_mission(mission), "--sim", self.binary])
        self.assertEqual(code, 0)
        self.assertIn(f"{len(mission['steps'])} steps done", out.getvalue())
        self.assertIn("True final pose", out.getvalue())

    def test_the_command_line_reports_a_failed_mission(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main([self.write_mission(compiled()), "--sim", self.binary, "--sim-option=--motor-scale", "--sim-option=0.1"])
        self.assertEqual(code, 1)
        self.assertIn("mission failed", err.getvalue())
        self.assertIn("STEP_TIMEOUT", err.getvalue())

    def test_a_missing_simulator_binary_is_a_failed_mission_not_a_traceback(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main([self.write_mission(compiled()), "--sim", os.path.join(tempfile.gettempdir(), "no_such_rover_sim.exe")])
        self.assertEqual(code, 1)
        self.assertIn("mission failed", err.getvalue())


def rover_line(payload):
    return frame(payload)


class FakeRover:
    """Scripted transport: replies maps a sent word to the lines the rover answers with."""

    def __init__(self, replies, wait_replies=()):
        self.replies = replies
        self.wait_replies = list(wait_replies)
        self.sent = []
        self.waited = 0

    def send(self, payload):
        self.sent.append(payload)
        word = payload.split()[0]
        reply = self.replies.get(word, ["ACK 0"])
        if callable(reply):
            reply = reply(payload)
        return [rover_line(line) if not line.endswith("!raw") else line[:-4] for line in reply]

    def wait(self, ms):
        self.waited += ms
        return self.wait_replies.pop(0) if self.wait_replies else []


HELLO = "HELLO fw=sim proto=1 max_speed_mm_s=250 pump_max_s=30"


def small_mission():
    return compiled()


def step_ack(payload):
    return [f"ACK {payload.split()[1]}"]


class TestHostRobustness(unittest.TestCase):
    def rover(self, wait_replies=(), **overrides):
        replies = {"HELLO": [HELLO], "MOVE": step_ack, "TURN": step_ack, "EDGE": step_ack}
        replies.update(overrides)
        return FakeRover(replies, wait_replies)

    def test_wait_ms_that_would_starve_the_watchdog_or_hang_is_refused(self):
        for bad in (0, 5, 251, 500, 600, -10, True, 1.5):
            with self.subTest(wait_ms=bad):
                fake = self.rover()
                with self.assertRaisesRegex(MissionError, "wait_ms"):
                    run_mission(fake, small_mission(), wait_ms=bad)
                self.assertEqual(fake.sent, [])

    def test_a_corrupted_hello_is_a_mission_error(self):
        for hello in ("HELLO fw=sim proto=1 pump_max_s=30", "HELLO fw=sim proto=1 max_speed_mm_s=abc pump_max_s=30",
                      "HELLO junk proto=1 max_speed_mm_s=250 pump_max_s=30"):
            with self.subTest(hello=hello):
                with self.assertRaises(MissionError):
                    run_mission(self.rover(HELLO=[hello]), small_mission())

    def test_a_protocol_version_mismatch_is_refused(self):
        with self.assertRaisesRegex(MissionError, "protocol 2"):
            run_mission(self.rover(HELLO=["HELLO fw=sim proto=2 max_speed_mm_s=250 pump_max_s=30"]), small_mission())

    def test_a_line_with_a_bad_crc_is_skipped_as_noise(self):
        mission = small_mission()
        first = mission["steps"][0]
        done = f"DONE {first['i']} 1.0 2.0 3.0"
        fake = self.rover(wait_replies=[["PONG*00!raw", rover_line(done)]])
        trimmed = {key: value for key, value in mission.items() if key != "checksum"}
        trimmed["steps"] = mission["steps"][:1]
        trimmed["checksum"] = _checksum(trimmed)
        records = run_mission(fake, trimmed)
        self.assertEqual(records[0]["reported"], {"x": 1.0, "y": 2.0, "heading": 3.0})

    def test_a_malformed_done_is_a_mission_error(self):
        mission = small_mission()
        first = mission["steps"][0]
        fake = self.rover(wait_replies=[[rover_line(f"DONE {first['i']} 1.0")]])
        with self.assertRaisesRegex(MissionError, "malformed DONE"):
            run_mission(fake, mission)
        self.assertEqual(fake.sent[-1], "STOP")

    def test_an_estop_at_command_time_reports_the_estop_not_a_badstate_nak(self):
        mission = small_mission()
        estop = lambda payload: ["FAULT ESTOP", "STATE FAULT", f"NAK {payload.split()[1]} BADSTATE"]
        with self.assertRaisesRegex(MissionError, "fault ESTOP"):
            run_mission(self.rover(MOVE=estop, TURN=estop, EDGE=estop), mission)

    def test_a_fault_arriving_with_the_heartbeat_reply_stops_the_run(self):
        fake = self.rover(PING=["FAULT STALL"])
        with self.assertRaisesRegex(MissionError, "STALL"):
            run_mission(fake, small_mission())
        self.assertEqual(fake.sent[-1], "STOP")

    def test_a_step_that_never_finishes_times_out_and_stops_the_rover(self):
        fake = self.rover()
        with self.assertRaisesRegex(MissionError, "did not finish"):
            run_mission(fake, small_mission(), max_step_ms=300)
        self.assertEqual(fake.sent[-1], "STOP")
        self.assertLessEqual(fake.waited, 300)

    def test_a_done_for_another_step_is_ignored(self):
        fake = self.rover(wait_replies=[[rover_line("DONE 99 1.0 2.0 3.0")]])
        with self.assertRaisesRegex(MissionError, "did not finish"):
            run_mission(fake, small_mission(), max_step_ms=200)

    def test_an_ack_for_another_step_does_not_count(self):
        fake = self.rover(MOVE=["ACK 99"], TURN=["ACK 99"], EDGE=["ACK 99"])
        with self.assertRaisesRegex(MissionError, "no ACK"):
            run_mission(fake, small_mission())

    def test_a_refused_setup_command_names_the_command_and_the_reason(self):
        for word in ("SPEED", "POSE", "ARM", "WALL"):
            with self.subTest(word=word):
                with self.assertRaisesRegex(MissionError, f"{word} refused: NAK 0 BADSTATE"):
                    run_mission(self.rover(**{word: ["NAK 0 BADSTATE"]}), small_mission())

    def test_a_profile_without_a_pump_limit_still_runs(self):
        profile = load_profile()
        profile.pop("pump_max_continuous_s")
        mission = compiled(profile=profile)
        fake = self.rover(wait_replies=[[rover_line(f"DONE {mission['steps'][0]['i']} 1.0 2.0 3.0")]])
        with self.assertRaisesRegex(MissionError, "did not finish"):  # the fake never finishes step 2
            run_mission(fake, mission, max_step_ms=100)
        self.assertIn("ARM", fake.sent)


class TestPoseErrors(unittest.TestCase):
    def test_largest_difference_with_heading_wrap_around(self):
        records = [
            {"expected": {"x": 0.0, "y": 0.0, "heading": 180.0}, "reported": {"x": 1.0, "y": -2.0, "heading": -179.0}},
            {"expected": {"x": 5.0, "y": 5.0, "heading": 0.0}, "reported": {"x": 5.0, "y": 8.0, "heading": 0.5}},
        ]
        self.assertEqual(pose_errors(records), {"x": 1.0, "y": 3.0, "heading": 1.0})


if __name__ == "__main__":
    unittest.main()
