import unittest

from rover_support import SimTestCase, words
from rover_link import parse_line


def done_pose(lines):
    """x, y, heading from the DONE line in `lines`."""
    for line in lines:
        word, args = parse_line(line)
        if word == "DONE":
            return float(args[1]), float(args[2]), float(args[3])
    raise AssertionError(f"no DONE in {lines}")


def heading_difference(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


class TestMove(SimTestCase):
    def test_move_1000_is_within_two_percent_and_reports_where_it_ended(self):
        t = self.sim("--start", "0", "125", "0")
        self.arm(t)
        lines, _ = self.run_step(t, "MOVE 1 1000.0 0")
        self.assertEqual(words(lines)[:2], ["ACK 1", "STATE RUNNING"])
        self.assertEqual(words(lines)[-1], "STATE ARMED")
        truth = t.truth()
        self.assertAlmostEqual(truth["x"], 1000.0, delta=20.0)
        self.assertAlmostEqual(truth["y"], 125.0, delta=5.0)
        x, y, heading = done_pose(lines)
        self.assertAlmostEqual(x, truth["x"], delta=3.0)  # no slip: the estimate is the truth
        self.assertAlmostEqual(y, truth["y"], delta=3.0)

    def test_the_pose_is_streamed_about_ten_times_a_second_while_a_step_runs(self):
        t = self.sim("--start", "0", "125", "0")
        self.arm(t)
        t.send("MOVE 1 1000.0 0")
        lines = []
        for _ in range(10):
            lines += t.advance(100)
            t.send("PING")
        poses = [line for line in lines if line.startswith("POSE")]
        self.assertTrue(9 <= len(poses) <= 11, len(poses))
        self.assertLess(float(parse_line(poses[0])[1][0]), float(parse_line(poses[-1])[1][0]))  # x grows

    def test_the_heading_is_held_when_the_estimate_starts_off_the_true_heading(self):
        t = self.sim("--start", "0", "125", "3")  # true heading 3 degrees, the rover is told 0
        self.arm(t)
        self.run_step(t, "MOVE 1 1500.0 0")
        self.assertLess(abs(t.truth()["heading"] - 3.0), 0.5)  # it holds the heading it started with

    def test_heading_hold_keeps_the_rover_straight_when_the_wheels_slip_unevenly(self):
        # 5% slip, a different amount on each wheel: with no correction the rover would curve away.
        t = self.sim("--start", "0", "125", "0", "--slip", "0.05", "--seed", "5")
        self.arm(t)
        self.run_step(t, "MOVE 1 2000.0 0")
        truth = t.truth()
        self.assertLess(abs(truth["heading"]), 1.0)
        self.assertLess(abs(truth["y"] - 125.0), 15.0)

    def test_moves_work_in_every_direction(self):
        for heading in (0.0, 90.0, 180.0, -90.0):
            with self.subTest(heading=heading):
                t = self.sim("--start", "1200", "1200", heading)
                self.arm(t, pose=(1200.0, 1200.0, heading))
                self.run_step(t, "MOVE 1 500.0 0")
                truth = t.truth()
                import math
                self.assertAlmostEqual(truth["x"], 1200 + 500 * math.cos(math.radians(heading)), delta=12.0)
                self.assertAlmostEqual(truth["y"], 1200 + 500 * math.sin(math.radians(heading)), delta=12.0)


class TestTurn(SimTestCase):
    def test_turns_land_within_two_degrees(self):
        for degrees in (90.0, -90.0, 180.0, -180.0, 45.0, 10.0, 135.0):
            with self.subTest(degrees=degrees):
                t = self.sim("--start", "1200", "1200", "0")
                self.arm(t, pose=(1200.0, 1200.0, 0.0))
                lines, _ = self.run_step(t, f"TURN 1 {degrees}")
                self.assertEqual(words(lines)[:2], ["ACK 1", "STATE RUNNING"])
                self.assertLess(heading_difference(t.truth()["heading"], degrees), 2.0)
                self.assertLess(heading_difference(done_pose(lines)[2], degrees), 2.0)

    def test_a_turn_stays_on_the_spot(self):
        t = self.sim("--start", "1200", "1200", "0")
        self.arm(t, pose=(1200.0, 1200.0, 0.0))
        self.run_step(t, "TURN 1 90.0")
        truth = t.truth()
        self.assertAlmostEqual(truth["x"], 1200.0, delta=3.0)
        self.assertAlmostEqual(truth["y"], 1200.0, delta=3.0)

    def test_four_quarter_turns_make_a_full_circle(self):
        t = self.sim("--start", "1200", "1200", "0")
        self.arm(t, pose=(1200.0, 1200.0, 0.0))
        for step in range(1, 5):
            self.run_step(t, f"TURN {step} 90.0")
        self.assertLess(heading_difference(t.truth()["heading"], 0.0), 3.0)

    def test_the_pose_heading_is_normalised_in_reports(self):
        t = self.sim("--start", "1200", "1200", "170")
        self.arm(t, pose=(1200.0, 1200.0, 170.0))
        lines, _ = self.run_step(t, "TURN 1 20.0")
        self.assertLess(heading_difference(done_pose(lines)[2], -170.0), 2.0)
        self.assertLessEqual(abs(done_pose(lines)[2]), 180.0)


class TestEdge(SimTestCase):
    def test_edge_sets_the_coordinate_and_squares_the_heading(self):
        # The rover is 2 degrees off its nominal heading, short of the edge. Pushing into the edge
        # with both bumpers squares it up, so the heading is the nominal one afterwards.
        t = self.sim("--start", "2300", "125", "2")
        self.arm(t, pose=(2300.0, 125.0, 0.0))
        self.run_step(t, "MOVE 1 120.0 0")
        lines, _ = self.run_step(t, "EDGE 2 x 2438.4")
        x, y, heading = done_pose(lines)
        self.assertEqual((x, heading), (2438.4, 0.0))
        truth = t.truth()
        self.assertAlmostEqual(truth["x"], 2438.4, delta=0.1)
        self.assertAlmostEqual(truth["heading"], 0.0, delta=0.01)

    def test_edge_with_the_edge_already_touching_finishes_at_once(self):
        t = self.sim("--start", "2438.4", "125", "0")
        self.arm(t, pose=(2438.4, 125.0, 0.0))
        lines, elapsed = self.run_step(t, "EDGE 1 x 2438.4")
        self.assertEqual(words(lines)[-2:], ["DONE 1 2438.4 125.0 0.0", "STATE ARMED"])
        self.assertLessEqual(elapsed, 200)

    def test_edge_corrects_a_position_estimate_that_is_wrong(self):
        # 3% wheel slip: after a full row the rover believes it is at the edge but is not there yet.
        t = self.sim("--start", "0", "125", "0", "--slip", "0.03", "--seed", "3")
        self.arm(t)
        self.run_step(t, "MOVE 1 2438.4 0")
        short_of_the_edge = 2438.4 - t.truth()["x"]
        self.assertGreater(short_of_the_edge, 30.0)
        lines, _ = self.run_step(t, "EDGE 2 x 2438.4")
        self.assertEqual(done_pose(lines)[0], 2438.4)
        self.assertAlmostEqual(t.truth()["x"], 2438.4, delta=0.1)


class TestSpray(SimTestCase):
    def test_no_paint_without_the_spray_flag(self):
        t = self.sim("--start", "0", "125", "0")
        self.arm(t)
        self.run_step(t, "MOVE 1 500.0 0")
        self.assertEqual(t.truth()["painted"], 0)

    def test_pump_and_valve_run_during_a_spray_move_and_the_stripe_has_the_move_length(self):
        t = self.sim("--start", "0", "125", "0")
        self.arm(t)
        t.send("MOVE 1 500.0 1")
        for _ in range(10):
            t.advance(100)
            t.send("PING")
        during = t.truth()
        self.assertEqual((during["pump"], during["valve"]), (1, 1))
        lines = []
        while not any(line.startswith("DONE") for line in lines):
            lines = t.advance(100) + t.send("PING")
        after = t.truth()
        self.assertEqual((after["pump"], after["valve"]), (0, 0))
        self.assertAlmostEqual(after["painted"], 500.0, delta=15.0)


class TestPause(SimTestCase):
    def test_pause_stops_the_rover_and_resume_finishes_the_move(self):
        t = self.sim("--start", "0", "125", "0")
        self.arm(t)
        t.send("MOVE 1 1000.0 1")
        for _ in range(8):
            t.advance(100)
            t.send("PING")
        self.assertEqual(words(t.send("PAUSE")), ["ACK 0", "STATE PAUSED"])
        t.advance(300)
        t.send("PING")
        first = t.truth()
        t.advance(300)
        t.send("PING")
        second = t.truth()
        self.assertAlmostEqual(first["x"], second["x"], delta=0.5)  # standing still
        self.assertEqual((second["pump"], second["valve"]), (0, 0))
        self.assertEqual(words(t.send("RESUME")), ["ACK 0", "STATE RUNNING"])
        lines = []
        while not any(line.startswith(("DONE", "FAULT")) for line in lines):
            lines = t.advance(100) + t.send("PING")
        self.assertTrue(any(line.startswith("DONE") for line in lines))
        self.assertAlmostEqual(t.truth()["x"], 1000.0, delta=20.0)

    def test_the_watchdog_still_applies_while_paused(self):
        t = self.sim("--start", "0", "125", "0")
        self.arm(t)
        t.send("MOVE 1 1000.0 0")
        t.advance(200)
        t.send("PAUSE")
        self.assertIn("FAULT WATCHDOG", words(t.advance(700)))


if __name__ == "__main__":
    unittest.main()
