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


def run_for(transport, ms):
    """Let `ms` of simulated time pass in 100 ms steps, feeding the watchdog. Returns the lines (without PONG)."""
    lines = []
    for _ in range(ms // 100):
        lines += transport.advance(100) + transport.send("PING")
    return [line for line in lines if not line.startswith("PONG")]


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

    def test_edge_search_scales_with_the_wall(self):
        # On a 4 m wall, a full row with 5% slip ends about 190 mm short of the edge: more than the
        # old fixed 150 mm search, within the 150 mm + 5% of the wall = 350 mm the search now allows.
        t = self.sim("--start", "0", "125", "0", "--wall", "4000", "4000", "--slip", "0.05", "--seed", "5")
        self.arm(t, wall=(4000.0, 4000.0))
        self.run_step(t, "MOVE 1 4000.0 0")
        lines, _ = self.run_step(t, "EDGE 2 x 4000.0")
        self.assertFalse(any(line.startswith("FAULT") for line in lines), lines)
        self.assertEqual(words(lines)[-2:][1], "STATE ARMED")
        self.assertTrue(words(lines)[-2].startswith("DONE 2 "), lines)


class TestEdgeSensor(SimTestCase):
    def test_edge_ignores_an_edge_behind_the_rover(self):
        # The rover faces east at x = 0: the nearby boundary is behind it and must not count as contact.
        t = self.sim("--start", "0", "125", "0")
        self.arm(t)
        lines, elapsed = self.run_step(t, "EDGE 1 x 2438.4")
        self.assertEqual(words(lines)[-2:], ["FAULT EDGE_NOT_FOUND", "STATE FAULT"])
        # It gives up after 150 mm + 5% of the 2438.4 mm wall = 271.9 mm of search.
        self.assertAlmostEqual(t.truth()["x"], 271.9, delta=15.0)
        # The search runs at 30 mm/s, so about 9.1 s; the step timeout (2 x that + 2 s = 20.1 s) was not what ended it.
        self.assertAlmostEqual(elapsed, 271.9 / 30.0 * 1000.0, delta=1000.0)


class TestEdgeRequests(SimTestCase):
    def reply(self, heading, command):
        t = self.sim("--start", "1200", "2300", heading)
        self.arm(t, pose=(1200.0, 2300.0, heading))
        return words(t.send(command))[0]

    def test_edge_needs_a_matching_axis_and_value(self):
        cases = [
            (90.0, "EDGE 1 x 2438.4", "NAK 1 BADARG"),  # facing north, an x edge cannot be in front
            (90.0, "EDGE 1 y 2438.4", "ACK 1"),
            (90.0, "EDGE 1 y 0.0", "NAK 1 BADARG"),     # that edge is behind
            (0.0, "EDGE 1 x 500.0", "NAK 1 BADARG"),
            (0.0, "EDGE 1 x 2439.0", "ACK 1"),          # within 1 mm of the edge in front
            (0.0, "EDGE 1 x 2440.0", "NAK 1 BADARG"),
            (180.0, "EDGE 1 x 0.0", "ACK 1"),
            (180.0, "EDGE 1 x 2438.4", "NAK 1 BADARG"),
        ]
        for heading, command, expected in cases:
            with self.subTest(heading=heading, command=command):
                self.assertEqual(self.reply(heading, command), expected)


class TestEdgeMismatch(SimTestCase):
    def test_edge_mismatch_when_the_estimate_is_far_from_the_edge(self):
        # The rover is really at x = 2300 but believes it is at 1000: its contact with the edge
        # would move the estimate by more than 1 m, so something is wrong.
        t = self.sim("--start", "2300", "125", "0")
        self.arm(t, pose=(1000.0, 125.0, 0.0))
        lines, _ = self.run_step(t, "EDGE 1 x 2438.4")
        self.assertEqual(words(lines)[-2:], ["FAULT EDGE_MISMATCH", "STATE FAULT"])

    def test_edge_mismatch_when_the_heading_correction_is_large(self):
        # Really facing 12 degrees off while the pose says 0: squaring up to the edge corrects the
        # heading by 12 degrees, more than the 10 degrees an edge contact may correct.
        t = self.sim("--start", "2300", "125", "12")
        self.arm(t, pose=(2300.0, 125.0, 0.0))
        lines, _ = self.run_step(t, "EDGE 1 x 2438.4")
        self.assertEqual(words(lines)[-2:], ["FAULT EDGE_MISMATCH", "STATE FAULT"])


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


class TestSprayLimits(SimTestCase):
    def test_spray_move_at_the_pump_limit_completes(self):
        t = self.sim("--start", "0", "125", "0", "--wall", "4000", "4000")
        self.arm(t, wall=(4000.0, 4000.0), speed=100.0)
        lines, _ = self.run_step(t, "MOVE 1 2700.0 1")
        self.assertTrue(words(lines)[-2].startswith("DONE 1 "), lines)
        self.assertEqual(words(lines)[-1], "STATE ARMED")
        self.assertFalse(any(line.startswith("FAULT") for line in lines), lines)
        self.assertEqual((t.truth()["pump"], t.truth()["valve"]), (0, 0))


class TestStall(SimTestCase):
    def test_stall_is_detected_at_the_minimum_speed(self):
        t = self.sim("--start", "0", "125", "0")
        self.arm(t, speed=25.0)
        t.send("MOVE 1 1000.0 0")
        t.advance(300)
        t.send("PING")
        t.set_stall(True)
        lines = []
        elapsed = 0
        while elapsed < 2200 and not any(line.startswith("FAULT") for line in lines):
            lines += t.advance(100) + t.send("PING")
            elapsed += 100
        self.assertIn("FAULT STALL", words(lines))


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

    def test_pause_and_resume_during_a_turn_and_during_an_edge_search_complete(self):
        t = self.sim("--start", "1200", "1200", "0")
        self.arm(t, pose=(1200.0, 1200.0, 0.0))
        lines = list(t.send("TURN 1 90.0"))
        lines += run_for(t, 500)
        self.assertEqual(words(t.send("PAUSE")), ["ACK 0", "STATE PAUSED"])
        run_for(t, 300)
        first = t.truth()["heading"]
        run_for(t, 300)
        self.assertAlmostEqual(t.truth()["heading"], first, delta=0.5)  # standing still
        self.assertGreater(first, 5.0)
        self.assertLess(first, 85.0)  # it was paused part-way through the turn
        self.assertEqual(words(t.send("RESUME")), ["ACK 0", "STATE RUNNING"])
        lines, _ = self.finish_step(t, lines)
        self.assertEqual(words(lines)[-2:][1], "STATE ARMED")
        self.assertLess(heading_difference(t.truth()["heading"], 90.0), 2.0)

        t = self.sim("--start", "2300", "125", "0")
        self.arm(t, pose=(2300.0, 125.0, 0.0))
        lines = list(t.send("EDGE 1 x 2438.4"))
        lines += run_for(t, 1000)
        self.assertEqual(words(t.send("PAUSE")), ["ACK 0", "STATE PAUSED"])
        run_for(t, 300)
        first = t.truth()["x"]
        run_for(t, 300)
        self.assertAlmostEqual(t.truth()["x"], first, delta=0.5)
        self.assertLess(first, 2438.0)  # paused before reaching the edge
        self.assertEqual(words(t.send("RESUME")), ["ACK 0", "STATE RUNNING"])
        lines, _ = self.finish_step(t, lines)
        self.assertEqual(words(lines)[-2:], ["DONE 1 2438.4 125.0 0.0", "STATE ARMED"])

    def test_the_watchdog_still_applies_while_paused(self):
        t = self.sim("--start", "0", "125", "0")
        self.arm(t)
        t.send("MOVE 1 1000.0 0")
        t.advance(200)
        t.send("PAUSE")
        self.assertIn("FAULT WATCHDOG", words(t.advance(700)))


if __name__ == "__main__":
    unittest.main()
