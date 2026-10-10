import unittest

from rover_support import SimTestCase, words


class TestFaults(SimTestCase):
    def assert_faulted_and_off(self, transport, lines, fault):
        self.assertEqual(words(lines)[-2:], [f"FAULT {fault}", "STATE FAULT"])
        truth = transport.truth()
        self.assertEqual((truth["pump"], truth["valve"], truth["duty_left"], truth["duty_right"]), (0, 0, 0, 0), truth)
        self.assertEqual(words(transport.send("MOVE 9 100.0 0")), ["NAK 9 BADSTATE"])  # latched

    def test_stall(self):
        t = self.sim("--start", "0", "125", "0")
        self.arm(t)
        t.send("MOVE 1 1000.0 1")
        t.advance(300)
        t.set_stall(True)
        lines, elapsed = self.run_step(t, "PING")
        self.assert_faulted_and_off(t, lines, "STALL")
        self.assertGreaterEqual(elapsed, 900)   # not before a full second without motion
        self.assertLessEqual(elapsed, 2100)

    def test_stall_during_a_turn(self):
        t = self.sim("--start", "1200", "1200", "0", "--motor-scale", "0.001")
        self.arm(t, pose=(1200.0, 1200.0, 0.0))
        lines, _ = self.run_step(t, "TURN 1 90.0")
        self.assert_faulted_and_off(t, lines, "STALL")

    def test_step_timeout_when_the_motors_cannot_keep_up(self):
        t = self.sim("--start", "0", "125", "0", "--motor-scale", "0.1")
        self.arm(t, speed=100.0)
        lines, elapsed = self.run_step(t, "MOVE 1 500.0 0")
        self.assert_faulted_and_off(t, lines, "STEP_TIMEOUT")
        self.assertTrue(11000 <= elapsed <= 13000, elapsed)  # 2 x 5 s expected + 2 s

    def test_out_of_bounds_when_the_estimate_leaves_the_wall(self):
        t = self.sim("--start", "0", "125", "0")
        self.arm(t, wall=(500.0, 500.0))
        lines, _ = self.run_step(t, "MOVE 1 600.0 0")  # allowed: wall + 100 mm
        self.assert_faulted_and_off(t, lines, "OUT_OF_BOUNDS")
        self.assertLess(t.truth()["x"], 600.0)

    def test_heading_when_the_gyro_runs_away(self):
        t = self.sim("--start", "0", "125", "0", "--gyro-bias", "60")
        self.arm(t)
        lines, _ = self.run_step(t, "MOVE 1 2000.0 0")
        self.assert_faulted_and_off(t, lines, "HEADING")

    def test_edge_not_found_when_nothing_is_touched(self):
        t = self.sim("--start", "1000", "125", "0")
        self.arm(t, pose=(1000.0, 125.0, 0.0))
        lines, _ = self.run_step(t, "EDGE 1 x 2438.4")
        self.assert_faulted_and_off(t, lines, "EDGE_NOT_FOUND")
        self.assertAlmostEqual(t.truth()["x"], 1000.0 + 150.0 + 0.05 * 2438.4, delta=15.0)  # gave up after 150 mm plus 5% of the wall

    def test_pump_timeout(self):
        # Wide wall, weak motors: a spray move that is allowed (24 s at the set speed) but takes
        # about 33 s, longer than the 30 s pump limit and shorter than the step timeout.
        t = self.sim("--start", "0", "125", "0", "--wall", "4000", "4000", "--motor-scale", "0.3")
        self.arm(t, wall=(4000.0, 4000.0), speed=125.0)
        lines, elapsed = self.run_step(t, "MOVE 1 3000.0 1")
        self.assert_faulted_and_off(t, lines, "PUMP_TIMEOUT")
        self.assertTrue(29000 <= elapsed <= 31500, elapsed)

    def test_a_fault_clears_with_clear_and_the_rover_can_be_armed_again(self):
        t = self.sim("--start", "0", "125", "0", "--gyro-bias", "60")
        self.arm(t)
        lines, _ = self.run_step(t, "MOVE 1 2000.0 0")
        self.assertIn("FAULT HEADING", words(lines))
        self.assertEqual(words(t.send("CLEAR")), ["ACK 0", "STATE IDLE"])
        self.assertEqual(words(self.arm(t))[-1], "STATE ARMED")


if __name__ == "__main__":
    unittest.main()
