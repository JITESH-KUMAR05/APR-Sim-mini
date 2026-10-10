import unittest

from rover_support import SimTestCase, words
from rover_link import frame, tenths


class TestSetupAndRefusals(SimTestCase):
    def test_boot_reports_idle(self):
        self.assertEqual(words(self.sim().log), ["STATE BOOT", "STATE IDLE"])

    def test_hello_and_ping(self):
        t = self.sim()
        self.assertEqual(words(t.send("HELLO")), ["HELLO fw=0.1 proto=1 max_speed_mm_s=250 pump_max_s=30"])
        self.assertEqual(words(t.send("PING")), ["PONG"])

    def test_arm_needs_the_complete_setup(self):
        setup = ["HELLO", "WALL 2438.4 2438.4", "SPEED 250.0", "POSE 0.0 125.0 0.0"]
        for skipped in range(len(setup)):
            with self.subTest(skipped=setup[skipped]):
                t = self.sim()
                for index, command in enumerate(setup):
                    if index != skipped:
                        t.send(command)
                self.assertEqual(words(t.send("ARM")), ["NAK 0 BADSTATE"])

    def test_arm_with_the_complete_setup(self):
        self.assertEqual(words(self.arm(self.sim())), ["ACK 0", "STATE ARMED"])

    def test_commands_that_do_not_fit_the_state_are_naked_with_their_id(self):
        t = self.sim()
        self.assertEqual(words(t.send("MOVE 7 100.0 0")), ["NAK 7 BADSTATE"])
        self.assertEqual(words(t.send("TURN 8 10.0")), ["NAK 8 BADSTATE"])
        self.assertEqual(words(t.send("EDGE 9 x 10.0")), ["NAK 9 BADSTATE"])
        self.assertEqual(words(t.send("PAUSE")), ["NAK 0 BADSTATE"])
        self.assertEqual(words(t.send("RESUME")), ["NAK 0 BADSTATE"])
        self.assertEqual(words(t.send("CLEAR")), ["NAK 0 BADSTATE"])
        self.arm(t)
        self.assertEqual(words(t.send("WALL 100.0 100.0")), ["NAK 0 BADSTATE"])
        self.assertEqual(words(t.send("SPEED 100.0")), ["NAK 0 BADSTATE"])
        self.assertEqual(words(t.send("POSE 0.0 0.0 0.0")), ["NAK 0 BADSTATE"])
        self.assertEqual(words(t.send("ARM")), ["NAK 0 BADSTATE"])

    def test_malformed_lines_are_naked_and_never_executed(self):
        t = self.sim()
        self.arm(t)
        self.assertEqual(words(t.send_raw("MOVE 1 100.0 0*00")), ["NAK 0 BADCRC"])
        self.assertEqual(words(t.send("FLY 3")), ["NAK 0 UNKNOWN"])
        self.assertEqual(words(t.send("WALL 100")), ["NAK 0 BADARG"])
        self.assertEqual(words(t.send("MOVE 3 -5.0 0")), ["NAK 3 BADARG"])
        self.assertEqual(words(t.send("TURN 4 270.0")), ["NAK 4 BADARG"])
        self.assertEqual(words(t.send("EDGE 5 q 1.0")), ["NAK 5 BADARG"])
        self.assertEqual(words(t.advance(100)), [])  # still ARMED, nothing started

    def test_a_line_longer_than_the_simulator_buffer_is_one_bad_command(self):
        t = self.sim()
        self.assertEqual(words(t.send_raw("PING " + "x" * 400)), ["NAK 0 BADCRC"])  # not split into two commands
        self.assertEqual(words(t.send("PING")), ["PONG"])

    def test_speed_above_the_firmware_maximum_is_refused(self):
        t = self.sim()
        t.send("HELLO")
        self.assertEqual(words(t.send("SPEED 250.1")), ["NAK 0 SPEED"])
        self.assertEqual(words(t.send("SPEED 250.0")), ["ACK 0"])

    def test_moves_longer_than_the_wall_or_the_pump_limit_are_too_long(self):
        t = self.sim()
        self.arm(t, wall=(2438.4, 2000.0))
        self.assertEqual(words(t.send("MOVE 1 2538.5 0")), ["NAK 1 TOO_LONG"])
        self.assertEqual(words(t.send("MOVE 1 2538.4 0"))[0], "ACK 1")  # exactly wall + 100 mm is fine

        t = self.sim()
        self.arm(t, wall=(4000.0, 4000.0), speed=100.0)
        self.assertEqual(words(t.send("MOVE 1 3100.0 1")), ["NAK 1 TOO_LONG"])  # 31 s of pump at 100 mm/s
        self.assertEqual(words(t.send("MOVE 2 3100.0 0"))[0], "ACK 2")       # same distance without spray is fine

        # 28 s of travel plus the 2 s allowance for the ramp and the creep is exactly the 30 s pump limit:
        # accepted; a little more is not.
        t = self.sim()
        self.arm(t, wall=(4000.0, 4000.0), speed=100.0)
        self.assertEqual(words(t.send("MOVE 1 2810.0 1")), ["NAK 1 TOO_LONG"])
        self.assertEqual(words(t.send("MOVE 2 2800.0 1"))[0], "ACK 2")

    def test_speed_below_the_minimum_is_refused(self):
        # Below 25 mm/s the stall detector (which needs a 20 mm/s command) would be blind.
        t = self.sim()
        t.send("HELLO")
        self.assertEqual(words(t.send("SPEED 24.9")), ["NAK 0 SPEED"])
        self.assertEqual(words(t.send("SPEED 25.0")), ["ACK 0"])

    def test_pose_outside_the_wall_is_refused(self):
        t = self.sim()
        t.send("WALL 500.0 500.0")
        self.assertEqual(words(t.send("POSE 600.0 125.0 0.0")), ["NAK 0 BADARG"])
        self.assertEqual(words(t.send("POSE 550.1 125.0 0.0")), ["NAK 0 BADARG"])
        self.assertEqual(words(t.send("POSE 550.0 125.0 0.0")), ["ACK 0"])  # exactly the margin outside is fine
        self.assertEqual(words(t.send("POSE -50.0 -50.0 0.0")), ["ACK 0"])
        self.assertEqual(words(t.send("POSE -50.1 0.0 0.0")), ["NAK 0 BADARG"])
        self.assertEqual(words(t.send("POSE 0.0 550.1 0.0")), ["NAK 0 BADARG"])

    def test_a_refused_pose_cannot_be_armed(self):
        t = self.sim()
        t.send("HELLO")
        t.send("WALL 500.0 500.0")
        t.send("SPEED 100.0")
        self.assertEqual(words(t.send("POSE 100.0 100.0 0.0")), ["ACK 0"])
        # A host that ignores this NAK must not be able to ARM on the older pose.
        self.assertEqual(words(t.send("POSE 600.0 125.0 0.0")), ["NAK 0 BADARG"])
        self.assertEqual(words(t.send("ARM")), ["NAK 0 BADSTATE"])
        self.assertEqual(words(t.send("POSE 200.0 100.0 0.0")), ["ACK 0"])
        self.assertEqual(words(t.send("ARM")), ["ACK 0", "STATE ARMED"])

    def test_a_wall_that_leaves_the_pose_outside_is_refused_and_the_old_wall_stays(self):
        t = self.sim()
        t.send("HELLO")
        t.send("WALL 2438.4 2438.4")
        t.send("SPEED 250.0")
        self.assertEqual(words(t.send("POSE 600.0 125.0 0.0")), ["ACK 0"])
        self.assertEqual(words(t.send("WALL 500.0 500.0")), ["NAK 0 BADARG"])
        self.assertEqual(words(t.send("ARM")), ["ACK 0", "STATE ARMED"])
        # the old 2438.4 wall is still the one in force: 700 mm from x = 600 is not out of bounds
        lines, _ = self.run_step(t, "MOVE 1 700.0 0")
        self.assertTrue(any(word.startswith("DONE") for word in words(lines)), lines)
        self.assertNotIn("FAULT OUT_OF_BOUNDS", words(lines))

    def test_stop_returns_to_idle_and_the_setup_must_be_repeated(self):
        t = self.sim()
        self.arm(t)
        self.assertEqual(words(t.send("STOP")), ["ACK 0", "STATE IDLE"])
        self.assertEqual(words(t.send("ARM")), ["NAK 0 BADSTATE"])
        self.assertEqual(words(t.send("STOP")), ["ACK 0"])  # harmless when already idle
        self.assertEqual(words(self.arm(t))[-1], "STATE ARMED")  # HELLO is a one-time handshake


class TestWatchdog(SimTestCase):
    def test_silence_while_armed_is_a_fault(self):
        t = self.sim()
        self.arm(t)
        self.assertEqual(t.advance(490), [])
        self.assertEqual(words(t.advance(40)), ["FAULT WATCHDOG", "STATE FAULT"])

    def test_pings_keep_it_alive(self):
        t = self.sim()
        self.arm(t)
        for _ in range(60):
            self.assertEqual(t.advance(100), [])
            t.send("PING")

    def test_lines_with_a_bad_crc_do_not_count_as_a_heartbeat(self):
        t = self.sim()
        self.arm(t)
        lines = []
        for _ in range(8):
            lines += t.advance(100)
            lines += t.send_raw("PING*00")
        self.assertIn("FAULT WATCHDOG", words(lines))

    def test_there_is_no_watchdog_while_idle(self):
        t = self.sim()
        self.assertEqual(t.advance(5000), [])

    def test_commands_are_refused_in_fault_and_clear_needs_a_new_setup(self):
        t = self.sim()
        self.arm(t)
        t.advance(600)
        self.assertEqual(words(t.send("MOVE 1 100.0 0")), ["NAK 1 BADSTATE"])
        self.assertEqual(words(t.send("STOP")), ["NAK 0 BADSTATE"])
        self.assertEqual(words(t.send("PING")), ["PONG"])
        self.assertEqual(words(t.send("CLEAR")), ["ACK 0", "STATE IDLE"])
        self.assertEqual(words(t.send("ARM")), ["NAK 0 BADSTATE"])
        self.assertEqual(words(self.arm(t))[-1], "STATE ARMED")


class TestOutOfBounds(SimTestCase):
    def test_clear_after_out_of_bounds_does_not_fault_again(self):
        t = self.sim()
        self.arm(t, wall=(500.0, 500.0))
        lines, _ = self.run_step(t, "MOVE 1 600.0 0")
        self.assertIn("FAULT OUT_OF_BOUNDS", words(lines))
        self.assertEqual(words(t.send("CLEAR")), ["ACK 0", "STATE IDLE"])
        self.assertEqual(t.advance(200), [])  # the stale out-of-bounds estimate must not fault again
        self.assertEqual(words(self.arm(t, wall=(500.0, 500.0), pose=(0.0, 125.0, 0.0)))[-1], "STATE ARMED")


class TestEstop(SimTestCase):
    def test_estop_is_a_fault_in_every_state(self):
        for name in ("IDLE", "ARMED", "RUNNING", "PAUSED"):
            with self.subTest(state=name):
                t = self.sim("--start", "0", "125", "0")
                if name != "IDLE":
                    self.arm(t)
                if name in ("RUNNING", "PAUSED"):
                    t.send("MOVE 1 1000.0 1")
                    t.advance(300)
                if name == "PAUSED":
                    t.send("PAUSE")
                t.set_estop(True)
                self.assertEqual(words(t.advance(30)), ["FAULT ESTOP", "STATE FAULT"])
                truth = t.truth()
                self.assertEqual((truth["pump"], truth["valve"], truth["duty_left"], truth["duty_right"]), (0, 0, 0, 0))

    def test_estop_pressed_at_boot(self):
        t = self.sim()
        t.set_estop(True)
        self.assertEqual(words(t.advance(30)), ["FAULT ESTOP", "STATE FAULT"])

    def test_clear_needs_the_estop_released_and_a_new_setup(self):
        t = self.sim()
        self.arm(t)
        t.set_estop(True)
        t.advance(30)
        self.assertEqual(words(t.send("CLEAR")), ["NAK 0 BADSTATE"])
        t.set_estop(False)
        self.assertEqual(words(t.advance(100)), [])  # releasing does not clear anything by itself
        self.assertEqual(words(t.send("CLEAR")), ["ACK 0", "STATE IDLE"])
        self.assertEqual(words(t.send("ARM")), ["NAK 0 BADSTATE"])

    def test_arm_is_refused_while_the_estop_is_pressed(self):
        t = self.sim()
        t.send("HELLO")
        t.send("WALL 2438.4 2438.4")
        t.send("SPEED 250.0")
        t.send("POSE 0.0 125.0 0.0")
        t.set_estop(True)
        self.assertEqual(words(t.send("ARM")), ["NAK 0 BADSTATE"])

    def assert_everything_off_now(self, t):
        truth = t.truth()
        self.assertEqual((truth["pump"], truth["valve"], truth["duty_left"], truth["duty_right"]), (0, 0, 0, 0), truth)

    def test_estop_pressed_between_polls_stops_a_new_step_and_resume(self):
        # The e-stop goes down and a command arrives before the next tick: the command must not start anything.
        for command in ("MOVE 1 500.0 1", "TURN 1 90.0", "EDGE 1 x 2438.4"):
            with self.subTest(command=command):
                t = self.sim("--start", "0", "125", "0")
                self.arm(t)
                t.set_estop(True)
                self.assertEqual(words(t.send(command)), ["FAULT ESTOP", "STATE FAULT", "NAK 1 BADSTATE"])
                self.assert_everything_off_now(t)

        with self.subTest(command="RESUME"):
            t = self.sim("--start", "0", "125", "0")
            self.arm(t)
            t.send("MOVE 1 1000.0 1")
            t.advance(300)
            t.send("PAUSE")
            t.set_estop(True)
            self.assertEqual(words(t.send("RESUME")), ["FAULT ESTOP", "STATE FAULT", "NAK 0 BADSTATE"])
            self.assert_everything_off_now(t)


class TestOutputsAreOffUnlessRunning(SimTestCase):
    def assert_everything_off(self, truth):
        self.assertEqual((truth["pump"], truth["valve"], truth["duty_left"], truth["duty_right"]), (0, 0, 0, 0), truth)

    def test_idle_armed_paused_done_and_fault(self):
        t = self.sim("--start", "0", "125", "0")
        self.assert_everything_off(t.truth())  # IDLE

        self.arm(t)
        t.advance(200)
        self.assert_everything_off(t.truth())  # ARMED

        t.send("MOVE 1 1000.0 1")
        t.advance(300)
        t.send("PING")  # feed the watchdog so it cannot trip anywhere near this check
        running = t.truth()
        self.assertEqual((running["pump"], running["valve"]), (1, 1))  # RUNNING with spray
        self.assertGreater(running["duty_left"], 0)

        t.send("PAUSE")
        self.assert_everything_off(t.truth())  # PAUSED, before the next control tick: the motors stop at once
        t.advance(100)
        self.assert_everything_off(t.truth())

        t.send("RESUME")
        t.advance(100)
        t.send("PING")
        self.assertEqual(t.truth()["pump"], 1)  # RUNNING again

        lines = []
        while not any(line.startswith(("DONE", "FAULT")) for line in lines):
            lines = t.advance(100) + t.send("PING")
        self.assertTrue(any(line.startswith("DONE") for line in lines), lines)
        self.assert_everything_off(t.truth())  # DONE

        t.send("MOVE 2 1000.0 1")
        t.advance(100)
        self.assertEqual(t.truth()["pump"], 1)  # RUNNING once more
        lines = t.advance(700)  # no heartbeat: the watchdog trips while the pump is on
        self.assertIn("FAULT WATCHDOG", words(lines))
        self.assert_everything_off(t.truth())  # FAULT

    def test_stop_while_running_or_paused_switches_everything_off(self):
        for paused in (False, True):
            with self.subTest(paused=paused):
                t = self.sim("--start", "0", "125", "0")
                self.arm(t)
                t.send("MOVE 1 1000.0 1")
                t.advance(300)
                if paused:
                    t.send("PAUSE")
                else:
                    self.assertEqual(t.truth()["pump"], 1)
                self.assertEqual(words(t.send("STOP")), ["ACK 0", "STATE IDLE"])
                self.assert_everything_off(t.truth())
                self.assertEqual(t.advance(200), [])
                self.assert_everything_off(t.truth())


if __name__ == "__main__":
    unittest.main()
