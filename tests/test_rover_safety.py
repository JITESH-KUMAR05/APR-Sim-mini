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
        t.advance(500)
        running = t.truth()
        self.assertEqual((running["pump"], running["valve"]), (1, 1))  # RUNNING with spray
        self.assertGreater(running["duty_left"], 0)

        t.send("PAUSE")
        t.advance(100)
        self.assert_everything_off(t.truth())  # PAUSED

        t.send("RESUME")
        t.advance(100)
        self.assertEqual(t.truth()["pump"], 1)  # RUNNING again
        t.advance(700)  # no heartbeat: the watchdog trips while the pump is on
        self.assert_everything_off(t.truth())  # FAULT


if __name__ == "__main__":
    unittest.main()
