import subprocess
import unittest

from rover_support import proto_tool_binary
from rover_link import MAX_LINE, crc8, frame, parse_line, tenths


def ask(binary, *commands):
    result = subprocess.run([binary], input="\n".join(list(commands) + ["QUIT"]) + "\n", capture_output=True, text=True, timeout=20)
    return result.stdout.splitlines()


def framed(payload):
    return frame(payload)


class ProtocolTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = proto_tool_binary()

    def parse(self, payload):
        return ask(self.binary, "PARSE " + framed(payload))[0]


class TestCrc(ProtocolTestCase):
    def test_the_standard_check_value(self):
        self.assertEqual(crc8("123456789"), 0xF4)  # CRC-8, polynomial 0x07, initial value 0
        self.assertEqual(ask(self.binary, "CRC 123456789"), ["CRC F4"])

    def test_python_and_firmware_agree(self):
        texts = ["", "HELLO", "MOVE 12 2438.4 1", "FAULT PUMP_TIMEOUT", "POSE -90.0 0.5 180.0", "x" * 60]
        for text in texts:
            with self.subTest(text=text):
                answer = ask(self.binary, "CRC " + text)[0]
                self.assertEqual(answer, f"CRC {crc8(text):02X}")


class TestNumbers(ProtocolTestCase):
    def test_parse_tenths(self):
        cases = {
            "0": 0, "90": 900, "-90": -900, "2438.4": 24384, "0.25": 3, "0.24": 2, "-0.25": -3,
            "205.0": 2050, "12.345": 123, "12.35": 124, "1000.04": 10000, "7.": None, ".5": None,
            "1.2345": None, "abc": None, "": None, "--1": None, "1e3": None, "5 ": None,
        }
        answers = ask(self.binary, *["TENTHS " + text for text in cases])
        for (text, expected), answer in zip(cases.items(), answers):
            with self.subTest(text=text):
                self.assertEqual(answer, "BAD" if expected is None else f"OK {expected}")

    def test_format_tenths_round_trips_with_python(self):
        for value in (0, 5, -5, 24384, -900, 10, 1999999):
            with self.subTest(value=value):
                text = ask(self.binary, f"FMT {value}")[0]
                self.assertEqual(text, tenths(value / 10))
                self.assertEqual(ask(self.binary, "TENTHS " + text), [f"OK {value}"])

    def test_python_tenths_rounds_half_away_from_zero(self):
        self.assertEqual([tenths(v) for v in (0.04, 0.05, -0.05, 2438.44, 90)], ["0.0", "0.1", "-0.1", "2438.4", "90.0"])


class TestFraming(ProtocolTestCase):
    def test_format_matches_python_frame(self):
        for payload in ("ACK 3", "DONE 4 2438.4 125.0 0.0", "STATE ARMED"):
            with self.subTest(payload=payload):
                self.assertEqual(ask(self.binary, "FORMAT " + payload), [frame(payload)])

    def test_a_line_that_would_exceed_the_limit_is_refused(self):
        self.assertEqual(ask(self.binary, "FORMAT " + "A" * 78), ["TOOLONG"])
        self.assertEqual(len(ask(self.binary, "FORMAT " + "A" * 77)[0]), MAX_LINE)  # exactly at the limit is fine
        self.assertEqual(len(frame("A" * 77)), MAX_LINE)
        with self.assertRaises(ValueError):
            frame("A" * 78)

    def test_parse_line_checks_the_crc(self):
        self.assertEqual(parse_line(frame("DONE 4 1.0 2.0 3.0")), ("DONE", ["4", "1.0", "2.0", "3.0"]))
        with self.assertRaises(ValueError):
            parse_line("DONE 4 1.0 2.0 3.0*00")
        with self.assertRaises(ValueError):
            parse_line("DONE 4 1.0 2.0 3.0")


class TestParse(ProtocolTestCase):
    def test_every_command(self):
        self.assertEqual(self.parse("HELLO"), "OK HELLO id=0 a=0 b=0 c=0 axis=-")
        self.assertEqual(self.parse("WALL 2438.4 2438.4"), "OK WALL id=0 a=24384 b=24384 c=0 axis=-")
        self.assertEqual(self.parse("SPEED 250.0"), "OK SPEED id=0 a=2500 b=0 c=0 axis=-")
        self.assertEqual(self.parse("POSE 0.0 125.0 -90.0"), "OK POSE id=0 a=0 b=1250 c=-900 axis=-")
        self.assertEqual(self.parse("ARM"), "OK ARM id=0 a=0 b=0 c=0 axis=-")
        self.assertEqual(self.parse("MOVE 12 2438.4 1"), "OK MOVE id=12 a=24384 b=1 c=0 axis=-")
        self.assertEqual(self.parse("TURN 13 -90.0"), "OK TURN id=13 a=-900 b=0 c=0 axis=-")
        self.assertEqual(self.parse("EDGE 14 x 2438.4"), "OK EDGE id=14 a=24384 b=0 c=0 axis=x")
        for word in ("PAUSE", "RESUME", "STOP", "PING", "CLEAR"):
            with self.subTest(word=word):
                self.assertEqual(self.parse(word), f"OK {word} id=0 a=0 b=0 c=0 axis=-")

    def test_extra_spaces_are_tolerated(self):
        self.assertEqual(self.parse("MOVE   5  100.0   0"), "OK MOVE id=5 a=1000 b=0 c=0 axis=-")

    def test_bad_arguments_report_the_step_id_when_it_can_be_read(self):
        cases = {
            "WALL 100": "BADARG id=0", "WALL 0 100": "BADARG id=0", "WALL -5 100": "BADARG id=0",
            "SPEED 0": "BADARG id=0", "SPEED fast": "BADARG id=0", "POSE 1 2": "BADARG id=0",
            "MOVE 3 -10.0 0": "BADARG id=3", "MOVE 3 10.0 2": "BADARG id=3", "MOVE 0 10.0 0": "BADARG id=0",
            "MOVE x 10.0 0": "BADARG id=0", "MOVE 3 10.0": "BADARG id=3",
            "TURN 4 270.0": "BADARG id=4", "TURN 4 -180.1": "BADARG id=4",
            "EDGE 5 z 10.0": "BADARG id=5", "EDGE 5 x": "BADARG id=5",
            "HELLO now": "BADARG id=0", "PING 1": "BADARG id=0", "ARM 1 2 3 4 5 6 7": "BADARG id=0",
        }
        for payload, expected in cases.items():
            with self.subTest(payload=payload):
                self.assertEqual(self.parse(payload), expected)

    def test_limits_are_inclusive(self):
        self.assertEqual(self.parse("TURN 1 180.0"), "OK TURN id=1 a=1800 b=0 c=0 axis=-")
        self.assertEqual(self.parse("TURN 1 -180.0"), "OK TURN id=1 a=-1800 b=0 c=0 axis=-")

    def test_unknown_words_and_empty_lines(self):
        self.assertEqual(self.parse("FLY 3"), "UNKNOWN")
        self.assertEqual(self.parse("hello"), "UNKNOWN")
        answers = ask(self.binary, "PARSE *00", "PARSE " + frame(" "))
        self.assertEqual(answers[0], "BADCRC")
        self.assertEqual(answers[1], "UNKNOWN")

    def test_bad_crc_and_malformed_framing(self):
        good = frame("MOVE 1 100.0 0")
        lines = [
            good[:-2] + "00",          # wrong CRC
            good[:-3],                  # no CRC at all
            good[:-2] + "ZZ",           # not hex
            good.replace("*", " "),     # no asterisk
            "MOVE 1 " + "9" * 90 + "*00",  # longer than the limit
            "",
        ]
        answers = ask(self.binary, *["PARSE " + line for line in lines])
        self.assertEqual(answers, ["BADCRC"] * len(lines))

    def test_lower_case_crc_digits_are_accepted(self):
        line = frame("PING")
        self.assertEqual(ask(self.binary, "PARSE " + line[:-2] + line[-2:].lower())[0][:7], "OK PING")


if __name__ == "__main__":
    unittest.main()
