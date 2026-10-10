# Rover Firmware and Protocol (P2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build everything of the rover's Arduino firmware and serial protocol that can be built and tested without hardware: the protocol, the safety state machine, the motion control, a PC simulator running the same C++ core, a host tool that streams a compiled mission to it, and the tests.

**Architecture:** A portable C++11 core (`rover/firmware/core/`, no Arduino code, no heap) talks to the world only through the `Hal` interface. A simulated differential-drive rover implements `Hal`, and `rover_sim` runs the core against it on the PC with the protocol on stdin/stdout. Python tests build `rover_sim` with `g++` and drive it. `rover/tools/run_mission.py` is the stand-in for the Raspberry Pi and streams a P1 mission file to it. The board drivers, `main.cpp`, tuning and an AVR build are plan P2b, after the parts exist.

**Tech Stack:** C++11 (MinGW g++ 6.3 is installed), Python 3.12 run with `uv run python`, unittest. No new Python or system dependencies.

Spec: `docs/superpowers/specs/2026-10-09-rover-firmware-protocol-design.md`. P1 (the mission compiler) is merged. All code below was written and run (197 tests passing) before this plan was written: when a step says "create exactly as shown", transcribe it exactly.

## Global Constraints

- Run Python only as `uv run python ...` from the repo root. Never `python3`, never `pip`, never `uv run --no-project`.
- No new dependencies of any kind. In particular no `pyserial`: the only transport in P2 is the simulator (a serial transport comes with the board in P2b).
- The firmware core in `rover/firmware/core/` must build with `g++ -std=c++11 -Wall -Wextra -Werror -fno-exceptions -fno-rtti -O2`, use no heap (`new`, `malloc`), no standard containers and no Arduino headers, and talk to the outside world only through `hal.h`.
- Protocol: one ASCII line `WORDS args*CC`, `CC` = CRC-8 (polynomial 0x07, initial value 0) in two upper-case hex digits, at most 80 characters including `*CC`, USB serial at 115200 baud. Distances in millimetres and angles in degrees, with one decimal on the wire (fixed point in tenths inside the firmware).
- Wall frame: origin bottom-left, x right, y up, headings counter-clockwise from +x, positions are the spray nozzle centre.
- Safety limits (starting values, in `config.h`): watchdog 500 ms, stall 1 s with under 1 mm of wheel motion, bounds margin 50 mm, heading fault 15 degrees, edge search 150 mm, pump limit 30 s, step timeout twice the expected time plus 2 s, control tick 10 ms, pose stream 100 ms.
- Motors, pump and valve are off in every state except RUNNING. Only the Arduino enforces safety.
- AI output never reaches a motor, valve or adhesion command.
- On this machine a freshly built executable is sometimes blocked at random by Windows Application Control ("An Application Control policy has blocked this file"). `tests/rover_support.py` rebuilds and retries, and the tests skip with a clear message if no build ever runs. Never disable or work around the policy any other way.
- Tests run with `uv run python -m unittest discover -s tests -p "<file>" -v`. Baseline before this plan: 125 tests pass, 1 skipped.
- Commit trailer on every commit: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`, exactly this text, in its own paragraph (use two `-m` arguments as in the commit commands). After committing run `git log -1 --format='%(trailers:key=Co-Authored-By)'`: it must print the trailer line.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `rover/firmware/core/protocol.h`, `protocol.cpp` | Create | Line framing, CRC-8, fixed-point numbers, command parsing |
| `rover/firmware/core/config.h` | Create | Every tunable constant |
| `rover/firmware/core/hal.h` | Create | The `Hal` and `Sink` interfaces |
| `rover/firmware/core/safety.h` | Create | State and fault enums with their names |
| `rover/firmware/core/motion.h`, `motion.cpp` | Create | Pose estimate and the MOVE / TURN / EDGE executor |
| `rover/firmware/core/rover.h`, `rover.cpp` | Create | Command handling, the safety state machine, watchdog, replies |
| `rover/sim/plant.h`, `plant.cpp` | Create | The simulated rover implementing `Hal` |
| `rover/sim/sim_main.cpp` | Create | `rover_sim`: the core plus the plant on stdin/stdout |
| `rover/sim/proto_tool.cpp` | Create | `proto_tool`: exposes the protocol module to the Python tests |
| `rover/tools/rover_link.py` | Create | Host side of the protocol and the `SimTransport` |
| `rover/tools/run_mission.py` | Create | Streams a mission file to a rover |
| `rover_profile.json`, `rover_mission.py` | Modify | New `pump_max_continuous_s`, file contract text |
| `docs/rover-protocol.md` | Create | The protocol, single source of truth |
| `tests/rover_support.py` | Create | Builds the C++ test programs with retry, base test class |
| `tests/test_rover_protocol.py`, `test_rover_safety.py`, `test_rover_motion.py`, `test_rover_faults.py`, `test_rover_run_mission.py` | Create | The firmware and host tool tests |
| `tests/test_rover_mission.py` | Modify | Pump limit cases for the profile |
| `.gitignore` | Modify | Ignore `rover/sim/build/` |
| both rover specs | Modify | Correct statements that building P2 showed to be wrong |

---

### Task 1: Build support, protocol layer and host link

**Files:**
- Create: `rover/firmware/core/protocol.h`, `rover/firmware/core/protocol.cpp`, `rover/sim/proto_tool.cpp`, `rover/tools/rover_link.py`, `tests/rover_support.py`
- Modify: `.gitignore`
- Test: `tests/test_rover_protocol.py`

**Interfaces:**
- Produces (C++): `proto::crc8`, `proto::parse(line, Message&, idOut) -> ParseResult`, `proto::parseTenths`, `proto::formatLine`, `proto::formatTenths`, `proto::kMaxLine = 80` and the `Message` / `Cmd` types in `protocol.h`. `proto_tool`: one command per stdin line (`CRC`, `PARSE`, `TENTHS`, `FORMAT`, `FMT`, `QUIT`), one answer line each.
- Produces (Python): `rover_link.crc8`, `tenths`, `frame`, `parse_line`, `MAX_LINE`, and `SimTransport` (used from Task 2 on). `tests/rover_support.py`: `build(name, sources, probe)`, `simulator_binary()`, `proto_tool_binary()`, `words(lines)` and the `SimTestCase` base class (used from Task 2 on).

- [ ] **Step 1: Write the tests and the test support**

Create `tests/rover_support.py`:

```python
"""
Shared helpers for the rover firmware tests: build the C++ test programs with g++ and drive them.

Windows Application Control on some machines blocks a freshly built executable at random
("An Application Control policy has blocked this file"), even for an identical source. Each build
gets a new binary, so build() simply tries again, up to a limit, and the tests skip if no build ever
runs. A compile error is a real failure and is never retried or skipped.
"""

import os
import shutil
import subprocess
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "rover", "tools"))
sys.path.insert(0, ROOT)

from rover_link import SimTransport, tenths  # noqa: E402

CORE_DIR = os.path.join(ROOT, "rover", "firmware", "core")
SIM_DIR = os.path.join(ROOT, "rover", "sim")
BUILD_DIR = os.path.join(SIM_DIR, "build")
FLAGS = ["-std=c++11", "-Wall", "-Wextra", "-Werror", "-fno-exceptions", "-fno-rtti", "-O2"]

_binaries = {}


def _source_files():
    paths = []
    for folder in (CORE_DIR, SIM_DIR):
        for name in sorted(os.listdir(folder)):
            if name.endswith((".h", ".cpp")):
                paths.append(os.path.join(folder, name))
    return paths


def _runs(binary, probe):
    try:
        result = subprocess.run([binary], input=probe, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def build(name, sources, probe, attempts=12):
    """Return the path of a runnable test program built from `sources`, or raise unittest.SkipTest."""
    if name in _binaries:
        return _binaries[name]
    gpp = shutil.which("g++")
    if gpp is None:
        raise unittest.SkipTest("g++ was not found; install MinGW or GCC to run the rover firmware tests")

    os.makedirs(BUILD_DIR, exist_ok=True)
    binary = os.path.join(BUILD_DIR, name + (".exe" if os.name == "nt" else ""))
    newest = max(os.path.getmtime(path) for path in _source_files())
    if os.path.exists(binary) and os.path.getmtime(binary) > newest and _runs(binary, probe):
        _binaries[name] = binary
        return binary

    for _ in range(attempts):
        if os.path.exists(binary):
            os.remove(binary)
        result = subprocess.run([gpp] + FLAGS + sources + ["-o", binary], capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(f"the rover firmware does not compile:\n{result.stderr}")
        if _runs(binary, probe):
            _binaries[name] = binary
            return binary
    raise unittest.SkipTest(f"{name} was blocked by the operating system after {attempts} builds")


def simulator_binary():
    core = [os.path.join(CORE_DIR, f) for f in ("protocol.cpp", "motion.cpp", "rover.cpp")]
    sim = [os.path.join(SIM_DIR, f) for f in ("plant.cpp", "sim_main.cpp")]
    return build("rover_sim", core + sim, "@QUIT\n")


def proto_tool_binary():
    return build("proto_tool", [os.path.join(CORE_DIR, "protocol.cpp"), os.path.join(SIM_DIR, "proto_tool.cpp")], "QUIT\n")


def words(lines):
    """Protocol lines without their CRC, for readable assertions: 'FAULT STALL*AB' -> 'FAULT STALL'."""
    return [line.rsplit("*", 1)[0] for line in lines]


class SimTestCase(unittest.TestCase):
    """Base class: builds rover_sim once and gives each test its own simulated rover."""

    @classmethod
    def setUpClass(cls):
        cls.binary = simulator_binary()

    def sim(self, *options):
        transport = SimTransport(self.binary, [str(option) for option in options])
        self.addCleanup(transport.close)
        return transport

    def arm(self, transport, wall=(2438.4, 2438.4), speed=250.0, pose=(0.0, 125.0, 0.0)):
        """HELLO, WALL, SPEED, POSE, ARM. Returns the replies to ARM."""
        transport.send("HELLO")
        transport.send(f"WALL {tenths(wall[0])} {tenths(wall[1])}")
        transport.send(f"SPEED {tenths(speed)}")
        transport.send(f"POSE {tenths(pose[0])} {tenths(pose[1])} {tenths(pose[2])}")
        return transport.send("ARM")

    def run_step(self, transport, payload, max_ms=90000):
        """
        Send one step and let simulated time pass, feeding the watchdog, until DONE or FAULT.
        Returns (lines, elapsed_ms) without the POSE stream and PONG replies.
        """
        lines = list(transport.send(payload))
        elapsed = 0
        while elapsed < max_ms:
            batch = transport.advance(100) + transport.send("PING")
            elapsed += 100
            lines += batch
            if any(line.startswith(("DONE", "FAULT")) for line in batch):
                break
        return [line for line in lines if not line.startswith(("POSE", "PONG"))], elapsed
```

Create `tests/test_rover_protocol.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m unittest discover -s tests -p "test_rover_protocol.py" -v`
Expected: ERROR `ModuleNotFoundError: No module named 'rover_link'`.

- [ ] **Step 3: Implement**

Create `rover/tools/rover_link.py`:

```python
"""
Host side of the rover protocol (docs/rover-protocol.md): line framing, CRC-8, fixed-point
numbers, and a transport that drives the PC simulator `rover_sim`.

Quantities on the wire are millimetres and degrees in tenths, written as decimal text.
"""

import subprocess
from typing import Dict, List, Optional, Tuple

MAX_LINE = 80


def crc8(text: str) -> int:
    crc = 0
    for byte in text.encode("ascii"):
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def tenths(value: float) -> str:
    """Format a number as the protocol's one-decimal text, rounding half away from zero."""
    scaled = int(abs(value) * 10 + 0.5)
    sign = "-" if value < 0 and scaled else ""
    return f"{sign}{scaled // 10}.{scaled % 10}"


def frame(payload: str) -> str:
    line = f"{payload}*{crc8(payload):02X}"
    if len(line) > MAX_LINE:
        raise ValueError(f"line longer than {MAX_LINE} characters: {line}")
    return line


def parse_line(line: str) -> Tuple[str, List[str]]:
    """Check the CRC of a received line and split it into (word, arguments)."""
    line = line.strip()
    if len(line) < 4 or line[-3] != "*":
        raise ValueError(f"missing CRC: {line!r}")
    payload, given = line[:-3], line[-2:]
    if f"{crc8(payload):02X}" != given.upper():
        raise ValueError(f"bad CRC: {line!r}")
    words = payload.split()
    if not words:
        raise ValueError(f"empty line: {line!r}")
    return words[0], words[1:]


class SimTransport:
    """Runs rover_sim as a subprocess. Time only advances when asked to (advance())."""

    def __init__(self, binary: str, options: Optional[List[str]] = None):
        self._process = subprocess.Popen(
            [binary] + (options or []),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.log: List[str] = []
        self.log.extend(self._read_until_end())  # what the rover wrote while booting

    def _read_until_end(self) -> List[str]:
        lines = []
        while True:
            line = self._process.stdout.readline()
            if line == "":
                raise RuntimeError("rover_sim exited")
            line = line.rstrip("\r\n")
            if line == "@END":
                return lines
            lines.append(line)

    def _send_raw(self, text: str) -> List[str]:
        self._process.stdin.write(text + "\n")
        self._process.stdin.flush()
        lines = self._read_until_end()
        self.log.extend(lines)
        return lines

    def send(self, payload: str) -> List[str]:
        """Send one protocol line (CRC added) and return what the rover wrote in reply."""
        return self._send_raw(frame(payload))

    def send_raw(self, line: str) -> List[str]:
        """Send a line exactly as given, for tests of bad input."""
        return self._send_raw(line)

    def advance(self, ms: int) -> List[str]:
        return self._send_raw(f"@ADVANCE {ms}")

    def wait(self, ms: int) -> List[str]:
        """Let time pass and return what the rover wrote meanwhile. (A serial transport sleeps and reads.)"""
        return self.advance(ms)

    def truth(self) -> Dict[str, float]:
        lines = self._send_raw("@TRUTH")
        fields = lines[-1].split()
        names = ["x", "y", "heading", "pump", "valve", "duty_left", "duty_right", "painted", "t_ms"]
        return {name: float(value) for name, value in zip(names, fields[1:])}

    def set_estop(self, pressed: bool) -> List[str]:
        return self._send_raw(f"@ESTOP {1 if pressed else 0}")

    def set_stall(self, blocked: bool) -> List[str]:
        return self._send_raw(f"@STALL {1 if blocked else 0}")

    def close(self) -> None:
        try:
            self._process.stdin.write("@QUIT\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        self._process.wait(timeout=5)
        self._process.stdout.close()
        self._process.stdin.close()
```

Create `rover/firmware/core/protocol.h`:

```cpp
#pragma once
#include <stddef.h>
#include <stdint.h>

// Line protocol, see docs/rover-protocol.md. All quantities are fixed point: millimetres and
// degrees in tenths, so the core needs no float printing (which AVR's snprintf lacks).
namespace proto {

const size_t kMaxLine = 80;

enum Cmd {
  CMD_HELLO, CMD_WALL, CMD_SPEED, CMD_POSE, CMD_ARM, CMD_MOVE, CMD_TURN, CMD_EDGE,
  CMD_PAUSE, CMD_RESUME, CMD_STOP, CMD_PING, CMD_CLEAR
};

enum ParseResult { PARSE_OK, PARSE_BADCRC, PARSE_BADARG, PARSE_UNKNOWN };

struct Message {
  Cmd cmd;
  int32_t id;     // step id, 0 for commands without one
  int32_t a;      // first number: mm, deg, width, speed or x, in tenths (spray flag for MOVE is in b)
  int32_t b;
  int32_t c;
  char axis;      // 'x' or 'y' for EDGE
};

uint8_t crc8(const char* data, size_t length);  // polynomial 0x07, initial value 0

// Parse one received line (without newline). On PARSE_OK `out` is filled. `idOut` is the step
// id when it could be read, else 0, for the NAK reply.
ParseResult parse(const char* line, Message& out, int32_t& idOut);

// "-90", "2438.4", "0.25": optional sign, digits, optional fraction of up to 3 digits, rounded
// to tenths. Returns false for anything else.
bool parseTenths(const char* text, int32_t& out);

// Write "payload*CC" into buf. Returns false if it does not fit.
bool formatLine(char* buf, size_t capacity, const char* payload);

// Write a tenths value as decimal text ("-90.0", "2438.4"). Returns the length.
size_t formatTenths(char* buf, size_t capacity, int32_t tenths);

}  // namespace proto
```

Create `rover/firmware/core/protocol.cpp`:

```cpp
#include "protocol.h"

#include <stdio.h>
#include <string.h>

namespace proto {

uint8_t crc8(const char* data, size_t length) {
  uint8_t crc = 0;
  for (size_t i = 0; i < length; i++) {
    crc ^= (uint8_t)data[i];
    for (int bit = 0; bit < 8; bit++) {
      crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
    }
  }
  return crc;
}

bool parseTenths(const char* text, int32_t& out) {
  if (text == NULL || *text == '\0') return false;
  bool negative = false;
  if (*text == '-') {
    negative = true;
    text++;
  }
  if (*text < '0' || *text > '9') return false;
  int32_t whole = 0;
  while (*text >= '0' && *text <= '9') {
    whole = whole * 10 + (*text - '0');
    if (whole > 1000000) return false;
    text++;
  }
  int32_t tenths = 0;
  if (*text == '.') {
    text++;
    int digits = 0;
    int32_t fraction = 0;  // up to 3 digits, scaled to thousandths
    while (*text >= '0' && *text <= '9') {
      if (digits == 3) return false;
      fraction = fraction * 10 + (*text - '0');
      digits++;
      text++;
    }
    if (digits == 0) return false;
    while (digits < 3) {
      fraction *= 10;
      digits++;
    }
    tenths = (fraction + 50) / 100;  // round half up on the magnitude
  }
  if (*text != '\0') return false;
  int32_t value = whole * 10 + tenths;
  out = negative ? -value : value;
  return true;
}

static bool parseUnsigned(const char* text, int32_t& out) {
  if (text == NULL || *text == '\0') return false;
  int32_t value = 0;
  while (*text) {
    if (*text < '0' || *text > '9') return false;
    value = value * 10 + (*text - '0');
    if (value > 1000000) return false;
    text++;
  }
  out = value;
  return true;
}

static int hexValue(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

static const int kMaxTokens = 5;

ParseResult parse(const char* line, Message& out, int32_t& idOut) {
  idOut = 0;
  size_t length = strlen(line);
  if (length < 4 || length > kMaxLine) return PARSE_BADCRC;
  if (line[length - 3] != '*') return PARSE_BADCRC;
  int high = hexValue(line[length - 2]);
  int low = hexValue(line[length - 1]);
  if (high < 0 || low < 0) return PARSE_BADCRC;
  if (crc8(line, length - 3) != (uint8_t)(high * 16 + low)) return PARSE_BADCRC;

  char payload[kMaxLine + 1];
  size_t payloadLength = length - 3;
  memcpy(payload, line, payloadLength);
  payload[payloadLength] = '\0';

  char* tokens[kMaxTokens + 1];
  int count = 0;
  char* cursor = payload;
  while (*cursor) {
    while (*cursor == ' ') cursor++;
    if (*cursor == '\0') break;
    if (count == kMaxTokens + 1) return PARSE_BADARG;
    tokens[count++] = cursor;
    while (*cursor && *cursor != ' ') cursor++;
    if (*cursor) *cursor++ = '\0';
  }
  if (count == 0) return PARSE_UNKNOWN;

  out.id = 0;
  out.a = out.b = out.c = 0;
  out.axis = 0;
  const char* word = tokens[0];

  if (strcmp(word, "MOVE") == 0 || strcmp(word, "TURN") == 0 || strcmp(word, "EDGE") == 0) {
    int32_t id;
    if (count >= 2 && parseUnsigned(tokens[1], id)) idOut = id;
  }

  struct Shape { const char* word; Cmd cmd; int args; };
  static const Shape shapes[] = {
    {"HELLO", CMD_HELLO, 0}, {"WALL", CMD_WALL, 2}, {"SPEED", CMD_SPEED, 1}, {"POSE", CMD_POSE, 3},
    {"ARM", CMD_ARM, 0}, {"MOVE", CMD_MOVE, 3}, {"TURN", CMD_TURN, 2}, {"EDGE", CMD_EDGE, 3},
    {"PAUSE", CMD_PAUSE, 0}, {"RESUME", CMD_RESUME, 0}, {"STOP", CMD_STOP, 0}, {"PING", CMD_PING, 0},
    {"CLEAR", CMD_CLEAR, 0},
  };
  const Shape* shape = NULL;
  for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
    if (strcmp(word, shapes[i].word) == 0) shape = &shapes[i];
  }
  if (shape == NULL) return PARSE_UNKNOWN;
  if (count - 1 != shape->args) return PARSE_BADARG;
  out.cmd = shape->cmd;

  switch (shape->cmd) {
    case CMD_WALL:
      if (!parseTenths(tokens[1], out.a) || !parseTenths(tokens[2], out.b)) return PARSE_BADARG;
      if (out.a <= 0 || out.b <= 0) return PARSE_BADARG;
      break;
    case CMD_SPEED:
      if (!parseTenths(tokens[1], out.a) || out.a <= 0) return PARSE_BADARG;
      break;
    case CMD_POSE:
      if (!parseTenths(tokens[1], out.a) || !parseTenths(tokens[2], out.b) || !parseTenths(tokens[3], out.c)) {
        return PARSE_BADARG;
      }
      break;
    case CMD_MOVE:
      if (!parseUnsigned(tokens[1], out.id) || out.id < 1) return PARSE_BADARG;
      if (!parseTenths(tokens[2], out.a) || out.a <= 0) return PARSE_BADARG;
      if (strcmp(tokens[3], "0") != 0 && strcmp(tokens[3], "1") != 0) return PARSE_BADARG;
      out.b = tokens[3][0] - '0';
      break;
    case CMD_TURN:
      if (!parseUnsigned(tokens[1], out.id) || out.id < 1) return PARSE_BADARG;
      if (!parseTenths(tokens[2], out.a)) return PARSE_BADARG;
      if (out.a < -1800 || out.a > 1800) return PARSE_BADARG;
      break;
    case CMD_EDGE:
      if (!parseUnsigned(tokens[1], out.id) || out.id < 1) return PARSE_BADARG;
      if (strcmp(tokens[2], "x") != 0 && strcmp(tokens[2], "y") != 0) return PARSE_BADARG;
      out.axis = tokens[2][0];
      if (!parseTenths(tokens[3], out.a)) return PARSE_BADARG;
      break;
    default:
      break;
  }
  return PARSE_OK;
}

bool formatLine(char* buf, size_t capacity, const char* payload) {
  size_t length = strlen(payload);
  if (length + 3 > kMaxLine || length + 4 > capacity) return false;
  memcpy(buf, payload, length);
  uint8_t crc = crc8(payload, length);
  snprintf(buf + length, capacity - length, "*%02X", (unsigned)crc);
  return true;
}

size_t formatTenths(char* buf, size_t capacity, int32_t tenths) {
  bool negative = tenths < 0;
  int32_t magnitude = negative ? -tenths : tenths;
  int written = snprintf(buf, capacity, "%s%ld.%ld", negative ? "-" : "", (long)(magnitude / 10), (long)(magnitude % 10));
  return written < 0 ? 0 : (size_t)written;
}

}  // namespace proto
```

Create `rover/sim/proto_tool.cpp`:

```cpp
// proto_tool: lets the Python tests exercise the firmware's protocol module directly.
// One command per stdin line, one answer line each:
//   CRC <text>        -> CRC <two hex digits>
//   PARSE <line>      -> OK <word> id=.. a=.. b=.. c=.. axis=..   |  BADCRC  |  BADARG id=..  |  UNKNOWN
//   TENTHS <text>     -> OK <integer tenths>  |  BAD
//   FORMAT <payload>  -> the payload with "*CC" appended (or TOOLONG)
//   FMT <integer>     -> a tenths value as text
//   QUIT
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../firmware/core/protocol.h"

static const char* wordOf(proto::Cmd cmd) {
  switch (cmd) {
    case proto::CMD_HELLO: return "HELLO";
    case proto::CMD_WALL: return "WALL";
    case proto::CMD_SPEED: return "SPEED";
    case proto::CMD_POSE: return "POSE";
    case proto::CMD_ARM: return "ARM";
    case proto::CMD_MOVE: return "MOVE";
    case proto::CMD_TURN: return "TURN";
    case proto::CMD_EDGE: return "EDGE";
    case proto::CMD_PAUSE: return "PAUSE";
    case proto::CMD_RESUME: return "RESUME";
    case proto::CMD_STOP: return "STOP";
    case proto::CMD_PING: return "PING";
    case proto::CMD_CLEAR: return "CLEAR";
  }
  return "?";
}

int main() {
  char line[256];
  while (fgets(line, sizeof(line), stdin)) {
    size_t length = strlen(line);
    while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r')) line[--length] = '\0';

    if (strcmp(line, "QUIT") == 0) return 0;
    if (strncmp(line, "CRC ", 4) == 0) {
      printf("CRC %02X\n", (unsigned)proto::crc8(line + 4, strlen(line + 4)));
    } else if (strncmp(line, "PARSE ", 6) == 0) {
      proto::Message message;
      int32_t id = 0;
      switch (proto::parse(line + 6, message, id)) {
        case proto::PARSE_OK:
          printf("OK %s id=%ld a=%ld b=%ld c=%ld axis=%c\n", wordOf(message.cmd), (long)message.id, (long)message.a,
                 (long)message.b, (long)message.c, message.axis ? message.axis : '-');
          break;
        case proto::PARSE_BADCRC: printf("BADCRC\n"); break;
        case proto::PARSE_BADARG: printf("BADARG id=%ld\n", (long)id); break;
        case proto::PARSE_UNKNOWN: printf("UNKNOWN\n"); break;
      }
    } else if (strncmp(line, "TENTHS ", 7) == 0) {
      int32_t value;
      if (proto::parseTenths(line + 7, value)) printf("OK %ld\n", (long)value); else printf("BAD\n");
    } else if (strncmp(line, "FORMAT ", 7) == 0) {
      char out[proto::kMaxLine + 8];
      if (proto::formatLine(out, sizeof(out), line + 7)) printf("%s\n", out); else printf("TOOLONG\n");
    } else if (strncmp(line, "FMT ", 4) == 0) {
      char out[24];
      proto::formatTenths(out, sizeof(out), (int32_t)atol(line + 4));
      printf("%s\n", out);
    } else {
      printf("?\n");
    }
    fflush(stdout);
  }
  return 0;
}
```

Append this line to `.gitignore`:

```
rover/sim/build/
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m unittest discover -s tests -p "test_rover_protocol.py" -v`
Expected: 15 tests PASS. The first run compiles `proto_tool` with `g++` and may take a few seconds. If a test class is skipped with `blocked by the operating system`, run it again once; if it is still skipped, stop and report BLOCKED with the full message. A compile error means a typo in the code above: fix it, do not skip.

Run: `uv run python -m unittest discover -s tests -p "test_*.py"`
Expected: all PASS (140 tests), 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add .gitignore rover/firmware/core/protocol.h rover/firmware/core/protocol.cpp rover/sim/proto_tool.cpp rover/tools/rover_link.py tests/rover_support.py tests/test_rover_protocol.py
git commit -m "Add the rover protocol layer, host link and firmware test support" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git log -1 --format='%(trailers:key=Co-Authored-By)'
```

---

### Task 2: The firmware core and the simulator

**Files:**
- Create: `rover/firmware/core/config.h`, `hal.h`, `safety.h`, `motion.h`, `motion.cpp`, `rover.h`, `rover.cpp`, `rover/sim/plant.h`, `plant.cpp`, `sim_main.cpp`
- Test: `tests/test_rover_safety.py`, `tests/test_rover_motion.py`

**Interfaces:**
- Consumes: everything from Task 1.
- Produces: `class Rover { Rover(Hal&, Sink&); begin(); handleLine(const char*); tick(); }` (the caller calls `tick()` every `cfg::kTickMs`); `Hal` and `Sink` as in `hal.h`; `Plant : Hal` with `PlantOptions`; the `rover_sim` program and its simulator commands (`@ADVANCE`, `@TRUTH`, `@ESTOP`, `@STALL`, `@QUIT`, each input line answered by `@END`), options `--start x y heading`, `--wall w h`, `--slip f`, `--seed n`, `--gyro-bias deg_per_s`, `--valve-delay ms`, `--motor-scale f`. The protocol itself is described in `docs/rover-protocol.md` (written in Task 5).

How the pieces behave, so the tests make sense: the rover keeps a nominal heading (the `POSE` heading plus every commanded `TURN`), `MOVE` and `EDGE` hold it with a P+I heading loop on the gyro, and an `EDGE` contact squares the rover to the edge and resets the heading to the nominal one. The plant models wheel slip, a first-order motor lag, encoder quantisation, gyro bias drift and a wall rectangle the rover cannot leave.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rover_safety.py`:

```python
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
```

Create `tests/test_rover_motion.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m unittest discover -s tests -p "test_rover_safety.py" -v`
Expected: ERROR `AssertionError: the rover firmware does not compile:` followed by g++ saying `rover/firmware/core/motion.cpp: No such file or directory`.

- [ ] **Step 3: Implement**

Create these files exactly as shown.

Create `rover/firmware/core/config.h`:

```cpp
#pragma once
#include <stdint.h>

// Tunable constants. The geometry values are initial numbers for the simulator;
// plan P2b replaces them with the real rover's numbers and tunes the gains on the bench.
namespace cfg {

const char kFirmwareVersion[] = "0.1";
const int kProtocolVersion = 1;

// Geometry
const float kWheelDiameterMm = 80.0f;
const float kTicksPerWheelRev = 1000.0f;
const float kWheelbaseMm = 180.0f;
const float kMmPerTick = 3.14159265f * kWheelDiameterMm / kTicksPerWheelRev;
const float kMaxWheelSpeedMmS = 300.0f;  // wheel surface speed at duty 1.0

// Timing
const uint32_t kTickMs = 10;               // control loop period
const uint32_t kPoseStreamMs = 100;        // POSE line period while a step runs
const uint32_t kWatchdogMs = 500;          // no valid line for this long: FAULT WATCHDOG

// Speed limits and profile
const int32_t kMaxSpeedMmS = 250;          // largest accepted SPEED
const float kAccelMmS2 = 150.0f;
const float kMinMoveSpeedMmS = 25.0f;      // creep speed at the end of a MOVE
const float kStopToleranceMm = 1.0f;
const float kEdgeSpeedMmS = 30.0f;
const float kEdgeSearchMm = 150.0f;

// Turning
const float kTurnRateDegS = 60.0f;
const float kTurnAccelDegS2 = 240.0f;
const float kTurnMinRateDegS = 8.0f;
const float kTurnToleranceDeg = 0.5f;

// Control gains
const float kWheelKp = 0.0020f;            // duty per mm/s of speed error
const float kWheelKi = 0.0100f;            // duty per mm/s*s of speed error
const float kSpeedFilter = 0.3f;           // EMA weight of the newest wheel speed sample
const float kHeadingKp = 3.0f;             // mm/s of wheel speed difference per degree of heading error
const float kHeadingKi = 1.5f;             // mm/s per degree-second of accumulated heading error
const float kHeadingMaxIntegralMmS = 20.0f;
const float kHeadingMaxTrimMmS = 40.0f;

// Safety limits
const float kBoundsMarginMm = 50.0f;       // OUT_OF_BOUNDS beyond the wall rectangle by more than this
const float kHeadingFaultDeg = 15.0f;
const uint32_t kStallMs = 1000;
const float kStallMinMotionMm = 1.0f;      // wheel motion needed within kStallMs
const float kStallMinCommandMmS = 20.0f;
const uint32_t kPumpMaxMs = 30000;
const int kPumpMaxSeconds = 30;
const float kMoveMarginMm = 100.0f;        // MOVE longer than max(wall) + this is TOO_LONG

// Spray timing, calibrated in Stage 2 of the integration design
const uint32_t kValveOpenDelayMs = 0;      // wait after opening the valve before moving
const float kValveCloseLeadMm = 0.0f;      // close the valve this far before the end of a spray MOVE

}  // namespace cfg
```

Create `rover/firmware/core/hal.h`:

```cpp
#pragma once
#include <stdint.h>

// Everything the firmware core needs from the outside world. The simulator implements
// Hal (rover/sim/plant.h); plan P2b implements it for the real board. The core never
// calls Arduino functions.
class Hal {
 public:
  virtual uint32_t nowMs() = 0;
  virtual void setMotorDuty(float left, float right) = 0;  // -1..1
  virtual int32_t encoderTicksLeft() = 0;                  // cumulative, signed
  virtual int32_t encoderTicksRight() = 0;
  virtual float gyroHeadingDeg() = 0;                      // integrated yaw, counter-clockwise positive, continuous
  virtual void setPump(bool on) = 0;
  virtual void setValve(bool open) = 0;
  virtual bool edgeContact() = 0;
  virtual bool estopPressed() = 0;
};

// Where the core writes protocol lines (no trailing newline).
class Sink {
 public:
  virtual void writeLine(const char* line) = 0;
};
```

Create `rover/firmware/core/safety.h`:

```cpp
#pragma once

// Rover states and fault codes, see docs/rover-protocol.md.
enum State { STATE_BOOT, STATE_IDLE, STATE_ARMED, STATE_RUNNING, STATE_PAUSED, STATE_FAULT };

enum Fault {
  FAULT_NONE,
  FAULT_WATCHDOG,
  FAULT_ESTOP,
  FAULT_STALL,
  FAULT_STEP_TIMEOUT,
  FAULT_OUT_OF_BOUNDS,
  FAULT_HEADING,
  FAULT_EDGE_NOT_FOUND,
  FAULT_PUMP_TIMEOUT
};

inline const char* stateName(State state) {
  switch (state) {
    case STATE_BOOT: return "BOOT";
    case STATE_IDLE: return "IDLE";
    case STATE_ARMED: return "ARMED";
    case STATE_RUNNING: return "RUNNING";
    case STATE_PAUSED: return "PAUSED";
    case STATE_FAULT: return "FAULT";
  }
  return "?";
}

inline const char* faultName(Fault fault) {
  switch (fault) {
    case FAULT_NONE: return "NONE";
    case FAULT_WATCHDOG: return "WATCHDOG";
    case FAULT_ESTOP: return "ESTOP";
    case FAULT_STALL: return "STALL";
    case FAULT_STEP_TIMEOUT: return "STEP_TIMEOUT";
    case FAULT_OUT_OF_BOUNDS: return "OUT_OF_BOUNDS";
    case FAULT_HEADING: return "HEADING";
    case FAULT_EDGE_NOT_FOUND: return "EDGE_NOT_FOUND";
    case FAULT_PUMP_TIMEOUT: return "PUMP_TIMEOUT";
  }
  return "?";
}
```

Create `rover/firmware/core/motion.h`:

```cpp
#pragma once
#include <stdint.h>

#include "hal.h"
#include "safety.h"

// Executes one step at a time (MOVE, TURN or EDGE) with closed loops on the encoders and the
// gyro, and keeps the pose estimate. All distances are millimetres, angles degrees.
class Motion {
 public:
  enum Event { EV_NONE, EV_DONE, EV_FAULT };

  explicit Motion(Hal& hal);

  void setWall(float widthMm, float heightMm);
  void setSpeed(float mmPerSecond);
  void setPose(float xMm, float yMm, float headingDeg);

  void startMove(float mm, bool spray);
  void startTurn(float degrees);
  void startEdge(char axis, float valueMm);

  void pause();
  void resume();
  void abort();       // drop the step and switch everything off
  void outputsOff();  // motors off, pump off, valve closed

  Event tick();       // call every cfg::kTickMs
  Fault fault() const { return fault_; }
  void clearFault() { fault_ = FAULT_NONE; }

  bool busy() const { return kind_ != KIND_NONE; }
  float x() const { return x_; }
  float y() const { return y_; }
  float headingDeg() const;  // normalised to (-180, 180]

 private:
  enum Kind { KIND_NONE, KIND_MOVE, KIND_TURN, KIND_EDGE };

  void updatePose(float& leftMm, float& rightMm);
  float wheelDuty(float target, float measured, float& integral);
  void drive(float leftMmS, float rightMmS);
  Event raise(Fault fault);
  Event finish();
  bool checkStall(float commandMmS, float wheelMotionMm);
  void closeSpray();
  float headingTrim(float errorDeg);

  Hal& hal_;
  Kind kind_;
  Fault fault_;
  bool paused_;

  // pose
  float x_, y_, heading_, gyroOffset_;
  int32_t lastLeftTicks_, lastRightTicks_;
  bool poseKnown_;
  float wallW_, wallH_;
  bool wallKnown_;
  float speed_;

  // wheel loops
  float leftMeasured_, rightMeasured_;
  float leftIntegral_, rightIntegral_;

  // current step
  float targetMm_, traveled_, vCmd_, holdHeading_;
  float targetHeading_, rateCmd_;
  float nominalHeading_, headingIntegral_;  // nominal = pose heading plus every commanded turn
  char edgeAxis_;
  float edgeValue_;
  bool spray_, sprayOpen_;
  int settledTicks_;
  uint32_t waitUntilMs_;
  uint32_t runMs_, timeoutMs_;
  uint32_t pumpOnSinceMs_;
  bool pumpOn_;
  uint32_t stallWindowMs_;
  float stallMotionMm_;
};
```

Create `rover/firmware/core/motion.cpp`:

```cpp
#include "motion.h"

#include <math.h>

#include "config.h"

namespace {
const float kDt = cfg::kTickMs / 1000.0f;
const float kDegToRad = 3.14159265f / 180.0f;

float clampf(float value, float low, float high) {
  return value < low ? low : (value > high ? high : value);
}
}  // namespace

Motion::Motion(Hal& hal)
    : hal_(hal), kind_(KIND_NONE), fault_(FAULT_NONE), paused_(false),
      x_(0), y_(0), heading_(0), gyroOffset_(0), lastLeftTicks_(0), lastRightTicks_(0), poseKnown_(false),
      wallW_(0), wallH_(0), wallKnown_(false), speed_(100.0f),
      leftMeasured_(0), rightMeasured_(0), leftIntegral_(0), rightIntegral_(0),
      targetMm_(0), traveled_(0), vCmd_(0), holdHeading_(0), targetHeading_(0), rateCmd_(0),
      nominalHeading_(0), headingIntegral_(0),
      edgeAxis_('x'), edgeValue_(0), spray_(false), sprayOpen_(false), settledTicks_(0), waitUntilMs_(0),
      runMs_(0), timeoutMs_(0), pumpOnSinceMs_(0), pumpOn_(false), stallWindowMs_(0), stallMotionMm_(0) {}

void Motion::setWall(float widthMm, float heightMm) {
  wallW_ = widthMm;
  wallH_ = heightMm;
  wallKnown_ = true;
}

void Motion::setSpeed(float mmPerSecond) { speed_ = mmPerSecond; }

void Motion::setPose(float xMm, float yMm, float headingDeg) {
  x_ = xMm;
  y_ = yMm;
  heading_ = headingDeg;
  nominalHeading_ = headingDeg;
  gyroOffset_ = hal_.gyroHeadingDeg() - headingDeg;
  lastLeftTicks_ = hal_.encoderTicksLeft();
  lastRightTicks_ = hal_.encoderTicksRight();
  leftMeasured_ = rightMeasured_ = 0;
  poseKnown_ = true;
}

float Motion::headingDeg() const {
  float h = fmodf(heading_, 360.0f);
  if (h > 180.0f) h -= 360.0f;
  if (h <= -180.0f) h += 360.0f;
  return h;
}

void Motion::updatePose(float& leftMm, float& rightMm) {
  int32_t leftTicks = hal_.encoderTicksLeft();
  int32_t rightTicks = hal_.encoderTicksRight();
  leftMm = (leftTicks - lastLeftTicks_) * cfg::kMmPerTick;
  rightMm = (rightTicks - lastRightTicks_) * cfg::kMmPerTick;
  lastLeftTicks_ = leftTicks;
  lastRightTicks_ = rightTicks;

  leftMeasured_ += cfg::kSpeedFilter * (leftMm / kDt - leftMeasured_);
  rightMeasured_ += cfg::kSpeedFilter * (rightMm / kDt - rightMeasured_);

  heading_ = hal_.gyroHeadingDeg() - gyroOffset_;
  float mean = (leftMm + rightMm) * 0.5f;
  x_ += mean * cosf(heading_ * kDegToRad);
  y_ += mean * sinf(heading_ * kDegToRad);
}

float Motion::wheelDuty(float target, float measured, float& integral) {
  float error = target - measured;
  integral = clampf(integral + cfg::kWheelKi * error * kDt, -0.4f, 0.4f);
  return clampf(target / cfg::kMaxWheelSpeedMmS + cfg::kWheelKp * error + integral, -1.0f, 1.0f);
}

void Motion::drive(float leftMmS, float rightMmS) {
  float left = wheelDuty(leftMmS, leftMeasured_, leftIntegral_);
  float right = wheelDuty(rightMmS, rightMeasured_, rightIntegral_);
  hal_.setMotorDuty(left, right);
}

void Motion::outputsOff() {
  hal_.setMotorDuty(0, 0);
  hal_.setPump(false);
  hal_.setValve(false);
  pumpOn_ = false;
  leftIntegral_ = rightIntegral_ = 0;
}

void Motion::closeSpray() {
  hal_.setValve(false);
  hal_.setPump(false);
  pumpOn_ = false;
  sprayOpen_ = false;
}

void Motion::abort() {
  kind_ = KIND_NONE;
  paused_ = false;
  spray_ = false;
  sprayOpen_ = false;
  outputsOff();
}

Motion::Event Motion::raise(Fault fault) {
  fault_ = fault;
  abort();
  return EV_FAULT;
}

Motion::Event Motion::finish() {
  kind_ = KIND_NONE;
  paused_ = false;
  spray_ = false;
  sprayOpen_ = false;
  outputsOff();
  return EV_DONE;
}

static void resetStall(uint32_t& windowMs, float& motionMm) {
  windowMs = 0;
  motionMm = 0;
}

void Motion::startMove(float mm, bool spray) {
  kind_ = KIND_MOVE;
  paused_ = false;
  targetMm_ = mm;
  traveled_ = 0;
  vCmd_ = 0;
  holdHeading_ = nominalHeading_;
  headingIntegral_ = 0;
  spray_ = spray;
  sprayOpen_ = false;
  runMs_ = 0;
  uint32_t now = hal_.nowMs();
  waitUntilMs_ = now;
  if (spray) {
    hal_.setPump(true);
    hal_.setValve(true);
    pumpOn_ = true;
    pumpOnSinceMs_ = now;
    sprayOpen_ = true;
    waitUntilMs_ = now + cfg::kValveOpenDelayMs;
  }
  timeoutMs_ = (uint32_t)(2.0f * (mm / speed_) * 1000.0f) + 2000 + cfg::kValveOpenDelayMs;
  resetStall(stallWindowMs_, stallMotionMm_);
}

void Motion::startTurn(float degrees) {
  kind_ = KIND_TURN;
  paused_ = false;
  nominalHeading_ += degrees;
  targetHeading_ = nominalHeading_;
  rateCmd_ = 0;
  settledTicks_ = 0;
  runMs_ = 0;
  float expectedMs = fabsf(degrees) / cfg::kTurnRateDegS * 1000.0f;
  timeoutMs_ = (uint32_t)(2.0f * expectedMs) + 2000;
  resetStall(stallWindowMs_, stallMotionMm_);
}

void Motion::startEdge(char axis, float valueMm) {
  kind_ = KIND_EDGE;
  paused_ = false;
  edgeAxis_ = axis;
  edgeValue_ = valueMm;
  traveled_ = 0;
  holdHeading_ = nominalHeading_;
  headingIntegral_ = 0;
  runMs_ = 0;
  timeoutMs_ = (uint32_t)(2.0f * (cfg::kEdgeSearchMm / cfg::kEdgeSpeedMmS) * 1000.0f) + 2000;
  resetStall(stallWindowMs_, stallMotionMm_);
}

void Motion::pause() {
  if (kind_ == KIND_NONE || paused_) return;
  paused_ = true;
  hal_.setPump(false);
  hal_.setValve(false);
  pumpOn_ = false;
}

void Motion::resume() {
  if (!paused_) return;
  paused_ = false;
  vCmd_ = 0;
  rateCmd_ = 0;
  resetStall(stallWindowMs_, stallMotionMm_);
  if (spray_ && sprayOpen_) {
    uint32_t now = hal_.nowMs();
    hal_.setPump(true);
    hal_.setValve(true);
    pumpOn_ = true;
    pumpOnSinceMs_ = now;
    waitUntilMs_ = now + cfg::kValveOpenDelayMs;
  }
}

float Motion::headingTrim(float errorDeg) {
  headingIntegral_ = clampf(headingIntegral_ + cfg::kHeadingKi * errorDeg * kDt,
                            -cfg::kHeadingMaxIntegralMmS, cfg::kHeadingMaxIntegralMmS);
  return clampf(cfg::kHeadingKp * errorDeg + headingIntegral_, -cfg::kHeadingMaxTrimMmS, cfg::kHeadingMaxTrimMmS);
}

bool Motion::checkStall(float commandMmS, float wheelMotionMm) {
  if (commandMmS < cfg::kStallMinCommandMmS) {
    resetStall(stallWindowMs_, stallMotionMm_);
    return false;
  }
  stallWindowMs_ += cfg::kTickMs;
  stallMotionMm_ += wheelMotionMm;
  if (stallWindowMs_ < cfg::kStallMs) return false;
  bool stalled = stallMotionMm_ < cfg::kStallMinMotionMm;
  resetStall(stallWindowMs_, stallMotionMm_);
  return stalled;
}

Motion::Event Motion::tick() {
  float leftMm, rightMm;
  updatePose(leftMm, rightMm);
  float mean = (leftMm + rightMm) * 0.5f;
  float wheelMotion = (fabsf(leftMm) + fabsf(rightMm)) * 0.5f;
  uint32_t now = hal_.nowMs();

  if (pumpOn_ && now - pumpOnSinceMs_ > cfg::kPumpMaxMs) return raise(FAULT_PUMP_TIMEOUT);

  if (kind_ != KIND_EDGE && wallKnown_ && poseKnown_) {
    float m = cfg::kBoundsMarginMm;
    if (x_ < -m || x_ > wallW_ + m || y_ < -m || y_ > wallH_ + m) return raise(FAULT_OUT_OF_BOUNDS);
  }

  if (kind_ == KIND_NONE || paused_) {
    hal_.setMotorDuty(0, 0);
    leftIntegral_ = rightIntegral_ = 0;
    return EV_NONE;
  }

  runMs_ += cfg::kTickMs;
  if (runMs_ > timeoutMs_) return raise(FAULT_STEP_TIMEOUT);

  if (kind_ == KIND_MOVE) {
    if (now < waitUntilMs_) {
      drive(0, 0);
      return EV_NONE;
    }
    traveled_ += mean;
    float remaining = targetMm_ - traveled_;
    if (remaining <= cfg::kStopToleranceMm) {
      closeSpray();
      return finish();
    }
    if (sprayOpen_ && remaining <= cfg::kValveCloseLeadMm) {
      hal_.setValve(false);
      sprayOpen_ = false;
    }
    float profile = fminf(speed_, sqrtf(2.0f * cfg::kAccelMmS2 * remaining) + cfg::kMinMoveSpeedMmS);
    vCmd_ = fminf(vCmd_ + cfg::kAccelMmS2 * kDt, profile);
    float error = holdHeading_ - heading_;
    if (fabsf(error) > cfg::kHeadingFaultDeg) return raise(FAULT_HEADING);
    float trim = headingTrim(error);
    drive(vCmd_ - trim, vCmd_ + trim);
    if (checkStall(vCmd_, wheelMotion)) return raise(FAULT_STALL);
    return EV_NONE;
  }

  if (kind_ == KIND_TURN) {
    float error = targetHeading_ - heading_;
    if (fabsf(error) <= cfg::kTurnToleranceDeg) {
      if (++settledTicks_ >= 3) return finish();
      drive(0, 0);
      return EV_NONE;
    }
    settledTicks_ = 0;
    float profile = fminf(cfg::kTurnRateDegS, sqrtf(2.0f * cfg::kTurnAccelDegS2 * fabsf(error)) + cfg::kTurnMinRateDegS);
    float magnitude = fminf(fabsf(rateCmd_) + cfg::kTurnAccelDegS2 * kDt, profile);
    rateCmd_ = (error > 0 ? 1.0f : -1.0f) * magnitude;
    float wheel = rateCmd_ * kDegToRad * cfg::kWheelbaseMm * 0.5f;
    drive(-wheel, wheel);
    if (checkStall(fabsf(wheel), wheelMotion)) return raise(FAULT_STALL);
    return EV_NONE;
  }

  // KIND_EDGE
  if (hal_.edgeContact()) {
    if (edgeAxis_ == 'x') x_ = edgeValue_; else y_ = edgeValue_;
    // Both bumpers touching means the rover is square to the edge: the heading is the nominal one.
    gyroOffset_ = hal_.gyroHeadingDeg() - nominalHeading_;
    heading_ = nominalHeading_;
    return finish();
  }
  traveled_ += mean;
  if (traveled_ > cfg::kEdgeSearchMm) return raise(FAULT_EDGE_NOT_FOUND);
  float error = holdHeading_ - heading_;
  if (fabsf(error) > cfg::kHeadingFaultDeg) return raise(FAULT_HEADING);
  float trim = headingTrim(error);
  drive(cfg::kEdgeSpeedMmS - trim, cfg::kEdgeSpeedMmS + trim);
  if (checkStall(cfg::kEdgeSpeedMmS, wheelMotion)) return raise(FAULT_STALL);
  return EV_NONE;
}
```

Create `rover/firmware/core/rover.h`:

```cpp
#pragma once
#include <stdint.h>

#include "hal.h"
#include "motion.h"
#include "protocol.h"
#include "safety.h"

// The whole firmware core: protocol handling, the safety state machine and the step executor.
// The caller feeds it received lines and calls tick() every cfg::kTickMs.
class Rover {
 public:
  Rover(Hal& hal, Sink& sink);

  void begin();                       // BOOT, then IDLE
  void handleLine(const char* line);  // one received line, without the newline
  void tick();

  State state() const { return state_; }
  Fault fault() const { return fault_; }

 private:
  void emit(const char* payload);
  void emitState();
  void emitNak(int32_t id, const char* reason);
  void emitAck(int32_t id);
  void emitPose(const char* word, int32_t id, bool withId);
  void setState(State state);
  void enterFault(Fault fault);
  void clearSetup();
  void startStep(const proto::Message& message);

  Hal& hal_;
  Sink& sink_;
  Motion motion_;
  State state_;
  Fault fault_;

  bool helloSeen_, wallSet_, speedSet_, poseSet_;
  int32_t wallWTenths_, wallHTenths_;
  int32_t speedTenths_;
  int32_t currentId_;
  uint32_t lastRxMs_;
  uint32_t lastPoseMs_;
};
```

Create `rover/firmware/core/rover.cpp`:

```cpp
#include "rover.h"

#include <stdio.h>
#include <string.h>

#include "config.h"

namespace {
float fromTenths(int32_t tenths) { return tenths / 10.0f; }

int32_t toTenths(float value) {
  return (int32_t)(value * 10.0f + (value >= 0 ? 0.5f : -0.5f));
}
}  // namespace

Rover::Rover(Hal& hal, Sink& sink)
    : hal_(hal), sink_(sink), motion_(hal), state_(STATE_BOOT), fault_(FAULT_NONE),
      helloSeen_(false), wallSet_(false), speedSet_(false), poseSet_(false),
      wallWTenths_(0), wallHTenths_(0), speedTenths_(0), currentId_(0), lastRxMs_(0), lastPoseMs_(0) {}

void Rover::emit(const char* payload) {
  char line[proto::kMaxLine + 8];
  if (proto::formatLine(line, sizeof(line), payload)) sink_.writeLine(line);
}

void Rover::emitState() {
  char payload[24];
  snprintf(payload, sizeof(payload), "STATE %s", stateName(state_));
  emit(payload);
}

void Rover::emitAck(int32_t id) {
  char payload[24];
  snprintf(payload, sizeof(payload), "ACK %ld", (long)id);
  emit(payload);
}

void Rover::emitNak(int32_t id, const char* reason) {
  char payload[40];
  snprintf(payload, sizeof(payload), "NAK %ld %s", (long)id, reason);
  emit(payload);
}

void Rover::emitPose(const char* word, int32_t id, bool withId) {
  char x[16], y[16], h[16];
  proto::formatTenths(x, sizeof(x), toTenths(motion_.x()));
  proto::formatTenths(y, sizeof(y), toTenths(motion_.y()));
  proto::formatTenths(h, sizeof(h), toTenths(motion_.headingDeg()));
  char payload[64];
  if (withId) {
    snprintf(payload, sizeof(payload), "%s %ld %s %s %s", word, (long)id, x, y, h);
  } else {
    snprintf(payload, sizeof(payload), "%s %s %s %s", word, x, y, h);
  }
  emit(payload);
}

void Rover::setState(State state) {
  state_ = state;
  emitState();
}

void Rover::clearSetup() {
  wallSet_ = speedSet_ = poseSet_ = false;  // HELLO is a one-time handshake, not part of the setup
}

void Rover::enterFault(Fault fault) {
  if (state_ == STATE_FAULT) return;
  motion_.abort();
  fault_ = fault;
  clearSetup();
  char payload[40];
  snprintf(payload, sizeof(payload), "FAULT %s", faultName(fault));
  emit(payload);
  setState(STATE_FAULT);
}

void Rover::begin() {
  setState(STATE_BOOT);
  motion_.outputsOff();
  lastRxMs_ = hal_.nowMs();
  setState(STATE_IDLE);
}

void Rover::startStep(const proto::Message& message) {
  currentId_ = message.id;
  float speedMmS = fromTenths(speedTenths_);
  motion_.setSpeed(speedMmS);
  switch (message.cmd) {
    case proto::CMD_MOVE: motion_.startMove(fromTenths(message.a), message.b == 1); break;
    case proto::CMD_TURN: motion_.startTurn(fromTenths(message.a)); break;
    default: motion_.startEdge(message.axis, fromTenths(message.a)); break;
  }
  emitAck(message.id);
  setState(STATE_RUNNING);
}

void Rover::handleLine(const char* line) {
  proto::Message message;
  int32_t id = 0;
  proto::ParseResult result = proto::parse(line, message, id);
  if (result == proto::PARSE_BADCRC) {
    emitNak(0, "BADCRC");
    return;
  }
  lastRxMs_ = hal_.nowMs();
  if (result == proto::PARSE_UNKNOWN) {
    emitNak(0, "UNKNOWN");
    return;
  }
  if (result == proto::PARSE_BADARG) {
    emitNak(id, "BADARG");
    return;
  }

  switch (message.cmd) {
    case proto::CMD_HELLO: {
      helloSeen_ = true;
      char payload[72];
      snprintf(payload, sizeof(payload), "HELLO fw=%s proto=%d max_speed_mm_s=%ld pump_max_s=%d",
               cfg::kFirmwareVersion, cfg::kProtocolVersion, (long)cfg::kMaxSpeedMmS, cfg::kPumpMaxSeconds);
      emit(payload);
      return;
    }
    case proto::CMD_PING:
      emit("PONG");
      return;
    case proto::CMD_WALL:
    case proto::CMD_SPEED:
    case proto::CMD_POSE:
      if (state_ != STATE_IDLE) {
        emitNak(0, "BADSTATE");
        return;
      }
      if (message.cmd == proto::CMD_WALL) {
        wallWTenths_ = message.a;
        wallHTenths_ = message.b;
        motion_.setWall(fromTenths(message.a), fromTenths(message.b));
        wallSet_ = true;
      } else if (message.cmd == proto::CMD_SPEED) {
        if (message.a > cfg::kMaxSpeedMmS * 10) {
          emitNak(0, "SPEED");
          return;
        }
        speedTenths_ = message.a;
        speedSet_ = true;
      } else {
        motion_.setPose(fromTenths(message.a), fromTenths(message.b), fromTenths(message.c));
        poseSet_ = true;
      }
      emitAck(0);
      return;
    case proto::CMD_ARM:
      if (state_ != STATE_IDLE || !helloSeen_ || !wallSet_ || !speedSet_ || !poseSet_ || hal_.estopPressed()) {
        emitNak(0, "BADSTATE");
        return;
      }
      emitAck(0);
      setState(STATE_ARMED);
      return;
    case proto::CMD_MOVE:
    case proto::CMD_TURN:
    case proto::CMD_EDGE:
      if (state_ != STATE_ARMED) {
        emitNak(message.id, "BADSTATE");
        return;
      }
      if (message.cmd == proto::CMD_MOVE) {
        int32_t longestTenths = wallWTenths_ > wallHTenths_ ? wallWTenths_ : wallHTenths_;
        int32_t marginTenths = (int32_t)(cfg::kMoveMarginMm * 10.0f);
        float seconds = fromTenths(message.a) / fromTenths(speedTenths_);
        if (message.a > longestTenths + marginTenths || (message.b == 1 && seconds * 1000.0f > cfg::kPumpMaxMs)) {
          emitNak(message.id, "TOO_LONG");
          return;
        }
      }
      startStep(message);
      return;
    case proto::CMD_PAUSE:
      if (state_ != STATE_RUNNING) {
        emitNak(0, "BADSTATE");
        return;
      }
      motion_.pause();
      emitAck(0);
      setState(STATE_PAUSED);
      return;
    case proto::CMD_RESUME:
      if (state_ != STATE_PAUSED) {
        emitNak(0, "BADSTATE");
        return;
      }
      motion_.resume();
      emitAck(0);
      setState(STATE_RUNNING);
      return;
    case proto::CMD_STOP:
      if (state_ == STATE_FAULT) {
        emitNak(0, "BADSTATE");
        return;
      }
      motion_.abort();
      clearSetup();
      emitAck(0);
      if (state_ != STATE_IDLE) setState(STATE_IDLE);
      return;
    case proto::CMD_CLEAR:
      if (state_ != STATE_FAULT || hal_.estopPressed()) {
        emitNak(0, "BADSTATE");
        return;
      }
      motion_.clearFault();
      fault_ = FAULT_NONE;
      clearSetup();
      emitAck(0);
      setState(STATE_IDLE);
      return;
  }
}

void Rover::tick() {
  uint32_t now = hal_.nowMs();

  if (hal_.estopPressed() && state_ != STATE_FAULT) enterFault(FAULT_ESTOP);

  bool watched = state_ == STATE_ARMED || state_ == STATE_RUNNING || state_ == STATE_PAUSED;
  if (watched && now - lastRxMs_ > cfg::kWatchdogMs) enterFault(FAULT_WATCHDOG);

  Motion::Event event = motion_.tick();
  if (event == Motion::EV_FAULT) {
    enterFault(motion_.fault());
  } else if (event == Motion::EV_DONE && state_ == STATE_RUNNING) {
    emitPose("DONE", currentId_, true);
    setState(STATE_ARMED);
  }

  if (state_ != STATE_RUNNING) motion_.outputsOff();

  if (state_ == STATE_RUNNING && now - lastPoseMs_ >= cfg::kPoseStreamMs) {
    lastPoseMs_ = now;
    emitPose("POSE", 0, false);
  }
}
```

Create `rover/sim/plant.h`:

```cpp
#pragma once
#include <stdint.h>

#include "../firmware/core/hal.h"

// A differential-drive rover on a flat wall rectangle, standing in for the real hardware.
// Wheels follow the duty command with a first-order lag and lose a fixed fraction of
// their travel to slip (the encoders count wheel rotation, the ground does not move that far).
// The gyro reports the true heading plus a constant bias drift. The rover cannot leave the
// rectangle: at the boundary the edge sensor reports contact.
struct PlantOptions {
  float slip;            // fraction of wheel travel lost, 0.03 means 3%
  uint32_t seed;
  float wallWidthMm;
  float wallHeightMm;
  float startXMm, startYMm, startHeadingDeg;
  float gyroBiasDegPerS;
  uint32_t valveDelayMs;
  float motorScale;      // 1.0 normal; below 1 the motors reach only that fraction of the commanded speed
  PlantOptions()
      : slip(0.0f), seed(1), wallWidthMm(2438.4f), wallHeightMm(2438.4f),
        startXMm(0.0f), startYMm(125.0f), startHeadingDeg(0.0f), gyroBiasDegPerS(0.0f), valveDelayMs(0), motorScale(1.0f) {}
};

class Plant : public Hal {
 public:
  explicit Plant(const PlantOptions& options);

  // Hal
  uint32_t nowMs();
  void setMotorDuty(float left, float right);
  int32_t encoderTicksLeft();
  int32_t encoderTicksRight();
  float gyroHeadingDeg();
  void setPump(bool on);
  void setValve(bool open);
  bool edgeContact();
  bool estopPressed();

  // Simulation
  void step(uint32_t dtMs);
  void setEstop(bool pressed) { estop_ = pressed; }
  void setStall(bool blocked) { stalled_ = blocked; }

  float trueX() const { return x_; }
  float trueY() const { return y_; }
  float trueHeadingDeg() const { return headingDeg_; }
  bool pumpOn() const { return pump_; }
  bool valveOpen() const { return valve_; }
  float dutyLeft() const { return dutyLeft_; }
  float dutyRight() const { return dutyRight_; }
  float paintedMm() const { return paintedMm_; }

 private:
  float nextUniform();

  PlantOptions options_;
  uint32_t rng_;
  float slipLeft_, slipRight_;
  uint32_t nowMs_;
  float dutyLeft_, dutyRight_;
  float wheelLeft_, wheelRight_;      // wheel surface speeds, mm/s
  double ticksLeft_, ticksRight_;
  float x_, y_, headingDeg_, gyroBiasDeg_;
  bool pump_, valve_, estop_, stalled_;
  uint32_t valveOpenedAtMs_;
  bool valveEffective_;
  float paintedMm_;
};
```

Create `rover/sim/plant.cpp`:

```cpp
#include "plant.h"

#include <math.h>

#include "../firmware/core/config.h"

namespace {
const float kLagSeconds = 0.04f;
const float kDegToRad = 3.14159265f / 180.0f;
}  // namespace

Plant::Plant(const PlantOptions& options)
    : options_(options), rng_(options.seed ? options.seed : 1), slipLeft_(0), slipRight_(0), nowMs_(0),
      dutyLeft_(0), dutyRight_(0), wheelLeft_(0), wheelRight_(0), ticksLeft_(0), ticksRight_(0),
      x_(options.startXMm), y_(options.startYMm), headingDeg_(options.startHeadingDeg), gyroBiasDeg_(0),
      pump_(false), valve_(false), estop_(false), stalled_(false), valveOpenedAtMs_(0),
      valveEffective_(false), paintedMm_(0) {
  slipLeft_ = options.slip * (0.75f + 0.5f * nextUniform());
  slipRight_ = options.slip * (0.75f + 0.5f * nextUniform());
}

float Plant::nextUniform() {
  rng_ = rng_ * 1664525u + 1013904223u;
  return (float)(rng_ >> 8) / 16777216.0f;
}

uint32_t Plant::nowMs() { return nowMs_; }
void Plant::setMotorDuty(float left, float right) { dutyLeft_ = left; dutyRight_ = right; }
int32_t Plant::encoderTicksLeft() { return (int32_t)floor(ticksLeft_); }
int32_t Plant::encoderTicksRight() { return (int32_t)floor(ticksRight_); }
float Plant::gyroHeadingDeg() { return headingDeg_ + gyroBiasDeg_; }
bool Plant::estopPressed() { return estop_; }

void Plant::setPump(bool on) { pump_ = on; }

void Plant::setValve(bool open) {
  if (open && !valve_) valveOpenedAtMs_ = nowMs_;
  valve_ = open;
}

bool Plant::edgeContact() {
  const float eps = 0.01f;
  return x_ <= eps || x_ >= options_.wallWidthMm - eps || y_ <= eps || y_ >= options_.wallHeightMm - eps;
}

void Plant::step(uint32_t dtMs) {
  float dt = dtMs / 1000.0f;
  float targetLeft = dutyLeft_ * cfg::kMaxWheelSpeedMmS * options_.motorScale;
  float targetRight = dutyRight_ * cfg::kMaxWheelSpeedMmS * options_.motorScale;
  if (estop_ || stalled_) targetLeft = targetRight = 0;
  float alpha = dt / (kLagSeconds + dt);
  wheelLeft_ += alpha * (targetLeft - wheelLeft_);
  wheelRight_ += alpha * (targetRight - wheelRight_);
  if (stalled_) wheelLeft_ = wheelRight_ = 0;

  ticksLeft_ += wheelLeft_ * dt / cfg::kMmPerTick;
  ticksRight_ += wheelRight_ * dt / cfg::kMmPerTick;

  float groundLeft = wheelLeft_ * (1.0f - slipLeft_);
  float groundRight = wheelRight_ * (1.0f - slipRight_);
  float speed = (groundLeft + groundRight) * 0.5f;
  float omega = (groundRight - groundLeft) / cfg::kWheelbaseMm;  // rad/s, counter-clockwise positive

  headingDeg_ += omega / kDegToRad * dt;
  float newX = x_ + speed * cosf(headingDeg_ * kDegToRad) * dt;
  float newY = y_ + speed * sinf(headingDeg_ * kDegToRad) * dt;
  bool hitX = false, hitY = false;
  if (newX < 0) { newX = 0; hitX = true; }
  if (newX > options_.wallWidthMm) { newX = options_.wallWidthMm; hitX = true; }
  if (newY < 0) { newY = 0; hitY = true; }
  if (newY > options_.wallHeightMm) { newY = options_.wallHeightMm; hitY = true; }
  // Pushing into an edge with both bumpers squares the rover to it.
  // (only when the rover is really driving into that edge, not sliding along it or turning on the spot)
  bool driving = fabsf(speed) > 15.0f;
  if (driving && hitX && fabsf(cosf(headingDeg_ * kDegToRad)) > 0.7f) headingDeg_ = 180.0f * roundf(headingDeg_ / 180.0f);
  if (driving && hitY && fabsf(sinf(headingDeg_ * kDegToRad)) > 0.7f) headingDeg_ = 90.0f + 180.0f * roundf((headingDeg_ - 90.0f) / 180.0f);
  float moved = sqrtf((newX - x_) * (newX - x_) + (newY - y_) * (newY - y_));
  x_ = newX;
  y_ = newY;
  gyroBiasDeg_ += options_.gyroBiasDegPerS * dt;

  nowMs_ += dtMs;
  valveEffective_ = valve_ && nowMs_ - valveOpenedAtMs_ >= options_.valveDelayMs;
  if (pump_ && valveEffective_) paintedMm_ += moved;
}
```

Create `rover/sim/sim_main.cpp`:

```cpp
// rover_sim: the firmware core on the PC, against a simulated rover.
//
// stdin: protocol lines (they go to the core) and simulator commands that start with '@':
//   @ADVANCE ms   run the simulation for ms milliseconds (multiples of 10)
//   @TRUTH        print the true state of the simulated rover
//   @ESTOP 0|1    release or press the e-stop
//   @STALL 0|1    block or free the wheels
// stdout: every line the core writes, then "@END" after each processed input line.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../firmware/core/config.h"
#include "../firmware/core/rover.h"
#include "plant.h"

class StdoutSink : public Sink {
 public:
  void writeLine(const char* line) { printf("%s\n", line); }
};

static bool parseOptions(int argc, char** argv, PlantOptions& options) {
  for (int i = 1; i < argc; i++) {
    const char* flag = argv[i];
    if (strcmp(flag, "--slip") == 0 && i + 1 < argc) options.slip = (float)atof(argv[++i]);
    else if (strcmp(flag, "--seed") == 0 && i + 1 < argc) options.seed = (uint32_t)atol(argv[++i]);
    else if (strcmp(flag, "--gyro-bias") == 0 && i + 1 < argc) options.gyroBiasDegPerS = (float)atof(argv[++i]);
    else if (strcmp(flag, "--valve-delay") == 0 && i + 1 < argc) options.valveDelayMs = (uint32_t)atol(argv[++i]);
    else if (strcmp(flag, "--motor-scale") == 0 && i + 1 < argc) options.motorScale = (float)atof(argv[++i]);
    else if (strcmp(flag, "--wall") == 0 && i + 2 < argc) {
      options.wallWidthMm = (float)atof(argv[++i]);
      options.wallHeightMm = (float)atof(argv[++i]);
    } else if (strcmp(flag, "--start") == 0 && i + 3 < argc) {
      options.startXMm = (float)atof(argv[++i]);
      options.startYMm = (float)atof(argv[++i]);
      options.startHeadingDeg = (float)atof(argv[++i]);
    } else {
      fprintf(stderr, "unknown or incomplete option: %s\n", flag);
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  PlantOptions options;
  if (!parseOptions(argc, argv, options)) return 2;

  Plant plant(options);
  StdoutSink sink;
  Rover rover(plant, sink);
  rover.begin();
  printf("@END\n");
  fflush(stdout);

  char line[256];
  while (fgets(line, sizeof(line), stdin)) {
    size_t length = strlen(line);
    while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r')) line[--length] = '\0';

    if (line[0] == '@') {
      if (strncmp(line, "@ADVANCE ", 9) == 0) {
        long ms = atol(line + 9);
        for (long elapsed = 0; elapsed + (long)cfg::kTickMs <= ms; elapsed += cfg::kTickMs) {
          plant.step(cfg::kTickMs);
          rover.tick();
        }
      } else if (strcmp(line, "@TRUTH") == 0) {
        printf("TRUTH %.3f %.3f %.3f %d %d %.3f %.3f %.3f %lu\n", plant.trueX(), plant.trueY(),
               plant.trueHeadingDeg(), plant.pumpOn() ? 1 : 0, plant.valveOpen() ? 1 : 0, plant.dutyLeft(),
               plant.dutyRight(), plant.paintedMm(), (unsigned long)plant.nowMs());
      } else if (strncmp(line, "@ESTOP ", 7) == 0) {
        plant.setEstop(atoi(line + 7) != 0);
      } else if (strncmp(line, "@STALL ", 7) == 0) {
        plant.setStall(atoi(line + 7) != 0);
      } else if (strcmp(line, "@QUIT") == 0) {
        return 0;
      } else {
        printf("@ERR unknown simulator command\n");
      }
    } else if (length > 0) {
      rover.handleLine(line);
    }
    printf("@END\n");
    fflush(stdout);
  }
  return 0;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m unittest discover -s tests -p "test_rover_safety.py" -v`
Expected: 19 tests PASS.

Run: `uv run python -m unittest discover -s tests -p "test_rover_motion.py" -v`
Expected: 16 tests PASS.

Run: `uv run python -m unittest discover -s tests -p "test_*.py"`
Expected: all PASS (175 tests), 1 skipped. If a test class is skipped with `blocked by the operating system`, see the note in Task 1 Step 4.

- [ ] **Step 5: Commit**

```bash
git add rover/firmware/core rover/sim tests/test_rover_safety.py tests/test_rover_motion.py
git commit -m "Add the rover firmware core and the PC simulator" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git log -1 --format='%(trailers:key=Co-Authored-By)'
```

---

### Task 3: Fault tests and a mutation check

**Files:**
- Test: `tests/test_rover_faults.py`

No source changes: Task 2 already contains every fault check. This task proves the tests would notice if one of them broke.

**Interfaces:**
- Consumes: `SimTestCase`, `words` (Task 1), `rover_sim` and its options `--motor-scale`, `--gyro-bias`, `--wall` (Task 2).

- [ ] **Step 1: Write the tests**

Create `tests/test_rover_faults.py`:

```python
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
        self.assertAlmostEqual(t.truth()["x"], 1150.0, delta=15.0)  # gave up after about 150 mm

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
```

- [ ] **Step 2: Run them**

Run: `uv run python -m unittest discover -s tests -p "test_rover_faults.py" -v`
Expected: 8 tests PASS.

Run: `uv run python -m unittest discover -s tests -p "test_*.py"`
Expected: all PASS (183 tests), 1 skipped.

- [ ] **Step 3: Mutation check**

Apply each change below to the named file, one at a time. After each, run `uv run python -m unittest discover -s tests -p "test_rover_*.py"` and confirm it FAILS (the tests that must fail are named). Then restore the file with `git checkout -- <file>` before the next change. Write the result of each one in `.superpowers/sdd/task-3-report.md`. All seven must be caught. If one is not, the matching test is too weak: strengthen it and say what you changed.

| # | File | Replace | With | Tests that must fail |
|---|---|---|---|---|
| 1 | `rover/firmware/core/config.h` | `const uint32_t kWatchdogMs = 500;` | `const uint32_t kWatchdogMs = 5000;` | `test_silence_while_armed_is_a_fault` and others in `TestWatchdog` |
| 2 | `rover/firmware/core/motion.cpp` | `  if (pumpOn_ && now - pumpOnSinceMs_ > cfg::kPumpMaxMs) return raise(FAULT_PUMP_TIMEOUT);` | (delete the line) | `test_pump_timeout` |
| 3 | `rover/firmware/core/motion.cpp` | `  bool stalled = stallMotionMm_ < cfg::kStallMinMotionMm;` | `  bool stalled = false;` | `test_stall`, `test_stall_during_a_turn` |
| 4 | `rover/firmware/core/motion.cpp` | `    heading_ = nominalHeading_;` followed by `    return finish();` (inside the EDGE contact branch) | delete only the `heading_ = nominalHeading_;` line | `test_edge_sets_the_coordinate_and_squares_the_heading` |
| 5 | `rover/firmware/core/rover.cpp` | `  if (hal_.estopPressed() && state_ != STATE_FAULT) enterFault(FAULT_ESTOP);` | (delete the line) | the `TestEstop` tests |
| 6 | `rover/firmware/core/motion.cpp` | `  return clampf(cfg::kHeadingKp * errorDeg + headingIntegral_, -cfg::kHeadingMaxTrimMmS, cfg::kHeadingMaxTrimMmS);` | `  return 0.0f;` | `test_heading_hold_keeps_the_rover_straight_when_the_wheels_slip_unevenly` |
| 7 | `rover/firmware/core/rover.cpp` | `if (message.a > longestTenths + marginTenths \|\|` | `if (message.a > longestTenths + marginTenths + 100000 \|\|` | `test_moves_longer_than_the_wall_or_the_pump_limit_are_too_long` |

(In the table `\|\|` stands for the two characters `||`.) Finally run `git status --short`: it must show no modified source files.

Note for the reviewer: the line `if (state_ != STATE_RUNNING) motion_.outputsOff();` at the end of `Rover::tick()` is a deliberate second layer. Today `Motion` already switches everything off, so removing that line changes no test result. It is not a gap to fix.

- [ ] **Step 4: Commit**

```bash
git add tests/test_rover_faults.py
git commit -m "Add the rover firmware fault tests" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git log -1 --format='%(trailers:key=Co-Authored-By)'
```

---

### Task 4: The host tool and the profile's pump limit

**Files:**
- Create: `rover/tools/run_mission.py`
- Modify: `rover_profile.json`, `rover_mission.py`, `tests/test_rover_mission.py`
- Test: `tests/test_rover_run_mission.py`, `tests/test_rover_mission.py`

**Interfaces:**
- Consumes: `SimTransport` and `parse_line`, `tenths` (Task 1); `rover_sim` (Task 2); `verify_checksum`, `compile_rover_mission`, `load_profile` (P1).
- Produces: `run_mission(transport, mission, wait_ms=100, max_step_ms=120000, on_step=None) -> list of {i, op, expected, reported}` (raises `MissionError`), `pose_errors(records)`, `MissionError`, `main(argv) -> int` (command line: `uv run python rover/tools/run_mission.py exports/apr_rover_mission.json --sim rover/sim/build/rover_sim.exe`). A transport is any object with `send(payload) -> lines` and `wait(ms) -> lines`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rover_run_mission.py`:

```python
import contextlib
import copy
import io
import json
import os
import tempfile
import unittest

from rover_support import SimTestCase
from geometry_engine import START_CORNERS, GeometryEngine
from rover_mission import compile_rover_mission, load_profile
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
                self.assertAlmostEqual(t.truth()["painted"], sprayed, delta=0.02 * sprayed)
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
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([self.write_mission(compiled()), "--sim", self.binary, "--sim-option=--motor-scale", "--sim-option=0.1"])
        self.assertEqual(code, 1)
        self.assertIn("mission failed", out.getvalue())
        self.assertIn("STEP_TIMEOUT", out.getvalue())


class TestPoseErrors(unittest.TestCase):
    def test_largest_difference_with_heading_wrap_around(self):
        records = [
            {"expected": {"x": 0.0, "y": 0.0, "heading": 180.0}, "reported": {"x": 1.0, "y": -2.0, "heading": -179.0}},
            {"expected": {"x": 5.0, "y": 5.0, "heading": 0.0}, "reported": {"x": 5.0, "y": 8.0, "heading": 0.5}},
        ]
        self.assertEqual(pose_errors(records), {"x": 1.0, "y": 3.0, "heading": 1.0})


if __name__ == "__main__":
    unittest.main()
```

Then make the edit to `tests/test_rover_mission.py` shown under "Edit `tests/test_rover_mission.py`" in Step 3 (the new pump-limit test goes in now).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m unittest discover -s tests -p "test_rover_run_mission.py" -v`
Expected: ERROR `ModuleNotFoundError: No module named 'run_mission'`.

Run: `uv run python -m unittest discover -s tests -p "test_rover_mission.py" -v`
Expected: FAIL in `test_the_shipped_profile_loads` (`KeyError: 'pump_max_continuous_s'`) and in the new pump-limit test.

- [ ] **Step 3: Implement**

Create `rover/tools/run_mission.py`:

```python
"""
Stand-in for the Raspberry Pi: stream a rover mission file (apr-rover-mission/1) to a rover,
one step at a time, and check what the rover reports.

    uv run python rover/tools/run_mission.py exports/apr_rover_mission.json --sim build/rover_sim

Today the only transport is the PC simulator. A serial transport arrives with the real
board in plan P2b.
"""

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from rover_link import SimTransport, parse_line, tenths  # noqa: E402
from rover_mission import verify_checksum  # noqa: E402

PROTOCOL_VERSION = 1


class MissionError(Exception):
    """The rover refused a command, reported a fault, or did not finish."""


def _expect_ack(lines: List[str], what: str) -> None:
    for line in lines:
        word, args = parse_line(line)
        if word == "NAK":
            raise MissionError(f"{what} refused: NAK {' '.join(args)}")
        if word == "ACK":
            return
    raise MissionError(f"{what}: no ACK received")


def _hello(transport: Any) -> Dict[str, str]:
    for line in transport.send("HELLO"):
        word, args = parse_line(line)
        if word == "HELLO":
            return dict(arg.split("=", 1) for arg in args)
    raise MissionError("no HELLO reply from the rover")


def _step_payload(step: Dict[str, Any]) -> str:
    op = step["op"]
    if op == "MOVE":
        return f"MOVE {step['i']} {tenths(step['mm'])} {step['spray']}"
    if op == "TURN":
        return f"TURN {step['i']} {tenths(step['deg'])}"
    if op == "EDGE_ALIGN":
        return f"EDGE {step['i']} {step['axis']} {tenths(step['value_mm'])}"
    raise MissionError(f"unknown mission step {op}")


def run_mission(
    transport: Any,
    mission: Dict[str, Any],
    wait_ms: int = 100,
    max_step_ms: int = 120000,
    on_step: Optional[Callable[[Dict[str, Any], Dict[str, float]], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Run every step. Returns one record per step with the expected and the reported pose.
    Raises MissionError on any NAK or FAULT. Sends PING between waits so the rover's
    500 ms watchdog stays fed.
    """
    if not verify_checksum(mission):
        raise MissionError("mission checksum does not match its content")

    hello = _hello(transport)
    if int(hello.get("proto", "0")) != PROTOCOL_VERSION:
        raise MissionError(f"rover speaks protocol {hello.get('proto')}, expected {PROTOCOL_VERSION}")
    speed_mm_s = mission["speed_mps"] * 1000.0
    if speed_mm_s > float(hello["max_speed_mm_s"]):
        raise MissionError(f"mission speed {speed_mm_s:.0f} mm/s is above the rover's {hello['max_speed_mm_s']} mm/s")
    pump_limit = mission["profile"].get("pump_max_continuous_s")
    if pump_limit is not None and pump_limit > float(hello["pump_max_s"]):
        raise MissionError("the profile allows a longer pump run than the firmware does")

    wall = mission["wall"]
    pose = mission["start_pose"]
    _expect_ack(transport.send(f"WALL {tenths(wall['width_mm'])} {tenths(wall['height_mm'])}"), "WALL")
    _expect_ack(transport.send(f"SPEED {tenths(speed_mm_s)}"), "SPEED")
    _expect_ack(
        transport.send(f"POSE {tenths(pose['x_mm'])} {tenths(pose['y_mm'])} {tenths(pose['heading_deg'])}"), "POSE"
    )
    _expect_ack(transport.send("ARM"), "ARM")

    records: List[Dict[str, Any]] = []
    for step in mission["steps"]:
        step_id = step["i"]
        _expect_ack(transport.send(_step_payload(step)), f"step {step_id}")
        waited = 0
        done: Optional[Dict[str, float]] = None
        while done is None:
            if waited >= max_step_ms:
                raise MissionError(f"step {step_id} did not finish within {max_step_ms} ms")
            lines = transport.wait(wait_ms) + transport.send("PING")
            waited += wait_ms
            for line in lines:
                word, args = parse_line(line)
                if word == "FAULT":
                    raise MissionError(f"rover fault during step {step_id}: {args[0]}")
                if word == "DONE" and int(args[0]) == step_id:
                    done = {"x": float(args[1]), "y": float(args[2]), "heading": float(args[3])}
        record = {
            "i": step_id,
            "op": step["op"],
            "expected": {"x": step["x_mm"], "y": step["y_mm"], "heading": step["heading_deg"]},
            "reported": done,
        }
        records.append(record)
        if on_step:
            on_step(step, done)

    _expect_ack(transport.send("STOP"), "STOP")
    return records


def pose_errors(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Largest difference between the rover's reported pose and the mission's expected pose."""
    worst = {"x": 0.0, "y": 0.0, "heading": 0.0}
    for record in records:
        for key in worst:
            diff = record["reported"][key] - record["expected"][key]
            if key == "heading":
                diff = (diff + 180.0) % 360.0 - 180.0
            worst[key] = max(worst[key], abs(diff))
    return worst


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mission", help="path to apr_rover_mission.json")
    parser.add_argument("--sim", required=True, help="path to the rover_sim binary")
    parser.add_argument("--sim-option", action="append", default=[], help="extra rover_sim option, repeatable")
    args = parser.parse_args(argv)

    with open(args.mission, encoding="utf-8") as f:
        mission = json.load(f)
    pose = mission["start_pose"]
    options = ["--start", str(pose["x_mm"]), str(pose["y_mm"]), str(pose["heading_deg"]),
               "--wall", str(mission["wall"]["width_mm"]), str(mission["wall"]["height_mm"])] + args.sim_option
    transport = SimTransport(args.sim, options)
    try:
        records = run_mission(transport, mission)
        truth = transport.truth()
    except MissionError as error:
        print(f"mission failed: {error}")
        return 1
    finally:
        transport.close()

    errors = pose_errors(records)
    print(f"{len(records)} steps done. Largest reported-versus-expected pose difference: "
          f"x {errors['x']:.1f} mm, y {errors['y']:.1f} mm, heading {errors['heading']:.1f} deg")
    print(f"True final pose in the simulator: x {truth['x']:.1f} mm, y {truth['y']:.1f} mm, heading {truth['heading']:.1f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Edit `rover_profile.json`:** replace

```json
  "max_speed_mps": 0.25
```

with

```json
  "max_speed_mps": 0.25,
  "pump_max_continuous_s": 30.0
```

**Edit `rover_mission.py`:**

1. In the module docstring replace this bullet:

```
- EDGE_ALIGN: the rover drives slowly along its current heading until its edge sensor triggers
  (firmware faults if it does not trigger within a firmware-defined distance), then sets the
  coordinate on `axis` to value_mm. The step's own x_mm/y_mm are the expected pose BEFORE that
  reset. The offset between the edge sensor and the nozzle centre belongs to the rover profile
  (added in plan P2).
```

with:

```
- EDGE_ALIGN: the rover drives slowly along its current heading until its edge sensors trigger
  (firmware faults if they do not trigger within 150 mm), then sets the coordinate on `axis` to
  value_mm. Pushing into the edge with both sensors squares the rover to it, so its heading is
  also reset to the nominal heading. That resets gyro drift once per row. The step's own
  x_mm/y_mm are the expected pose BEFORE that reset. The offset between the edge sensors and the
  nozzle centre is a rover constant, added with the real hardware in plan P2b.
```

2. In the same docstring replace the last three lines of the checksum bullet:

```
  its `checksum` key. Float text is Python's repr (for example 0.0, 90.0), so a verifier in
  another language must reproduce that formatting. A language-neutral canonical form is to be
  decided in the P2 spec, while the schema is still /1 with no consumers.
```

with:

```
  its `checksum` key. Float text is Python's repr (for example 0.0, 90.0), so a verifier in
  another language must reproduce that formatting. Only Python code verifies it today (the host
  tool, and the Pi agent in plan P3); a language-neutral canonical form is needed only if a
  non-Python verifier appears.
```

3. In `load_profile` replace:

```python
    for key in ("wheel_diameter_mm", "max_speed_mps"):
        value = profile.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"rover profile {key} must be a positive number")
```

with:

```python
    for key, required in (("wheel_diameter_mm", True), ("max_speed_mps", True), ("pump_max_continuous_s", False)):
        value = profile.get(key)
        if value is None and not required:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"rover profile {key} must be a positive number")
```

**Edit `tests/test_rover_mission.py`:** in `TestProfile.test_the_shipped_profile_loads` replace

```python
        self.assertEqual(profile["max_speed_mps"], 0.25)

```

with

```python
        self.assertEqual(profile["max_speed_mps"], 0.25)
        self.assertEqual(profile["pump_max_continuous_s"], 30.0)

    def test_the_pump_limit_is_optional_but_must_be_a_positive_number_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "profile.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"schema": "apr-rover-profile/1", "wheel_diameter_mm": 80.0, "max_speed_mps": 0.25}, f)
            self.assertNotIn("pump_max_continuous_s", load_profile(path))
            for bad in (0, -1, True, "30", float("inf")):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    load_profile(write_profile(tmp, pump_max_continuous_s=bad))

```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m unittest discover -s tests -p "test_rover_run_mission.py" -v`
Expected: 13 tests PASS.

Run: `uv run python -m unittest discover -s tests -p "test_rover_mission.py" -v`
Expected: 16 tests PASS.

Run: `uv run python -m unittest discover -s tests -p "test_*.py"`
Expected: all PASS (197 tests), 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add rover/tools/run_mission.py rover_profile.json rover_mission.py tests/test_rover_run_mission.py tests/test_rover_mission.py
git commit -m "Add the rover mission host tool and the profile pump limit" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git log -1 --format='%(trailers:key=Co-Authored-By)'
```

---

### Task 5: The protocol document, corrected specs and final verification

**Files:**
- Create: `docs/rover-protocol.md`
- Modify: `docs/superpowers/specs/2026-10-09-rover-firmware-protocol-design.md`, `docs/superpowers/specs/2026-10-09-rover-hardware-integration-design.md`

- [ ] **Step 1: Write the protocol document**

Create `docs/rover-protocol.md`:

````markdown
# Rover protocol (version 1)

The line protocol between a host (a laptop today, the Raspberry Pi in plan P3) and the rover's Arduino. The Arduino firmware core (`rover/firmware/core/`) implements it, the simulator (`rover/sim/`) runs that same core on the PC, and `rover/tools/rover_link.py` is the host side. This document is the single source of truth: if code and document disagree, fix one of them in the same commit.

## Framing

One ASCII line per message, ending with a newline on the wire. A line is `WORDS args*CC`:

- `CC` is a CRC-8 (polynomial 0x07, initial value 0, no reflection, no final xor) over every character before the `*`, written as two upper-case hex digits. Lower case is accepted when receiving. The check value for `123456789` is `F4`.
- A line is at most 80 characters including `*CC`.
- Fields are separated by single spaces. Extra spaces are tolerated when receiving.
- USB serial, 115200 baud, 8N1.
- A received line with a missing or wrong CRC, bad hex, or more than 80 characters is never executed and gets `NAK 0 BADCRC`. It does not count as a heartbeat.

## Numbers

Distances are millimetres and angles are degrees, always written with one decimal (`2438.4`, `-90.0`, `0.0`). The firmware also accepts up to three decimals and rounds half away from zero to a tenth, so `0.25` is `0.3`. Integers (step ids) have no decimals. Headings are counter-clockwise from the wall's +x axis. The wall frame has its origin at the bottom-left corner, x to the right, y up, and all positions refer to the spray nozzle centre.

## Host to rover

| Line | Allowed in | Effect |
|---|---|---|
| `HELLO` | any state | Reply `HELLO fw=<version> proto=1 max_speed_mm_s=<n> pump_max_s=<n>`. Required once before `ARM`. |
| `WALL w h` | IDLE | Wall size in mm. Sets the bounds for the `OUT_OF_BOUNDS` check and the `TOO_LONG` limit. |
| `SPEED v` | IDLE | Speed for `MOVE` and `EDGE` in mm/s. `NAK 0 SPEED` above `max_speed_mm_s`. |
| `POSE x y heading` | IDLE | Where the operator placed the rover. Sets the pose estimate and the nominal heading. |
| `ARM` | IDLE | IDLE to ARMED. Needs `HELLO`, `WALL`, `SPEED` and `POSE` since the last `STOP` or `CLEAR`, and the e-stop released. |
| `MOVE id mm spray` | ARMED | Drive `mm` along the nominal heading. `spray` is `0` or `1`. |
| `TURN id deg` | ARMED | Turn on the spot to the nominal heading plus `deg` (-180.0 to 180.0, counter-clockwise positive). |
| `EDGE id axis value` | ARMED | `axis` is `x` or `y`. Creep along the nominal heading until both edge sensors touch the wall edge (at most 150 mm), then set that coordinate to `value` and square the heading to the nominal one. |
| `PAUSE` | RUNNING | Stop the step, switch motors, pump and valve off. |
| `RESUME` | PAUSED | Continue the step. |
| `STOP` | IDLE, ARMED, RUNNING, PAUSED | Abort the step, outputs off, go to IDLE. The setup (`WALL`, `SPEED`, `POSE`) is forgotten. |
| `PING` | any state | Heartbeat. Reply `PONG`. |
| `CLEAR` | FAULT | Only with the e-stop released. Go to IDLE and forget the setup. |

Step ids are positive integers chosen by the host (the mission file's step number). A step id of `0` in a reply means the line had no id.

## Rover to host

| Line | Meaning |
|---|---|
| `ACK id` | Accepted. `id` is `0` for commands without a step id. |
| `NAK id reason` | Refused, nothing executed. Reasons: `BADCRC`, `UNKNOWN` (not a command), `BADARG` (wrong argument count, number, or range), `BADSTATE` (not allowed in the current state), `SPEED` (above the maximum), `TOO_LONG` (see below). |
| `DONE id x y heading` | Step finished, with the pose after it (after the reset for `EDGE`). Heading is in (-180, 180]. |
| `POSE x y heading` | The pose estimate, every 100 ms while RUNNING. |
| `STATE name` | Sent on every state change: `BOOT`, `IDLE`, `ARMED`, `RUNNING`, `PAUSED`, `FAULT`. |
| `FAULT code` | Sent once when entering FAULT. |
| `HELLO ...`, `PONG` | Replies to `HELLO` and `PING`. |

`TOO_LONG`: a `MOVE` longer than the longer wall side plus 100 mm, or a spray `MOVE` that would keep the pump on longer than `pump_max_s` at the set speed.

## States

BOOT, IDLE, ARMED, RUNNING, PAUSED and a latched FAULT.

- On power-up the rover sends `STATE BOOT` and `STATE IDLE`.
- A step command in ARMED gives `ACK id` and `STATE RUNNING`. When the step ends: `DONE ...` and `STATE ARMED`.
- Motors, pump and valve are off in every state except RUNNING. The pump and valve are on only during a `MOVE` with spray 1.
- FAULT is left only with `CLEAR`, which needs the e-stop released and a person's decision on the host. After it the setup (`WALL`, `SPEED`, `POSE`, `ARM`) must be sent again. `HELLO` does not need repeating.

## Safety rules, enforced in the firmware

| Rule | Value |
|---|---|
| Watchdog: no valid line (any line with a good CRC, including `PING`) while ARMED, RUNNING or PAUSED | 500 ms, `FAULT WATCHDOG`. Hosts should send `PING` at least every 200 ms. |
| E-stop input pressed, in any state | `FAULT ESTOP`. A physical relay also cuts motor and pump power, independent of the firmware. |
| `STALL`: motion commanded (20 mm/s or more) but under 1 mm of wheel motion in 1 s | `FAULT STALL` |
| `STEP_TIMEOUT`: a step runs longer than twice its expected time plus 2 s | `FAULT STEP_TIMEOUT` |
| `OUT_OF_BOUNDS`: estimated position more than 50 mm outside the `WALL` rectangle (not checked during `EDGE`) | `FAULT OUT_OF_BOUNDS` |
| `HEADING`: heading more than 15 degrees from the nominal heading during `MOVE` or `EDGE` | `FAULT HEADING` |
| `EDGE_NOT_FOUND`: no edge contact within 150 mm of creeping | `FAULT EDGE_NOT_FOUND` |
| `PUMP_TIMEOUT`: pump on continuously for more than 30 s | `FAULT PUMP_TIMEOUT` |

All limits are starting values in `rover/firmware/core/config.h` and expected to change after the bench tests (plan P2b).

## Example session

Lines the host sends start with `>`, lines the rover sends with `<`. This is a real transcript from `rover_sim`.

```
< STATE BOOT*E6
< STATE IDLE*C8
> HELLO*35
< HELLO fw=0.1 proto=1 max_speed_mm_s=250 pump_max_s=30*F7
> WALL 2438.4 2438.4*1F
< ACK 0*5F
> SPEED 250.0*A3
< ACK 0*5F
> POSE 0.0 125.0 0.0*32
< ACK 0*5F
> ARM*2F
< ACK 0*5F
< STATE ARMED*CC
> MOVE 1 300.0 1*10
< ACK 1*58
< STATE RUNNING*99
< POSE 0.5 125.0 0.0*51
< POSE 2.5 125.0 0.0*7E
> PING*1F
< PONG*62
    ... a POSE line every 100 ms, a PING from the host at least every 200 ms ...
< DONE 1 299.1 125.0 0.0*BC
< STATE ARMED*CC
```

## Simulator commands

`rover_sim` (built from `rover/sim/sim_main.cpp`) accepts the protocol above on stdin, plus simulator commands that start with `@`: `@ADVANCE ms` (run simulated time, in multiples of 10 ms), `@TRUTH` (print the true state of the simulated rover), `@ESTOP 0|1`, `@STALL 0|1`, `@QUIT`. It prints every line the firmware writes, then `@END` after each processed input line. Simulated time passes only through `@ADVANCE`. Options: `--start x y heading`, `--wall w h`, `--slip f`, `--seed n`, `--gyro-bias deg_per_s`, `--valve-delay ms`, `--motor-scale f`. The simulator models wheel slip, gyro bias, a first-order motor lag, encoder quantisation, a wall rectangle the rover cannot leave, and squaring against an edge it drives into.
````

- [ ] **Step 2: Correct the specs**

Building P2 showed that some spec statements were wrong or incomplete: the state list, the NAK reasons, the EDGE semantics, the simulator options, the accuracy targets, and a missing y reference. Fix them.

Make these replacements in `docs/superpowers/specs/2026-10-09-rover-firmware-protocol-design.md`. Find each line by its starting words; the line numbers are only for orientation.

1. Line 77, the `EDGE` row of the host-to-rover table: replace the whole row with

```
| `EDGE id axis value` | creep into the wall edge until both edge sensors touch it, then set that coordinate to `value` and square the heading to the nominal one |
```

2. Line 86, the `NAK` row: change the list in brackets to ``(`BADCRC`, `UNKNOWN`, `BADSTATE`, `BADARG`, `TOO_LONG`, `SPEED`)``.

3. Line 96, the `States` bullet: replace it with

```
- **States:** BOOT, IDLE, ARMED, RUNNING, PAUSED, and a latched FAULT reachable from every state. There is no DONE state: the host sequences the steps and sends `STOP` at the end.
```

4. Line 120, the `EDGE` bullet of section 4: replace it with these two bullets

```
- **`EDGE`:** creep at `edge_speed_mm_s` until both edge sensors trigger (at most 150 mm), stop, set the coordinate on `axis` to `value`, square the heading to the nominal one and report the corrected pose.
- **Nominal heading:** the rover keeps a nominal heading, the `POSE` heading plus every commanded `TURN`. `MOVE` and `EDGE` hold it with a P+I loop on the gyro, and `TURN` aims at it, so the half-degree stopping tolerance of a turn does not add up over a mission.
```

5. Line 124, the simulator paragraph of section 5: replace the sentence ``Command-line switches inject trouble: `--slip`, `--seed`, `--stall-at-step N`, `--estop-at-ms T`.`` with ``Options: `--start`, `--wall`, `--slip`, `--seed`, `--gyro-bias`, `--valve-delay` and `--motor-scale` (weak motors, for the timeout faults). The simulator commands `@STALL` and `@ESTOP` block the wheels or press the e-stop at any moment. Squaring against an edge the rover drives into is modelled, because without it gyro drift ruins row spacing (see Findings).``

6. Line 132, test 3 of section 6: replace the whole item with

```
3. **Motion accuracy:** `MOVE 1000` within 2%, `TURN 90` within 2 degrees, and a full compiled 8 x 8 ft mission with 3% wheel slip and a gyro bias of 0.01 degrees per second: x re-zeroed within 1 mm at every row end, row-to-row spacing error under 30 mm, and an absolute y error under 4% of the vertical distance driven (about 65 mm measured). These are the Stage 0 and Stage 1 targets of the integration design, in simulation, except the absolute y position (see Findings).
```

7. Insert this new section directly before `## Risks`:

```
## Findings while building P2

- **Gyro drift ruins row spacing unless the heading is reset.** With a gyro bias of 0.01 degrees per second the true heading drifts about 1 degree over a 200 s mission, and rows then lean in alternating directions: the row-to-row spacing error reached 90 to 180 mm, far over the 30 mm target. Pushing into the wall edge with both edge sensors squares the rover to the edge, so `EDGE` now resets the heading as well as the coordinate. That needs two edge sensors (the hardware sizing table already lists two bumper switches), and the real rover must actually square up against the edge. With it the spacing error is under 20 mm at 3% slip.
- **Nothing re-references y.** The edge contacts re-zero x at every row end, but y is only dead-reckoned from the encoders, so its error is about the wheel slip times the vertical distance driven: about 65 mm at 3% slip on an 8 ft wall, 100 mm at 5%. The integration design's "end position within 50 mm" target cannot be met in y without a y reference. Options for P2b or later: a distance sensor to the floor or ceiling, an `EDGE` on the y axis at the end of the last row, or learning the slip factor online from how far past the planned point each row's edge contact happens (a deterministic estimator, and a good first learning-from-logs candidate).
- **The tests must tolerate Windows Application Control.** On this machine a freshly built executable is blocked at random, so `tests/rover_support.py` rebuilds and retries, and the tests skip if no build ever runs.
```

Make this replacement in `docs/superpowers/specs/2026-10-09-rover-hardware-integration-design.md` (line 96, the `Stage A targets` bullet):

```
- Stage A targets, proposed and to be confirmed after Stage 1: row-to-row error within 30 mm and heading within 2 degrees over a row (both met in the simulator at 3% wheel slip), and x within 5 mm at every row end. Absolute end position in y is not on this list: nothing re-references y, so it drifts by about the slip times the vertical distance driven (about 65 mm for 3% slip on an 8 ft wall) until a y reference exists (see the firmware spec).
```

- [ ] **Step 3: Final verification**

Run: `uv run python -m unittest discover -s tests -p "test_*.py"`
Expected: all PASS (197 tests), 1 skipped.

Run: `uv lock --check`
Expected: exits 0 (no dependency changed).

Run: `grep -inE "torch|ultralytics|pyserial" requirements.txt pyproject.toml`
Expected: no output.

Run: `uv run python -c "import app; c = app.app.test_client(); print([c.get(p).status_code for p in ('/', '/about')])"`
Expected: `[200, 200]`

Run the end-to-end tool once by hand, with a mission compiled by the app's own modules. Run this with `uv run python`:

```python
import json, sys
sys.path.insert(0, "tests")
from rover_support import simulator_binary
from geometry_engine import GeometryEngine
from rover_mission import compile_rover_mission, load_profile
wps, st = GeometryEngine().plan_coverage_path(2438.4, 2438.4, 250.0, 12.0, 60.0, [], "bottom_left")
mission = compile_rover_mission(wps, 2438.4, 2438.4, 250.0, st["effective_step_mm"], "bottom_left", load_profile())
with open("exports/demo_rover_mission.json", "w") as f:
    json.dump(mission, f)
print(simulator_binary())
```

then: `uv run python rover/tools/run_mission.py exports/demo_rover_mission.json --sim <the path printed above>`
Expected: `57 steps done.` followed by the pose differences and the true final pose, exit code 0. Delete `exports/demo_rover_mission.json` afterwards (`exports/` is git-ignored).

Run `git status --short`: only the files named in this task may show as modified or new.

- [ ] **Step 4: Commit**

```bash
git add docs/rover-protocol.md docs/superpowers/specs/2026-10-09-rover-firmware-protocol-design.md docs/superpowers/specs/2026-10-09-rover-hardware-integration-design.md
git commit -m "Add the rover protocol document and correct the specs" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git log -1 --format='%(trailers:key=Co-Authored-By)'
```

- [ ] **Step 5: Report to the user, then stop before pushing**

Summarise what shipped. Say plainly that none of this has run on real hardware, that the firmware has not been built for the Arduino (no AVR build exists until plan P2b), and that the simulator's numbers prove the logic and the protocol, not tuned gains. Hand over these open items:

1. Buy the parts and start P2b (drivers, `main.cpp`, PlatformIO build for the Mega, gain tuning, Stage 0 bench criteria).
2. The rover needs two edge sensors that can square it against a wall edge, and a decision on how y gets a reference (see Findings in the firmware spec).
3. The simulator tests can skip on a machine whose Application Control policy blocks new executables.
