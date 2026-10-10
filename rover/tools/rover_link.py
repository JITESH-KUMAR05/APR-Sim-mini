"""
Host side of the rover protocol (docs/rover-protocol.md): line framing, CRC-8, fixed-point
numbers, and a transport that drives the PC simulator `rover_sim`.

Quantities on the wire are millimetres and degrees in tenths, written as decimal text.
"""

import math
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
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"cannot format {value!r} as a protocol number")
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
    if len(line) > MAX_LINE:
        raise ValueError(f"line longer than {MAX_LINE} characters: {line!r}")
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
        try:
            self.log.extend(self._read_until_end())  # what the rover wrote while booting
        except BaseException:
            if self._process.poll() is None:
                self._process.kill()
            self.close()  # reap the child and close its pipes
            raise

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
        names = ["x", "y", "heading", "pump", "valve", "duty_left", "duty_right", "painted", "t_ms"]
        fields = lines[-1].split() if lines else []
        if not fields or fields[0] != "TRUTH" or len(fields) != len(names) + 1:
            raise RuntimeError(f"expected a TRUTH line with {len(names)} values, got {lines[-1] if lines else 'no reply'!r}")
        return {name: float(value) for name, value in zip(names, fields[1:])}

    def set_estop(self, pressed: bool) -> List[str]:
        return self._send_raw(f"@ESTOP {1 if pressed else 0}")

    def set_stall(self, blocked: bool) -> List[str]:
        return self._send_raw(f"@STALL {1 if blocked else 0}")

    def close(self) -> None:
        """Stop the simulator and release its pipes. Safe to call more than once."""
        process = self._process
        if process.poll() is None:
            try:
                process.stdin.write("@QUIT\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        for pipe in (process.stdout, process.stdin):
            try:
                pipe.close()
            except (OSError, ValueError):
                pass
