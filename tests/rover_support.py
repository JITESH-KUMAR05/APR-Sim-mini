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
_blocked = {}


def _source_files():
    paths = []
    for folder in (CORE_DIR, SIM_DIR):
        for name in sorted(os.listdir(folder)):
            if name.endswith((".h", ".cpp")):
                paths.append(os.path.join(folder, name))
    return paths


def _runs(binary, probe):
    """
    True if the program launched and exited cleanly on `probe`; False if the operating system
    refused to launch it (an OSError, which is what Application Control produces). A program that
    launches but crashes, exits non-zero or hangs is broken, not blocked: that raises AssertionError.
    """
    try:
        result = subprocess.run([binary], input=probe, capture_output=True, text=True, timeout=20)
    except OSError:
        return False
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        raise AssertionError(f"{binary} launched but did not finish within {error.timeout} s; stderr: {stderr!r}")
    if result.returncode != 0:
        raise AssertionError(f"{binary} launched but exited with return code {result.returncode}; stderr: {result.stderr!r}")
    return True


def build(name, sources, probe, attempts=12):
    """Return the path of a runnable test program built from `sources`, or raise unittest.SkipTest."""
    if name in _binaries:
        return _binaries[name]
    if name in _blocked:
        raise unittest.SkipTest(_blocked[name])
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
            try:
                os.remove(binary)
            except OSError:
                pass  # a locked old executable: g++ will say so if it cannot overwrite it
        result = subprocess.run([gpp] + FLAGS + sources + ["-o", binary], capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(f"the rover firmware does not compile:\n{result.stderr}")
        if _runs(binary, probe):
            _binaries[name] = binary
            return binary
    message = f"{name} was blocked by the operating system after {attempts} builds"
    _blocked[name] = message  # later test classes skip at once instead of paying for every build again
    raise unittest.SkipTest(message)


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
        return self.finish_step(transport, list(transport.send(payload)), max_ms)

    def finish_step(self, transport, lines, max_ms=90000):
        """Continue a step that is already running (`lines` are the replies so far), same return as run_step()."""
        lines = list(lines)
        elapsed = 0
        while elapsed < max_ms:
            batch = transport.advance(100) + transport.send("PING")
            elapsed += 100
            lines += batch
            if any(line.startswith(("DONE", "FAULT")) for line in batch):
                break
        return [line for line in lines if not line.startswith(("POSE", "PONG"))], elapsed
