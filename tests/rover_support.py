"""
Shared helpers for the rover firmware tests: build the C++ test programs with g++ and drive them.

The build logic lives in rover/tools/sim_build.py (the demo uses it too). Here an unavailable
simulator, either no g++ or an operating system that blocks every new executable, becomes a skipped
test. A compile error is a real failure and is never skipped.
"""

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "rover", "tools"))
sys.path.insert(0, ROOT)

import sim_build  # noqa: E402
from rover_link import SimTransport, tenths  # noqa: E402

CORE_DIR = sim_build.CORE_DIR
SIM_DIR = sim_build.SIM_DIR
BUILD_DIR = sim_build.BUILD_DIR


def build(name, sources, probe, attempts=12):
    """Return the path of a runnable test program built from `sources`, or raise unittest.SkipTest."""
    try:
        return sim_build.build_program(name, sources, probe, attempts)
    except sim_build.SimulatorUnavailable as error:
        raise unittest.SkipTest(str(error))


def simulator_binary():
    try:
        return sim_build.simulator_binary()
    except sim_build.SimulatorUnavailable as error:
        raise unittest.SkipTest(str(error))


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
