"""
Replay a compiled rover mission (apr-rover-mission/1) through the C++ rover simulator, the same program
and the same host tool (run_mission.py) that will drive the real board, and report what happened.

The simulator has wheel slip and gyro drift, so the true path differs from the planned one. Edge contacts
at the row ends pull x (or y for vertical stripes) back; nothing re-references the other axis, which is
the known gap listed in docs/rover-protocol.md.
"""

import math
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import sim_build  # noqa: E402
from rover_link import SimTransport  # noqa: E402
from rover_mission import replay_rover_mission  # noqa: E402
from run_mission import MissionError, run_mission  # noqa: E402

SimulatorUnavailable = sim_build.SimulatorUnavailable


def _number(name: str, value: Any, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a number from {low} to {high}")
    return float(value)


def _angle_difference(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def replay(
    mission: Dict[str, Any],
    slip: float = 0.03,
    gyro_bias: float = 0.01,
    seed: int = 1,
    extra_options: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Returns {"ok", "error", "start", "steps", "summary"}. Each step has the planned pose, the pose the rover
    reported and the true pose in the simulator. If the run fails, ok is False, error says why and the steps
    done so far are still there. Raises ValueError for bad numbers and SimulatorUnavailable if the simulator
    cannot run on this machine.
    """
    slip = _number("slip", slip, 0.0, 0.15)
    gyro_bias = _number("gyro_bias", gyro_bias, -0.5, 0.5)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 1000000:
        raise ValueError("seed must be a whole number from 0 to 1000000")

    binary = sim_build.simulator_binary()
    pose = mission["start_pose"]
    wall = mission["wall"]
    options = ["--start", str(pose["x_mm"]), str(pose["y_mm"]), str(pose["heading_deg"]),
               "--wall", str(wall["width_mm"]), str(wall["height_mm"]),
               "--slip", repr(slip), "--seed", str(seed), "--gyro-bias", repr(gyro_bias)] + list(extra_options or [])

    transport = SimTransport(binary, options)
    steps: List[Dict[str, Any]] = []
    error = None
    try:
        start = transport.truth()

        def on_step(step: Dict[str, Any], reported: Dict[str, float]) -> None:
            truth = transport.truth()
            steps.append({
                "i": step["i"],
                "op": step["op"],
                "expected": {"x": step["x_mm"], "y": step["y_mm"], "heading": step["heading_deg"]},
                "reported": reported,
                "truth": {"x": truth["x"], "y": truth["y"], "heading": truth["heading"]},
            })

        try:
            run_mission(transport, mission, on_step=on_step)
        except (MissionError, ValueError, KeyError, RuntimeError) as exc:
            error = str(exc) if isinstance(exc, MissionError) else repr(exc)
        final = transport.truth()
    finally:
        transport.close()

    planned = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b, spray in replay_rover_mission(mission)["moves"] if spray)
    worst_truth = {"x": 0.0, "y": 0.0, "heading": 0.0}
    worst_reported = {"x": 0.0, "y": 0.0, "heading": 0.0}
    for step in steps:
        for axis in worst_truth:
            diff = step["truth"][axis] - step["expected"][axis]
            if axis == "heading":
                diff = _angle_difference(step["truth"][axis], step["expected"][axis])
            worst_truth[axis] = max(worst_truth[axis], abs(diff))
            diff = step["reported"][axis] - step["expected"][axis]
            if axis == "heading":
                diff = _angle_difference(step["reported"][axis], step["expected"][axis])
            worst_reported[axis] = max(worst_reported[axis], abs(diff))
    return {
        "ok": error is None,
        "error": error,
        "start": {"x": start["x"], "y": start["y"], "heading": start["heading"]},
        "steps": steps,
        "summary": {
            "planned_spray_mm": planned,
            "painted_mm": final["painted"],
            "duration_s": final["t_ms"] / 1000.0,
            "final_truth": {"x": final["x"], "y": final["y"], "heading": final["heading"]},
            "max_truth_error_mm": worst_truth,
            "max_reported_error_mm": worst_reported,
            "slip": slip,
            "gyro_bias_deg_s": gyro_bias,
            "seed": seed,
        },
    }
