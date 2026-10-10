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
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from rover_link import SimTransport, parse_line, tenths  # noqa: E402
from rover_mission import verify_checksum  # noqa: E402

PROTOCOL_VERSION = 1
WATCHDOG_MS = 500  # the rover faults after this much silence
MIN_WAIT_MS = 10  # the simulator advances in 10 ms ticks
MAX_WAIT_MS = WATCHDOG_MS // 2  # a PING must reach the rover at least twice per watchdog period


class MissionError(Exception):
    """The rover refused a command, reported a fault, or did not finish."""


def _parsed(lines: List[str]) -> List[Tuple[str, List[str]]]:
    """Parse the rover's lines. A line with a bad CRC is line noise: skip it."""
    out = []
    for line in lines:
        try:
            out.append(parse_line(line))
        except ValueError:
            continue
    return out


def _expect_ack(lines: List[str], what: str, step_id: Optional[int] = None) -> None:
    """A FAULT in the reply beats a NAK: after an e-stop the rover answers FAULT ESTOP, STATE FAULT, NAK BADSTATE."""
    fault = nak = None
    acked = False
    for word, args in _parsed(lines):
        if word == "FAULT" and args and fault is None:
            fault = args[0]
        elif word == "NAK" and nak is None:
            nak = " ".join(args)
        elif word == "ACK" and (step_id is None or args[:1] == [str(step_id)]):
            acked = True
    if fault is not None:
        raise MissionError(f"{what}: rover fault {fault}" + (f" (NAK {nak})" if nak else ""))
    if nak is not None:
        raise MissionError(f"{what} refused: NAK {nak}")
    if not acked:
        raise MissionError(f"{what}: no ACK received")


def _hello(transport: Any) -> Dict[str, str]:
    for word, args in _parsed(transport.send("HELLO")):
        if word == "HELLO":
            return dict(arg.split("=", 1) for arg in args if "=" in arg)
    raise MissionError("no HELLO reply from the rover")


def _number(text: str, what: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise MissionError(f"rover sent {text!r} where {what} was expected") from None


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
    Raises MissionError on any NAK, FAULT, unusable reply or timeout. Sends PING after every
    wait (wait_ms must be 10 to 250 ms) so the rover's 500 ms watchdog stays fed. If the run
    fails after the handshake, a best-effort STOP is sent so the motors do not wait for the
    watchdog.
    """
    if isinstance(wait_ms, bool) or not isinstance(wait_ms, int) or not MIN_WAIT_MS <= wait_ms <= MAX_WAIT_MS:
        raise MissionError(f"wait_ms must be {MIN_WAIT_MS} to {MAX_WAIT_MS} ms to keep the {WATCHDOG_MS} ms watchdog fed")
    if not verify_checksum(mission):
        raise MissionError("mission checksum does not match its content")

    hello = _hello(transport)
    if hello.get("proto") != str(PROTOCOL_VERSION):
        raise MissionError(f"rover speaks protocol {hello.get('proto')}, expected {PROTOCOL_VERSION}")
    speed_mm_s = mission["speed_mps"] * 1000.0
    max_speed = _number(hello.get("max_speed_mm_s", ""), "max_speed_mm_s")
    if round(speed_mm_s, 1) > max_speed:
        raise MissionError(f"mission speed {speed_mm_s:.0f} mm/s is above the rover's {hello['max_speed_mm_s']} mm/s")
    pump_limit = mission["profile"].get("pump_max_continuous_s")
    if pump_limit is not None and pump_limit > _number(hello.get("pump_max_s", ""), "pump_max_s"):
        raise MissionError("the profile allows a longer pump run than the firmware does")

    try:
        return _run_steps(transport, mission, speed_mm_s, wait_ms, max_step_ms, on_step)
    except Exception:
        try:
            transport.send("STOP")
        except Exception:
            pass
        raise


def _run_steps(
    transport: Any,
    mission: Dict[str, Any],
    speed_mm_s: float,
    wait_ms: int,
    max_step_ms: int,
    on_step: Optional[Callable[[Dict[str, Any], Dict[str, float]], None]],
) -> List[Dict[str, Any]]:

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
        _expect_ack(transport.send(_step_payload(step)), f"step {step_id}", step_id)
        waited = 0
        done: Optional[Dict[str, float]] = None
        while done is None:
            if waited >= max_step_ms:
                raise MissionError(f"step {step_id} did not finish within {max_step_ms} ms")
            lines = transport.wait(wait_ms) + transport.send("PING")
            waited += wait_ms
            for word, args in _parsed(lines):
                if word == "FAULT" and args:
                    raise MissionError(f"rover fault during step {step_id}: {args[0]}")
                if word == "DONE" and args[:1] == [str(step_id)]:
                    if len(args) != 4:
                        raise MissionError(f"step {step_id}: malformed DONE {' '.join(args)}")
                    done = {"x": _number(args[1], "x"), "y": _number(args[2], "y"), "heading": _number(args[3], "heading")}
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
    parser.add_argument("--sim-option", action="append", default=[],
                        help="extra rover_sim option, repeatable; use the = form for a value that starts with '-', "
                             "for example --sim-option=--slip --sim-option=0.03")
    args = parser.parse_args(argv)

    with open(args.mission, encoding="utf-8") as f:
        mission = json.load(f)
    pose = mission["start_pose"]
    options = ["--start", str(pose["x_mm"]), str(pose["y_mm"]), str(pose["heading_deg"]),
               "--wall", str(mission["wall"]["width_mm"]), str(mission["wall"]["height_mm"])] + args.sim_option
    transport = None
    try:
        transport = SimTransport(args.sim, options)
        records = run_mission(transport, mission)
        truth = transport.truth()
    except (MissionError, ValueError, KeyError, RuntimeError, OSError) as error:
        print(f"mission failed: {error if isinstance(error, MissionError) else repr(error)}", file=sys.stderr)
        return 1
    finally:
        if transport is not None:
            transport.close()

    errors = pose_errors(records)
    print(f"{len(records)} steps done. Largest reported-versus-expected pose difference: "
          f"x {errors['x']:.1f} mm, y {errors['y']:.1f} mm, heading {errors['heading']:.1f} deg")
    print(f"True final pose in the simulator: x {truth['x']:.1f} mm, y {truth['y']:.1f} mm, heading {truth['heading']:.1f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
