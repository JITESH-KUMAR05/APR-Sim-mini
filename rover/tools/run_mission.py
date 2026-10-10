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
