"""
Compile planner waypoints into rover steps (schema apr-rover-mission/1).

The planner emits absolute waypoints in the wall frame (millimetres, origin bottom-left,
x right, y up). A differential-drive rover that can turn on the spot runs simpler steps:

    MOVE        drive mm in the current heading, spray 0 or 1
    TURN        rotate deg on the spot, positive is counter-clockwise
    EDGE_ALIGN  touch the wall edge on the given axis and reset that coordinate to value_mm

TURN and EDGE_ALIGN always run with the spray off. Headings are degrees counter-clockwise
from +x. Every step carries the pose the rover is expected to have after it, so a console
can compare planned and measured position. replay_rover_mission() recomputes the path from
the steps alone, which is what the tests use to prove the compiled mission matches the planner.

File contract

- x_mm/y_mm and start_pose are the position of the spray nozzle centre in the wall frame.
  start_pose is where the operator places the rover (nozzle centre) and the direction it faces.
- EDGE_ALIGN: the rover drives slowly along its current heading until its edge sensor triggers
  (firmware faults if it does not trigger within a firmware-defined distance), then sets the
  coordinate on `axis` to value_mm. The step's own x_mm/y_mm are the expected pose BEFORE that
  reset. The offset between the edge sensor and the nozzle centre belongs to the rover profile
  (added in plan P2).
- speed_mps is the speed MOVE steps are driven at, spray on or off, and equals the profile's
  max_speed_mps.
- Obstacles are no-paint zones, not no-go zones: a MOVE with spray 0 may cross a keep-out box,
  including at the start corner where the first pass can begin inside one.
- Checksum: sha256 (hex) of the UTF-8 bytes of Python
  json.dumps(body, sort_keys=True, separators=(",", ":")), where body is the mission dict without
  its `checksum` key. Float text is Python's repr (for example 0.0, 90.0), so a verifier in
  another language must reproduce that formatting. A language-neutral canonical form is to be
  decided in the P2 spec, while the schema is still /1 with no consumers.
"""

import hashlib
import json
import math
import os
from typing import Any, Dict, List

SCHEMA = "apr-rover-mission/1"
PROFILE_SCHEMA = "apr-rover-profile/1"
DEFAULT_PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rover_profile.json")

_EPS_MM = 0.5  # shorter moves are planner join points, not motion


def load_profile(path: str = DEFAULT_PROFILE_PATH) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        profile = json.load(f)
    if not isinstance(profile, dict) or profile.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"rover profile must have schema {PROFILE_SCHEMA}")
    for key in ("wheel_diameter_mm", "max_speed_mps"):
        value = profile.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"rover profile {key} must be a positive number")
    if profile["max_speed_mps"] > 0.5:
        raise ValueError("rover profile max_speed_mps must be at most 0.5")
    return profile


def _normalize_heading(deg: float) -> float:
    deg = math.fmod(deg, 360.0)
    if deg > 180.0:
        deg -= 360.0
    elif deg <= -180.0:
        deg += 360.0
    return deg


def _heading_of(dx: float, dy: float) -> float:
    if abs(dx) > _EPS_MM and abs(dy) > _EPS_MM:
        raise ValueError("planner segment is not axis-aligned")
    if abs(dx) > _EPS_MM:
        return 0.0 if dx > 0 else 180.0
    return 90.0 if dy > 0 else -90.0


def _checksum(body: Dict[str, Any]) -> str:
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_checksum(mission: Dict[str, Any]) -> bool:
    body = {key: value for key, value in mission.items() if key != "checksum"}
    return mission.get("checksum") == _checksum(body)


def compile_rover_mission(
    waypoints: List[Dict[str, Any]],
    wall_w_mm: float,
    wall_h_mm: float,
    spray_width_mm: float,
    row_pitch_mm: float,
    start_corner: str,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    segments = []
    for p, q in zip(waypoints, waypoints[1:]):
        if math.hypot(q["x"] - p["x"], q["y"] - p["y"]) > _EPS_MM:
            segments.append((p, q, bool(p.get("spray_active")) and bool(q.get("spray_active"))))
    if not segments:
        raise ValueError("mission has no motion segments")

    x, y = segments[0][0]["x"], segments[0][0]["y"]
    heading = _heading_of(segments[0][1]["x"] - x, segments[0][1]["y"] - y)
    start_pose = {"x_mm": x, "y_mm": y, "heading_deg": heading}

    steps: List[Dict[str, Any]] = []
    for _, q, spray in segments:
        target = _heading_of(q["x"] - x, q["y"] - y)
        turn = _normalize_heading(target - heading)
        if abs(turn) > 0.01:
            heading = target
            steps.append({"i": len(steps) + 1, "op": "TURN", "deg": turn,
                          "x_mm": x, "y_mm": y, "heading_deg": heading})
        # Distance is the projection on the move axis and only that coordinate is updated, so the
        # stored pose is exactly what replay_rover_mission() computes from the steps.
        horizontal = heading in (0.0, 180.0)
        distance = round(abs(q["x"] - x) if horizontal else abs(q["y"] - y), 3)
        if horizontal:
            x = q["x"]
        else:
            y = q["y"]
        steps.append({"i": len(steps) + 1, "op": "MOVE", "mm": distance, "spray": 1 if spray else 0,
                      "x_mm": x, "y_mm": y, "heading_deg": heading})
        if heading in (0.0, 180.0) and (abs(x) <= _EPS_MM or abs(x - wall_w_mm) <= _EPS_MM):
            edge = 0.0 if abs(x) <= _EPS_MM else wall_w_mm
            steps.append({"i": len(steps) + 1, "op": "EDGE_ALIGN", "axis": "x", "value_mm": edge,
                          "x_mm": x, "y_mm": y, "heading_deg": heading})

    body = {
        "schema": SCHEMA,
        "profile": dict(profile),
        "wall": {"width_mm": wall_w_mm, "height_mm": wall_h_mm, "start_corner": start_corner},
        "spray": {"width_mm": spray_width_mm, "row_pitch_mm": row_pitch_mm},
        "speed_mps": profile["max_speed_mps"],
        "start_pose": start_pose,
        "steps": steps,
    }
    body["checksum"] = _checksum(body)
    return body


def replay_rover_mission(mission: Dict[str, Any]) -> Dict[str, Any]:
    pose = mission["start_pose"]
    x, y, heading = float(pose["x_mm"]), float(pose["y_mm"]), float(pose["heading_deg"])
    moves, poses = [], []
    for step in mission["steps"]:
        op = step["op"]
        if op == "TURN":
            heading = _normalize_heading(heading + step["deg"])
        elif op == "MOVE":
            radians = math.radians(heading)
            nx = x + step["mm"] * math.cos(radians)
            ny = y + step["mm"] * math.sin(radians)
            moves.append(((x, y), (nx, ny), bool(step["spray"])))
            x, y = nx, ny
        elif op != "EDGE_ALIGN":
            raise ValueError(f"unknown rover step: {op}")
        poses.append((x, y, heading))
    return {"moves": moves, "poses": poses}
