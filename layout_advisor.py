"""
Pick the start corner and the stripe direction for a wall from the obstacles in it.

Every layout (4 corners x 2 directions) is planned with the real planner and scored with rules that can
be explained to an operator, not with a learned model:

  1. least wall left unpainted because the rover cannot reach it (a keep-out from wall edge to wall edge
     cuts the wall in two, and only the side the rover starts on is painted),
  2. then least non-painting travel (rounded to 0.1 m),
  3. then fewest turns,
  4. then the shortest total path, and after that the default order (bottom-left, horizontal first).

The rover never drives over a keep-out: the planner paints the wall region by region (region_planner.py).

A layout the rover would refuse is ranked below every layout it accepts, whatever its score: the firmware
answers `NAK TOO_LONG` to a spray MOVE that would run the pump over its limit (about 7 m at 250 mm/s) and
to any MOVE longer than the longest wall side plus 100 mm (docs/rover-protocol.md).

Only the operator's `auto` choices are searched. A fixed corner or direction stays as chosen.
"""

import math
from typing import Any, Dict, List, Optional

from geometry_engine import DEFAULT_DIRECTION, DEFAULT_START_CORNER, DIRECTIONS, START_CORNERS, GeometryEngine

AUTO = "auto"
PUMP_ALLOWANCE_S = 2.0  # the firmware's allowance for the speed ramp and the creep at the end of a spray MOVE
MOVE_MARGIN_MM = 100.0  # a MOVE may be this much longer than the longest wall side
_CORNER_WORDS = {"bottom_left": "bottom-left", "bottom_right": "bottom-right", "top_left": "top-left", "top_right": "top-right"}


def _turns(waypoints: List[Dict[str, Any]]) -> int:
    """Number of changes of travel direction along the path (each one is a TURN step for the rover)."""
    heading = None
    turns = 0
    for p, q in zip(waypoints, waypoints[1:]):
        dx, dy = q["x"] - p["x"], q["y"] - p["y"]
        if math.hypot(dx, dy) < 0.5:
            continue
        new = (1 if dx > 0 else -1, 0) if abs(dx) > abs(dy) else (0, 1 if dy > 0 else -1)
        if heading is not None and new != heading:
            turns += 1
        heading = new
    return turns


def _feasible(waypoints: List[Dict[str, Any]], wall_w_mm: float, wall_h_mm: float,
              max_speed_mps: float, pump_max_s: float) -> bool:
    """True if every MOVE the planner implies is one the firmware accepts."""
    spray_limit = max(0.0, pump_max_s - PUMP_ALLOWANCE_S) * max_speed_mps * 1000.0
    move_limit = max(wall_w_mm, wall_h_mm) + MOVE_MARGIN_MM
    for p, q in zip(waypoints, waypoints[1:]):
        length = math.hypot(q["x"] - p["x"], q["y"] - p["y"])
        if length > move_limit or (p["spray_active"] and q["spray_active"] and length > spray_limit):
            return False
    return True


def evaluate_layouts(
    engine: GeometryEngine,
    wall_w_mm: float,
    wall_h_mm: float,
    spray_width_mm: float,
    overlap_pct: float,
    safety_buffer_mm: float,
    obstacles: List[Dict[str, Any]],
    corners=START_CORNERS,
    directions=DIRECTIONS,
    max_speed_mps: float = 0.25,
    pump_max_s: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Score every corner and direction given. Returns the layouts best first, with a `rank` field and a
    `feasible` field (False if the rover's firmware would refuse a move of this layout).
    """
    options = []
    for order, (direction, corner) in enumerate((d, c) for d in directions for c in corners):
        waypoints, stats = engine.plan_coverage_path(
            wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm,
            [dict(o) for o in obstacles], corner, direction,
        )
        options.append({
            "start_corner": corner,
            "direction": direction,
            "regions": stats["num_regions"],
            "unreached_m2": stats["unreached_area_m2"],
            "transit_m": stats["transit_distance_m"],
            "spray_m": stats["spray_distance_m"],
            "total_m": stats["total_distance_m"],
            "turns": _turns(waypoints),
            "feasible": _feasible(waypoints, wall_w_mm, wall_h_mm, max_speed_mps, pump_max_s),
            "_order": order,
        })
    options.sort(key=lambda o: (not o["feasible"], o["unreached_m2"], round(o["transit_m"], 1), o["turns"], o["total_m"], o["_order"]))
    for rank, option in enumerate(options, 1):
        option["rank"] = rank
        del option["_order"]
    return options


def describe(option: Dict[str, Any]) -> str:
    regions = option["regions"]
    text = (f"{option['direction']} stripes from the {_CORNER_WORDS[option['start_corner']]} corner: "
            f"{regions} region{'' if regions == 1 else 's'}, "
            f"{option['transit_m']:.1f} m of non-painting travel, {option['turns']} turns")
    if option["unreached_m2"] > 0:
        text += f", {option['unreached_m2']:.2f} m² cut off by a keep-out and left unpainted"
    return text


def choose_layout(
    engine: GeometryEngine,
    wall_w_mm: float,
    wall_h_mm: float,
    spray_width_mm: float,
    overlap_pct: float,
    safety_buffer_mm: float,
    obstacles: List[Dict[str, Any]],
    start_corner: Optional[str],
    direction: Optional[str],
    max_speed_mps: float = 0.25,
    pump_max_s: float = 30.0,
) -> Dict[str, Any]:
    """
    start_corner and direction are a value, "auto", or missing (the default). Returns the chosen
    layout, the scored options that were searched, the mode ("auto" if anything was searched) and a
    one-sentence reason.
    """
    auto_corner = start_corner == AUTO
    auto_direction = direction == AUTO
    corner = DEFAULT_START_CORNER if auto_corner else engine.parse_start_corner(start_corner)
    stripes = DEFAULT_DIRECTION if auto_direction else engine.parse_direction(direction)
    corners = START_CORNERS if auto_corner else (corner,)
    directions = DIRECTIONS if auto_direction else (stripes,)

    options = evaluate_layouts(engine, wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm,
                               obstacles, corners, directions, max_speed_mps, pump_max_s)
    best = options[0]
    mode = "auto" if (auto_corner or auto_direction) else "manual"
    if not best["feasible"]:
        reason = ("No layout fits the rover's spray limit (a stripe this long would run the pump too long or "
                  "exceed the wall size); the least bad one is " + describe(best) + ".")
    elif mode == "manual":
        reason = "Layout chosen by the operator: " + describe(best) + "."
    elif len(options) == 1:
        reason = "Only one layout was possible: " + describe(best) + "."
    else:
        worst = options[-1]
        reason = (f"Best of {len(options)} layouts: {describe(best)}. "
                  f"The worst was {describe(worst)}.")
    return {"mode": mode, "start_corner": best["start_corner"], "direction": best["direction"],
            "options": options, "reason": reason}
