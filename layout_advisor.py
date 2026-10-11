"""
Pick the start corner and the stripe direction for a wall from the obstacles in it.

Every layout (4 corners x 2 directions) is planned with the real planner and scored with rules that can
be explained to an operator, not with a learned model:

  1. fewest obstacle crossings: stretches where the rover must drive across a no-paint zone with spray off,
  2. then least non-painting travel (rounded to 0.1 m),
  3. then fewest turns,
  4. then the shortest total path, and after that the default order (bottom-left, horizontal first).

Only the operator's `auto` choices are searched. A fixed corner or direction stays as chosen.
"""

import math
from typing import Any, Dict, List, Optional

from geometry_engine import DEFAULT_DIRECTION, DEFAULT_START_CORNER, DIRECTIONS, START_CORNERS, GeometryEngine

AUTO = "auto"
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
) -> List[Dict[str, Any]]:
    """Score every corner and direction given. Returns the layouts best first, with a `rank` field."""
    options = []
    for order, (direction, corner) in enumerate((d, c) for d in directions for c in corners):
        waypoints, stats = engine.plan_coverage_path(
            wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm,
            [dict(o) for o in obstacles], corner, direction,
        )
        options.append({
            "start_corner": corner,
            "direction": direction,
            "crossings": sum(1 for wp in waypoints if wp["type"] == "TRANSIT_START"),
            "transit_m": stats["transit_distance_m"],
            "spray_m": stats["spray_distance_m"],
            "total_m": stats["total_distance_m"],
            "turns": _turns(waypoints),
            "_order": order,
        })
    options.sort(key=lambda o: (o["crossings"], round(o["transit_m"], 1), o["turns"], o["total_m"], o["_order"]))
    for rank, option in enumerate(options, 1):
        option["rank"] = rank
        del option["_order"]
    return options


def describe(option: Dict[str, Any]) -> str:
    crossings = option["crossings"]
    return (f"{option['direction']} stripes from the {_CORNER_WORDS[option['start_corner']]} corner: "
            f"{crossings} obstacle crossing{'' if crossings == 1 else 's'}, "
            f"{option['transit_m']:.1f} m of non-painting travel, {option['turns']} turns")


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
                               obstacles, corners, directions)
    best = options[0]
    mode = "auto" if (auto_corner or auto_direction) else "manual"
    if mode == "manual":
        reason = "Layout chosen by the operator: " + describe(best) + "."
    elif len(options) == 1:
        reason = "Only one layout was possible: " + describe(best) + "."
    else:
        worst = options[-1]
        reason = (f"Best of {len(options)} layouts: {describe(best)}. "
                  f"The worst was {describe(worst)}.")
    return {"mode": mode, "start_corner": best["start_corner"], "direction": best["direction"],
            "options": options, "reason": reason}
