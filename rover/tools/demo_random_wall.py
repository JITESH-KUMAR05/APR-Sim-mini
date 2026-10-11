"""
End-to-end check of the demo on random walls, without the web page: draw a wall with known obstacles, detect
them with the app's own detector, choose the layout, plan and compile the mission, and optionally replay it in
the rover simulator. Prints, for every wall and in total, what matters for painting safely:

  coverage       share of the planted obstacle area that the detected keep-out zones cover
  false zone     share of the detected keep-out area that covers nothing planted
  paint on       millimetres of planned spray that run over a planted obstacle (should be 0)
  planted/found  obstacles planted, and how many have at least half their area covered

    .venv/Scripts/python.exe rover/tools/demo_random_wall.py --count 20 --seed 100
"""

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometry_engine import GeometryEngine  # noqa: E402
from layout_advisor import choose_layout  # noqa: E402
from obstacle_detector import load_default_detector  # noqa: E402
from rover_mission import compile_rover_mission, load_profile  # noqa: E402
from wall_synth import random_scene  # noqa: E402

CELL_MM = 10.0


def _mask(boxes: List[Dict[str, Any]], width_mm: float, height_mm: float) -> np.ndarray:
    mask = np.zeros((int(math.ceil(height_mm / CELL_MM)), int(math.ceil(width_mm / CELL_MM))), bool)
    for b in boxes:
        x1, x2 = int(max(0, b["x"]) // CELL_MM), int(min(width_mm, b["x"] + b["w"]) // CELL_MM) + 1
        y1, y2 = int(max(0, b["y"]) // CELL_MM), int(min(height_mm, b["y"] + b["h"]) // CELL_MM) + 1
        mask[y1:y2, x1:x2] = True
    return mask


def coverage(truth, detected, width_mm: float, height_mm: float) -> Dict[str, float]:
    """Area shares. coverage is 1.0 when there is nothing to cover; false_zone is 0.0 when nothing was detected."""
    t, d = _mask(truth, width_mm, height_mm), _mask(detected, width_mm, height_mm)
    both = (t & d).sum()
    found = 0
    for box in truth:
        m = _mask([box], width_mm, height_mm)
        if m.sum() and (m & d).sum() >= 0.5 * m.sum():
            found += 1
    return {
        "coverage": 1.0 if t.sum() == 0 else float(both / t.sum()),
        "false_zone": 0.0 if d.sum() == 0 else float((d & ~t).sum() / d.sum()),
        "planted": len(truth),
        "found": found,
    }


def paint_on_obstacles(waypoints: List[Dict[str, Any]], truth: List[Dict[str, Any]]) -> float:
    """Millimetres of planned spray that run over a planted obstacle, sampled every 5 mm."""
    over = 0.0
    for p, q in zip(waypoints, waypoints[1:]):
        if not (p["spray_active"] and q["spray_active"]):
            continue
        length = math.hypot(q["x"] - p["x"], q["y"] - p["y"])
        steps = max(1, int(length // 5))
        for i in range(steps):
            t = (i + 0.5) / steps
            x, y = p["x"] + t * (q["x"] - p["x"]), p["y"] + t * (q["y"] - p["y"])
            if any(b["x"] < x < b["x"] + b["w"] and b["y"] < y < b["y"] + b["h"] for b in truth):
                over += length / steps
    return over


def run_one(engine: GeometryEngine, seed: int, simulate: bool, sensitivity: float = 0.5) -> Dict[str, Any]:
    image, width_mm, height_mm, truth = random_scene(seed)
    detection = engine.detect(image, width_mm, height_mm, sensitivity)
    obstacles = detection["obstacles"]
    layout = choose_layout(engine, width_mm, height_mm, 250.0, 12.0, 60.0, obstacles, "auto", "auto")
    waypoints, stats = engine.plan_coverage_path(
        width_mm, height_mm, 250.0, 12.0, 60.0, [dict(o) for o in obstacles], layout["start_corner"], layout["direction"]
    )
    result = coverage(truth, obstacles, width_mm, height_mm)
    result.update({
        "seed": seed, "detector": detection["detector"], "wall": (width_mm, height_mm),
        "layout": f"{layout['direction']} {layout['start_corner']}", "zones": len(obstacles),
        "paint_on_mm": paint_on_obstacles(waypoints, truth),
    })
    if simulate:
        from sim_replay import replay

        mission = compile_rover_mission(waypoints, width_mm, height_mm, 250.0, stats["effective_step_mm"],
                                        layout["start_corner"], load_profile())
        run = replay(mission)
        result["sim_ok"] = run["ok"]
        result["painted_share"] = run["summary"]["painted_mm"] / max(1.0, run["summary"]["planned_spray_mm"])
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--no-sim", action="store_true", help="skip the rover simulator replay")
    parser.add_argument("--sensitivity", type=float, default=0.5)
    args = parser.parse_args(argv)

    detector = load_default_detector()
    engine = GeometryEngine(detector=detector)
    print("detector:", "trained model" if detector else "classical fallback (no model file)")
    rows = []
    for seed in range(args.seed, args.seed + args.count):
        row = run_one(engine, seed, not args.no_sim, args.sensitivity)
        rows.append(row)
        sim = "" if args.no_sim else f"  sim {'ok' if row['sim_ok'] else 'FAILED'} painted {row['painted_share']:.2f}"
        print(f"seed {seed:>6}  {row['planted']:>2} planted, {row['found']:>2} found  coverage {row['coverage']:.2f}  "
              f"false zone {row['false_zone']:.2f}  paint on obstacles {row['paint_on_mm']:.0f} mm  {row['layout']}{sim}")
    n = len(rows)
    planted = sum(r["planted"] for r in rows)
    print(f"\n{n} walls: {sum(r['found'] for r in rows)} of {planted} obstacles found, "
          f"mean coverage {sum(r['coverage'] for r in rows) / n:.2f}, mean false zone {sum(r['false_zone'] for r in rows) / n:.2f}, "
          f"walls with paint on an obstacle: {sum(1 for r in rows if r['paint_on_mm'] > 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
