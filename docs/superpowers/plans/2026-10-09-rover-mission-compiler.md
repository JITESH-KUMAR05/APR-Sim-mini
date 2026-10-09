# Rover Mission Compiler (P1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the operator pick the rover's start corner, and turn the planner's waypoints into a rover-executable mission file (`apr-rover-mission/1`) that the app can download, with honest export parameters.

**Architecture:** `GeometryEngine.plan_coverage_path` gains a `start_corner` argument and mirrors its existing top-left plan to the chosen corner. A new, framework-free module `rover_mission.py` compiles waypoints into `MOVE` / `TURN` / `EDGE_ALIGN` steps and can replay them, so a test proves the compiled steps reproduce the planner's path. `app.py` wires the start corner and writes the new export; the page gets a start-corner dropdown and a download button.

**Tech Stack:** Python 3.12, Flask, unittest, vanilla JS. No new dependencies.

Spec: `docs/superpowers/specs/2026-10-09-rover-hardware-integration-design.md` (sections 2 and 6). This is plan P1 of that spec. It needs no rover hardware.

## Global Constraints

- Wall frame: origin bottom-left, millimetres, x right, y up. Headings are degrees counter-clockwise from +x.
- Start corners, in this exact spelling: `bottom_left`, `bottom_right`, `top_left`, `top_right`. Default `bottom_left` (the rover's (0,0)).
- Mission schema string: `apr-rover-mission/1`. Profile schema string: `apr-rover-profile/1`.
- Mission steps are only `MOVE` (millimetres, spray 0 or 1), `TURN` (degrees, positive is counter-clockwise) and `EDGE_ALIGN` (axis `x`, value in millimetres). `TURN` and `EDGE_ALIGN` always run with spray off.
- Replaying a compiled mission must reproduce the planner's motion segments within 1 mm (the tests demand 0.01 mm).
- Never add PyTorch or ultralytics to `requirements.txt` or `pyproject.toml`. This plan adds no dependencies at all.
- AI output never reaches motor or valve commands. Nothing in this plan touches detection.
- Existing exports (DXF, OBJ, MTL, ZIP, mission JSON, waypoints CSV) keep working. The mission JSON and CSV keep their current keys and columns.
- The app has no browser-automation tool available. Frontend changes are verified by `node --check`, Flask test-client checks that the served HTML and script contain the new pieces, and a human checklist (Task 6).
- Tests run with `uv run python -m unittest discover -s tests -p "<file>" -v`. Baseline before this plan: 88 tests pass, 1 skipped.
- Commit trailer on every commit: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`, exactly this text, in its own paragraph. Use two `-m` arguments as in the commit commands below (a trailer glued to the subject line is not a trailer). After committing run `git log -1 --format='%(trailers:key=Co-Authored-By)'`: it must print the trailer line.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `geometry_engine.py` | Modify | `START_CORNERS`, `DEFAULT_START_CORNER`, `parse_start_corner`, `plan_coverage_path(..., start_corner)`; old planner body renamed `_plan_coverage_from_top_left` |
| `rover_profile.json` | Create | Rover constants shared by the compiler and (later) firmware |
| `rover_mission.py` | Create | `load_profile`, `compile_rover_mission`, `replay_rover_mission`, `verify_checksum` |
| `cad_exporter.py` | Modify | `export_mission_json` writes the real spray width, overlap, buffer, speed and start corner |
| `app.py` | Modify | Start corner in `/api/analyze` and `/api/replan`, writes `apr_rover_mission.json`, download type `rover` |
| `templates/index.html`, `static/app.js`, `static/style.css` | Modify | Start-corner dropdown, Rover Mission download button, honest start label |
| `tests/test_start_corner.py`, `tests/test_rover_mission.py`, `tests/test_mission_manifest.py`, `tests/test_rover_ui.py` | Create | New tests |
| `tests/test_api.py` | Modify | API tests for start corner, rover mission download, honest exports |
| `docs/superpowers/specs/2026-10-09-rover-hardware-integration-design.md` | Modify | Align section 2 with what P1 builds |

---

### Task 1: Planner start corner

**Files:**
- Modify: `geometry_engine.py` (module constants after `LEGACY_OBSTACLE_INFO`; a static method and the planner wrapper in `GeometryEngine`)
- Test: `tests/test_start_corner.py`

**Interfaces:**
- Produces: `geometry_engine.START_CORNERS: tuple[str, ...]`; `geometry_engine.DEFAULT_START_CORNER = "bottom_left"`; `GeometryEngine.parse_start_corner(raw) -> str` (static; `None` or `""` returns the default; anything else not in `START_CORNERS` raises `ValueError`); `GeometryEngine.plan_coverage_path(wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles, start_corner="bottom_left") -> (waypoints, stats)`, same waypoint dict shape as today.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_start_corner.py`:

```python
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry_engine import DEFAULT_START_CORNER, START_CORNERS, GeometryEngine

WALL_W, WALL_H = 4000.0, 2800.0
# Deliberately not symmetric about the wall centre, so a wrong mirror would put it somewhere else.
BOX = {"id": "obs_1", "type": "window", "label": "Window #1", "x": 500.0, "y": 300.0, "w": 700.0, "h": 500.0}


def plan(corner, obstacles=()):
    return GeometryEngine().plan_coverage_path(WALL_W, WALL_H, 250.0, 12.0, 60.0, list(obstacles), corner)


class TestStartCorner(unittest.TestCase):
    def test_default_is_bottom_left_with_the_first_row_near_the_floor(self):
        self.assertEqual(DEFAULT_START_CORNER, "bottom_left")
        waypoints, _ = GeometryEngine().plan_coverage_path(WALL_W, WALL_H, 250.0, 12.0, 60.0, [])
        first = waypoints[0]
        self.assertEqual((first["x"], first["y"]), (0.0, 125.0))
        self.assertEqual(first["type"], "PASS_START")

    def test_each_corner_starts_in_that_corner_and_the_first_pass_crosses_the_wall(self):
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                waypoints, _ = plan(corner)
                vertical, horizontal = corner.split("_")
                start_x = 0.0 if horizontal == "left" else WALL_W
                start_y = 125.0 if vertical == "bottom" else WALL_H - 125.0
                self.assertEqual((waypoints[0]["x"], waypoints[0]["y"]), (start_x, start_y))
                self.assertEqual(waypoints[1]["x"], WALL_W - start_x)

    def test_rows_progress_away_from_the_start_side(self):
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                waypoints, stats = plan(corner)
                row_y = [next(w["y"] for w in waypoints if w["row"] == r) for r in range(1, stats["num_rows"] + 1)]
                steps = [b - a for a, b in zip(row_y, row_y[1:])]
                if corner.startswith("bottom"):
                    self.assertTrue(all(s > 0 for s in steps), row_y)
                else:
                    self.assertTrue(all(s < 0 for s in steps), row_y)

    def test_distances_do_not_depend_on_the_corner_for_a_clear_wall(self):
        reference = plan("top_left")[1]
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                self.assertEqual(plan(corner)[1], reference)

    def test_spray_passes_avoid_the_box_at_its_real_position_from_every_corner(self):
        buffer_mm = 60.0
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                waypoints, _ = plan(corner, [BOX])
                rows_aligned_with_the_box = 0
                for start, end in zip(waypoints, waypoints[1:]):
                    if start["type"] == "PASS_START" and end["type"] == "PASS_END":
                        if BOX["y"] - buffer_mm <= start["y"] <= BOX["y"] + BOX["h"] + buffer_mm:
                            rows_aligned_with_the_box += 1
                            low, high = sorted((start["x"], end["x"]))
                            crosses = low < BOX["x"] + BOX["w"] + buffer_mm - 0.5 and high > BOX["x"] - buffer_mm + 0.5
                            self.assertFalse(crosses, f"{corner}: pass {low}..{high} crosses the box")
                self.assertGreater(rows_aligned_with_the_box, 0)

    def test_the_obstacle_dicts_passed_in_are_not_modified(self):
        before = dict(BOX)
        plan("top_right", [BOX])
        self.assertEqual(BOX, before)

    def test_parse_start_corner(self):
        self.assertEqual(GeometryEngine.parse_start_corner(None), "bottom_left")
        self.assertEqual(GeometryEngine.parse_start_corner(""), "bottom_left")
        self.assertEqual(GeometryEngine.parse_start_corner("top_right"), "top_right")
        for bad in ("middle", "TOP_RIGHT", 5, ["top_left"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                GeometryEngine.parse_start_corner(bad)

    def test_planner_rejects_an_unknown_corner(self):
        with self.assertRaises(ValueError):
            plan("middle")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m unittest discover -s tests -p "test_start_corner.py" -v`
Expected: ERROR `ImportError: cannot import name 'DEFAULT_START_CORNER' from 'geometry_engine'`.

- [ ] **Step 3: Implement**

In `geometry_engine.py`, add these two constants directly after the `LEGACY_OBSTACLE_INFO = { ... }` block (before `class GeometryEngine:`):

```python
START_CORNERS = ("bottom_left", "bottom_right", "top_left", "top_right")
DEFAULT_START_CORNER = "bottom_left"  # the rover's (0,0)
```

Add this static method inside `GeometryEngine`, directly above the existing `def plan_coverage_path(`:

```python
    @staticmethod
    def parse_start_corner(raw: Any) -> str:
        """Return a valid start corner. A missing or empty value means the default (bottom-left)."""
        if raw is None or raw == "":
            return DEFAULT_START_CORNER
        if not isinstance(raw, str) or raw not in START_CORNERS:
            raise ValueError(f"start_corner must be one of {', '.join(START_CORNERS)}")
        return raw

    def plan_coverage_path(
        self,
        wall_w_mm: float,
        wall_h_mm: float,
        spray_width_mm: float,
        overlap_pct: float,
        safety_buffer_mm: float,
        obstacles: List[Dict[str, Any]],
        start_corner: str = DEFAULT_START_CORNER
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Boustrophedon coverage that starts in the given wall corner: rows progress away from
        the start side and the first pass runs from the start corner across the wall.
        The path is planned from the top-left and mirrored, so obstacles are mirrored into
        that frame first (copies, the caller's dicts are not changed).
        """
        start_corner = self.parse_start_corner(start_corner)
        flip_x = start_corner.endswith("right")
        flip_y = start_corner.startswith("bottom")
        if not (flip_x or flip_y):
            return self._plan_coverage_from_top_left(
                wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles
            )

        mirrored_obstacles = []
        for obs in obstacles:
            mirrored = dict(obs)
            if flip_x:
                mirrored["x"] = wall_w_mm - (obs["x"] + obs["w"])
            if flip_y:
                mirrored["y"] = wall_h_mm - (obs["y"] + obs["h"])
            mirrored_obstacles.append(mirrored)

        waypoints, stats = self._plan_coverage_from_top_left(
            wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, mirrored_obstacles
        )
        for wp in waypoints:
            if flip_x:
                wp["x"] = round(wall_w_mm - wp["x"], 1)
            if flip_y:
                wp["y"] = round(wall_h_mm - wp["y"], 1)
        return waypoints, stats
```

Then rename the old planner. Replace this text (the old definition line, signature and the first docstring line):

```python
    def plan_coverage_path(
        self,
        wall_w_mm: float,
        wall_h_mm: float,
        spray_width_mm: float,
        overlap_pct: float,
        safety_buffer_mm: float,
        obstacles: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate continuous Boustrophedon (serpentine zig-zag) coverage trajectory
```

with:

```python
    def _plan_coverage_from_top_left(
        self,
        wall_w_mm: float,
        wall_h_mm: float,
        spray_width_mm: float,
        overlap_pct: float,
        safety_buffer_mm: float,
        obstacles: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate continuous Boustrophedon (serpentine zig-zag) coverage trajectory
```

(Only the method name changes. Leave the rest of the body as it is. After this edit `grep -n "def plan_coverage_path" geometry_engine.py` must show exactly one match, the new wrapper.)

- [ ] **Step 4: Run the tests**

Run: `uv run python -m unittest discover -s tests -p "test_start_corner.py" -v`
Expected: all PASS.

Run: `uv run python -m unittest discover -s tests -p "test_*.py" -v`
Expected: all PASS, 1 skipped (existing planner callers use the default corner and their assertions do not depend on the corner).

- [ ] **Step 5: Commit**

```bash
git add geometry_engine.py tests/test_start_corner.py
git commit -m "Add start corner to the coverage planner" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git log -1 --format='%(trailers:key=Co-Authored-By)'
```

---

### Task 2: Rover profile and mission compiler

**Files:**
- Create: `rover_profile.json`, `rover_mission.py`
- Modify: `tests/fakes.py` (shared `planner_moves` helper, also used by `tests/test_api.py` in Task 4)
- Test: `tests/test_rover_mission.py`

**Interfaces:**
- Consumes: waypoint dicts from `GeometryEngine.plan_coverage_path` (keys `x`, `y`, `spray_active`, `type`, `row`), and `stats["effective_step_mm"]`.
- Produces: `rover_mission.SCHEMA = "apr-rover-mission/1"`; `load_profile(path=DEFAULT_PROFILE_PATH) -> dict` (raises `ValueError`); `compile_rover_mission(waypoints, wall_w_mm, wall_h_mm, spray_width_mm, row_pitch_mm, start_corner, profile) -> dict` (raises `ValueError`); `replay_rover_mission(mission) -> {"moves": [((x0, y0), (x1, y1), spray_bool), ...], "poses": [(x, y, heading), ...]}`; `verify_checksum(mission) -> bool`. Mission dict keys: `schema`, `profile`, `wall` (`width_mm`, `height_mm`, `start_corner`), `spray` (`width_mm`, `row_pitch_mm`), `speed_mps`, `start_pose` (`x_mm`, `y_mm`, `heading_deg`), `steps`, `checksum`. Every step has `i`, `op`, and the expected `x_mm`, `y_mm`, `heading_deg` after it.

- [ ] **Step 1: Create the profile**

Create `rover_profile.json`. The 80 mm wheel and 0.25 m/s come from section 4 of the spec. They are initial design values: update the file when the real parts are chosen.

```json
{
  "schema": "apr-rover-profile/1",
  "name": "apr-floor-demo",
  "drive": "differential-turn-in-place",
  "wheel_diameter_mm": 80.0,
  "max_speed_mps": 0.25
}
```

- [ ] **Step 2: Add the shared test helper, then write the failing tests**

In `tests/fakes.py`, add `import math` as the first line (the file has no imports yet, leave a blank line after it) and append at the end of the file:

```python


def planner_moves(waypoints):
    """The motion segments of a planner path as ((x0, y0), (x1, y1), spray): zero-length join points are skipped."""
    moves = []
    for p, q in zip(waypoints, waypoints[1:]):
        if math.hypot(q["x"] - p["x"], q["y"] - p["y"]) > 0.5:
            moves.append(((p["x"], p["y"]), (q["x"], q["y"]), bool(p["spray_active"] and q["spray_active"])))
    return moves
```

Create `tests/test_rover_mission.py`:

```python
import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fakes import planner_moves
from geometry_engine import START_CORNERS, GeometryEngine
from rover_mission import SCHEMA, compile_rover_mission, load_profile, replay_rover_mission, verify_checksum

WALL = 2438.4  # 8 ft square
OBSTACLES = [
    {"id": "obs_1", "type": "window", "x": 1000.0, "y": 600.0, "w": 700.0, "h": 800.0},
    {"id": "obs_2", "type": "door", "x": 0.0, "y": 0.0, "w": 500.0, "h": 900.0},
]


def plan_and_compile(corner, obstacles=()):
    waypoints, stats = GeometryEngine().plan_coverage_path(WALL, WALL, 250.0, 12.0, 60.0, list(obstacles), corner)
    mission = compile_rover_mission(waypoints, WALL, WALL, 250.0, stats["effective_step_mm"], corner, load_profile())
    return waypoints, stats, mission


def write_profile(tmp, **overrides):
    profile = {"schema": "apr-rover-profile/1", "wheel_diameter_mm": 80.0, "max_speed_mps": 0.25}
    profile.update(overrides)
    path = os.path.join(tmp, "profile.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f)
    return path


class TestProfile(unittest.TestCase):
    def test_the_shipped_profile_loads(self):
        profile = load_profile()
        self.assertEqual(profile["schema"], "apr-rover-profile/1")
        self.assertEqual(profile["max_speed_mps"], 0.25)

    def test_bad_profiles_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for overrides in ({"schema": "other/1"}, {"max_speed_mps": 0}, {"max_speed_mps": 0.9}, {"wheel_diameter_mm": -1}, {"max_speed_mps": True}):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    load_profile(write_profile(tmp, **overrides))


class TestCompile(unittest.TestCase):
    def test_header(self):
        _, stats, mission = plan_and_compile("bottom_left")
        self.assertEqual(mission["schema"], SCHEMA)
        self.assertEqual(mission["wall"], {"width_mm": WALL, "height_mm": WALL, "start_corner": "bottom_left"})
        self.assertEqual(mission["spray"], {"width_mm": 250.0, "row_pitch_mm": 220.0})
        self.assertEqual(mission["speed_mps"], 0.25)
        self.assertEqual(mission["profile"], load_profile())
        self.assertEqual(stats["effective_step_mm"], 220.0)

    def test_start_pose_for_each_corner(self):
        expected = {
            "bottom_left": (0.0, 125.0, 0.0),
            "bottom_right": (WALL, 125.0, 180.0),
            "top_left": (0.0, 2313.4, 0.0),
            "top_right": (WALL, 2313.4, 180.0),
        }
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                pose = plan_and_compile(corner)[2]["start_pose"]
                self.assertEqual((pose["x_mm"], pose["y_mm"], pose["heading_deg"]), expected[corner])

    def test_serpentine_shape_on_a_clear_wall(self):
        _, _, mission = plan_and_compile("bottom_left")
        steps = mission["steps"]
        self.assertEqual(
            [s["op"] for s in steps[:11]],
            ["MOVE", "EDGE_ALIGN", "TURN", "MOVE", "TURN", "MOVE", "EDGE_ALIGN", "TURN", "MOVE", "TURN", "MOVE"],
        )
        self.assertEqual([steps[i]["spray"] for i in (0, 3, 5, 8, 10)], [1, 0, 1, 0, 1])
        self.assertEqual([steps[i]["deg"] for i in (2, 4, 7, 9)], [90.0, 90.0, -90.0, -90.0])
        self.assertEqual([s["i"] for s in steps], list(range(1, len(steps) + 1)))

    def test_replaying_the_steps_reproduces_the_planner_path_from_every_corner(self):
        for corner in START_CORNERS:
            for obstacles in ((), OBSTACLES):
                with self.subTest(corner=corner, obstacles=len(obstacles)):
                    waypoints, _, mission = plan_and_compile(corner, obstacles)
                    expected = planner_moves(waypoints)
                    actual = replay_rover_mission(mission)["moves"]
                    self.assertEqual(len(actual), len(expected))
                    for (e_start, e_end, e_spray), (a_start, a_end, a_spray) in zip(expected, actual):
                        self.assertEqual(e_spray, a_spray)
                        for e, a in ((e_start, a_start), (e_end, a_end)):
                            self.assertAlmostEqual(e[0], a[0], delta=0.01)
                            self.assertAlmostEqual(e[1], a[1], delta=0.01)

    def test_stored_expected_pose_matches_the_replay_after_every_step(self):
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                _, _, mission = plan_and_compile(corner, OBSTACLES)
                poses = replay_rover_mission(mission)["poses"]
                self.assertEqual(len(poses), len(mission["steps"]))
                for step, (x, y, heading) in zip(mission["steps"], poses):
                    self.assertAlmostEqual(step["x_mm"], x, delta=0.01)
                    self.assertAlmostEqual(step["y_mm"], y, delta=0.01)
                    self.assertAlmostEqual(step["heading_deg"], heading, delta=0.01)

    def test_one_edge_align_per_full_row_pass_on_a_clear_wall(self):
        for corner in START_CORNERS:
            with self.subTest(corner=corner):
                _, stats, mission = plan_and_compile(corner)
                steps = mission["steps"]
                aligns = [i for i, s in enumerate(steps) if s["op"] == "EDGE_ALIGN"]
                self.assertEqual(len(aligns), stats["num_rows"])
                for i in aligns:
                    self.assertEqual(steps[i - 1]["op"], "MOVE")
                    self.assertIn(steps[i]["value_mm"], (0.0, WALL))
                    self.assertEqual(steps[i]["axis"], "x")
                    self.assertEqual(steps[i]["x_mm"], steps[i]["value_mm"])

    def test_checksum_detects_tampering(self):
        _, _, mission = plan_and_compile("bottom_left", OBSTACLES)
        self.assertTrue(verify_checksum(mission))
        tampered = copy.deepcopy(mission)
        tampered["steps"][0]["mm"] += 1.0
        self.assertFalse(verify_checksum(tampered))

    def test_mission_survives_a_json_round_trip(self):
        _, _, mission = plan_and_compile("top_right", OBSTACLES)
        again = json.loads(json.dumps(mission))
        self.assertEqual(again, mission)
        self.assertTrue(verify_checksum(again))

    def test_diagonal_waypoints_are_rejected(self):
        diagonal = [
            {"x": 0.0, "y": 0.0, "spray_active": True},
            {"x": 100.0, "y": 100.0, "spray_active": True},
        ]
        with self.assertRaises(ValueError):
            compile_rover_mission(diagonal, WALL, WALL, 250.0, 220.0, "bottom_left", load_profile())

    def test_a_mission_with_no_motion_is_rejected(self):
        still = [{"x": 5.0, "y": 5.0, "spray_active": True}, {"x": 5.0, "y": 5.2, "spray_active": True}]
        with self.assertRaises(ValueError):
            compile_rover_mission(still, WALL, WALL, 250.0, 220.0, "bottom_left", load_profile())

    def test_replay_rejects_an_unknown_step(self):
        _, _, mission = plan_and_compile("bottom_left")
        mission["steps"].append({"i": 99, "op": "JUMP"})
        with self.assertRaises(ValueError):
            replay_rover_mission(mission)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run python -m unittest discover -s tests -p "test_rover_mission.py" -v`
Expected: ERROR `ModuleNotFoundError: No module named 'rover_mission'`.

- [ ] **Step 4: Implement**

Create `rover_mission.py`:

```python
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
        distance = round(math.hypot(q["x"] - x, q["y"] - y), 3)
        x, y = q["x"], q["y"]
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
```

- [ ] **Step 5: Run the tests**

Run: `uv run python -m unittest discover -s tests -p "test_rover_mission.py" -v`
Expected: all PASS.

Run: `uv run python -m unittest discover -s tests -p "test_*.py" -v`
Expected: all PASS, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add rover_profile.json rover_mission.py tests/fakes.py tests/test_rover_mission.py
git commit -m "Add rover profile and mission compiler" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git log -1 --format='%(trailers:key=Co-Authored-By)'
```

---

### Task 3: Honest mission manifest

**Files:**
- Modify: `cad_exporter.py` (`export_mission_json`, around lines 369-420)
- Test: `tests/test_mission_manifest.py`

**Interfaces:**
- Produces: `CADExporter.export_mission_json(wall_w_mm, wall_h_mm, obstacles, waypoints, metrics, *, spray_width_mm=250.0, overlap_pct=12.0, safety_buffer_mm=60.0, speed_mps=0.25, start_corner="bottom_left") -> str`. The new keyword-only arguments default to the values that were hardcoded before, so existing callers are unaffected. New manifest keys: `spray_parameters.overlap_pct` and top-level `coverage.start_corner`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mission_manifest.py`:

```python
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cad_exporter import CADExporter

WAYPOINTS = [
    {"seq": 1, "x": 0.0, "y": 125.0, "z": 250.0, "spray_active": True, "type": "PASS_START", "row": 1},
    {"seq": 2, "x": 4000.0, "y": 125.0, "z": 250.0, "spray_active": True, "type": "PASS_END", "row": 1},
]
METRICS = {
    "gross_area_m2": 11.2, "net_paintable_area_m2": 10.2, "masked_area_m2": 1.0,
    "paint_volume_liters": 2.0, "cycle_time_formatted": "05:00",
}
OBSTACLE = {"id": "obs_1", "type": "window", "label": "Window #1", "x": 1000.0, "y": 1000.0, "w": 500.0, "h": 500.0}


def manifest(**kwargs):
    text = CADExporter().export_mission_json(4000.0, 2800.0, [OBSTACLE], WAYPOINTS, METRICS, **kwargs)
    return json.loads(text)


class TestMissionManifest(unittest.TestCase):
    def test_defaults_keep_the_previous_values(self):
        data = manifest()
        self.assertEqual(data["spray_parameters"]["spray_width_mm"], 250.0)
        self.assertEqual(data["spray_parameters"]["nominal_speed_mps"], 0.25)
        self.assertEqual(data["no_paint_obstacles"][0]["safety_buffer_mm"], 60.0)
        self.assertEqual(data["coverage"]["start_corner"], "bottom_left")

    def test_real_parameters_are_written(self):
        data = manifest(spray_width_mm=300.0, overlap_pct=20.0, safety_buffer_mm=90.0, speed_mps=0.2, start_corner="top_right")
        self.assertEqual(data["spray_parameters"]["spray_width_mm"], 300.0)
        self.assertEqual(data["spray_parameters"]["overlap_pct"], 20.0)
        self.assertEqual(data["spray_parameters"]["nominal_speed_mps"], 0.2)
        self.assertEqual(data["no_paint_obstacles"][0]["safety_buffer_mm"], 90.0)
        self.assertEqual(data["coverage"]["start_corner"], "top_right")

    def test_existing_keys_are_still_there(self):
        data = manifest()
        for key in ("project", "coordinate_frame", "wall_geometry", "spray_parameters", "no_paint_obstacles", "waypoint_count", "waypoints"):
            self.assertIn(key, data)
        self.assertEqual(data["waypoint_count"], 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m unittest discover -s tests -p "test_mission_manifest.py" -v`
Expected: FAIL/ERROR: `KeyError: 'coverage'` and `TypeError: ... unexpected keyword argument 'spray_width_mm'`.

- [ ] **Step 3: Implement**

In `cad_exporter.py`, replace the head of `export_mission_json`:

```python
        waypoints: List[Dict[str, Any]],
        metrics: Dict[str, Any]
    ) -> str:
        """
        Generate ROS 2 / APR Mission Planner execution manifest JSON schema.
```

with:

```python
        waypoints: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        *,
        spray_width_mm: float = 250.0,
        overlap_pct: float = 12.0,
        safety_buffer_mm: float = 60.0,
        speed_mps: float = 0.25,
        start_corner: str = "bottom_left"
    ) -> str:
        """
        Generate ROS 2 / APR Mission Planner execution manifest JSON schema.
```

Replace the `spray_parameters` block start:

```python
            "spray_parameters": {
                "spray_width_mm": 250.0,
                "target_dft_um": 50.0,
                "nominal_speed_mps": 0.25,
```

with:

```python
            "coverage": {
                "start_corner": start_corner
            },
            "spray_parameters": {
                "spray_width_mm": spray_width_mm,
                "overlap_pct": overlap_pct,
                "target_dft_um": 50.0,
                "nominal_speed_mps": speed_mps,
```

Replace `                    "safety_buffer_mm": 60.0` (inside the `no_paint_obstacles` list comprehension) with `                    "safety_buffer_mm": safety_buffer_mm`.

- [ ] **Step 4: Run the tests**

Run: `uv run python -m unittest discover -s tests -p "test_mission_manifest.py" -v`
Expected: all PASS.

Run: `uv run python -m unittest discover -s tests -p "test_*.py" -v`
Expected: all PASS, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add cad_exporter.py tests/test_mission_manifest.py
git commit -m "Write real spray parameters and start corner into the mission manifest" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git log -1 --format='%(trailers:key=Co-Authored-By)'
```

---

### Task 4: Wire the start corner and the rover mission into the API

**Files:**
- Modify: `app.py`
- Test: `tests/test_api.py` (append a new class before `if __name__ == "__main__":`)

**Interfaces:**
- Consumes: `GeometryEngine.parse_start_corner`, `plan_coverage_path(..., start_corner)`, `DEFAULT_START_CORNER` (Task 1); `compile_rover_mission`, `load_profile` (Task 2); `export_mission_json(..., spray_width_mm=, overlap_pct=, safety_buffer_mm=, speed_mps=, start_corner=)` (Task 3).
- Produces: optional `start_corner` form field on `/api/analyze` and JSON field on `/api/replan` (missing means `bottom_left`, invalid means HTTP 400); `wall.start_corner` in the `/api/analyze` response; `GET /api/download/rover` serves `apr_rover_mission.json`, rewritten on every analyze and replan. `build_mission(wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles, start_corner=DEFAULT_START_CORNER)`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_api.py`, change the existing line `from fakes import FakeDetector` to `from fakes import FakeDetector, planner_moves`, and add below `from obstacle_detector import Detection`:

```python
from rover_mission import replay_rover_mission, verify_checksum
```

Append before `if __name__ == "__main__":`:

```python
class TestStartCornerAndRoverMission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app_module.app.test_client()

    def analyze(self, detections=(), **extra):
        with mock.patch.object(app_module.geometry_engine, "detector", FakeDetector(list(detections))):
            return self.client.post("/api/analyze", data={**FORM, **extra})

    def download_json(self, kind):
        res = self.client.get(f"/api/download/{kind}")
        try:
            return json.loads(res.data)
        finally:
            res.close()

    def test_default_start_is_bottom_left(self):
        data = json.loads(self.analyze().data)
        first = data["waypoints"][0]
        self.assertEqual(data["wall"]["start_corner"], "bottom_left")
        self.assertEqual((first["x"], first["y"]), (0.0, 125.0))

    def test_top_right_start(self):
        data = json.loads(self.analyze(start_corner="top_right").data)
        first = data["waypoints"][0]
        self.assertEqual(data["wall"]["start_corner"], "top_right")
        self.assertEqual((first["x"], first["y"]), (4000.0, 2675.0))

    def test_bad_start_corner_is_a_400(self):
        res = self.analyze(start_corner="middle")
        self.assertEqual(res.status_code, 400)
        self.assertFalse(json.loads(res.data)["success"])

    def test_rover_mission_download_replays_to_the_response_waypoints(self):
        data = json.loads(self.analyze(
            detections=[Detection("window", 0.9, 100, 100, 400, 400)], start_corner="bottom_right"
        ).data)
        mission = self.download_json("rover")
        self.assertTrue(verify_checksum(mission))
        self.assertEqual(mission["wall"]["start_corner"], "bottom_right")
        expected = planner_moves(data["waypoints"])
        actual = replay_rover_mission(mission)["moves"]
        self.assertEqual(len(actual), len(expected))
        for (e_start, e_end, e_spray), (a_start, a_end, a_spray) in zip(expected, actual):
            self.assertEqual(e_spray, a_spray)
            self.assertAlmostEqual(e_end[0], a_end[0], delta=0.01)
            self.assertAlmostEqual(e_end[1], a_end[1], delta=0.01)

    def test_exports_use_the_requested_spray_settings(self):
        self.analyze(
            detections=[Detection("window", 0.9, 100, 100, 400, 400)],
            spray_width_mm="300", overlap_pct="20", safety_buffer_mm="90", start_corner="top_left",
        )
        manifest = self.download_json("json")
        self.assertEqual(manifest["spray_parameters"]["spray_width_mm"], 300.0)
        self.assertEqual(manifest["spray_parameters"]["overlap_pct"], 20.0)
        self.assertEqual(manifest["coverage"]["start_corner"], "top_left")
        self.assertEqual(manifest["no_paint_obstacles"][0]["safety_buffer_mm"], 90.0)
        self.assertEqual(self.download_json("rover")["spray"]["width_mm"], 300.0)

    def test_replan_honours_the_start_corner_and_defaults_to_bottom_left(self):
        with_corner = self.client.post("/api/replan", json={**WALL, "obstacles": [], "start_corner": "top_right"})
        self.assertEqual(json.loads(with_corner.data)["waypoints"][0]["x"], 4000.0)

        default = json.loads(self.client.post("/api/replan", json={**WALL, "obstacles": []}).data)
        self.assertEqual((default["waypoints"][0]["x"], default["waypoints"][0]["y"]), (0.0, 125.0))

        bad = self.client.post("/api/replan", json={**WALL, "obstacles": [], "start_corner": "middle"})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(self.download_json("rover")["wall"]["start_corner"], "bottom_left")
```

(`WALL` is the module-level dict already defined in this file; `FORM`, `FakeDetector`, `Detection` and `mock` are already imported.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m unittest discover -s tests -p "test_api.py" -v`
Expected: FAIL: `KeyError: 'start_corner'` for the new tests (the existing tests still pass).

- [ ] **Step 3: Implement**

In `app.py`:

1. Add `import json` on the line after `import io`.

2. Replace the project imports:

```python
from geometry_engine import GeometryEngine
from cad_exporter import CADExporter
from obstacle_detector import load_default_detector
```

with:

```python
from geometry_engine import DEFAULT_START_CORNER, GeometryEngine
from cad_exporter import CADExporter
from obstacle_detector import load_default_detector
from rover_mission import compile_rover_mission, load_profile
```

3. Replace:

```python
geometry_engine = GeometryEngine(detector=load_default_detector())
cad_exporter = CADExporter()
```

with:

```python
geometry_engine = GeometryEngine(detector=load_default_detector())
cad_exporter = CADExporter()
ROVER_PROFILE = load_profile()
```

4. In `build_mission`, replace the head:

```python
def build_mission(wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles):
    """Plan coverage, compute metrics, write every export file and refresh the session cache."""
    waypoints, path_stats = geometry_engine.plan_coverage_path(
        wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles
    )
```

with:

```python
def build_mission(wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles,
                  start_corner=DEFAULT_START_CORNER):
    """Plan coverage, compute metrics, write every export file and refresh the session cache."""
    waypoints, path_stats = geometry_engine.plan_coverage_path(
        wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles, start_corner
    )
```

5. Replace the manifest line:

```python
    mission_json = cad_exporter.export_mission_json(wall_w_mm, wall_h_mm, obstacles, waypoints, metrics)
```

with:

```python
    mission_json = cad_exporter.export_mission_json(
        wall_w_mm, wall_h_mm, obstacles, waypoints, metrics,
        spray_width_mm=spray_width_mm,
        overlap_pct=overlap_pct,
        safety_buffer_mm=safety_buffer_mm,
        speed_mps=metrics["nominal_speed_mps"],
        start_corner=start_corner,
    )
```

6. Replace:

```python
    csv_path = os.path.join(EXPORTS_DIR, "apr_waypoints.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(motion_csv)

    current_session.update({
```

with:

```python
    csv_path = os.path.join(EXPORTS_DIR, "apr_waypoints.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(motion_csv)

    rover_mission = compile_rover_mission(
        waypoints, wall_w_mm, wall_h_mm, spray_width_mm, path_stats["effective_step_mm"], start_corner, ROVER_PROFILE
    )
    rover_path = os.path.join(EXPORTS_DIR, "apr_rover_mission.json")
    with open(rover_path, "w", encoding="utf-8") as f:
        json.dump(rover_mission, f, indent=2)

    current_session.update({
```

7. Replace:

```python
        "stats": path_stats,
    })
    return waypoints, path_stats, metrics
```

with:

```python
        "stats": path_stats,
        "start_corner": start_corner,
    })
    return waypoints, path_stats, metrics
```

8. In `analyze_wall`, replace `        sample_id = request.form.get("sample_id", None)` with:

```python
        sample_id = request.form.get("sample_id", None)
        start_corner = geometry_engine.parse_start_corner(request.form.get("start_corner"))
```

9. In `analyze_wall`, replace:

```python
        waypoints, path_stats, metrics = build_mission(
            wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles
        )
        current_session["confidence"] = confidence
```

with:

```python
        waypoints, path_stats, metrics = build_mission(
            wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles, start_corner
        )
        current_session["confidence"] = confidence
```

10. In the `/api/analyze` response, replace:

```python
                "safety_buffer_mm": safety_buffer_mm
            },
```

with:

```python
                "safety_buffer_mm": safety_buffer_mm,
                "start_corner": start_corner
            },
```

11. In `replan`, replace:

```python
        obstacles = geometry_engine.normalize_obstacles(payload.get("obstacles"), wall_w_mm, wall_h_mm)

        waypoints, path_stats, metrics = build_mission(
            wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles
        )
```

with:

```python
        obstacles = geometry_engine.normalize_obstacles(payload.get("obstacles"), wall_w_mm, wall_h_mm)
        start_corner = geometry_engine.parse_start_corner(payload.get("start_corner"))

        waypoints, path_stats, metrics = build_mission(
            wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles, start_corner
        )
```

12. In `download_file`, replace `        "csv": ("apr_waypoints.csv", "text/csv", "apr_waypoints.csv")` with:

```python
        "csv": ("apr_waypoints.csv", "text/csv", "apr_waypoints.csv"),
        "rover": ("apr_rover_mission.json", "application/json", "apr_rover_mission.json")
```

- [ ] **Step 4: Run the tests**

Run: `uv run python -m unittest discover -s tests -p "test_api.py" -v`
Expected: all PASS.

Run: `uv run python -m unittest discover -s tests -p "test_*.py" -v`
Expected: all PASS, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_api.py
git commit -m "Serve a rover mission export and accept a start corner" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git log -1 --format='%(trailers:key=Co-Authored-By)'
```

---

### Task 5: Start-corner dropdown and rover download in the page

**Files:**
- Modify: `templates/index.html`, `static/app.js`, `static/style.css` (append)
- Test: `tests/test_rover_ui.py`

**Interfaces:**
- Consumes: `/api/analyze` `start_corner` form field and `wall.start_corner` (Task 4); `/api/replan` `start_corner` JSON field; `/api/download/rover`.
- Produces: `<select id="startCorner">` with the four corner values; a `data-export="rover"` button (the existing `[data-export]` click handler downloads it).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rover_ui.py`:

```python
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module
from geometry_engine import START_CORNERS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


class TestRoverUi(unittest.TestCase):
    def test_page_has_the_start_corner_select_and_the_rover_download(self):
        html = app_module.app.test_client().get("/").get_data(as_text=True)
        self.assertIn('id="startCorner"', html)
        for corner in START_CORNERS:
            self.assertIn(f'value="{corner}"', html)
        self.assertIn('data-export="rover"', html)

    def test_script_sends_the_start_corner_on_analyze_and_replan(self):
        js = read("static", "app.js")
        self.assertIn("const startCornerInput = $('startCorner');", js)
        self.assertIn("formData.append('start_corner', startCornerInput.value);", js)
        self.assertIn("start_corner: wall.start_corner,", js)

    def test_start_label_is_no_longer_hardcoded_to_the_origin(self):
        self.assertNotIn("START (0,0)", read("static", "app.js"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m unittest discover -s tests -p "test_rover_ui.py" -v`
Expected: FAIL: `AssertionError: 'id="startCorner"' not found in ...`.

- [ ] **Step 3: Implement the page**

Match the HTML and JS anchors below by content. The files indent with tabs (HTML) and four spaces (JS); keep each file's own indentation and do not reformat anything else.

In `templates/index.html`, replace:

```html
						<button class="primary-btn" id="generate">Generate Wall Map & CAD &nbsp;→</button>
```

with:

```html
						<div class="param-group">
							<label for="startCorner">Rover Start Corner</label>
							<select id="startCorner">
								<option value="bottom_left" selected>Bottom-left (0,0)</option>
								<option value="bottom_right">Bottom-right</option>
								<option value="top_left">Top-left</option>
								<option value="top_right">Top-right</option>
							</select>
						</div>

						<button class="primary-btn" id="generate">Generate Wall Map & CAD &nbsp;→</button>
```

In `templates/index.html`, replace:

```html
						<button type="button" class="export-btn" data-export="csv">
							↓ Waypoints CSV
						</button>
```

with:

```html
						<button type="button" class="export-btn" data-export="csv">
							↓ Waypoints CSV
						</button>
						<button type="button" class="export-btn" data-export="rover" style="color:var(--amber);border-color:#654c1f;">
							↓ Rover Mission (.json)
						</button>
```

Append to `static/style.css`:

```css
/* Rover start corner */
.param-group select {
	width: 100%;
	border: 1px solid var(--line);
	border-radius: 5px;
	background: #071318;
	color: var(--text);
	padding: 8px 10px;
	outline: none;
}
```

- [ ] **Step 4: Implement the script**

In `static/app.js`:

1. Replace `    const sensitivityInput = $('sensitivity');` with:

```javascript
    const sensitivityInput = $('sensitivity');
    const startCornerInput = $('startCorner');
```

2. Replace `        formData.append('sensitivity', sensitivityInput.value);` with:

```javascript
        formData.append('sensitivity', sensitivityInput.value);
        formData.append('start_corner', startCornerInput.value);
```

3. In `replan`, replace `                safety_buffer_mm: wall.safety_buffer_mm,` with:

```javascript
                safety_buffer_mm: wall.safety_buffer_mm,
                start_corner: wall.start_corner,
```

4. Replace the START label line:

```javascript
            ctx2d.fillText('START (0,0)', fx + 10, fy - 5);
```

with:

```javascript
            const startLabel = `START (${Math.round(first.x)}, ${Math.round(first.y)})`;
            const labelX = fx > width / 2 ? fx - 10 - ctx2d.measureText(startLabel).width : fx + 10;
            ctx2d.fillText(startLabel, labelX, fy - 5);
```

- [ ] **Step 5: Run the tests and the syntax check**

Run: `uv run python -m unittest discover -s tests -p "test_rover_ui.py" -v`
Expected: all PASS.

Run: `node --check static/app.js` (skip if Node is not installed)
Expected: no output.

Run: `uv run python -m unittest discover -s tests -p "test_*.py" -v`
Expected: all PASS, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add templates/index.html static/app.js static/style.css tests/test_rover_ui.py
git commit -m "Add start corner dropdown and rover mission download to the page" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git log -1 --format='%(trailers:key=Co-Authored-By)'
```

---

### Task 6: Align the spec and hand off for a browser check

**Files:**
- Modify: `docs/superpowers/specs/2026-10-09-rover-hardware-integration-design.md`

P1 built the start corner as one setting and verifies simulator equivalence with a replay test, not by changing `rover_sim.js`. The spec must say so.

- [ ] **Step 1: Update the spec**

Make these five replacements in the spec file. Each old text is the full line as it stands (line numbers are for orientation only, match by content).

1. Replace the whole list item that starts `1. **Operator-chosen start.**` (one paragraph) with:

```markdown
1. **Operator-chosen start.** `plan_coverage_path` gains `start_corner`, one of the four wall corners (default `bottom_left`, the rover's (0,0)). The corner fixes both the side the first pass starts from and the direction rows progress: rows move away from the start side, so a bottom corner paints bottom-up and a top corner paints top-down. Top-down avoids drips on fresh paint on a vertical wall, which is one reason the choice is per job. The operator picks the corner in the app, places the rover there, and the planner builds the path from that start. No trained model decides the start: that keeps the frozen rule that AI never decides motion. A deterministic "suggest a start" heuristic (for example fewest transit metres) can be added later and would still only suggest.
```

2. In the line that starts `   - Each step carries the expected (x, y) after it`, replace `The file has a header (wall size, spray width, row pitch, speed, paint estimate) and a checksum.` with `The file has a header (wall size, start corner, spray width, row pitch, speed, the rover profile) and a checksum. The paint estimate stays in the existing mission manifest JSON.`

3. Replace the line `3. **`rover_profile.json`.** Wheel diameter, ticks per revolution, wheelbase, max speed, pump maximum continuous run time. The compiler and firmware both read it.` with:

```markdown
3. **`rover_profile.json`.** Constants shared by the compiler and the firmware. P1 ships the wheel diameter and max speed. P2 adds ticks per revolution, wheelbase and the pump's maximum continuous run time once the parts are chosen.
```

4. Replace the line `5. Existing DXF, OBJ, CSV and JSON exports are unchanged. The 2D map gets a rover-path overlay. `static/rover_sim.js` plays the compiled primitives so the preview matches what the rover receives.` with:

```markdown
5. Existing DXF, OBJ, CSV and JSON exports are unchanged apart from the honest parameters in item 4. The 2D map's start marker shows the real start position. The simulation keeps playing the planner's waypoints. A replay test proves that the compiled rover steps reproduce those waypoints within 0.01 mm, so the preview matches what the rover receives without a second playback path in `static/rover_sim.js`.
```

5. Replace the plan-table row `| P1 | Planner start corner and row order, `rover_profile.json`, `rover_mission.py`, new export, simulator playback, honest export parameters | None. Can start immediately. |` with:

```markdown
| P1 | Planner start corner, `rover_profile.json`, `rover_mission.py`, new export, start-corner dropdown, honest export parameters | None. Can start immediately. |
```

- [ ] **Step 2: Final verification**

Run: `uv run python -m unittest discover -s tests -p "test_*.py" -v`
Expected: all PASS, 1 skipped.

Run: `uv lock --check`
Expected: exits 0 (no dependency changed).

Run: `grep -inE "torch|ultralytics" requirements.txt pyproject.toml`
Expected: no output.

Run: `uv run python -c "import app; c = app.app.test_client(); print([c.get(p).status_code for p in ('/', '/about')])"`
Expected: `[200, 200]`

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-10-09-rover-hardware-integration-design.md
git commit -m "Align the rover spec with what P1 builds" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git log -1 --format='%(trailers:key=Co-Authored-By)'
```

- [ ] **Step 4: Report to the user, then stop before pushing**

Summarise what shipped. State plainly that the page changes were not seen in a browser (no browser tool is available), and hand the user this checklist, to run with `uv run python app.py` and a hard refresh:

1. The left panel shows a "Rover Start Corner" dropdown, styled like the other inputs, default "Bottom-left (0,0)".
2. Generate with Bottom-left: the 2D map's START dot is at the bottom-left with a label like `START (0, 125)`, and the first pass runs left to right along the bottom row.
3. Switch to Top-right and generate: the START dot moves to the top-right, the label sits to the left of the dot and is not clipped, and the first pass runs right to left along the top row.
4. Dismiss or confirm an uncertain obstacle in the review panel: the re-plan keeps the same start corner.
5. Click "Rover Mission (.json)": a file `apr_rover_mission.json` downloads, starts with `"schema": "apr-rover-mission/1"`, and its `wall.start_corner` matches the dropdown.
6. Click "ROS 2 Mission JSON": it contains your real spray width and overlap values and a `coverage.start_corner` entry.

Known limit to mention: if an obstacle covers the chosen start corner, the first pass crosses the keep-out zone with the spray off. The operator should pick another corner. A start-position check can be added later.

Pushing is the user's call: ask before pushing.
