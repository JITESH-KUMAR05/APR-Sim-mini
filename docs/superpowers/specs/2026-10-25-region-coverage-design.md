# Region-by-region coverage: the rover never drives over a keep-out

Date: 2026-10-25. Chosen by the project lead over "detour around each keep-out" and "leave as is".

## Problem

The old planner ran every stripe from wall edge to wall edge and only switched the spray off over a
keep-out. On the real wall the rover cannot cross a window, an AC unit or a pipe, so those plans were
not executable.

## Design

Implemented in `region_planner.py`; `GeometryEngine.plan_coverage_path` keeps its corner and
direction handling (mirror and transpose into one frame) and calls it.

- **Rows.** Same rows as before: one nozzle pitch apart, a half nozzle width inside the wall.
- **Free pieces.** Each keep-out, grown by the safety margin, cuts the rows it touches.
- **Regions.** Pieces of consecutive rows that overlap one to one form a region (boustrophedon cell
  decomposition). Where a keep-out begins or ends, pieces split or merge and new regions start.
- **Order.** A region starts only when every region directly above it (in the planning frame) is
  done. Of the ready regions, the nearest entry point is next. Each region is one zigzag.
- **Moves.** Between rows and between regions the rover takes the shortest route of horizontal and
  vertical segments over free wall (rover missions accept nothing else), with a 150 mm cost per turn.
  The routes run on the grid through every keep-out edge. Grown keep-outs are not clipped at the
  wall edge, so a keep-out touching the floor cannot be passed underneath.
- **Cut-off wall.** A keep-out that runs from wall edge to wall edge splits the wall. The part that
  no route reaches is left out and reported (`unreached_regions`, `unreached_area_m2`).

On a clear wall the result is exactly the old zigzag.

## What changed elsewhere

- Waypoint types: `PASS_START`, `PASS_END`, `ROW_TRANSITION`, and the new `REGION_MOVE`.
  `TRANSIT_START` and `TRANSIT_END` are no longer produced. Spray waypoints carry `pass`, the pass
  number in painting order.
- Stats add `num_regions`, `num_passes`, `unreached_regions`, `unreached_area_m2`.
- `layout_advisor`: "obstacle crossings" is gone. Layouts rank by unreached area, then non-painting
  travel, then turns. Options carry `regions` and `unreached_m2`.
- Console: passes numbered in painting order, "Regions" in the layout table, and a note when part
  of the wall is cut off.

## Known limits

- The physics replay shows true-path drift of about 110 mm at 3% wheel slip, more than the default
  60 mm keep-out margin. Passes that end at a keep-out do not touch a wall edge, so fewer
  edge-contact position resets happen there.
- Painting a cut-off part needs a second plan, with the rover placed on that side by hand.
- Planning time on a wall with 15 keep-outs is about 0.6 to 0.9 s for all eight layouts.
