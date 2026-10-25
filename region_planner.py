"""
Coverage planning region by region, so the rover never drives over a keep-out.

The plan is made in one frame: horizontal stripes (rows) one nozzle pitch apart, numbered from the
top, the first one starting at the left. GeometryEngine mirrors and transposes into this frame for
the other corners and for vertical stripes.

A keep-out, grown by the safety margin, cuts every row it touches into free pieces. Pieces of
consecutive rows that overlap one to one form a region (a boustrophedon cell decomposition); where a
keep-out begins or ends, pieces split or merge and new regions begin. Each region is painted as its
own zigzag from its top row down. A region is started only once every region directly above it is
done, and of the regions that are ready the nearest one is next.

Moves between rows and between regions are routed over free wall with horizontal and vertical
segments only, because the rover mission accepts nothing else. A move takes the shortest route,
with a small cost for each turn. The rover's centre stays between the first and the last row and
inside the wall. A region that no route reaches (cut off by a keep-out from wall edge to wall edge)
is left out and reported in the stats.
"""

import heapq
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]  # x1, y1, x2, y2

TURN_COST_MM = 150.0  # a turn costs as much as this much straight travel when choosing a route
MIN_PIECE_MM = 1.0  # a free piece of a row shorter than this is not painted
_EPS = 1e-6


def row_positions(wall_h_mm: float, spray_width_mm: float, step_mm: float) -> List[float]:
    """Centre line of every row, top first, kept a half nozzle width inside the wall."""
    num_rows = max(2, int(math.ceil(wall_h_mm / step_mm)))
    ys = []
    for r in range(num_rows):
        y = wall_h_mm - (r + 0.5) * step_mm
        y = min(wall_h_mm - spray_width_mm * 0.5, y)
        ys.append(max(spray_width_mm * 0.5, y))
    return ys


def free_pieces(y: float, wall_w_mm: float, boxes: Sequence[Box]) -> List[Tuple[float, float]]:
    """The parts of the row at height y that no grown keep-out touches, left to right."""
    cuts = sorted((max(0.0, b[0]), min(wall_w_mm, b[2])) for b in boxes if b[1] <= y <= b[3])
    pieces = []
    x = 0.0
    for start, end in cuts:
        if start > x and start - x >= MIN_PIECE_MM:
            pieces.append((x, start))
        x = max(x, end)
    if wall_w_mm - x >= MIN_PIECE_MM:
        pieces.append((x, wall_w_mm))
    return pieces


def find_regions(ys: Sequence[float], wall_w_mm: float, boxes: Sequence[Box]) -> List[Dict[str, Any]]:
    """
    Regions of the wall, each a list of (row, x_start, x_end) top to bottom, with the regions
    directly above it as `parents`.
    """
    regions: List[Dict[str, Any]] = []
    active: List[Tuple[int, float, float]] = []  # (region, x_start, x_end) of the previous row
    for r, y in enumerate(ys):
        pieces = free_pieces(y, wall_w_mm, boxes)
        below = {i: [] for i in range(len(active))}
        above = []
        for j, (a, b) in enumerate(pieces):
            links = [i for i, (_, pa, pb) in enumerate(active) if pa < b and a < pb]
            above.append(links)
            for i in links:
                below[i].append(j)
        next_active = []
        for j, (a, b) in enumerate(pieces):
            links = above[j]
            if len(links) == 1 and len(below[links[0]]) == 1:
                region = active[links[0]][0]
                regions[region]["rows"].append((r, a, b))
            else:
                region = len(regions)
                regions.append({"rows": [(r, a, b)], "parents": {active[i][0] for i in links}})
            next_active.append((region, a, b))
        active = next_active
    return regions


class Router:
    """Shortest horizontal-and-vertical routes over free wall, around the grown keep-outs."""

    def __init__(self, boxes: Sequence[Box], x_lo: float, x_hi: float, y_lo: float, y_hi: float):
        self.boxes = list(boxes)
        self.x_lo, self.x_hi, self.y_lo, self.y_hi = x_lo, x_hi, y_lo, y_hi

    def _inside(self, x: float, y: float) -> bool:
        return any(b[0] + _EPS < x < b[2] - _EPS and b[1] + _EPS < y < b[3] - _EPS for b in self.boxes)

    def free(self, p: Point, q: Point) -> bool:
        """True if the straight move from p to q (horizontal or vertical) stays off every keep-out."""
        (px, py), (qx, qy) = p, q
        if abs(py - qy) <= _EPS:
            lo, hi = sorted((px, qx))
            return not any(b[1] + _EPS < py < b[3] - _EPS and b[0] + _EPS < hi and lo < b[2] - _EPS for b in self.boxes)
        if abs(px - qx) <= _EPS:
            lo, hi = sorted((py, qy))
            return not any(b[0] + _EPS < px < b[2] - _EPS and b[1] + _EPS < hi and lo < b[3] - _EPS for b in self.boxes)
        return False

    def route(self, p: Point, q: Point) -> Optional[List[Point]]:
        """Corner points from p to q, both included, or None if q cannot be reached."""
        if abs(p[0] - q[0]) <= _EPS and abs(p[1] - q[1]) <= _EPS:
            return [p]
        if (abs(p[0] - q[0]) <= _EPS or abs(p[1] - q[1]) <= _EPS) and self.free(p, q):
            return [p, q]
        for corner in ((q[0], p[1]), (p[0], q[1])):
            if not self._inside(*corner) and self.free(p, corner) and self.free(corner, q):
                return [p, corner, q]
        return self._search(p, q)

    def _search(self, p: Point, q: Point) -> Optional[List[Point]]:
        # The grid through every keep-out edge and both end points holds a shortest such route.
        xs = sorted({self.x_lo, self.x_hi, p[0], q[0], *(v for b in self.boxes for v in (b[0], b[2]) if self.x_lo < v < self.x_hi)})
        ys = sorted({self.y_lo, self.y_hi, p[1], q[1], *(v for b in self.boxes for v in (b[1], b[3]) if self.y_lo < v < self.y_hi)})
        start, goal = (xs.index(p[0]), ys.index(p[1])), (xs.index(q[0]), ys.index(q[1]))
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
        best = {(start, -1): 0.0}
        back = {}
        queue = [(0.0, start, -1)]
        while queue:
            cost, node, heading = heapq.heappop(queue)
            if cost > best.get((node, heading), math.inf):
                continue
            if node == goal:
                path = [node]
                state = (node, heading)
                while state in back:
                    state = back[state]
                    path.append(state[0])
                points = [(xs[i], ys[j]) for i, j in reversed(path)]
                return _corners_only(points)
            i, j = node
            for k, (di, dj) in enumerate(steps):
                ni, nj = i + di, j + dj
                if not (0 <= ni < len(xs) and 0 <= nj < len(ys)):
                    continue
                a, b = (xs[i], ys[j]), (xs[ni], ys[nj])
                if self._inside(*b) or not self.free(a, b):
                    continue
                step = abs(b[0] - a[0]) + abs(b[1] - a[1]) + (TURN_COST_MM if heading not in (-1, k) else 0.0)
                state = ((ni, nj), k)
                if cost + step < best.get(state, math.inf):
                    best[state] = cost + step
                    back[state] = (node, heading)
                    heapq.heappush(queue, (cost + step, (ni, nj), k))
        return None


def _corners_only(points: List[Point]) -> List[Point]:
    """Drop the points a route passes straight through."""
    kept = [points[0]]
    for prev, here, nxt in zip(points, points[1:], points[2:]):
        straight = (abs(prev[0] - here[0]) <= _EPS and abs(here[0] - nxt[0]) <= _EPS) or \
                   (abs(prev[1] - here[1]) <= _EPS and abs(here[1] - nxt[1]) <= _EPS)
        if not straight:
            kept.append(here)
    if len(points) > 1:
        kept.append(points[-1])
    return kept


def plan_regions(
    wall_w_mm: float,
    wall_h_mm: float,
    spray_width_mm: float,
    overlap_pct: float,
    safety_buffer_mm: float,
    obstacles: List[Dict[str, Any]],
    standoff_mm: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Waypoints and stats for the wall in the top-left frame (see the module docstring)."""
    overlap = max(0.0, min(0.4, overlap_pct / 100.0))
    step = spray_width_mm * (1.0 - overlap)
    ys = row_positions(wall_h_mm, spray_width_mm, step)
    boxes = [(o["x"] - safety_buffer_mm, o["y"] - safety_buffer_mm,
              o["x"] + o["w"] + safety_buffer_mm, o["y"] + o["h"] + safety_buffer_mm) for o in obstacles]
    regions = find_regions(ys, wall_w_mm, boxes)
    router = Router(boxes, 0.0, wall_w_mm, min(ys), max(ys))

    waypoints: List[Dict[str, Any]] = []
    dist = {"spray": 0.0, "transit": 0.0}
    passes = 0

    def add(x: float, y: float, spray: bool, kind: str, row: int, pass_no: Optional[int] = None) -> None:
        if waypoints:
            last = waypoints[-1]
            length = abs(x - last["x"]) + abs(y - last["y"])
            dist["spray" if spray and last["spray_active"] else "transit"] += length
        point = {"seq": len(waypoints) + 1, "x": round(x, 1), "y": round(y, 1), "z": standoff_mm,
                 "spray_active": spray, "type": kind, "row": row}
        if pass_no is not None:
            point["pass"] = pass_no
        waypoints.append(point)

    def move(points: List[Point], kind: str, row: int) -> None:
        for x, y in points[1:]:
            add(x, y, False, kind, row)

    done, cut_off = set(), set()
    pos: Optional[Point] = None
    while True:
        ready = [k for k in range(len(regions))
                 if k not in done and k not in cut_off and regions[k]["parents"] <= done | cut_off]
        if not ready:
            break
        here = pos if pos is not None else (0.0, ys[0])
        choices = sorted(
            (abs(x - here[0]) + abs(ys[r0] - here[1]), k, x)
            for k in ready
            for r0, a, b in regions[k]["rows"][:1]
            for x in (a, b)
        )
        chosen = None
        for _, k, x in choices:
            if k in cut_off:
                continue
            entry = (x, ys[regions[k]["rows"][0][0]])
            path = [entry] if pos is None else router.route(pos, entry)
            if path is not None:
                chosen = (k, x, path)
                break
            # Both ends of a region's top row are joined by free wall, so neither can be reached.
            cut_off.add(k)
        if chosen is None:
            continue
        k, x, path = chosen
        region = regions[k]
        first_row = region["rows"][0][0]
        move(path, "REGION_MOVE", first_row + 1)
        left_to_right = abs(x - region["rows"][0][1]) <= _EPS
        for index, (r, a, b) in enumerate(region["rows"]):
            start, end = (a, b) if left_to_right else (b, a)
            if index > 0:
                hop = router.route(pos, (start, ys[r]))
                if hop is None:  # a sliver between rows is blocked; finish the rest of the region from here
                    break
                move(hop, "ROW_TRANSITION", r + 1)
            passes += 1
            add(start, ys[r], True, "PASS_START", r + 1, passes)
            add(end, ys[r], True, "PASS_END", r + 1, passes)
            pos = (end, ys[r])
            left_to_right = not left_to_right
        done.add(k)

    unreached_mm2 = sum((b - a) * step for k in cut_off for _, a, b in regions[k]["rows"])
    stats = {
        "num_rows": len(ys),
        "effective_step_mm": round(step, 1),
        "total_distance_m": round((dist["spray"] + dist["transit"]) / 1000.0, 2),
        "spray_distance_m": round(dist["spray"] / 1000.0, 2),
        "transit_distance_m": round(dist["transit"] / 1000.0, 2),
        "waypoint_count": len(waypoints),
        "num_regions": len(done),
        "num_passes": passes,
        "unreached_regions": len(cut_off),
        "unreached_area_m2": round(unreached_mm2 / 1e6, 2),
    }
    return waypoints, stats
