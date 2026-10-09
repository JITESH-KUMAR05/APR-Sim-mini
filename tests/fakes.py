import math


class FakeDetector:
    """Stands in for ObstacleDetector: returns preset detections at or above min_score."""

    def __init__(self, detections):
        self.detections = list(detections)
        self.min_scores = []

    def detect(self, image_bgr, min_score):
        self.min_scores.append(min_score)
        return [d for d in self.detections if d.confidence >= min_score]


def planner_moves(waypoints):
    """The motion segments of a planner path as ((x0, y0), (x1, y1), spray): zero-length join points are skipped."""
    moves = []
    for p, q in zip(waypoints, waypoints[1:]):
        if math.hypot(q["x"] - p["x"], q["y"] - p["y"]) > 0.5:
            moves.append(((p["x"], p["y"]), (q["x"], q["y"]), bool(p["spray_active"] and q["spray_active"])))
    return moves
