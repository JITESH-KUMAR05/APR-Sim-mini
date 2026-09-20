class FakeDetector:
    """Stands in for ObstacleDetector: returns preset detections at or above min_score."""

    def __init__(self, detections):
        self.detections = list(detections)
        self.min_scores = []

    def detect(self, image_bgr, min_score):
        self.min_scores.append(min_score)
        return [d for d in self.detections if d.confidence >= min_score]
