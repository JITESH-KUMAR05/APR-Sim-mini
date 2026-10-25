"""
APR (Automated Painting Rover) - Geometry & Computer Vision Engine
Traced to MSME Grant INC25ETS084848 / WBS 1.2 & 1.3 Subsystems.
Handles wall image rectification, obstacle segmentation (windows, doors, switchboards),
Boustrophedon coverage path planning, and quantitative engineering metrics.
"""

import cv2
import numpy as np
import logging
import math
import json
from typing import Dict, List, Tuple, Any, Optional

from obstacle_detector import ObstacleDetector, thresholds_for_sensitivity
from region_planner import plan_regions

log = logging.getLogger(__name__)

# type -> (label prefix, depth_mm). depth_mm only drives the 3D export.
OBSTACLE_CLASS_INFO = {
    "window": ("Glazed Window", 120.0),
    "door": ("Door Opening", 80.0),
    "ac_unit": ("AC Unit", 400.0),
    "meter_panel": ("Meter Panel", 100.0),
    "pipe": ("Pipe", 80.0),
    "grill": ("Grill", 60.0),
}
UNVERIFIED_INFO = ("Unverified object", 50.0)
LEGACY_OBSTACLE_INFO = {  # types still emitted by the classical fallback
    "switchboard": ("Utility Panel", 35.0),
    "fixture": ("Architectural Opening", 50.0),
}
START_CORNERS = ("bottom_left", "bottom_right", "top_left", "top_right")
DEFAULT_START_CORNER = "bottom_left"  # the rover's (0,0)
DIRECTIONS = ("horizontal", "vertical")  # stripe direction
DEFAULT_DIRECTION = "horizontal"
# A start corner seen from the transposed wall (x and y swapped): (0,0) stays, (W,0) becomes top-left, ...
_TRANSPOSED_CORNER = {"bottom_left": "bottom_left", "bottom_right": "top_left",
                      "top_left": "bottom_right", "top_right": "top_right"}


class GeometryEngine:
    MAX_OBSTACLES = 200  # request-size guard for client-submitted obstacle lists (/api/replan)

    def __init__(self, detector: Optional[ObstacleDetector] = None):
        self.detector = detector
        # Default physical parameters mapped to APR Design Review (WBS 1.2 §3)
        self.defaults = {
            "width_mm": 4000.0,
            "height_mm": 2800.0,
            "wall_depth_mm": 200.0,
            "spray_width_mm": 250.0,       # 20-40 cm adjustable range (WBS 1.2 §3)
            "overlap_pct": 12.0,            # 10-30% range
            "safety_buffer_mm": 60.0,       # Keep-out buffer around obstacles
            "standoff_dist_mm": 250.0,      # Target 200-300 mm (Slide 10)
            "rover_speed_mps": 0.25,        # 0.1-0.5 m/s (WBS 1.2 §2)
            "target_dft_um": 50.0,          # 50 µm ± 15 µm dry film thickness (WBS 1.2 §3)
            "paint_solids_pct": 45.0,       # Water-based exterior emulsion solids
            "transfer_efficiency": 0.90     # Airless nozzle: <=10% overspray loss
        }

    def detect(
        self,
        image_bgr: np.ndarray,
        wall_w_mm: float,
        wall_h_mm: float,
        sensitivity: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect obstacles in wall millimetre coordinates (origin bottom-left).
        Uses the trained detector when loaded, otherwise the classical fallback.
        Returns {"obstacles", "confidence", "detector"}; confidence is a percentage
        or None when the model kept no detections.
        """
        detections = None
        review_threshold, accept_threshold = thresholds_for_sensitivity(sensitivity)
        if self.detector is not None:
            try:
                detections = self.detector.detect(image_bgr, min_score=review_threshold)
            except Exception as exc:  # a broken model must never take the pipeline down
                log.warning("Obstacle detector failed (%s); using classical fallback", exc, exc_info=True)

        if detections is None:
            obstacles, confidence = self._detect_classical(image_bgr, wall_w_mm, wall_h_mm, sensitivity)
            return {"obstacles": obstacles, "confidence": confidence, "detector": "classical-fallback"}

        img_h, img_w = image_bgr.shape[:2]
        obstacles = self._obstacles_from_detections(
            detections, img_w, img_h, wall_w_mm, wall_h_mm, accept_threshold
        )
        confidence = None
        if detections:
            confidence = round(float(np.mean([d.confidence for d in detections])) * 100.0, 1)
        return {"obstacles": obstacles, "confidence": confidence, "detector": "yolo11n-onnx"}

    def detect_obstacles_from_image(
        self,
        image_bgr: np.ndarray,
        wall_w_mm: float,
        wall_h_mm: float,
        sensitivity: float = 0.5
    ) -> Tuple[List[Dict[str, Any]], Optional[float]]:
        result = self.detect(image_bgr, wall_w_mm, wall_h_mm, sensitivity)
        return result["obstacles"], result["confidence"]

    def _obstacles_from_detections(
        self,
        detections: List[Any],
        img_w: int,
        img_h: int,
        wall_w_mm: float,
        wall_h_mm: float,
        accept_threshold: float
    ) -> List[Dict[str, Any]]:
        obstacles: List[Dict[str, Any]] = []
        for det in sorted(detections, key=lambda d: d.x1):
            norm_x = det.x1 / img_w
            norm_y = (img_h - det.y2) / img_h   # bottom-left origin
            norm_w = (det.x2 - det.x1) / img_w
            norm_h = (det.y2 - det.y1) / img_h

            if det.confidence >= accept_threshold and det.class_name in OBSTACLE_CLASS_INFO:
                obs_type = det.class_name
                base_label, depth_mm = OBSTACLE_CLASS_INFO[det.class_name]
            else:
                # Confidence below the accept threshold, or a class name the app doesn't
                # know about (e.g. a future model with a class we haven't wired up yet):
                # fall back to "unverified" rather than raising a bare KeyError.
                obs_type = "unverified"
                base_label, depth_mm = UNVERIFIED_INFO

            number = len(obstacles) + 1
            obstacle = {
                "id": f"obs_{number}",
                "type": obs_type,
                "label": f"{base_label} #{number}",
                "x": round(norm_x * wall_w_mm, 1),
                "y": round(norm_y * wall_h_mm, 1),
                "w": round(norm_w * wall_w_mm, 1),
                "h": round(norm_h * wall_h_mm, 1),
                "depth_mm": depth_mm,
                "confidence": round(det.confidence, 3),
                "norm": {
                    "x": round(norm_x, 4),
                    "y": round(norm_y, 4),
                    "w": round(norm_w, 4),
                    "h": round(norm_h, 4)
                }
            }
            if obs_type == "unverified":
                obstacle["guess"] = det.class_name
            obstacles.append(obstacle)
        return obstacles

    @staticmethod
    def parse_corners(raw: Any) -> List[Tuple[float, float]]:
        """
        Parse a JSON list of four [x, y] image fractions (0 to 1), ordered
        top-left, top-right, bottom-right, bottom-left. Raises ValueError.
        """
        try:
            points = [(float(x), float(y)) for x, y in json.loads(raw)]
        except (TypeError, ValueError):
            raise ValueError("corners must be a JSON list of four [x, y] pairs")
        if len(points) != 4:
            raise ValueError("corners must contain exactly four points")
        if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for point in points for v in point):
            raise ValueError("corner coordinates must be between 0 and 1")

        # Shoelace area is positive for top-left, top-right, bottom-right, bottom-left in image coordinates
        signed_area = sum(
            x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        ) / 2.0
        if signed_area < 0.01:
            raise ValueError(
                "corners must go top-left, top-right, bottom-right, bottom-left and enclose the wall"
            )
        if not cv2.isContourConvex(np.float32(points).reshape(-1, 1, 2)):
            raise ValueError("corners must form a convex four-sided shape")
        return points

    def rectify_wall(
        self,
        image_bgr: np.ndarray,
        corners_norm: List[Tuple[float, float]],
        wall_w_mm: float,
        wall_h_mm: float,
        max_dim: int = 1200
    ) -> np.ndarray:
        """Perspective-correct the photo so the marked wall fills the frame at the wall's aspect ratio."""
        img_h, img_w = image_bgr.shape[:2]
        src = np.float32([[x * img_w, y * img_h] for x, y in corners_norm])

        aspect = wall_w_mm / wall_h_mm
        if aspect >= 1.0:
            out_w, out_h = max_dim, max(1, int(round(max_dim / aspect)))
        else:
            out_w, out_h = max(1, int(round(max_dim * aspect))), max_dim
        dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])

        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image_bgr, matrix, (out_w, out_h))

    def normalize_obstacles(
        self,
        obstacles: Any,
        wall_w_mm: float = 4000.0,
        wall_h_mm: float = 2800.0,
    ) -> List[Dict[str, Any]]:
        """
        Validate client-supplied obstacles and refresh id, label and depth from the
        obstacle type, so the server stays the single source of truth for those.
        x/y/w/h are clamped (not rejected) into [0, wall_w_mm] / [0, wall_h_mm] as a
        defensive bound, so an out-of-range client value can never propagate an
        infinite or huge number into the exported mission manifest.
        Raises ValueError with a message safe to show to the operator.
        """
        if not isinstance(obstacles, list):
            raise ValueError("obstacles must be a list")
        if len(obstacles) > self.MAX_OBSTACLES:
            raise ValueError(f"too many obstacles (max {self.MAX_OBSTACLES})")

        cleaned: List[Dict[str, Any]] = []
        for number, obs in enumerate(obstacles, start=1):
            if not isinstance(obs, dict):
                raise ValueError(f"obstacle {number} must be an object")
            try:
                x, y, w, h = (float(obs[key]) for key in ("x", "y", "w", "h"))
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"obstacle {number} needs numeric x, y, w and h")
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                raise ValueError(f"obstacle {number} has an invalid size or position")

            x = max(0.0, min(wall_w_mm, x))
            y = max(0.0, min(wall_h_mm, y))
            w = max(0.0, min(wall_w_mm, w))
            h = max(0.0, min(wall_h_mm, h))

            obs_type = obs.get("type", "unverified")
            if obs_type in OBSTACLE_CLASS_INFO:
                base_label, depth_mm = OBSTACLE_CLASS_INFO[obs_type]
            elif obs_type in LEGACY_OBSTACLE_INFO:
                base_label, depth_mm = LEGACY_OBSTACLE_INFO[obs_type]
            elif obs_type == "unverified":
                base_label, depth_mm = UNVERIFIED_INFO
            else:
                raise ValueError(f"obstacle {number} has unknown type '{obs_type}'")

            cleaned_obs = {
                "id": f"obs_{number}",
                "type": obs_type,
                "label": f"{base_label} #{number}",
                "x": round(x, 1),
                "y": round(y, 1),
                "w": round(w, 1),
                "h": round(h, 1),
                "depth_mm": depth_mm,
            }
            if "confidence" in obs:
                cleaned_obs["confidence"] = obs["confidence"]
            if obs_type == "unverified" and obs.get("guess") in OBSTACLE_CLASS_INFO:
                cleaned_obs["guess"] = obs["guess"]
            cleaned.append(cleaned_obs)
        return cleaned

    def _detect_classical(
        self,
        image_bgr: np.ndarray,
        wall_w_mm: float,
        wall_h_mm: float,
        sensitivity: float = 0.5
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Fallback used when no trained model is loaded: segment openings and fixtures
        with adaptive edge and contour hierarchy analysis. Returns obstacles in wall
        millimetre coordinates and a heuristic confidence score.
        """
        img_h, img_w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Bilateral filter to smooth texture while keeping structural edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Dynamic Canny thresholds tuned by sensitivity parameter
        low_thresh = int(max(20, 50 * (1.2 - sensitivity)))
        high_thresh = int(min(220, 150 * (1.5 - sensitivity)))
        edges = cv2.Canny(filtered, low_thresh, high_thresh)

        # Morphological close to join broken rectangular contour segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = img_w * img_h

        obstacles = []
        wall_aspect = wall_w_mm / max(1.0, wall_h_mm)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            # Filter noise (ignore items < 1.0% and > 85% of total area)
            if area < total_area * 0.012 or area > total_area * 0.85:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            # Ensure within reasonable bounds
            if w < img_w * 0.05 or h < img_h * 0.05:
                continue

            # Convert to physical mm coordinates (Origin: bottom-left corner of wall)
            # In image: (0,0) is top-left. In robot wall coordinates: (0,0) is bottom-left.
            norm_x = x / img_w
            norm_y = (img_h - (y + h)) / img_h   # bottom-left y
            norm_w = w / img_w
            norm_h = h / img_h

            mm_x = round(norm_x * wall_w_mm, 1)
            mm_y = round(norm_y * wall_h_mm, 1)
            mm_w = round(norm_w * wall_w_mm, 1)
            mm_h = round(norm_h * wall_h_mm, 1)

            # Classify feature type based on aspect ratio and position
            aspect = mm_w / max(1.0, mm_h)
            if mm_y < wall_h_mm * 0.2 and aspect < 0.8:
                obs_type = "door"
                label = f"Door Opening #{len(obstacles)+1}"
                depth_mm = 80.0
            elif aspect > 0.5 and mm_h > wall_h_mm * 0.2:
                obs_type = "window"
                label = f"Glazed Window #{len(obstacles)+1}"
                depth_mm = 120.0
            elif mm_w < wall_w_mm * 0.15 and mm_h < wall_h_mm * 0.2:
                obs_type = "switchboard"
                label = f"Utility Panel #{len(obstacles)+1}"
                depth_mm = 35.0
            else:
                obs_type = "fixture"
                label = f"Architectural Opening #{len(obstacles)+1}"
                depth_mm = 50.0

            obstacles.append({
                "id": f"obs_{len(obstacles)+1}",
                "type": obs_type,
                "label": label,
                "x": mm_x,
                "y": mm_y,
                "w": mm_w,
                "h": mm_h,
                "depth_mm": depth_mm,
                "norm": {
                    "x": round(norm_x, 4),
                    "y": round(norm_y, 4),
                    "w": round(norm_w, 4),
                    "h": round(norm_h, 4)
                }
            })

        # Calculate edge detection confidence score
        confidence = min(98.5, max(82.0, 94.0 - len(obstacles) * 1.5 + (sensitivity * 5.0)))

        # Sort obstacles deterministically by X position
        obstacles.sort(key=lambda o: o["x"])
        return obstacles, round(confidence, 1)

    @staticmethod
    def parse_start_corner(raw: Any) -> str:
        """Return a valid start corner. A missing or empty value means the default (bottom-left)."""
        if raw is None or raw == "":
            return DEFAULT_START_CORNER
        if not isinstance(raw, str) or raw not in START_CORNERS:
            raise ValueError(f"start_corner must be one of {', '.join(START_CORNERS)}")
        return raw

    @staticmethod
    def parse_direction(raw: Any) -> str:
        """Return a valid stripe direction. A missing or empty value means horizontal."""
        if raw is None or raw == "":
            return DEFAULT_DIRECTION
        if not isinstance(raw, str) or raw not in DIRECTIONS:
            raise ValueError(f"direction must be one of {', '.join(DIRECTIONS)}")
        return raw

    def plan_coverage_path(
        self,
        wall_w_mm: float,
        wall_h_mm: float,
        spray_width_mm: float,
        overlap_pct: float,
        safety_buffer_mm: float,
        obstacles: List[Dict[str, Any]],
        start_corner: str = DEFAULT_START_CORNER,
        direction: str = DEFAULT_DIRECTION,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Boustrophedon coverage that starts in the given wall corner: rows progress away from
        the start side and the first pass runs from the start corner across the wall.
        The path is planned from the top-left and mirrored, so obstacles are mirrored into
        that frame first (copies, the caller's dicts are not changed).
        direction "vertical" plans the same pattern on the transposed wall (x and y swapped) and
        swaps the result back, so the stripes run up and down.
        """
        start_corner = self.parse_start_corner(start_corner)
        if self.parse_direction(direction) == "vertical":
            swapped = [dict(obs, x=obs["y"], y=obs["x"], w=obs["h"], h=obs["w"]) for obs in obstacles]
            waypoints, stats = self.plan_coverage_path(
                wall_h_mm, wall_w_mm, spray_width_mm, overlap_pct, safety_buffer_mm, swapped,
                _TRANSPOSED_CORNER[start_corner], "horizontal",
            )
            for wp in waypoints:
                wp["x"], wp["y"] = wp["y"], wp["x"]
            return waypoints, stats
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

    def _plan_coverage_from_top_left(
        self,
        wall_w_mm: float,
        wall_h_mm: float,
        spray_width_mm: float,
        overlap_pct: float,
        safety_buffer_mm: float,
        obstacles: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Region-by-region zigzag coverage in the top-left frame; the rover never drives over a keep-out."""
        return plan_regions(
            wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles,
            self.defaults["standoff_dist_mm"],
        )

    def compute_metrics(
        self,
        wall_w_mm: float,
        wall_h_mm: float,
        obstacles: List[Dict[str, Any]],
        path_stats: Dict[str, Any],
        speed_mps: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate complete engineering metrics compliant with APR requirements:
        Gross Area, Masked Area, Net Paintable Area, Paint Volume, and Cycle Time.
        speed_mps overrides the default rover speed (the rover profile's max speed).
        """
        gross_area_m2 = (wall_w_mm * wall_h_mm) / 1e6

        masked_area_m2 = 0.0
        for obs in obstacles:
            masked_area_m2 += (obs["w"] * obs["h"]) / 1e6

        net_paintable_area_m2 = max(0.1, gross_area_m2 - masked_area_m2)
        mask_pct = (masked_area_m2 / max(0.001, gross_area_m2)) * 100.0

        # Paint volume estimation (WBS 1.2 §3: 50 µm target dry film thickness)
        # Wet film thickness WFT = DFT / (solids% / 100)
        # Volume (liters) = Area (m²) * WFT (mm) / transfer_efficiency
        dft_mm = self.defaults["target_dft_um"] / 1000.0  # 0.050 mm
        solids_ratio = self.defaults["paint_solids_pct"] / 100.0
        wft_mm = dft_mm / solids_ratio
        paint_liters = (net_paintable_area_m2 * wft_mm) / self.defaults["transfer_efficiency"]

        # Cycle time estimation (Nominal rover speed = 0.25 m/s, plus row turn time of 3.5s)
        speed = speed_mps if speed_mps is not None else self.defaults["rover_speed_mps"]
        motion_time_sec = path_stats["total_distance_m"] / speed
        turns_time_sec = path_stats["num_rows"] * 3.5
        total_time_sec = motion_time_sec + turns_time_sec

        minutes = int(total_time_sec // 60)
        seconds = int(total_time_sec % 60)

        # Coverage throughput rate (m² / hr)
        hours = total_time_sec / 3600.0
        throughput_m2_per_hr = net_paintable_area_m2 / max(0.001, hours)

        return {
            "gross_area_m2": round(gross_area_m2, 2),
            "masked_area_m2": round(masked_area_m2, 2),
            "net_paintable_area_m2": round(net_paintable_area_m2, 2),
            "mask_percentage": round(mask_pct, 1),
            "paint_volume_liters": round(paint_liters, 2),
            "cycle_time_min": minutes,
            "cycle_time_sec": seconds,
            "cycle_time_formatted": f"{minutes:02d}:{seconds:02d}",
            "throughput_m2_per_hr": round(throughput_m2_per_hr, 1),
            "nominal_speed_mps": speed,
            "target_dft_um": self.defaults["target_dft_um"]
        }

    def generate_benchmark_wall_image(self, width: int = 1200, height: int = 840) -> np.ndarray:
        """
        Generate a photorealistic synthetic benchmark wall with architectural features
        (textured plaster facade, multi-pane window with sill, exterior door, and electrical box)
        for instant 1-click live demonstration.
        """
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Base plaster wall color (light architectural warm off-white / concrete)
        img[:] = (212, 218, 222)

        # Add subtle surface noise/texture
        noise = np.random.normal(0, 4, (height, width, 3)).astype(np.int16)
        img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = img_noisy

        # Add horizontal brick/panel mortar seams for realistic depth
        for y in range(60, height, 120):
            cv2.line(img, (0, y), (width, y), (185, 190, 195), 1)

        # Feature 1: Window (Center-Right)
        wx1, wy1, wx2, wy2 = int(width * 0.48), int(height * 0.22), int(width * 0.85), int(height * 0.65)
        # Window outer frame
        cv2.rectangle(img, (wx1 - 8, wy1 - 8), (wx2 + 8, wy2 + 8), (70, 75, 80), -1)
        # Glazing (tinted blue-grey glass reflection)
        cv2.rectangle(img, (wx1, wy1), (wx2, wy2), (130, 105, 85), -1)
        # Window mullions (cross-grid)
        mx = (wx1 + wx2) // 2
        my = (wy1 + wy2) // 2
        cv2.line(img, (mx, wy1), (mx, wy2), (210, 210, 215), 4)
        cv2.line(img, (wx1, my), (wx2, my), (210, 210, 215), 4)
        # Window sill
        cv2.rectangle(img, (wx1 - 14, wy2 + 8), (wx2 + 14, wy2 + 22), (55, 60, 65), -1)

        # Feature 2: Utility / Electrical Box (Top-Left)
        ex1, ey1, ex2, ey2 = int(width * 0.12), int(height * 0.18), int(width * 0.28), int(height * 0.38)
        cv2.rectangle(img, (ex1, ey1), (ex2, ey2), (90, 95, 100), -1)
        cv2.rectangle(img, (ex1 + 6, ey1 + 6), (ex2 - 6, ey2 - 6), (115, 120, 125), -1)
        cv2.circle(img, (ex2 - 16, (ey1 + ey2) // 2), 4, (40, 45, 50), -1)

        # Feature 3: Base Plinth / Lower skirting board
        cv2.rectangle(img, (0, height - 35), (width, height), (90, 95, 100), -1)

        return img