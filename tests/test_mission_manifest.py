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
