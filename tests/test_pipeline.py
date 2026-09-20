"""
Automated Painting Rover (APR) - Test Suite
Verifies CV segmentation, Boustrophedon path planning, CAD/OBJ exports, and Flask API.
Ref: MSME Grant INC25ETS084848 / WBS 1.2
"""

import os
import sys
import json
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry_engine import GeometryEngine
from cad_exporter import CADExporter
from app import app


class TestPaintPilotPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = GeometryEngine()
        cls.exporter = CADExporter()
        cls.client = app.test_client()

    def test_01_synthetic_image_generation(self):
        """Verify synthetic benchmark image generation creates valid 3-channel image."""
        img = self.engine.generate_benchmark_wall_image(1200, 840)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape, (840, 1200, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_02_obstacle_detection(self):
        """Verify OpenCV segmentation extracts obstacles and valid physical coordinates."""
        img = self.engine.generate_benchmark_wall_image(1200, 840)
        wall_w = 4000.0
        wall_h = 2800.0
        obstacles, confidence = self.engine.detect_obstacles_from_image(img, wall_w, wall_h, sensitivity=0.5)

        self.assertIsInstance(obstacles, list)
        self.assertGreaterEqual(len(obstacles), 1)
        self.assertGreaterEqual(confidence, 75.0)

        for obs in obstacles:
            self.assertIn("id", obs)
            self.assertIn("type", obs)
            self.assertIn("x", obs)
            self.assertIn("y", obs)
            self.assertIn("w", obs)
            self.assertIn("h", obs)
            self.assertGreater(obs["w"], 0)
            self.assertGreater(obs["h"], 0)
            self.assertLessEqual(obs["x"] + obs["w"], wall_w + 1.0)
            self.assertLessEqual(obs["y"] + obs["h"], wall_h + 1.0)

    def test_03_coverage_path_planning(self):
        """Verify Boustrophedon serpentine path generation and obstacle standoff clipping."""
        wall_w = 4000.0
        wall_h = 2800.0
        spray_w = 250.0
        overlap = 12.0
        buffer_mm = 60.0

        sample_obstacles = [
            {
                "id": "obs_1",
                "type": "window",
                "label": "Window",
                "x": 1500.0,
                "y": 1000.0,
                "w": 1000.0,
                "h": 800.0,
                "depth_mm": 120.0
            }
        ]

        waypoints, stats = self.engine.plan_coverage_path(
            wall_w, wall_h, spray_w, overlap, buffer_mm, sample_obstacles
        )

        self.assertIsInstance(waypoints, list)
        self.assertGreater(len(waypoints), 10)
        self.assertGreater(stats["num_rows"], 5)
        self.assertGreater(stats["total_distance_m"], 10.0)
        self.assertGreater(stats["spray_distance_m"], 0.0)

        # Check waypoint structure
        for wp in waypoints:
            self.assertIn("seq", wp)
            self.assertIn("x", wp)
            self.assertIn("y", wp)
            self.assertIn("z", wp)
            self.assertIn("spray_active", wp)
            self.assertIn("type", wp)
            self.assertIn("row", wp)
            self.assertGreaterEqual(wp["x"], -0.1)
            self.assertLessEqual(wp["x"], wall_w + 0.1)

    def test_04_engineering_metrics(self):
        """Verify calculation of paint volume (50um DFT) and cycle time (0.25 m/s)."""
        wall_w = 4000.0
        wall_h = 2800.0
        sample_obstacles = [
            {"id": "obs_1", "type": "window", "x": 1000, "y": 1000, "w": 1000, "h": 1000}
        ]
        path_stats = {
            "num_rows": 13,
            "effective_step_mm": 220.0,
            "total_distance_m": 50.0,
            "spray_distance_m": 38.0,
            "transit_distance_m": 12.0,
            "waypoint_count": 70
        }

        metrics = self.engine.compute_metrics(wall_w, wall_h, sample_obstacles, path_stats)

        self.assertEqual(metrics["gross_area_m2"], 11.2)
        self.assertEqual(metrics["masked_area_m2"], 1.0)
        self.assertEqual(metrics["net_paintable_area_m2"], 10.2)
        self.assertGreater(metrics["paint_volume_liters"], 0.5)
        self.assertGreater(metrics["throughput_m2_per_hr"], 10.0)
        self.assertEqual(metrics["target_dft_um"], 50.0)

    def test_05_cad_dxf_export(self):
        """Verify AutoCAD DXF format compliance and layer definitions."""
        dxf = self.exporter.export_dxf(
            4000.0, 2800.0,
            [{"id": "obs_1", "type": "window", "x": 500, "y": 500, "w": 500, "h": 500}],
            [
                {"seq": 1, "x": 0, "y": 125, "z": 250, "spray_active": True, "row": 1},
                {"seq": 2, "x": 4000, "y": 125, "z": 250, "spray_active": True, "row": 1}
            ]
        )
        self.assertIn("SECTION", dxf)
        self.assertIn("HEADER", dxf)
        self.assertIn("TABLES", dxf)
        self.assertIn("WALL_BOUNDARY", dxf)
        self.assertIn("NO_PAINT_OBSTACLES", dxf)
        self.assertIn("ROVER_TOOLPATH", dxf)
        self.assertIn("WAYPOINTS", dxf)
        self.assertIn("EOF", dxf)

    def test_06_3d_obj_mtl_export(self):
        """Verify 3D Wavefront OBJ mesh structure and MTL material bindings."""
        bundle = self.exporter.export_obj_and_mtl(
            4000.0, 2800.0,
            [{"id": "obs_1", "type": "window", "x": 500, "y": 500, "w": 500, "h": 500, "depth_mm": 120}],
            [
                {"seq": 1, "x": 0, "y": 125, "z": 250, "spray_active": True, "row": 1},
                {"seq": 2, "x": 4000, "y": 125, "z": 250, "spray_active": True, "row": 1}
            ]
        )
        obj = bundle["obj"]
        mtl = bundle["mtl"]

        self.assertIn("mtllib paintpilot_wall.mtl", obj)
        self.assertIn("o APR_Wall_Assembly", obj)
        self.assertIn("v ", obj)
        self.assertIn("g Wall_Backing_Structure", obj)
        self.assertIn("usemtl Wall_Concrete_Material", obj)
        self.assertIn("g Paintable_Target_Surface", obj)
        self.assertIn("usemtl Paintable_Surface_Material", obj)
        self.assertIn("f ", obj)

        self.assertIn("newmtl Wall_Concrete_Material", mtl)
        self.assertIn("newmtl Paintable_Surface_Material", mtl)
        self.assertIn("newmtl Window_Glazing_Material", mtl)
        self.assertIn("newmtl Toolpath_Spray_Active_Material", mtl)

    def test_07_flask_endpoints(self):
        """Verify Flask web server routes and API responses."""
        # 1. Index page
        res_idx = self.client.get("/")
        self.assertEqual(res_idx.status_code, 200)
        self.assertIn(b"PAINTPILOT", res_idx.data)

        # 2. Benchmark samples endpoint
        res_samp = self.client.get("/api/samples")
        self.assertEqual(res_samp.status_code, 200)
        data_samp = json.loads(res_samp.data)
        self.assertTrue(data_samp["success"])
        self.assertGreaterEqual(len(data_samp["samples"]), 2)

        # 3. Analyze wall endpoint
        res_ana = self.client.post("/api/analyze", data={
            "width_mm": "4000",
            "height_mm": "2800",
            "spray_width_mm": "250",
            "overlap_pct": "12",
            "safety_buffer_mm": "60",
            "sensitivity": "0.5",
            "sample_id": "residential"
        })
        self.assertEqual(res_ana.status_code, 200)
        data_ana = json.loads(res_ana.data)
        self.assertTrue(data_ana["success"])
        self.assertIn("wall", data_ana)
        self.assertIn("obstacles", data_ana)
        self.assertIn("waypoints", data_ana)
        self.assertIn("metrics", data_ana)
        self.assertIn("image_data_url", data_ana)

        # 4. Download endpoints
        for fmt in ["dxf", "obj", "mtl", "zip", "json", "csv"]:
            res_dl = self.client.get(f"/api/download/{fmt}")
            self.assertEqual(res_dl.status_code, 200, f"Failed download for {fmt}")
            self.assertGreater(len(res_dl.data), 50)

    def test_08_new_obstacle_classes_export_without_error(self):
        """New detector classes must not break the DXF/OBJ exporters."""
        waypoints = [
            {"seq": 1, "x": 0, "y": 125, "z": 250, "spray_active": True, "row": 1},
            {"seq": 2, "x": 4000, "y": 125, "z": 250, "spray_active": True, "row": 1},
        ]
        for obstacle_type in ("meter_panel", "ac_unit", "pipe", "grill", "unverified"):
            obstacles = [{"id": "obs_1", "type": obstacle_type, "x": 500, "y": 500, "w": 500, "h": 500, "depth_mm": 100}]
            bundle = self.exporter.export_obj_and_mtl(4000.0, 2800.0, obstacles, waypoints)
            self.assertIn("f ", bundle["obj"], obstacle_type)
            self.assertIn("NO_PAINT_OBSTACLES", self.exporter.export_dxf(4000.0, 2800.0, obstacles, waypoints))

        meter = [{"id": "obs_1", "type": "meter_panel", "x": 500, "y": 500, "w": 500, "h": 500, "depth_mm": 100}]
        obj = self.exporter.export_obj_and_mtl(4000.0, 2800.0, meter, waypoints)["obj"]
        self.assertIn("usemtl Electrical_Fixture_Material", obj)


if __name__ == "__main__":
    unittest.main()
