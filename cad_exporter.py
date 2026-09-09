"""
APR (Automated Painting Rover) - CAD & 3D Model Exporter
Traced to MSME Grant INC25ETS084848 / WBS 1.2 §3 & Software Subsystem Slide 5.
Generates:
1. AutoCAD-compliant ASCII DXF files with layered geometry.
2. 3D Wavefront (.OBJ / .MTL) digital twin meshes with extruded wall geometry and recessed openings.
3. ROS 2 Mission Execution Plan (JSON).
4. Motion Controller Waypoint Data (CSV).
"""

import json
import math
from typing import Dict, List, Any, Tuple


class CADExporter:
    def __init__(self):
        pass

    def export_dxf(
        self,
        wall_w_mm: float,
        wall_h_mm: float,
        obstacles: List[Dict[str, Any]],
        waypoints: List[Dict[str, Any]]
    ) -> str:
        """
        Generate AutoCAD-compliant ASCII DXF (Release 12 / 2000).
        Layers:
          - 0: Default
          - WALL_BOUNDARY: Cyan (Color 4), continuous line
          - NO_PAINT_OBSTACLES: Red (Color 1), keep-out boundary
          - ROVER_TOOLPATH: Yellow (Color 2), serpentine spray moves
          - TRANSIT_MOVES: Magenta (Color 6), non-spray transitions
          - WAYPOINTS: Green (Color 3), coordinates & tags
        """
        lines = []

        # DXF Header
        lines.extend([
            "0", "SECTION",
            "2", "HEADER",
            "9", "$ACADVER", "1", "AC1009",  # AutoCAD R12 compatibility
            "9", "$INSUNITS", "70", "4",     # 4 = Millimeters
            "0", "ENDSEC"
        ])

        # DXF Tables (Layer definitions)
        lines.extend([
            "0", "SECTION",
            "2", "TABLES",
            "0", "TABLE",
            "2", "LAYER",
            "70", "5",
            # Layer: WALL_BOUNDARY (Cyan=4)
            "0", "LAYER", "2", "WALL_BOUNDARY", "70", "0", "62", "4", "6", "CONTINUOUS",
            # Layer: NO_PAINT_OBSTACLES (Red=1)
            "0", "LAYER", "2", "NO_PAINT_OBSTACLES", "70", "0", "62", "1", "6", "CONTINUOUS",
            # Layer: ROVER_TOOLPATH (Yellow=2)
            "0", "LAYER", "2", "ROVER_TOOLPATH", "70", "0", "62", "2", "6", "CONTINUOUS",
            # Layer: TRANSIT_MOVES (Magenta=6)
            "0", "LAYER", "2", "TRANSIT_MOVES", "70", "0", "62", "6", "6", "DASHED",
            # Layer: WAYPOINTS (Green=3)
            "0", "LAYER", "2", "WAYPOINTS", "70", "0", "62", "3", "6", "CONTINUOUS",
            "0", "ENDTAB",
            "0", "ENDSEC"
        ])

        # DXF Entities
        lines.extend([
            "0", "SECTION",
            "2", "ENTITIES"
        ])

        def add_line(layer: str, x1: float, y1: float, x2: float, y2: float):
            return [
                "0", "LINE",
                "8", layer,
                "10", f"{x1:.2f}", "20", f"{y1:.2f}", "30", "0.0",
                "11", f"{x2:.2f}", "21", f"{y2:.2f}", "31", "0.0"
            ]

        def add_rect(layer: str, x: float, y: float, w: float, h: float):
            res = []
            res.extend(add_line(layer, x, y, x + w, y))
            res.extend(add_line(layer, x + w, y, x + w, y + h))
            res.extend(add_line(layer, x + w, y + h, x, y + h))
            res.extend(add_line(layer, x, y + h, x, y))
            return res

        # 1. Outer Wall Boundary
        lines.extend(add_rect("WALL_BOUNDARY", 0.0, 0.0, wall_w_mm, wall_h_mm))

        # 2. Obstacles (No-paint zones)
        for obs in obstacles:
            ox, oy, ow, oh = obs["x"], obs["y"], obs["w"], obs["h"]
            lines.extend(add_rect("NO_PAINT_OBSTACLES", ox, oy, ow, oh))

        # 3. Toolpath & Waypoint Entities
        for i in range(1, len(waypoints)):
            p1 = waypoints[i - 1]
            p2 = waypoints[i]
            # Use ROVER_TOOLPATH if both points are spray active and on the same row, else TRANSIT_MOVES
            layer = "ROVER_TOOLPATH" if (p1.get("spray_active") and p2.get("spray_active") and p1.get("row") == p2.get("row")) else "TRANSIT_MOVES"
            lines.extend(add_line(layer, p1["x"], p1["y"], p2["x"], p2["y"]))

        # 4. Waypoint Markers (Small circles + text sequence)
        for wp in waypoints:
            wx, wy = wp["x"], wp["y"]
            lines.extend([
                "0", "POINT",
                "8", "WAYPOINTS",
                "10", f"{wx:.2f}", "20", f"{wy:.2f}", "30", f"{wp.get('z', 250.0):.2f}"
            ])

        lines.extend([
            "0", "ENDSEC",
            "0", "EOF"
        ])

        return "\n".join(lines) + "\n"

    def export_obj_and_mtl(
        self,
        wall_w_mm: float,
        wall_h_mm: float,
        obstacles: List[Dict[str, Any]],
        waypoints: List[Dict[str, Any]],
        wall_depth_mm: float = 200.0
    ) -> Dict[str, str]:
        """
        Generate 3D Wavefront .OBJ and .MTL files.
        Models the physical wall as an extruded 3D volume, with recessed
        openings for windows/doors, extruded plates for switchboards,
        and 3D ribbons/tubes for the rover painting toolpath.
        """
        scale = 0.001  # Convert mm to meters for 3D engine standard units

        W = wall_w_mm * scale
        H = wall_h_mm * scale
        D = wall_depth_mm * scale

        obj_lines = [
            "# Automated Painting Rover (APR) - 3D Digital Twin Model",
            "# Reference: MSME Grant INC25ETS084848 / WBS 1.2 §3",
            "mtllib paintpilot_wall.mtl",
            "o APR_Wall_Assembly"
        ]

        # Vertices with 3D position and RGB vertex colors: (x, y, z, r, g, b)
        vertices: List[Tuple[float, float, float, float, float, float]] = []
        groups: List[Tuple[str, str, List[Tuple[int, ...]]]] = []

        def add_vertex(x: float, y: float, z: float, r: float = 0.5, g: float = 0.5, b: float = 0.5) -> int:
            vertices.append((x, y, z, r, g, b))
            return len(vertices)

        def add_face(group_name: str, mtl_name: str, face_v: Tuple[int, ...]):
            if not groups or groups[-1][0] != group_name or groups[-1][1] != mtl_name:
                groups.append((group_name, mtl_name, []))
            groups[-1][2].append(face_v)

        # Standard Color Palette for Vertex Colors & CAD Viewers
        CLR_WALL_CONCRETE = (0.42, 0.46, 0.50)    # Concrete Backing Grey
        CLR_WALL_PAINTABLE = (0.10, 0.72, 0.62)   # Vibrant Emerald / Cyan
        CLR_OBSTACLE_RED = (0.98, 0.18, 0.18)     # Bright Vibrant Crimson Red (Keep-Out Zone)
        CLR_OBSTACLE_BORDER = (0.75, 0.12, 0.12)  # Darker Red 3D Bevel Border
        CLR_WINDOW_GLASS = (0.12, 0.28, 0.48)     # Tinted Glazing
        CLR_SWITCHBOARD = (0.96, 0.55, 0.15)      # Amber / Orange Utility Panel
        CLR_PATH_SPRAY = (1.00, 0.78, 0.08)       # Golden Amber / Bright Yellow
        CLR_PATH_TRANSIT = (0.85, 0.20, 0.85)     # Magenta / Purple

        # 1. Base Wall Geometry (Subdivided into paintable face + side/back faces)
        # Back face (z = -D)
        v1 = add_vertex(0, 0, -D, *CLR_WALL_CONCRETE)
        v2 = add_vertex(W, 0, -D, *CLR_WALL_CONCRETE)
        v3 = add_vertex(W, H, -D, *CLR_WALL_CONCRETE)
        v4 = add_vertex(0, H, -D, *CLR_WALL_CONCRETE)
        add_face("Wall_Backing_Structure", "Wall_Concrete_Material", (v4, v3, v2, v1))

        # Left, Right, Bottom, Top border faces
        # Left
        v5 = add_vertex(0, 0, 0, *CLR_WALL_CONCRETE)
        v6 = add_vertex(0, H, 0, *CLR_WALL_CONCRETE)
        add_face("Wall_Backing_Structure", "Wall_Concrete_Material", (v1, v4, v6, v5))
        # Right
        v7 = add_vertex(W, 0, 0, *CLR_WALL_CONCRETE)
        v8 = add_vertex(W, H, 0, *CLR_WALL_CONCRETE)
        add_face("Wall_Backing_Structure", "Wall_Concrete_Material", (v7, v8, v3, v2))
        # Top
        add_face("Wall_Backing_Structure", "Wall_Concrete_Material", (v6, v4, v3, v8))
        # Bottom
        add_face("Wall_Backing_Structure", "Wall_Concrete_Material", (v5, v7, v2, v1))

        # Front Paintable Face (Emerald/Cyan Target Surface)
        pf1 = add_vertex(0, 0, 0.001, *CLR_WALL_PAINTABLE)
        pf2 = add_vertex(W, 0, 0.001, *CLR_WALL_PAINTABLE)
        pf3 = add_vertex(W, H, 0.001, *CLR_WALL_PAINTABLE)
        pf4 = add_vertex(0, H, 0.001, *CLR_WALL_PAINTABLE)
        add_face("Paintable_Target_Surface", "Paintable_Surface_Material", (pf1, pf2, pf3, pf4))

        # 2. Add Prominent Red Keep-Out Obstacle Zones & Fixtures
        for obs in obstacles:
            ox = obs["x"] * scale
            oy = obs["y"] * scale
            ow = obs["w"] * scale
            oh = obs["h"] * scale
            depth = obs.get("depth_mm", 80.0) * scale

            gname = f"KeepOut_Zone_{obs['id']}_{obs['type']}"
            zf = 0.008  # Placed 8mm in front of wall so red zone is vividly visible from all angles

            # 2A. Main Red Keep-Out Plate (Front Face)
            r1 = add_vertex(ox, oy, zf, *CLR_OBSTACLE_RED)
            r2 = add_vertex(ox + ow, oy, zf, *CLR_OBSTACLE_RED)
            r3 = add_vertex(ox + ow, oy + oh, zf, *CLR_OBSTACLE_RED)
            r4 = add_vertex(ox, oy + oh, zf, *CLR_OBSTACLE_RED)
            add_face(gname, "Obstacle_KeepOut_Red_Material", (r1, r2, r3, r4))

            # 2B. 3D Bevel Borders connecting Wall (z=0.001) to Plate (zf=0.008)
            b1 = add_vertex(ox, oy, 0.001, *CLR_OBSTACLE_BORDER)
            b2 = add_vertex(ox + ow, oy, 0.001, *CLR_OBSTACLE_BORDER)
            b3 = add_vertex(ox + ow, oy + oh, 0.001, *CLR_OBSTACLE_BORDER)
            b4 = add_vertex(ox, oy + oh, 0.001, *CLR_OBSTACLE_BORDER)

            add_face(gname + "_Border", "Obstacle_KeepOut_Red_Material", (b1, b2, r2, r1))  # Bottom
            add_face(gname + "_Border", "Obstacle_KeepOut_Red_Material", (r4, r3, b3, b4))  # Top
            add_face(gname + "_Border", "Obstacle_KeepOut_Red_Material", (b4, b1, r1, r4))  # Left
            add_face(gname + "_Border", "Obstacle_KeepOut_Red_Material", (r2, b2, b3, r3))  # Right

            # 2C. Inset Fixture Details inside the Red Zone
            if obs["type"] == "window":
                inset_x = ow * 0.09
                inset_y = oh * 0.09
                wx1 = ox + inset_x
                wy1 = oy + inset_y
                wx2 = ox + ow - inset_x
                wy2 = oy + oh - inset_y
                wz = zf + 0.003

                g1 = add_vertex(wx1, wy1, wz, *CLR_WINDOW_GLASS)
                g2 = add_vertex(wx2, wy1, wz, *CLR_WINDOW_GLASS)
                g3 = add_vertex(wx2, wy2, wz, *CLR_WINDOW_GLASS)
                g4 = add_vertex(wx1, wy2, wz, *CLR_WINDOW_GLASS)
                add_face(gname + "_Glazing", "Window_Glazing_Material", (g1, g2, g3, g4))

            elif obs["type"] == "switchboard":
                inset_x = ow * 0.12
                inset_y = oh * 0.12
                sx1 = ox + inset_x
                sy1 = oy + inset_y
                sx2 = ox + ow - inset_x
                sy2 = oy + oh - inset_y
                sz = zf + 0.015

                s1 = add_vertex(sx1, sy1, sz, *CLR_SWITCHBOARD)
                s2 = add_vertex(sx2, sy1, sz, *CLR_SWITCHBOARD)
                s3 = add_vertex(sx2, sy2, sz, *CLR_SWITCHBOARD)
                s4 = add_vertex(sx1, sy2, sz, *CLR_SWITCHBOARD)
                add_face(gname + "_Panel", "Electrical_Fixture_Material", (s1, s2, s3, s4))

        # 3. Add 3D Toolpath Waypoint Trajectory (Elevated bold ribbon with bright golden/magenta colors)
        standoff_m = 0.025  # 25mm elevation off wall face
        ribbon_w = 0.035    # 35mm ribbon width for high visibility

        for i in range(1, len(waypoints)):
            wp1 = waypoints[i - 1]
            wp2 = waypoints[i]

            x1, y1 = wp1["x"] * scale, wp1["y"] * scale
            x2, y2 = wp2["x"] * scale, wp2["y"] * scale

            # Direction vector
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < 1e-4:
                continue

            # Perpendicular vector
            px = -dy / length * ribbon_w
            py = dx / length * ribbon_w

            is_spray = wp1.get("spray_active") and wp2.get("spray_active") and (wp1.get("row") == wp2.get("row"))
            mtl_name = "Toolpath_Spray_Active_Material" if is_spray else "Toolpath_Transit_Material"
            clr = CLR_PATH_SPRAY if is_spray else CLR_PATH_TRANSIT

            r1 = add_vertex(x1 - px, y1 - py, standoff_m, *clr)
            r2 = add_vertex(x1 + px, y1 + py, standoff_m, *clr)
            r3 = add_vertex(x2 + px, y2 + py, standoff_m, *clr)
            r4 = add_vertex(x2 - px, y2 - py, standoff_m, *clr)
            add_face("Rover_Coverage_Toolpath", mtl_name, (r1, r2, r3, r4))

        # Write vertex list (with XYZ + RGB) and face groups
        out_lines = [
            "# Automated Painting Rover (APR) - 3D Digital Twin Model",
            "# Reference: MSME Grant INC25ETS084848 / WBS 1.2 §3",
            "mtllib paintpilot_wall.mtl",
            "o APR_Wall_Assembly"
        ]

        for v in vertices:
            out_lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {v[3]:.3f} {v[4]:.3f} {v[5]:.3f}")

        for gname, mtl_name, face_list in groups:
            out_lines.append(f"g {gname}")
            out_lines.append(f"usemtl {mtl_name}")
            for face in face_list:
                out_lines.append("f " + " ".join(str(idx) for idx in face))

        obj_content = "\n".join(out_lines) + "\n"

        # Material library content (.MTL) with vivid diffuse and ambient reflectance
        mtl_content = (
            "# Material Library for APR Wall Model\n"
            "# Created for PaintPilot Autonomous Painting Rover\n\n"
            "newmtl Wall_Concrete_Material\n"
            "Ka 0.42 0.46 0.50\n"
            "Kd 0.42 0.46 0.50\n"
            "Ks 0.10 0.10 0.10\n"
            "Ns 10.0\n"
            "d 1.0\n"
            "illum 2\n\n"
            "newmtl Paintable_Surface_Material\n"
            "Ka 0.10 0.72 0.62\n"
            "Kd 0.10 0.72 0.62\n"
            "Ks 0.25 0.25 0.25\n"
            "Ns 25.0\n"
            "d 1.0\n"
            "illum 2\n\n"
            "newmtl Obstacle_KeepOut_Red_Material\n"
            "Ka 0.98 0.18 0.18\n"
            "Kd 0.98 0.18 0.18\n"
            "Ks 0.20 0.20 0.20\n"
            "Ns 20.0\n"
            "d 1.0\n"
            "illum 2\n\n"
            "newmtl Window_Glazing_Material\n"
            "Ka 0.12 0.28 0.48\n"
            "Kd 0.12 0.28 0.48\n"
            "Ks 0.80 0.80 0.90\n"
            "Ns 80.0\n"
            "d 0.85\n"
            "illum 2\n\n"
            "newmtl Electrical_Fixture_Material\n"
            "Ka 0.96 0.55 0.15\n"
            "Kd 0.96 0.55 0.15\n"
            "Ks 0.30 0.30 0.30\n"
            "Ns 25.0\n"
            "d 1.0\n"
            "illum 2\n\n"
            "newmtl Toolpath_Spray_Active_Material\n"
            "Ka 1.00 0.78 0.08\n"
            "Kd 1.00 0.78 0.08\n"
            "Ks 0.50 0.50 0.20\n"
            "Ns 40.0\n"
            "d 1.0\n"
            "illum 2\n\n"
            "newmtl Toolpath_Transit_Material\n"
            "Ka 0.85 0.20 0.85\n"
            "Kd 0.85 0.20 0.85\n"
            "Ks 0.30 0.30 0.30\n"
            "Ns 15.0\n"
            "d 0.80\n"
            "illum 2\n"
        )

        return {"obj": obj_content, "mtl": mtl_content}

    def export_mission_json(
        self,
        wall_w_mm: float,
        wall_h_mm: float,
        obstacles: List[Dict[str, Any]],
        waypoints: List[Dict[str, Any]],
        metrics: Dict[str, Any]
    ) -> str:
        """
        Generate ROS 2 / APR Mission Planner execution manifest JSON schema.
        Directly integrates with `apr_mission_planner` and `apr_coverage_planner`.
        """
        manifest = {
            "project": "Automated Painting Rover (APR)",
            "grant_reference": "MSME INC25ETS084848",
            "wbs_module": "WBS 1.2 - Coverage Planning & Wall Geometry Subsystem",
            "coordinate_frame": {
                "system": "WALL-FRAME-CARTESIAN",
                "origin": "BOTTOM_LEFT_CORNER",
                "units": "millimeters",
                "axis_x": "HORIZONTAL_RIGHT",
                "axis_y": "VERTICAL_UP",
                "axis_z": "STANDOFF_NORMAL_AWAY_FROM_WALL"
            },
            "wall_geometry": {
                "width_mm": wall_w_mm,
                "height_mm": wall_h_mm,
                "gross_area_m2": metrics.get("gross_area_m2"),
                "net_paintable_area_m2": metrics.get("net_paintable_area_m2"),
                "masked_area_m2": metrics.get("masked_area_m2")
            },
            "spray_parameters": {
                "spray_width_mm": 250.0,
                "target_dft_um": 50.0,
                "nominal_speed_mps": 0.25,
                "estimated_paint_liters": metrics.get("paint_volume_liters"),
                "estimated_cycle_time": metrics.get("cycle_time_formatted")
            },
            "no_paint_obstacles": [
                {
                    "id": o["id"],
                    "type": o["type"],
                    "label": o["label"],
                    "bounding_box_mm": {"x": o["x"], "y": o["y"], "width": o["w"], "height": o["h"]},
                    "safety_buffer_mm": 60.0
                }
                for o in obstacles
            ],
            "waypoint_count": len(waypoints),
            "waypoints": waypoints
        }
        return json.dumps(manifest, indent=2)

    def export_motion_csv(self, waypoints: List[Dict[str, Any]]) -> str:
        """
        Generate machine-readable CSV for tachometer/encoder drive controller & spray solenoid gate.
        Columns: seq,x_mm,y_mm,z_mm,spray_active,pass_type,row
        """
        lines = ["seq,x_mm,y_mm,z_mm,spray_active,pass_type,row"]
        for wp in waypoints:
            spray = 1 if wp.get("spray_active", False) else 0
            lines.append(f"{wp['seq']},{wp['x']:.2f},{wp['y']:.2f},{wp.get('z', 250.0):.2f},{spray},{wp.get('type', 'PASS')},{wp.get('row', 1)}")
        return "\n".join(lines) + "\n"
