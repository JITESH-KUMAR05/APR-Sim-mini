"""
Automated Painting Rover (APR) - Wall-to-3D CAD & Surface Segmentation Server
Reference: MSME Grant INC25ETS084848 / WBS 1.2 §3 & Slide 5/6 Architecture.
Pair-programmed for presentation & engineering demonstration.
"""

import os
import io
import base64
import time
import zipfile
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

from geometry_engine import GeometryEngine
from cad_exporter import CADExporter

# Define absolute paths for static assets and templates (vital for Vercel/serverless environments)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

if os.environ.get("VERCEL") or not os.access(BASE_DIR, os.W_OK):
    EXPORTS_DIR = os.path.join("/tmp", "exports")
else:
    EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
os.makedirs(EXPORTS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR, static_url_path="/static")
CORS(app)

geometry_engine = GeometryEngine()
cad_exporter = CADExporter()

# In-memory cached active state for immediate downloads
current_session = {
    "wall_w_mm": 4000.0,
    "wall_h_mm": 2800.0,
    "obstacles": [],
    "waypoints": [],
    "metrics": {},
    "stats": {},
    "confidence": 94.2
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Explicit static handler for serverless environments (Vercel)."""
    return send_from_directory(STATIC_DIR, filename)


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return available pre-packaged benchmark wall samples."""
    samples = [
        {
            "id": "residential",
            "name": "Residential Exterior Facade",
            "desc": "Textured plaster finish, glazed multi-pane window & electrical enclosure (4.0m x 2.8m)",
            "width_mm": 4000,
            "height_mm": 2800,
            "file": "benchmark_residential_facade.jpg"
        },
        {
            "id": "commercial",
            "name": "Commercial Building Wall",
            "desc": "Smooth concrete finish with industrial access door & ventilation louvre (4.5m x 3.0m)",
            "width_mm": 4500,
            "height_mm": 3000,
            "file": "benchmark_commercial_facade.jpg"
        }
    ]
    return jsonify({"success": True, "samples": samples})


@app.route("/api/analyze", methods=["POST"])
def analyze_wall():
    """
    Core API endpoint:
    Processes wall photo, segments paintable vs keep-out obstacle zones,
    generates 3D CAD mesh + Boustrophedon rover toolpath, and prepares exports.
    """
    try:
        # Extract physical calibration parameters
        wall_w_mm = float(request.form.get("width_mm", 4000.0))
        wall_h_mm = float(request.form.get("height_mm", 2800.0))
        spray_width_mm = float(request.form.get("spray_width_mm", 250.0))
        overlap_pct = float(request.form.get("overlap_pct", 12.0))
        safety_buffer_mm = float(request.form.get("safety_buffer_mm", 60.0))
        sensitivity = float(request.form.get("sensitivity", 0.5))
        sample_id = request.form.get("sample_id", None)

        # Sanitize parameters
        wall_w_mm = max(1000.0, min(50000.0, wall_w_mm))
        wall_h_mm = max(1000.0, min(30000.0, wall_h_mm))
        spray_width_mm = max(150.0, min(500.0, spray_width_mm))
        overlap_pct = max(0.0, min(40.0, overlap_pct))
        safety_buffer_mm = max(10.0, min(300.0, safety_buffer_mm))

        # 1. Load image (from uploaded file or pre-packaged sample)
        img_bgr = None
        source_name = "Uploaded Image"

        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            source_name = file.filename
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        elif sample_id == "commercial":
            sample_path = os.path.join(SAMPLES_DIR, "benchmark_commercial_facade.jpg")
            if os.path.exists(sample_path):
                img_bgr = cv2.imread(sample_path)
                source_name = "Commercial Facade Benchmark"
        else:
            # Default or residential sample
            sample_path = os.path.join(SAMPLES_DIR, "benchmark_residential_facade.jpg")
            if os.path.exists(sample_path):
                img_bgr = cv2.imread(sample_path)
                source_name = "Residential Facade Benchmark"
            else:
                img_bgr = geometry_engine.generate_benchmark_wall_image()
                source_name = "Synthetic Calibrated Wall"

        if img_bgr is None:
            img_bgr = geometry_engine.generate_benchmark_wall_image()
            source_name = "Synthetic Calibrated Wall"

        # 2. Run computer vision segmentation
        obstacles, confidence = geometry_engine.detect_obstacles_from_image(
            img_bgr, wall_w_mm, wall_h_mm, sensitivity
        )

        # 3. Generate Boustrophedon coverage path
        waypoints, path_stats = geometry_engine.plan_coverage_path(
            wall_w_mm, wall_h_mm, spray_width_mm, overlap_pct, safety_buffer_mm, obstacles
        )

        # 4. Compute engineering metrics
        metrics = geometry_engine.compute_metrics(
            wall_w_mm, wall_h_mm, obstacles, path_stats
        )

        # 5. Export CAD and 3D files to disk for download
        dxf_content = cad_exporter.export_dxf(wall_w_mm, wall_h_mm, obstacles, waypoints)
        dxf_path = os.path.join(EXPORTS_DIR, "paintpilot_wall.dxf")
        with open(dxf_path, "w", encoding="utf-8") as f:
            f.write(dxf_content)

        obj_bundle = cad_exporter.export_obj_and_mtl(wall_w_mm, wall_h_mm, obstacles, waypoints)
        obj_path = os.path.join(EXPORTS_DIR, "paintpilot_wall.obj")
        mtl_path = os.path.join(EXPORTS_DIR, "paintpilot_wall.mtl")
        with open(obj_path, "w", encoding="utf-8") as f:
            f.write(obj_bundle["obj"])
        with open(mtl_path, "w", encoding="utf-8") as f:
            f.write(obj_bundle["mtl"])

        # Package 3D bundle (OBJ + MTL) into a single zip archive for 1-click import
        zip_path = os.path.join(EXPORTS_DIR, "paintpilot_wall_3d_bundle.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(obj_path, arcname="paintpilot_wall.obj")
            zf.write(mtl_path, arcname="paintpilot_wall.mtl")

        mission_json = cad_exporter.export_mission_json(wall_w_mm, wall_h_mm, obstacles, waypoints, metrics)
        json_path = os.path.join(EXPORTS_DIR, "apr_mission_manifest.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(mission_json)

        motion_csv = cad_exporter.export_motion_csv(waypoints)
        csv_path = os.path.join(EXPORTS_DIR, "apr_waypoints.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(motion_csv)

        # Update cached session state
        current_session["wall_w_mm"] = wall_w_mm
        current_session["wall_h_mm"] = wall_h_mm
        current_session["obstacles"] = obstacles
        current_session["waypoints"] = waypoints
        current_session["metrics"] = metrics
        current_session["stats"] = path_stats
        current_session["confidence"] = confidence

        # Encode image to base64 for 2D canvas overlay
        # Resize to max 1200px wide for snappy UI responsiveness
        h, w = img_bgr.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img_disp = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            img_disp = img_bgr

        _, buffer = cv2.imencode(".jpg", img_disp, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "success": True,
            "source_name": source_name,
            "wall": {
                "width_mm": wall_w_mm,
                "height_mm": wall_h_mm,
                "spray_width_mm": spray_width_mm,
                "overlap_pct": overlap_pct,
                "safety_buffer_mm": safety_buffer_mm
            },
            "obstacles": obstacles,
            "waypoints": waypoints,
            "stats": path_stats,
            "metrics": metrics,
            "confidence": confidence,
            "image_data_url": img_b64,
            "timestamp": time.strftime("%H:%M:%S")
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/download/<file_type>", methods=["GET"])
def download_file(file_type):
    """Serve generated CAD and mission files for download."""
    file_map = {
        "dxf": ("paintpilot_wall.dxf", "application/dxf", "paintpilot_wall_map.dxf"),
        "obj": ("paintpilot_wall.obj", "model/obj", "paintpilot_wall.obj"),
        "mtl": ("paintpilot_wall.mtl", "text/plain", "paintpilot_wall.mtl"),
        "zip": ("paintpilot_wall_3d_bundle.zip", "application/zip", "paintpilot_wall_3d_bundle.zip"),
        "json": ("apr_mission_manifest.json", "application/json", "apr_mission_manifest.json"),
        "csv": ("apr_waypoints.csv", "text/csv", "apr_waypoints.csv")
    }

    if file_type not in file_map:
        return jsonify({"error": f"Unknown file type: {file_type}"}), 404

    disk_name, mime_type, download_name = file_map[file_type]
    file_path = os.path.join(EXPORTS_DIR, disk_name)

    if not os.path.exists(file_path):
        return jsonify({"error": f"File not generated yet: {disk_name}"}), 404

    return send_file(
        file_path,
        mimetype=mime_type,
        as_attachment=True,
        download_name=download_name
    )


if __name__ == "__main__":
    print("==================================================================")
    print("  APR (Automated Painting Rover) - Wall-to-3D CAD Console")
    print("  Funded by MSME Grant Ref: INC25ETS084848")
    print("  WBS 1.2: Coverage Planning & 3D Wall Geometry Subsystem")
    print("  Server running at: http://127.0.0.1:5000")
    print("==================================================================")
    app.run(host="0.0.0.0", port=5000, debug=False)