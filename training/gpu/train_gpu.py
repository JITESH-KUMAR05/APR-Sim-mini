"""
Train YOLO11n on training/data/yolo with the local GPU, then export ONNX in the shape the app expects.

    training/gpu/.venv/Scripts/python.exe training/gpu/train_gpu.py --epochs 50 --name apr_v1

Exports to training/runs/<name>/weights/best.onnx (no NMS in the graph: obstacle_detector.py does it).
Copying the file to models/apr_obstacles.onnx is a separate step, done by hand after the evaluation and
check_export.py (the exported model must give the same boxes as the PyTorch weights).
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(ROOT, "training", "data", "yolo", "data.yaml"))
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="apr_v1")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=20,
        device=0,
        workers=4,
        project=os.path.join(ROOT, "training", "runs"),
        name=args.name,
        exist_ok=True,
        degrees=3.0,
        perspective=0.0005,
        mosaic=1.0,
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,
        fliplr=0.5,
        close_mosaic=8,
    )
    best = os.path.join(ROOT, "training", "runs", args.name, "weights", "best.pt")
    tuned = YOLO(best)
    onnx = tuned.export(format="onnx", imgsz=args.imgsz, simplify=True, nms=False, opset=12)
    print("exported", onnx)


if __name__ == "__main__":
    main()
