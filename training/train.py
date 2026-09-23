"""
Fine-tune YOLO11n on the merged dataset and export it as ONNX for the app.

Run in Colab (GPU runtime) after `pip install ultralytics`:
    python training/train.py                                   # stage 1: from pretrained yolo11n.pt
    python training/train.py --weights training/runs/apr_obstacles/weights/best.pt --name apr_stage2

Ultralytics is training-only and must never be added to requirements.txt or pyproject.toml.
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from obstacle_detector import CLASS_NAMES  # noqa: E402  (dataset.yaml is generated from the same list)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "training" / "data" / "merged" / "dataset.yaml"))
    parser.add_argument("--weights", default="yolo11n.pt", help="start point; use the previous stage's best.pt for stage 2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="apr_obstacles")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=20,
        project=str(ROOT / "training" / "runs"),
        name=args.name,
        exist_ok=True,
    )

    best = Path(model.trainer.best)
    tuned = YOLO(str(best))

    print("Held-out test split, per-class mAP50:")
    metrics = tuned.val(data=args.data, split="test", imgsz=args.imgsz)
    for class_index, ap50 in zip(metrics.box.ap_class_index, metrics.box.ap50):
        print(f"  {CLASS_NAMES[int(class_index)]}: {ap50:.3f}")

    # nms=False is explicit (not just relying on Ultralytics' current default): NMS runs in
    # our own code (obstacle_detector.py), not baked into the export, so a future Ultralytics
    # default change can't silently change what the exported model returns.
    onnx_path = Path(tuned.export(format="onnx", imgsz=args.imgsz, simplify=True, nms=False))
    target = ROOT / "models" / "apr_obstacles.onnx"
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(onnx_path, target)
    print(f"Exported {target}. Run training/evaluate.py next, then commit the file.")


if __name__ == "__main__":
    main()
