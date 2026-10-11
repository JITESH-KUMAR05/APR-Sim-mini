"""
Check that the exported ONNX computes the same thing as the PyTorch weights.

    training/gpu/.venv/Scripts/python.exe training/gpu/check_export.py --weights training/runs/apr_v1/weights/best.pt \
        --onnx models/apr_obstacles.onnx --images training/data/yolo/test/images

Both models get exactly the same 640 x 640 input tensor, made by the app's own letterbox. The raw outputs must
agree to within a small tolerance. (Ultralytics' predict() letterboxes differently from the app, so comparing
final boxes from predict() would measure the preprocessing and not the export.) Exits non-zero on a mismatch.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from obstacle_detector import ObstacleDetector, letterbox, to_input_tensor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=2e-2, help="largest allowed absolute difference of a class score")
    args = parser.parse_args()

    from ultralytics import YOLO

    net = YOLO(args.weights).model.float().cpu().eval()
    session = ObstacleDetector(model_path=args.onnx)._session
    input_name = session.get_inputs()[0].name
    names = sorted(n for n in os.listdir(args.images) if n.lower().endswith((".jpg", ".png")))[: args.limit]
    worst_score = worst_box = 0.0
    for name in names:
        image = cv2.imread(os.path.join(args.images, name))
        tensor = to_input_tensor(letterbox(image)[0])
        with torch.no_grad():
            out = net(torch.from_numpy(tensor))
        expected = (out[0] if isinstance(out, (list, tuple)) else out).numpy()
        actual = session.run(None, {input_name: tensor})[0]
        if expected.shape != actual.shape:
            print(f"shape differs: PyTorch {expected.shape}, ONNX {actual.shape}")
            return 1
        worst_box = max(worst_box, float(np.abs(expected[:, :4] - actual[:, :4]).max()))
        worst_score = max(worst_score, float(np.abs(expected[:, 4:] - actual[:, 4:]).max()))
    print(f"{len(names)} photos: largest class score difference {worst_score:.5f}, largest box difference {worst_box:.3f} px")
    return 0 if worst_score <= args.tolerance and worst_box <= 2.0 else 1


if __name__ == "__main__":
    sys.exit(main())
