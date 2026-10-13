"""
Evaluate one or more ONNX models per photo source on the held-out real test photos (human labels only), with the
app's own pipeline. The sources differ a lot: CMP Facade photos are frontal wall views like the app's use case,
Open Images photos are general building scenes with many tiny far-away windows, LVIS photos are mostly close
views of air conditioners.

    .venv/Scripts/python.exe training/eval_by_source.py models/apr_obstacles.onnx training/runs/apr_v3/weights/best.onnx

Needs training/data/pool (training/gpu/pool.py).
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from evaluate import evaluate, summarize  # noqa: E402
from obstacle_detector import CLASS_NAMES, ObstacleDetector, thresholds_for_sensitivity  # noqa: E402

DATA = ROOT / "training" / "data"


def group_by_source():
    index = json.load(open(DATA / "pool" / "splits.json"))
    groups = {}
    for image_id, rec in index.items():
        if rec["split"] == "test":
            groups.setdefault(rec["source"], []).append(image_id)
    out = DATA / "eval" / "by_source"
    if out.exists():
        shutil.rmtree(out)
    for source, ids in groups.items():
        (out / source / "images").mkdir(parents=True)
        (out / source / "labels").mkdir(parents=True)
        for image_id in ids:
            shutil.copyfile(DATA / "pool" / "images" / f"{image_id}.jpg", out / source / "images" / f"{image_id}.jpg")
            shutil.copyfile(DATA / "pool" / "labels_human" / f"{image_id}.txt", out / source / "labels" / f"{image_id}.txt")
    return out, {s: len(ids) for s, ids in groups.items()}


def main():
    models = sys.argv[1:]
    if not models:
        print(__doc__)
        return 1
    root, counts = group_by_source()
    review, _ = thresholds_for_sensitivity(0.5)
    for model in models:
        detector = ObstacleDetector(model_path=model)
        print(f"\n## {model}")
        print("| source (test photos) | class | ground truth | recall | precision |")
        print("|---|---|---|---|---|")
        for source in sorted(counts):
            stats = evaluate(detector, root / source / "images", root / source / "labels", CLASS_NAMES, review)
            summary = summarize(stats)
            for name in CLASS_NAMES:
                if stats[name]["gt"]:
                    m = summary[name]
                    print(f"| {source} ({counts[source]}) | {name} | {stats[name]['gt']} | {m['recall']:.2f} | {m['precision']:.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
