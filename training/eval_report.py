"""
Measure the exported ONNX model with the app's own pipeline (ObstacleDetector, default sensitivity) on photos
it never saw, and write the numbers to training/EVAL_REPORT.md.

    .venv/Scripts/python.exe training/eval_report.py --model models/apr_obstacles.onnx

Three sets, reported separately because they mean different things:
  real_human   held-out real photos, window and door boxes made by people (Open Images, CMP Facade, ADE20K).
               Annotators miss objects, so precision here is a lower bound.
  real_teacher held-out real photos, all six classes, where ac_unit, meter_panel, pipe and grill boxes come
               from the open-vocabulary teacher. Numbers against the teacher, not against the truth.
  synthetic    held-out synthetic walls, perfect labels. Shows what the model learned from the generator, and
               says nothing about real walls.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from evaluate import evaluate, summarize  # noqa: E402
from obstacle_detector import CLASS_NAMES, ObstacleDetector, thresholds_for_sensitivity  # noqa: E402

DATA = ROOT / "training" / "data"


def build_real_human():
    out = DATA / "eval" / "real_human"
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    index = json.load(open(DATA / "pool" / "splits.json"))
    count = 0
    for image_id, rec in index.items():
        if rec["split"] != "test":
            continue
        shutil.copyfile(DATA / "pool" / "images" / f"{image_id}.jpg", out / "images" / f"{image_id}.jpg")
        shutil.copyfile(DATA / "pool" / "labels_human" / f"{image_id}.txt", out / "labels" / f"{image_id}.txt")
        count += 1
    return out, count


def table(stats, summary):
    rows = ["| class | ground truth | recall | precision | objects with no box at all |", "|---|---|---|---|---|"]
    for name in CLASS_NAMES:
        s, m = stats[name], summary[name]
        fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
        rows.append(f"| {name} | {s['gt']} | {fmt(m['recall'])} | {fmt(m['precision'])} | {s['gt'] - s['found_any']} |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(ROOT / "models" / "apr_obstacles.onnx"))
    parser.add_argument("--sensitivity", type=float, default=0.5)
    parser.add_argument("--out", default=str(ROOT / "training" / "EVAL_REPORT.md"))
    args = parser.parse_args()

    detector = ObstacleDetector(model_path=args.model)
    review, accept = thresholds_for_sensitivity(args.sensitivity)
    human_dir, human_count = build_real_human()
    sets = [
        ("real_human", f"{human_count} held-out real photos, human window and door boxes", human_dir / "images", human_dir / "labels"),
        ("real_teacher", "held-out real photos, rare classes labelled by the teacher", DATA / "yolo" / "test" / "images", DATA / "yolo" / "test" / "labels"),
        ("synthetic", "held-out synthetic walls", DATA / "yolo" / "synth_test" / "images", DATA / "yolo" / "synth_test" / "labels"),
    ]
    lines = [f"# Evaluation report", "",
             f"Model `{os.path.relpath(args.model, ROOT)}`, app pipeline at sensitivity {args.sensitivity} "
             f"(review threshold {review:.2f}, accept threshold {accept:.2f}). A match needs IoU 0.5 and the same class.", ""]
    for name, description, images, labels in sets:
        stats = evaluate(detector, images, labels, CLASS_NAMES, review)
        summary = summarize(stats)
        print(f"== {name}: {description}")
        print(table(stats, summary))
        lines += [f"## {name}", "", description, "", table(stats, summary), ""]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
