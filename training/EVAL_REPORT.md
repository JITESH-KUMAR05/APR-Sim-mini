# Evaluation report: obstacle model v5

Model `models/apr_obstacles.onnx` (YOLO11n, 640 px). Measured with the app's own pipeline (`ObstacleDetector`, sensitivity 0.5, review threshold 0.25, accept threshold 0.50). A match needs IoU 0.5 and the same class. All real-photo numbers are on held-out photos the model never saw. Version 1 of this report is replaced: it described a model trained with two data bugs, listed below.

## Summary

- **Frontal facade photos (CMP Facade, the closest set to the app's use case):** windows recall 0.86, precision 0.92; doors recall 0.62, precision 0.48.
- **General building photos (Open Images, many tiny far-away windows, interiors, sheds):** windows recall 0.41, precision 0.65; doors recall 0.49, precision 0.29. Not the target domain, kept as a hard test.
- **Air conditioners on real photos (LVIS):** recall 0.28, precision 0.54, up from 0.02 before real AC data was added. On 9 real facade photos from Wikimedia Commons the model finds balcony AC units on an apartment block but still misses a close-up outdoor condenser.
- **meter_panel, pipe, grill:** no usable real-photo test exists, so there is **no real-world number** for them. They were learned from synthetic walls and a few teacher labels. Treat them as unproven.
- **Random synthetic walls, whole pipeline (detect, layout, plan, simulate):** 187 of 187 planted obstacles found, no planned paint over any of them (25 walls). This proves the chain works; it does not prove the model works on photographs.

## Pass or fail against the targets set before training

| class | target (`training/README.md` section 6) | result | verdict |
|---|---|---|---|
| window, frontal facades | recall at least 0.90, precision at least 0.85 | 0.86 / 0.92 | **close; recall fails by 0.04** |
| window, general building photos | same | 0.41 / 0.65 | **fail** |
| door | recall at least 0.90, precision at least 0.85 | 0.62 / 0.48 (frontal), 0.49 / 0.29 (general) | **fail** |
| ac_unit | mAP50 at least 0.70 | recall 0.28, precision 0.54 on real photos | **fail** |
| meter_panel, pipe, grill | mAP50 at least 0.70 | not measurable on real photos | **unknown, needs the team's photos** |

## Held-out real photos by source (human labels only)

| source (test photos) | class | ground truth | recall | precision |
|---|---|---|---|---|
| CMP Facade (46) | window | 1,410 | 0.86 | 0.92 |
| CMP Facade (46) | door | 40 | 0.62 | 0.48 |
| Open Images (195) | window | 1,900 | 0.41 | 0.65 |
| Open Images (195) | door | 72 | 0.49 | 0.29 |
| ADE20K (14) | window | 26 | 0.23 | 0.13 |
| ADE20K (14) | door | 11 | 0.18 | 0.14 |
| LVIS (41) | ac_unit | 100 | 0.28 | 0.54 |

Precision is a lower bound: the model also boxes real windows and doors that annotators did not label. ADE20K has only 14 photos, too few to read anything into. Written by `training/eval_by_source.py`.

## Against the classical fallback on the same real photos

Class-agnostic: a human window or door box counts as covered if any keep-out zone (any class, including `unverified`) covers at least half of it. `training/compare_detectors.py`.

| detector | window and door boxes covered | boxes over 3% of the photo covered | share of the photo marked keep-out |
|---|---|---|---|
| trained model v5 | 0.65 of 3,559 | 0.67 of 158 | 13% |
| classical fallback | 0.81 of 3,559 | 0.72 of 158 | 66% |

The classical detector looks better on coverage only because it marks two thirds of every photo as keep-out, which leaves almost nothing to paint. For the boxes that matter on a wall (over 3% of the photo) the two are close. The model's advantage on real photos is precision, not coverage.

## Held-out synthetic walls

Recall and precision are 1.00 for all six classes (one grill missed). This only shows the model learned the generator.

## Two data bugs found while improving it

1. **All 378 CMP Facade label files were empty** in version 1: the XML puts each label name on its own line, so no window or door matched. The model had been taught that frontal facades have no windows.
2. **CMP boxes had x and y swapped:** in those files the `<x>` pair is the vertical range. Found by comparing the boxes with the pixel label images and drawing them on photos. Now covered by a regression test (`tests/test_training_fusion.py`).

Fixing both took windows on frontal facades from recall 0.00 to 0.86. Lesson kept in `training/README.md`: draw labels on photos before every training run.

Also changed: the validation set is real photos only (it was 38% synthetic, which made the best-epoch choice favour walls the model already scored 1.0 on), and the LVIS `pipe` boxes were dropped after looking at them (exhaust pipes and plumbing, not wall pipes).

## Checks and known deviations

- ONNX output matches the PyTorch weights: largest class score difference 0.00001, largest box difference 0.010 px on 40 photos (`training/gpu/check_export.py`).
- Synthetic walls are 800 of 2,734 training images (29%).
- The share of teacher boxes that a person accepted on a contact sheet was not measured.
- The split is by photo id. Near copies of one photo under different ids, if any exist in the public sets, were not searched for.
- A test set of 296 photos, 3,336 of whose boxes are windows, says little about doors (123) and nothing about the rare classes.

## What would make it work on real walls

The team's own labelled photos, in the order set out in `docs/real-world-roadmap.md` section 3: 200 to 300 photos of the walls the rover will meet, machine pre-labels corrected in CVAT or Roboflow, a held-out 20%, then `training/gpu/pool.py yolo ...` and a retrain from these weights. The app now also saves every correction the operator makes in the review panel as a labelled photo (`exports/corrections`), so real jobs become training data.
