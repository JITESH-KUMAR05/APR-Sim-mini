# Evaluation report

Model `training\runs\apr_v1\weights\best.onnx`, app pipeline at sensitivity 0.5 (review threshold 0.25, accept threshold 0.50). A match needs IoU 0.5 and the same class.

## real_human

255 held-out real photos, human window and door boxes

| class | ground truth | recall | precision | objects with no box at all |
|---|---|---|---|---|
| window | 1926 | 0.43 | 0.57 | 1092 |
| door | 83 | 0.43 | 0.26 | 47 |
| ac_unit | 0 | n/a | 0.00 | 0 |
| meter_panel | 0 | n/a | 0.00 | 0 |
| pipe | 0 | n/a | n/a | 0 |
| grill | 0 | n/a | 0.00 | 0 |

## real_teacher

held-out real photos, rare classes labelled by the teacher

| class | ground truth | recall | precision | objects with no box at all |
|---|---|---|---|---|
| window | 1966 | 0.43 | 0.58 | 1116 |
| door | 131 | 0.43 | 0.41 | 73 |
| ac_unit | 20 | 0.00 | 0.00 | 19 |
| meter_panel | 4 | 0.00 | 0.00 | 4 |
| pipe | 1 | 0.00 | n/a | 1 |
| grill | 27 | 0.11 | 0.15 | 19 |

## synthetic

held-out synthetic walls

| class | ground truth | recall | precision | objects with no box at all |
|---|---|---|---|---|
| window | 893 | 1.00 | 1.00 | 0 |
| door | 154 | 1.00 | 1.00 | 0 |
| ac_unit | 362 | 1.00 | 1.00 | 0 |
| meter_panel | 239 | 1.00 | 1.00 | 0 |
| pipe | 271 | 1.00 | 1.00 | 0 |
| grill | 369 | 1.00 | 1.00 | 0 |

## How to read these numbers

- **synthetic** is perfect (1.00 everywhere). That only shows the model learned the generator. It says nothing about real walls.
- **real_human** is the honest number for `window` and `door` on real photos. The held-out photos are general building scenes from Open Images (sheds, interiors, street views, tiny far-away windows), not flat wall close-ups. Of the 1,926 window boxes, 1,155 cover under 0.2% of the photo. By size at the app's default threshold, window recall is 0.36 for those tiny ones, about 0.52 for windows covering 0.2% to 3% of the photo and 0.60 above 3%. Precision is a lower bound: the model also boxes real windows and doors that annotators did not label. A look at 12 of these photos (ground truth green, model red) shows most red boxes sitting on real windows and doors.
- **real_teacher** for `ac_unit`, `meter_panel`, `pipe`, `grill`: only 52 boxes exist in the held-out real photos and all of them come from the teacher, so these four rows are close to meaningless. The model finds almost none of them.

## Look at real photos outside the data (9 Wikimedia Commons photos, not used for training)

Window and door detection works on some facades (it found windows on a courtyard block and doors on two buildings). **AC units are not recognised on real photos**: a split AC unit on a wall was boxed as `grill` and `window`, a window AC as `window`. Meter boxes and pipes were not found in any of the 9 photos. The model has learned these four classes from cartoon-like synthetic walls and does not carry them over to real walls. This is the known limit described in `DATA_SOURCES.md`, not a surprise.

## End to end on random walls (`rover/tools/demo_random_wall.py --count 25 --seed 1000`)

Whole pipeline on 25 synthetic walls, same walls for both rows: draw a wall, detect, choose the layout, plan, compile, replay in the rover simulator.

| detector | planted obstacles found | mean area covered | walls where planned paint runs over a planted obstacle | simulator runs finished | paint laid / planned |
|---|---|---|---|---|---|
| trained model v1 | 187 of 187 | 1.00 | 1 of 25 (15 mm) | 25 of 25 | 0.97 |
| classical fallback | 107 of 187 | 0.68 | 20 of 25 (up to 7.6 m of spray over obstacles) | not run | not run |

This is a demonstration on walls drawn by the same generator the model trained on, so it will look as good as it can. It proves the pipeline works end to end and that the model is far better than the fallback on those walls. It does not prove anything about photographs.

## What would make it work on real walls

Real labelled photos of the four Indian-specific classes (`training/README.md`, sections 1 and 5): at least 40 to 80 labelled instances per class from the team's own photos, or a Roboflow dataset exported in YOLO format. Training can start from this version's weights.

## Targets set before training: pass or fail

| class | target (`training/README.md` section 6) | result on held-out real photos | verdict |
|---|---|---|---|
| window | recall at least 0.90 and precision at least 0.85 | recall 0.43, precision 0.57 (a lower bound) | **fail** |
| door | recall at least 0.90 and precision at least 0.85 | recall 0.43, precision 0.26 (a lower bound) | **fail** |
| ac_unit, meter_panel, pipe, grill | mAP50 at least 0.70 | cannot be measured on real photos: 52 teacher-made boxes, almost none found; a real AC was boxed as `grill` and `window` | **fail, needs the team's photos** |
| nothing in a test photo gets no box at all (windows and doors) | none missed | 1,092 of 1,926 window boxes and 47 of 83 door boxes got no box | **fail** |

The mAP50 values printed at the end of training (window 0.62, door 0.73, the four others about 0.95 to 0.98) are on a validation set that is 38% synthetic walls, which is why the four rare classes look good there. They are not real-photo numbers.

## Model against the classical fallback on the same real photos

Class-agnostic: a human window or door box counts as covered if a keep-out zone of any class (including `unverified`) covers at least half of it. Run with `training/compare_detectors.py`.

| detector | window and door boxes covered | boxes over 3% of the photo covered | share of the photo marked keep-out |
|---|---|---|---|
| trained model v1 | 0.52 of 2,009 | 0.68 of 151 | 11% |
| classical fallback | 0.85 of 2,009 | 0.72 of 151 | 63% |

The classical detector looks better on coverage only because it marks 63% of the photo as keep-out, which leaves almost nothing to paint. For the boxes that are large enough to matter for a wall (over 3% of the photo) the two are close, and the model is not better. **On real photos the trained model is not yet an improvement that justifies relying on it; on walls drawn by the generator it is.**

## Checks and known deviations

- The exported ONNX matches the PyTorch weights: on 40 held-out photos the largest class score difference is 0.00000 and the largest box difference 0.005 px (`training/gpu/check_export.py`, same input tensor made by the app's letterbox).
- Synthetic walls are 1,200 of 2,818 training images (43%), above the 30 to 40% the spec asked for. The validation set used to pick the best epoch is 150 synthetic of 395 images.
- The share of teacher boxes that a person accepted on a contact sheet was not measured.
- The train, validation and test split is by photo id. Near copies of one photo under different ids, if any exist in the public sets, were not searched for.
