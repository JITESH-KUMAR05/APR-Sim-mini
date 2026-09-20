# Real-world wall and obstacle detection — design

Date: 2026-09-20
Status: approved in conversation, pending spec review

## Goal

Make the photo-to-mission pipeline trustworthy on arbitrary real-world exterior walls, so the mission file it produces (waypoints CSV, mission JSON, DXF/OBJ) can be handed to the physical rover to paint.

Today the obstacle detector is classical CV: bilateral filter, Canny, morphology, `findContours`, then a guess at the class from bounding-box aspect ratio and position (`geometry_engine.py:31-128`). It works on the two bundled benchmark photos and will not generalise to real lighting, textures and clutter. The reported confidence is a formula (`94.0 - n*1.5 + sensitivity*5.0`), not a measurement.

## Context and constraints

- No rover hardware exists yet. This phase is software only.
- Timeline is weeks (an upcoming review/demo).
- The team can collect real wall photos and will also use public datasets.
- The team's frozen architecture (`APR_Final_Review_Document.pdf`) names the YOLO family as the perception model and states that AI perception only feeds the costmap; planning, control and safety stay deterministic. This design follows that: the detector proposes obstacle boxes, the existing deterministic coverage planner consumes them.
- Deployment target is Vercel serverless. A previous deploy failed because heavy binary dependencies did not install. Nothing in this design may add PyTorch or ultralytics to `requirements.txt` or `pyproject.toml`.
- Downstream code treats every obstacle as a rectangle with a uniform `safety_buffer_mm`. The obstacle `type` only drives the label text and `depth_mm` (checked in `geometry_engine.py:87-104`). Nothing safety-critical depends on it.

## Scope

In scope:

1. A learned obstacle detector (YOLO11n) trained offline, exported to ONNX, run in the app with `onnxruntime`.
2. A training pipeline in the repo (`training/`) that merges public datasets with the team's own photos.
3. Replacing the internals of `detect_obstacles_from_image` while keeping its output contract.
4. A review step for uncertain detections, and an endpoint to replan after the operator edits the obstacle list.
5. Real per-detection confidence replacing the formula.
6. Four-corner wall rectification, so mm coordinates are correct for photos that are not shot square-on. Sequenced last and cuttable (see Phasing).

Out of scope (deliberately, not forgotten):

- Closed-loop navigation simulation, ROS 2, SLAM, real-time onboard perception. Phase 2, and it does not feed the real rover.
- Automatic wall scale estimation. Wall width and height stay as manual inputs. Auto-scale needs depth sensing or a reference marker.
- Automatic wall boundary detection. The operator supplies the wall corners (item 6) or the whole frame is the wall.
- Manually drawing an obstacle the detector missed. Documented as a risk below and mitigated by a low review threshold; a follow-up.
- Wall-material classification, crack or damage detection.

## Classes

| Class | Source of training data | `depth_mm` default |
|---|---|---|
| `window` | public (Roboflow) + own photos | 120 |
| `door` | public (Roboflow) + own photos | 80 |
| `ac_unit` | own photos | 400 |
| `meter_panel` | own photos | 100 |
| `pipe` | own photos | 80 |
| `grill` | own photos | 60 |
| `unverified` | not a trained class; assigned at inference to low-confidence detections | 50 |

`depth_mm` values are adjustable defaults kept in one dict in `geometry_engine.py`, used only by the 3D export. Existing values for door, window and fixture are preserved; `meter_panel` replaces the old `switchboard` naming.

## Data strategy

Public datasets (verify each dataset's licence before use; Roboflow Universe datasets vary):

- Roboflow "Door and Window Detection" (Construction Plan), 5,382 images, YOLO format.
- Roboflow wall-door-window set `yolo-obb-1`, 997 images. Oriented boxes; convert to axis-aligned boxes or skip if conversion loses too much.
- CMP Facade Database (606), ECP (104), eTRIMS (60): optional. European facade styles and segmentation-style annotations make them a weak fit. Not required.

No public dataset was found for AC condensers, meter boxes, exterior pipes or grills. These classes depend on the team's own photos.

Own-photo collection guidance:

- At least 40 to 80 labelled instances per new class. More is better.
- Vary lighting (morning, noon, overcast, shade), distance, angle, wall colour and texture, and include clutter (wires, plants, signage, shadows).
- Include walls with no obstacles so the model learns what a clean wall looks like.
- Label in Roboflow's free tool or CVAT and export in YOLO format.
- Hold out at least 20% of the team's photos as a test set. They are never used for training or for tuning thresholds.

Class-name mapping from each source dataset to the classes above lives in `training/dataset.yaml` and the merge script, so the mapping is reviewable.

## Model and training

- Architecture: YOLO11n (nano). Input 640x640.
- Stage 1: fine-tune on the public window and door data.
- Stage 2: continue fine-tuning on the merged set including the team's photos and the new classes.
- Trained in free Google Colab (T4). The repo holds the script and config, not the weights' training environment.
- Export to ONNX (`opset` chosen at export time, recorded in the training README) and commit to `models/apr_obstacles.onnx`. A nano model is small enough to commit.

Licence risk: ultralytics and weights trained with it are AGPL-3.0, which the team's own architecture document already flags. Acceptable for the academic and prototype phase. Before any commercial deployment, either obtain an Ultralytics enterprise licence or retrain with a permissively licensed detector (for example an Apache-2.0 one). Recorded here so the decision is not lost.

## Inference integration

`GeometryEngine.detect_obstacles_from_image(image_bgr, wall_w_mm, wall_h_mm, sensitivity)` keeps its signature and return shape: `(obstacles, confidence)`, where each obstacle is `{id, type, label, x, y, w, h, depth_mm, norm{x,y,w,h}}` with origin bottom-left in mm. `plan_coverage_path`, `cad_exporter.py` and the Three.js viewer are unchanged by the swap.

New module `obstacle_detector.py`, kept separate so it is testable without Flask or a model file:

- `letterbox(image, size)`: resize with padding, return the image plus scale and padding.
- `decode(output, scale, pad, ...)`: turn raw ONNX output into boxes in original-image coordinates (inverse of the letterbox transform).
- `nms(boxes, scores, iou)`: non-maximum suppression, class-aware.
- `ObstacleDetector(model_path)`: loads the ONNX session once per process, exposes `detect(image_bgr) -> list[Detection]` with `class_name`, `confidence`, and a pixel box.

`GeometryEngine` maps detections to the obstacle dict: pixel box to `norm`, `norm` to mm using the wall dimensions (the existing conversion at `geometry_engine.py:75-85`), class to `label` and `depth_mm`.

Thresholds, derived from the existing `sensitivity` slider (0 to 1, default 0.5):

- `review_threshold = clamp(0.40 - 0.30 * sensitivity, 0.10, 0.40)`, which is 0.25 at the default.
- `accept_threshold = review_threshold + 0.25`, which is 0.50 at the default.
- Confidence at or above `accept_threshold`: obstacle keeps its predicted class.
- Between the two thresholds: obstacle becomes `type: "unverified"`, `label: "Unverified object #n"`, and is still treated as an obstacle by the planner (fail safe: paint avoids it until the operator says otherwise).
- Below `review_threshold`: discarded.

The classical size filters in the old detector (drop boxes under 1.2% of the area or narrower than 5% of the frame) do not apply to model output, because pipes are legitimately thin. A minimal filter drops boxes smaller than 0.5% of the frame in either dimension.

Each obstacle gains a `confidence` field (0 to 1). The aggregate `confidence` returned to the UI is the mean confidence of kept detections as a percentage. With zero detections it is `null`, not an invented number, because a clean wall is a legitimate result and a fake 100% would mislead.

Fallback: if `models/apr_obstacles.onnx` is missing or fails to load, the previous classical detector runs instead (kept as `_detect_classical`). The `/api/analyze` response carries `detector: "yolo11n-onnx"` or `"classical-fallback"`, and the UI shows a visible warning banner in the fallback case. This exists because weights will not exist until training is done, so development and the demo must not break in the meantime, and because a missing model file is a realistic deployment mistake.

## Review step for uncertain detections

- `/api/analyze` returns `needs_review` (count of `unverified` obstacles).
- The UI lists unverified obstacles with, for each, a class dropdown to confirm (which sets the class and keeps the box) and a dismiss button (which removes it).
- New endpoint `POST /api/replan` takes the wall parameters and the edited obstacle list and reruns `plan_coverage_path`, `compute_metrics` and the exports, without rerunning detection. It updates the cached session so downloads reflect the edited list.
- The mission is marked "unreviewed" in the UI, and the export buttons show a caution, while `needs_review > 0`. Exports are not blocked: the operator may deliberately accept the conservative plan.

## Wall rectification (four-corner)

The UI stepper already claims "Photo rectify" and the upload hint says "Auto-rectification", but no rectification exists in the code (verified by search). For real photos taken from the ground at an angle, box positions map to wrong millimetre coordinates, which defeats the purpose.

- The operator clicks the four wall corners on the uploaded image (top-left, top-right, bottom-right, bottom-left).
- Backend applies `cv2.getPerspectiveTransform` and `cv2.warpPerspective` to a wall-aspect-ratio image before detection.
- Corners are optional. If omitted, behaviour is unchanged (whole frame is the wall) and the UI notes that the photo should be shot square-on.
- The upload hint text and stepper label are corrected to match reality.

## Phasing

1. `obstacle_detector.py` with tests, using synthetic model output. No trained weights needed. Fallback wired in.
2. Training pipeline (`training/`), public-data stage, first ONNX model; integrate and verify end to end.
3. Team photo collection and labelling (runs in parallel with 1 and 2), stage-2 training, held-out evaluation.
4. Review UI plus `/api/replan`.
5. Four-corner rectification. Cuttable if time is short; the limitation is then stated in the UI.

## Success criteria

Measured on the held-out team photos, not the training data:

- `window` and `door`: recall at least 0.90, precision at least 0.85 at the default thresholds.
- `ac_unit`, `meter_panel`, `pipe`, `grill`: mAP@0.5 at least 0.70. These targets are set before training and reported honestly if missed.
- No obstacle present in a test photo is silently absent from the result at the default thresholds without appearing as either a classified or an unverified box, for windows and doors.
- End to end: a held-out photo goes in, and the exported waypoints CSV contains no waypoint inside any accepted or unverified obstacle plus safety buffer.

## Testing

Extends the pattern in `tests/test_pipeline.py`.

- Unit tests for `letterbox`, `decode`, `nms` and threshold banding with synthetic arrays. The letterbox inverse mapping is the most likely source of silent coordinate bugs, so it gets round-trip tests with non-square images.
- Unit test that detections map to the obstacle dict with correct mm coordinates and bottom-left origin.
- Fallback test: with no model file present, `detector == "classical-fallback"` and the old benchmark tests still pass.
- Integration test using the real ONNX model, skipped when the file is absent.
- `/api/replan` test: dismissing an unverified obstacle changes the waypoint set; confirming keeps the avoided region.
- Manual browser pass for the review UI and corner-click flow.

## Risks

- Missed detections are worse than false ones (paint over an AC vent). Mitigation: low review threshold, unverified boxes kept as obstacles, and a follow-up for manual box drawing.
- Small labelled datasets for the four Indian-specific classes may cap accuracy. Mitigation: held-out evaluation and honest reporting; collect more photos where a class underperforms.
- Domain shift between public datasets (mostly interior plans and European facades) and Indian exterior walls. Mitigation: the team's own photos dominate stage 2.
- Frontend touchpoints that need care: `static/app.js:390-391` formats `confidence` as `${confidence}%` and sizes a bar from it, so `null` must be handled; obstacle colouring and labels must cover the new classes and `unverified`.
- Uncommitted-model deployment: if the ONNX file is not committed, production silently runs the classical fallback. The warning banner and the `detector` field make this visible.
- Licences: AGPL (ultralytics) and per-dataset licences, see above.

## Decisions fixed here

- NMS is done in `nms()`, not baked into the ONNX export, so it can be unit tested.
- The `yolo-obb-1` set is used by converting each oriented box to its axis-aligned envelope, because the planner only handles axis-aligned rectangles. The ONNX opset is whatever the pinned ultralytics version exports by default and is recorded in the training README.
