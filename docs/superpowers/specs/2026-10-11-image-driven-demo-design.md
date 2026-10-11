# Image-driven demo design (P6): a trained obstacle model and a demo that works on any wall

Date: 2026-10-11
Status: approved in outline by the project lead on 2026-10-11 (data route and the three demo extras); details below for review.

## Context

The app already turns a wall photo into obstacles, plans a coverage path, compiles a rover mission and animates it (`docs/superpowers/specs/2026-09-20-real-world-obstacle-detection-design.md`). The learned detector slot exists (`obstacle_detector.py`, `models/apr_obstacles.onnx`) but **no trained model has ever existed**, so the app uses its classical detector. The firmware core, protocol and a PC simulator (`rover_sim`) exist (`docs/rover-protocol.md`). No rover hardware exists.

The project lead wants to show a demo on a random wall, in simulation, where the system decides from the image alone, and asked for the model to be trained as completely as possible.

## Honest limits (read first)

- A detector is only as good as its labelled photos. No Indian-wall photo set exists on this machine, and the Roboflow route needs the lead's account. So v1 of the model is trained on **public photos plus machine-made labels**, and it will be weaker on real Indian facades than a model trained on the team's own labelled photos.
- Window and door labels come from human annotators (Open Images, CMP Facade, ADE20K). The labels for `ac_unit`, `meter_panel`, `pipe` and `grill` are produced by an open-vocabulary detector (a "teacher") plus synthetic images, **not by people**. Their accuracy numbers are measured against the teacher, not against the truth, and are reported as such.
- The fail-safe rule stays: an uncertain detection becomes an `unverified` obstacle that the planner still avoids. A missed obstacle is worse than a false one.
- The numbers reported after training are the result of a measurement on photos the model never saw. They are not a promise for any other photo.

## Decisions made

| Question | Decision |
|---|---|
| Data route | public data and synthetic data now, trained on this PC (RTX 3050, 4 GB), team photos added later with the existing `training/` pipeline |
| Model | YOLO11n, 640 px, six classes in the order of `CLASS_NAMES`: window, door, ac_unit, meter_panel, pipe, grill. Exported to ONNX, loaded by the existing `ObstacleDetector`. |
| Training environment | a separate uv project `training/gpu/` (torch with CUDA, ultralytics, transformers). It never touches the root `pyproject.toml` or `requirements.txt` (the Vercel outage rule). |
| Licences | ultralytics is AGPL-3.0 (accepted for the prototype by the lead, see the obstacle-detection spec). Dataset licences are recorded in `training/DATA_SOURCES.md`; several are research or non-commercial and need review before commercial use. |
| Demo extras | (1) automatic choice of start corner and stripe direction with a shown reason, (2) a random-wall generator, (3) replay of the compiled mission through the C++ rover simulator inside the demo |

## 1. The data

| Source | Gives | Access |
|---|---|---|
| Open Images V7 (validation and test bounding boxes) | real photos with human boxes for `Window` and `Door`; about 4,200 images, 33,000 windows, 1,100 doors | key-free, images from the public bucket |
| CMP Facade database | 378 facade photos with human labels incl. window and door | key-free download |
| ADE20K scene parsing | masks for window and door, turned into boxes | key-free download (about 1 GB) |
| Open-vocabulary teacher on the same photos | boxes for `ac_unit`, `meter_panel`, `pipe`, `grill`, and extra window/door boxes that the humans missed | model downloaded once |
| Synthetic facade generator | walls with a random mix of all six classes, perfect labels, including Indian-style window grills, split and window AC units, meter boxes, drain pipes | written here, also used by the demo |
| Held-out real photos | evaluation only, never trained on | split by image id before anything else |

The train / validation / test split is made by image id before any augmentation or pseudo-labelling, so the same photo or a near copy cannot be on both sides. The two repo sample photos (`samples/`) are used for demonstration only, never for training.

## 2. The labels

- Human labels are kept as they are (window, door).
- The teacher boxes are filtered by confidence, per-class size limits and overlap rules, and a **random sample is saved as contact sheets** so a person can look at them. The report states the share of teacher boxes a quick visual check accepted.
- Teacher boxes on a window or door that overlap a human box are dropped (the human box wins).
- Synthetic images are 30 to 40% of the training set at most, so the model is not mostly learning the generator.

## 3. Training and evaluation

1. Stage 1: pretrained `yolo11n.pt`, all sources, early stopping on validation mAP50-95.
2. Evaluation with the app's own pipeline (`training/evaluate.py`): recall and precision per class at the app's default thresholds, how many objects get no box at all, and the same on the held-out teacher-labelled subset, reported **separately** from the human-labelled subset.
3. Export to ONNX with the output shape `obstacle_detector.decode` expects, check that the ONNX output matches the PyTorch output on test images, then run the app's `ObstacleDetector` on `samples/` as a smoke test.
4. Targets, set before training, same as `training/README.md`. They are reported pass or fail per class and a failing class is listed as "needs the team's photos", not hidden.

## 4. The demo features

1. **Automatic layout.** For each of the four start corners (and each stripe direction the planner supports) the planner is run, and a score picks the best: fewest obstacle crossings, then least non-painting travel, then fewest turns. The UI shows the table of scores and the reason in one sentence. It is rules, not learning, so it can be explained and tested. The operator can still override.
2. **Random-wall generator.** A button and a command-line tool make a wall photo from the synthetic generator (random size, obstacles, lighting, texture) and run detect, plan, compile and simulate end to end, so a demo does not need a photo.
3. **Replay in the C++ simulator.** The compiled mission is run through `rover_sim` by the same host tool that will drive the real board (`rover/tools/run_mission.py`) with optional trouble (wheel slip, gyro bias). The UI draws the planned path, the path the simulator reports and the true path, with the final error. This only works on the local app (a build of `rover_sim` is needed), not on Vercel; the page says so.

## 5. Out of scope

- Training on the team's own photos (the pipeline is ready for it, the photos are not here).
- Any change to the firmware or the protocol.
- Running the model on the rover.
- A learned policy for navigation (plan P5).

## 6. Order of work and checkpoints

1. Hardware requirements document (done, `docs/rover-hardware-requirements.md`).
2. Training pipeline: environment, data download, labelling, synthetic generator, training, evaluation, export. Checkpoint with the lead.
3. Demo features 1 to 3 with tests. Checkpoint.
4. Final review, merge locally (the lead pushes).

## Risks

- **Teacher quality.** If the teacher is bad at Indian-style objects, the labels are bad and so is the model. The contact sheets and the visual accept rate are there to show this early.
- **Synthetic gap.** A model can learn the generator's look. Mitigated by capping the share and by evaluating on real photos only.
- **4 GB of GPU memory.** YOLO11n at 640 px with a small batch fits. A larger model does not.
- **Download size.** About 1.5 GB of public data under `training/data/` (git-ignored).
- **Windows Application Control** can block freshly built `rover_sim.exe`; the demo falls back with a clear message.
