# From simulation to a rover that works on a real wall

Status: draft for the team, 2026-10-13. Written for the project lead and the four students, the faculty guide, and anyone who has to decide what to do next month. It is a plan with numbers and gates, not a promise: where something is a guess it says so.

## 1. Where the project really is

| Part | State | Evidence |
|---|---|---|
| Wall photo to obstacle map to coverage path | **Works in software.** Windows on frontal facade photos: recall 0.86, precision 0.92. Doors, air conditioners and the other classes are weak or unmeasured on real photos. | `training/EVAL_REPORT.md` |
| Start corner and stripe direction chosen from the obstacles | Works, rule-based, explains itself | `layout_advisor.py`, tests |
| Mission file, host tool, rover protocol, simulator | Works **in simulation only** (wheel slip, gyro drift, faults, e-stop) | 300+ tests, `docs/rover-protocol.md` |
| Arduino firmware | Written and tested as portable C++. **Never built for an Arduino. No hardware exists.** | `rover/firmware/core/` |
| Pi agent, laptop console that drives a real rover | Not written (needs the Pi) | plan P3 |
| Vertical wall climbing | Not designed. Adhesion, tether, power, safety review all open. | `docs/rover-hardware-requirements.md` section 7 |
| Learned decision-making on the rover | Not started, by design (needs run logs first) | plan P5 |

The honest summary: the software chain is ahead of everything it will connect to. The things that stop a real demo are (1) the detector on real walls, (2) buying and wiring the floor prototype, and (3) measuring real scale and real drift. None of these is solved by more code on this PC.

## 2. What blocks "works in real life", in order

1. **No hardware.** Nothing can be tuned or proven until the Stage 0 parts exist.
2. **The detector has not seen the team's walls.** Public photos are the wrong domain (street scenes, interiors, tiny far windows). Section 3 is the plan.
3. **Wall size comes from the operator typing it.** A wrong size ruins the path and the stripe overlap. Section 4.
4. **Nothing re-references y** (about 65 mm of drift at 3% slip on an 8 ft wall in simulation). Hardware doc section 4.
5. **No closed loop on the paint.** The rover paints what it was told to; nothing checks the result. Section 6.
6. **The vertical wall.** It is a separate project with a safety review, not a software task.

## 3. Perception: how to make the detector good on real walls

### 3.1 What public data can and cannot do

I searched the public sources. What I found and used, and what exists but needs an account or a manual download:

| Source | What it gives | Status |
|---|---|---|
| Open Images, CMP Facade, ADE20K | human window and door boxes | **used** |
| LVIS (COCO photos) | human boxes for `air_conditioner` (about 790 in training). Its `pipe` boxes were looked at and dropped: exhaust pipes and plumbing | **used** for air conditioners |
| [BFA-3D](https://arxiv.org/html/2409.04025) | 1,240 facade photos, 1200 x 1200, human-checked boxes for door, windows, balcony, **air conditioner unit**, billboard, glass curtain wall | **not used**: it is on a Baidu Pan share ([repository](https://github.com/CVEO/BFA-YOLO)), which needs a Baidu account and cannot be fetched by a script. A team member can download it by hand. Licence not stated. |
| Roboflow Universe ([air conditioning, 164 images](https://universe.roboflow.com/yolo-uv06o/air-conditioning-dataset), several door/window/AC sets of 250 to 1,600 images, [electric meter](https://universe.roboflow.com/object-detection-v0up3/electric-meter-detector) and [meter box](https://universe.roboflow.com/pemula-uf3rv/yolo_fix) sets, mostly CC BY 4.0) | small community sets, quality unchecked | **not used**: downloads need a free Roboflow account (an API key or a manual export). The importer is ready (3.3). |
| Objects365 (has an Air Conditioner class) and the full ADE20K (3,000+ classes) | large, real | not used: 700 GB, and a registration form |
| Window grills, drain pipes on walls, meter boxes on Indian walls | **no public set found** | the gap that only the team's own photos can close |

So public data will never be enough for `grill`, `meter_panel` and wall `pipe`. Plan for the team's own photos from the start.

### 3.2 The data flywheel (what to do, step by step)

1. **Capture protocol (one afternoon, 4 students).** Photograph real walls the rover will meet: the campus, homes, shops. 200 to 300 photos at first. Shoot square-on, from 2 to 4 m, wall filling the frame, a mix of lighting (morning, noon, overcast, shade), of wall types (plaster, paint, brick, tile) and of clutter. Include walls with **no obstacles**: the model has to learn what a clean wall is. Put an ArUco marker or a taped 1 m reference on some of them (section 4).
2. **Pre-label with the machine.** The teacher model (OWLv2, already in `training/gpu/`) proposes boxes. A person then corrects them in [CVAT](https://www.cvat.ai/) or Label Studio. Foundation-model labels are good for big objects and poor for small, rare or hard-to-describe ones, so every box is reviewed; a [Roboflow](https://blog.roboflow.com/launch-auto-label/) note makes the same point (full automation of labelling is not realistic yet).
3. **Label by class, exactly as the app names them:** window, door, ac_unit, meter_panel, pipe, grill. Export in YOLO format.
4. **Ingest and retrain** (one command each, 3.3). Hold out 20% of the team's photos as a test set that is never trained on, and never tune thresholds on it.
5. **Active learning loop.** Run the model on new photos, keep the ones where it is unsure or wrong (low score, or the operator corrected it in the review panel), label those first. Each round of 100 photos is cheaper and worth more than 100 random ones. The web app already records what the operator confirms or dismisses; saving those corrections as new labels is a small piece of work (about a day) and turns every real job into training data.
6. **Gate before trusting a model on a wall:** window and door recall at least 0.90 and precision at least 0.85 on the team's held-out photos for boxes that are at least 3% of the photo (the targets already set in `training/README.md`), and mAP50 at least 0.70 on each rare class. Below that the app keeps asking the operator to review.

Rule that does not change: a doubtful detection becomes an `unverified` obstacle and is still avoided. A missed obstacle (paint on a window) costs more than a false one.

### 3.3 The commands

The importer takes any YOLO export (the team's labelled photos, a Roboflow download) and maps its class names to the app's:

```
training/gpu/.venv/Scripts/python.exe training/gpu/pool.py yolo team DIR "window=window,aircon=ac_unit,meter=meter_panel,grill=grill,pipe=pipe" --keep-empty
training/gpu/.venv/Scripts/python.exe training/gpu/run_teacher.py
training/gpu/.venv/Scripts/python.exe training/gpu/fuse.py --synth 600
training/gpu/.venv/Scripts/python.exe training/gpu/train_gpu.py --weights training/runs/apr_v5/weights/best.pt --epochs 40 --name apr_v6
.venv/Scripts/python.exe training/eval_report.py --model training/runs/apr_v6/weights/best.onnx
```

Real, human-labelled photos always win over teacher boxes for the classes they label. The synthetic walls should shrink to a small share once real data exists.

### 3.4 Other ways to get better, in the order I would try them

1. More real data (above). Nothing else comes close.
2. Larger input size (960 px) and a larger model (YOLO11s). Small windows are the weak spot. Cost: `INPUT_SIZE` in `obstacle_detector.py` and CPU time per photo; try only after the data is better.
3. A segmentation model (SAM class) to get exact obstacle outlines instead of boxes, so the keep-out zone follows the real shape. Useful for arched windows and for pipes; not needed for the first rover.
4. Tiling: split a big photo into overlapping tiles so small objects are not lost at 640 px.

## 4. Getting the wall size right

The app assumes the whole photo (or the four marked corners) is the wall, and that the operator types the width and height. For a real job:

- **Preferred:** the operator measures the wall with a laser distance meter (cheap, accurate to a few millimetres) and types it. Keep this as the source of truth.
- **Check against the photo:** put a printed ArUco marker of known size on the wall. Its size in pixels gives millimetres per pixel, and the app can warn if the typed size and the photo disagree by more than 3%. A marker needs to be flat, big enough to detect, and sharp; a scale error from a small or blurred marker is the known failure. I found no source that quantifies the error of the whole chain (single photo, marker, rectification, planning), so measure it on 10 real walls before relying on it.
- **Do not** try to estimate absolute size from one photo without a reference.

## 5. The rover: stage gates

Follow `docs/rover-hardware-requirements.md`. The order, with the pass condition that lets you move on:

| Stage | Work | Pass condition |
|---|---|---|
| 0 | Mega, motors, driver, IMU, supply, fuse, e-stop on a bench, firmware built for the Mega (plan P2b) | `MOVE 1000` and `TURN 90` within 2% and 2 degrees; e-stop and heartbeat loss cut all outputs in under 0.5 s (the firmware fires about 510 ms after the last line: change that or the way it is measured) |
| 1 | Floor, dry, taped 8 x 8 ft | row spacing within 30 mm, heading within 2 degrees over a row, x within 5 mm at each row end; decide the y reference |
| 2 | Floor, water | continuous stripes; pump and valve delay measured and put in `config.h` |
| 3 | Pi agent and laptop console (plan P3) | an autonomous run started from the laptop with live position, pause, stop, e-stop |
| 4 | Real wall photo to real paint on the floor | the whole chain on a wall the model has never seen, with the operator's review step |
| 5 | Vertical wall | separate spec, written safety review by someone outside the team, low-height trials first |

Software that can be done before the parts arrive (no hardware needed): the Pi agent against the simulator (P3), saving operator corrections as labels (3.2), the ArUco scale check (4), the real-photo capture and labelling (3.2), and a log format for run data (6).

## 6. What "autonomous and smart" can honestly mean here

| Level | What it is | Needs |
|---|---|---|
| 0 | Operator draws or reviews everything | exists |
| 1 | Photo to obstacles to path to rover, operator reviews unverified items and presses start | exists in software; real rover needs Stage 3 |
| 2 | The rover checks its own work: a camera sees the painted stripes, reports gaps, the plan adds a repair pass | camera on the rover, labelled stripe photos |
| 3 | Learns from its logs: an online slip estimator from edge contacts; layout scoring tuned from real run time and paint use | 20 or more real runs with logs |
| 4 | A learned policy suggests the next move inside the safety envelope (plan P5) | the logs above, a simulator that matches the real rover, a separate spec |

The safety rule stays at every level: the Arduino enforces limits, the e-stop is a physical relay, and a learned model only ever suggests. Do not skip levels: a policy trained on a simulator that has never been compared with the real rover is a guess.

## 7. Risks, plainly

- **Dataset licences.** Several sources are research or non-commercial only (`training/DATA_SOURCES.md`). Before a company or a sale, retrain on the team's own photos and replace them.
- **Domain shift.** A model that scores well on public photos can fail on a plastered Hyderabad wall. Only the team's held-out photos measure that.
- **Paint on the electronics, battery fires, falls.** The floor prototype is safe to fail; a wall robot is not. The hardware document says what has to be decided first.
- **Scope.** Four students, one year. The floor prototype with a reliable detector is a complete, demonstrable result. The vertical wall is the project after it.

## 8. Next four weeks

1. Week 1: buy the Stage 0 parts; photograph 200 to 300 walls; create a free Roboflow or CVAT account; one person downloads BFA-3D by hand.
2. Week 2: label the photos (about 2 hours per 100 photos with machine pre-labels); retrain; read the report.
3. Week 3: Stage 0 on the bench; build the firmware for the Mega (P2b).
4. Week 4: Stage 1 on the floor; save operator corrections as labels; add the ArUco check.

## Sources

- Public datasets and papers: [BFA-YOLO and BFA-3D](https://arxiv.org/html/2409.04025), [BFA-YOLO repository](https://github.com/CVEO/BFA-YOLO), [Roboflow air conditioning set](https://universe.roboflow.com/yolo-uv06o/air-conditioning-dataset), [Roboflow electric meter set](https://universe.roboflow.com/object-detection-v0up3/electric-meter-detector), [Roboflow meter box set](https://universe.roboflow.com/pemula-uf3rv/yolo_fix), [LVIS in the Ultralytics docs](https://docs.ultralytics.com/datasets/detect/lvis), [Objects365 in the Ultralytics docs](https://docs.ultralytics.com/datasets/detect/objects365/), [ADE20K](https://ade20k.csail.mit.edu/).
- Labelling: [Roboflow Auto Label](https://blog.roboflow.com/launch-auto-label/), [MathWorks Grounding DINO and SAM workflow](https://jp.mathworks.com/help/vision/ug/automate-ground-truth-polygon-labeling-using-grounding-dino-and-segment-anything-model-or-grounded-sam-model.html).
- Wall climbing: [MEL-SPIDER vacuum wall climbing robot](https://www.iaarc.org/publications/proceedings_of_the_5th_isarc/fundamental_study_on_a_wallclimbing_robot.html), [bistable suction cups with a safety cable](https://arxiv.org/pdf/2609.21584), [adhesion monitoring review](https://doaj.org/article/63e5aea083604a2baa098605ab94d8ed).
- Scale from markers: [aruco-estimator](https://pypi.org/project/aruco-estimator/1.1.3/).
