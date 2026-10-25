# Data sources and licences for the obstacle model

What the trained model `models/apr_obstacles.onnx` was made from, where each part came from, and what each licence is as far as I could tell. **None of this has been checked by a lawyer.** Several sources are research or non-commercial only. Before the project is commercialised, the team has to review every row, or replace the data with its own photos.

| Source | Used for | Access | Licence (to verify on the source's own page) |
|---|---|---|---|
| Open Images V5/V7 (Google): bounding boxes and image labels | human `window` and `door` boxes on photos that annotators confirmed show a building (about 1,900 photos) | public CSV files and the public image bucket, no key | annotations CC BY 4.0; each image has its own licence, mostly CC BY 2.0, which needs credit to the author |
| CMP Facade database (Czech Technical University) | 378 frontal facade photos with human window and door boxes (about 12,000 windows). In its XML the `<x>` pair is the vertical range and the `<y>` pair the horizontal one. | public download | free for research with credit to Tyleček and Šára; not stated as commercial |
| LVIS v1 (annotations) with COCO photos | about 420 real photos with human `air_conditioner` boxes (788 in training), mostly window and wall units seen from outside and inside | annotation files and the COCO image server, no key | annotations CC BY 4.0; the COCO photos are from Flickr with their own licences |
| Door-and-Window Detection dataset (Sezen et al. 2022, github.com/MEnesAtik/Door-And-Window-Detection-Dataset), as a Roboflow YOLOv5 export ("tez - v1") | 727 Google Street View photos of apartment facades with human door and window boxes (the copy received had 727 of its 1,000 photos). Images are 416 x 416, stretched by the export. Imported with `pool.py yolo streetview_dw ...` | supplied by the team | no licence stated; Google Street View imagery falls under Google's terms. Research use only, and cite Sezen et al. (2022) and Isiler et al. (2023) |
| ADE20K scene parsing (MIT CSAIL) | window and door masks turned into boxes on photos where a building fills at least a quarter of the frame (about 140 photos) | public download | research and education, non-commercial |
| OWLv2 `google/owlv2-base-patch16-ensemble` | the "teacher" that proposes `ac_unit`, `meter_panel`, `pipe`, `grill` boxes, and some extra window and door boxes | model downloaded from Hugging Face | model card says Apache 2.0 |
| `wall_synth.py` in this repository | synthetic walls with exact labels | made here | ours |
| YOLO11n starting weights and ultralytics | the detector that was trained | downloaded by ultralytics | AGPL-3.0, accepted for the prototype by the project lead; revisit before commercial use |

## What is and is not human-labelled

- `window`, `door`, and `ac_unit` on the LVIS photos: boxes made by people on real photos. Annotators skip some objects, so the model is partly punished for finding real windows nobody boxed.
- `ac_unit` has about 790 human boxes on real photos from LVIS. `meter_panel`, `pipe`, `grill` have only a few dozen teacher boxes on real photos, and the teacher is not reliable on them. Almost all of what the model knows about these three comes from the synthetic walls, which are cartoon-like. **There is no real-world number for them.** LVIS `pipe` boxes were looked at and dropped: they are exhaust pipes and plumbing, not wall pipes. The only fix is the team's own labelled photos (see `training/README.md` and `docs/real-world-roadmap.md`).

## The team's own and the operator's photos

`exports/corrections` (written by the app when the operator changes the detected obstacles and leaves the "keep my corrections" box ticked) and any photos the team labels hold pictures of real people's homes, and may show faces and number plates. They stay on the computer, are never served back by the app, and are not in git (`exports/` is ignored). Review them before they are shared or trained on, and get the owner's consent for photos of private property. The labels in `exports/corrections` are the model's own detections plus the operator's changes, so look at a contact sheet of a batch before training on it.

## Splits

Every real photo is assigned to train (80%), validation (10%) or test (10%) from a hash of its id before any labelling or augmentation, so no photo can be on both sides. Synthetic walls use separate seed ranges, and a separate synthetic test set is kept apart from the real test photos. The two photos in `samples/` are not real photos (the app draws them) and are used only for demonstration.

## Rebuilding

```
uv sync --project training/gpu                         # once, needs an NVIDIA GPU driver
training/gpu/.venv/Scripts/python.exe training/gpu/fetch_openimages.py
# unzip CMP_facade_DB_base.zip into training/data/staging/cmp and put ADEChallengeData2016.zip in training/data/raw
training/gpu/.venv/Scripts/python.exe training/gpu/fetch_lvis.py       # needs lvis_v1_val.json and lvis_v1_train.json in training/data/raw
training/gpu/.venv/Scripts/python.exe training/gpu/pool.py
training/gpu/.venv/Scripts/python.exe training/gpu/run_teacher.py
training/gpu/.venv/Scripts/python.exe training/gpu/fuse.py --synth 800
training/gpu/.venv/Scripts/python.exe training/gpu/train_gpu.py --weights training/runs/apr_v1/weights/best.pt --epochs 50 --name apr_v5
.venv/Scripts/python.exe training/eval_report.py --model <path to best.onnx>
.venv/Scripts/python.exe training/eval_by_source.py <path to best.onnx>
```

`training/data/` and `training/runs/` are git-ignored. The raw downloads are about 1.5 GB.
