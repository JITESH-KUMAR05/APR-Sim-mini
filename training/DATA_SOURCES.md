# Data sources and licences for the obstacle model

What the trained model `models/apr_obstacles.onnx` was made from, where each part came from, and what each licence is as far as I could tell. **None of this has been checked by a lawyer.** Several sources are research or non-commercial only. Before the project is commercialised, the team has to review every row, or replace the data with its own photos.

| Source | Used for | Access | Licence (to verify on the source's own page) |
|---|---|---|---|
| Open Images V5/V7 (Google): bounding boxes and image labels | human `window` and `door` boxes on photos that annotators confirmed show a building (about 1,900 photos) | public CSV files and the public image bucket, no key | annotations CC BY 4.0; each image has its own licence, mostly CC BY 2.0, which needs credit to the author |
| CMP Facade database (Czech Technical University) | 378 facade photos with human window and door boxes | public download | free for research with credit to Tyleček and Šára; not stated as commercial |
| ADE20K scene parsing (MIT CSAIL) | window and door masks turned into boxes on photos where a building fills at least a quarter of the frame (about 140 photos) | public download | research and education, non-commercial |
| OWLv2 `google/owlv2-base-patch16-ensemble` | the "teacher" that proposes `ac_unit`, `meter_panel`, `pipe`, `grill` boxes, and some extra window and door boxes | model downloaded from Hugging Face | model card says Apache 2.0 |
| `wall_synth.py` in this repository | synthetic walls with exact labels | made here | ours |
| YOLO11n starting weights and ultralytics | the detector that was trained | downloaded by ultralytics | AGPL-3.0, accepted for the prototype by the project lead; revisit before commercial use |

## What is and is not human-labelled

- `window`, `door`: boxes made by people on real photos. Annotators skip some objects, so the model is partly punished for finding real windows nobody boxed.
- `ac_unit`, `meter_panel`, `pipe`, `grill`: the real training photos contain only a handful of boxes for these, all from the teacher, which is not reliable on them. Almost all of what the model knows about these four comes from the synthetic walls, which are cartoon-like. **The model is expected to be weak on these four classes on real walls.** The only fix is the team's own labelled photos (see `training/README.md`).

## Splits

Every real photo is assigned to train (80%), validation (10%) or test (10%) from a hash of its id before any labelling or augmentation, so no photo can be on both sides. Synthetic walls use separate seed ranges, and a separate synthetic test set is kept apart from the real test photos. The two photos in `samples/` are not real photos (the app draws them) and are used only for demonstration.

## Rebuilding

```
uv sync --project training/gpu                         # once, needs an NVIDIA GPU driver
training/gpu/.venv/Scripts/python.exe training/gpu/fetch_openimages.py
# unzip CMP_facade_DB_base.zip into training/data/staging/cmp and put ADEChallengeData2016.zip in training/data/raw
training/gpu/.venv/Scripts/python.exe training/gpu/pool.py
training/gpu/.venv/Scripts/python.exe training/gpu/run_teacher.py
training/gpu/.venv/Scripts/python.exe training/gpu/fuse.py --synth 1200
training/gpu/.venv/Scripts/python.exe training/gpu/train_gpu.py --epochs 50 --name apr_v1
.venv/Scripts/python.exe training/eval_report.py --model <path to best.onnx>
```

`training/data/` and `training/runs/` are git-ignored. The raw downloads are about 1.5 GB.
