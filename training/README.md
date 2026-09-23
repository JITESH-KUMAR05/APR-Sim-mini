# Training the obstacle detector

The app runs a YOLO11n model exported to ONNX (`models/apr_obstacles.onnx`). Training happens here, offline, in Google Colab. Only the exported `.onnx` file ships; PyTorch and ultralytics never go into `requirements.txt` or `pyproject.toml`.

Classes, in model order: `window`, `door`, `ac_unit`, `meter_panel`, `pipe`, `grill` (defined once in `obstacle_detector.py`).

## 1. Collect and label your own photos

Public datasets cover windows and doors. Nothing public covers AC condenser units, meter boxes, exterior pipes or grills, so those classes depend on your photos.

- At least 40 to 80 labelled instances per class. More is better.
- Vary lighting (morning, noon, overcast, shade), distance, angle, wall colour and texture. Include clutter such as wires, plants, signs and shadows.
- Include walls with no obstacles at all. The model needs to learn what a clean wall looks like.
- Label in [Roboflow](https://roboflow.com) (free tier) or CVAT, using exactly the class names above.
- Export as **YOLO** format with a train/valid/test split of about 70/10/20. The test split (20%) is your held-out set: it is never trained on and never used to tune thresholds.

Put the export at `training/data/sources/team/` so it contains `data.yaml`, `train/`, `valid/` and `test/`.

## 2. Download the public datasets

Check each dataset's licence on its page before using it.

- [Door and Window Detection](https://universe.roboflow.com/construction-plan/door-and-window-detection-fmw85) (5,382 images) → `training/data/sources/door_window_detection/`
- [yolo-obb-1 wall/door/window](https://universe.roboflow.com/test-v0q9r/yolo-obb-1) (997 images, oriented boxes) → `training/data/sources/wall_door_window_obb/`

Export each in YOLO format. Oriented boxes and polygons are converted to axis-aligned rectangles automatically.

## 3. Check the class map

Open each `data.yaml` and confirm its class names match the keys in `training/class_map.json`. Edit the map if a dataset names a class differently. `build_dataset.py` prints any source classes it drops so mistakes are visible.

## 4. Build the merged dataset

```bash
python training/build_dataset.py
```

Public datasets feed train and val (their test splits go to train). The `team` source keeps its test split as the merged held-out test set. The script warns if there is no test set.

## 5. Train

In Colab (Runtime → Change runtime type → T4 GPU):

```bash
git clone <this repo> && cd <repo>
pip install ultralytics onnx
python training/build_dataset.py
python training/train.py                       # stage 1, from pretrained yolo11n.pt
python training/train.py --weights training/runs/apr_obstacles/weights/best.pt --name apr_stage2   # stage 2
```

Stage 1 can run with only the public sources (leave out `team`), stage 2 adds your photos. The script prints per-class mAP50 on the held-out test split and writes `models/apr_obstacles.onnx`.

## 6. Evaluate with the app's own pipeline

```bash
python training/evaluate.py --model models/apr_obstacles.onnx --images training/data/merged/test/images
```

Prints recall and precision per class at the app's default thresholds, and how many ground-truth objects got no box at all. Targets, set before training:

- `window` and `door`: recall at least 0.90, precision at least 0.85. The script exits non-zero if missed.
- `ac_unit`, `meter_panel`, `pipe`, `grill`: mAP50 at least 0.70 (from the `train.py` output).
- Nothing in a test photo should get no box at all for windows and doors.

Report misses honestly and collect more photos for any class that falls short.

## 7. Ship the model

Copy `models/apr_obstacles.onnx` into the repo, commit it, and push. If the file is missing in production the app silently uses the weaker classical detector and shows a banner, so check the banner after deploying.

## Licences

Ultralytics and weights trained with it are AGPL-3.0. That is acceptable while testing. Before any commercial use, buy an Ultralytics enterprise licence or retrain with a permissively licensed detector. Roboflow datasets carry their own licences, so check each one.

## Known limits

- Wall width and height are entered by hand. Estimating scale from a photo needs depth sensing or a reference marker.
- Photos should be shot square-on, or the four wall corners marked in the UI so the photo is straightened.
- A missed detection is worse than a false one. The review threshold is deliberately low so doubtful objects surface as "unverified" instead of vanishing.
