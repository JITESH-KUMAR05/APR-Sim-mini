"""
Merge YOLO-format source datasets into one training set using APR's class list.

Layout expected under --sources:
    <sources>/<name>/{train,valid,test}/{images,labels}/    (a Roboflow "YOLO" export)
    <sources>/<name>/data.yaml                               (only its `names` entry is read)
--class-map is JSON: {"<name>": {"<source class>": "<APR class>", ...}, ...}
Source classes not listed are dropped; sources missing from the map are skipped.

Split rule: the source named --heldout (default "team") keeps its test split as the merged
test set. Every other split of every source feeds train or val. Only the held-out source's
test split is protected from training -- its train/valid splits feed training/val like any
other source.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from obstacle_detector import CLASS_NAMES  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_names(yaml_text: str) -> List[str]:
    """Read the `names` entry of a data.yaml (inline list, index mapping or block list)."""
    lines = yaml_text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("names:"):
            continue
        rest = line[len("names:"):].strip()
        if rest.startswith("["):
            return [n.strip().strip("'\"") for n in rest.strip("[]").split(",") if n.strip()]

        indexed: Dict[int, str] = {}
        listed: List[str] = []
        for follow in lines[i + 1:]:
            stripped = follow.strip()
            if not stripped or not (follow.startswith((" ", "\t")) or stripped.startswith("- ")):
                break
            if stripped.startswith("- "):
                listed.append(stripped[2:].strip().strip("'\""))
            elif ":" in stripped:
                key, value = stripped.split(":", 1)
                indexed[int(key)] = value.strip().strip("'\"")
        if indexed:
            return [indexed[k] for k in sorted(indexed)]
        if listed:
            return listed
        raise ValueError("could not read class names from data.yaml")
    raise ValueError("data.yaml has no names entry")


def remap_label_lines(
    lines: List[str], source_names: List[str], mapping: Dict[str, str], class_names: List[str]
) -> List[str]:
    """
    Convert YOLO label lines to APR class ids. Oriented boxes and polygons become their
    axis-aligned envelope because the planner only handles axis-aligned rectangles.
    """
    out = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
            # A negative class id (e.g. "-1") is out of range and must be dropped like any
            # other bad index -- without this check, Python's negative-index semantics would
            # silently wrap it into source_names instead of raising IndexError.
            if class_id < 0:
                continue
            target = mapping.get(source_names[class_id])
            coords = [float(v) for v in parts[1:]]
        except (IndexError, ValueError):
            continue
        if target is None:
            continue

        if len(coords) == 4:
            cx, cy, w, h = coords
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            xs, ys = coords[0::2], coords[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        else:
            continue
        out.append(f"{class_names.index(target)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def _split_targets(is_heldout: bool) -> Dict[str, str]:
    return {"train": "train", "valid": "val", "test": "test" if is_heldout else "train"}


def merge_sources(sources_dir, class_map, out_dir, class_names, heldout: str = "team") -> Dict[str, int]:
    out = Path(out_dir)
    counts = {"train": 0, "val": 0, "test": 0}

    for source in sorted(p for p in Path(sources_dir).iterdir() if p.is_dir()):
        mapping = class_map.get(source.name)
        if mapping is None:
            print(f"skip {source.name}: no entry in the class map")
            continue
        names = parse_names((source / "data.yaml").read_text(encoding="utf-8"))
        unmapped = sorted(set(names) - set(mapping))
        if unmapped:
            print(f"{source.name}: dropping source classes {unmapped}")

        for split, target in _split_targets(source.name == heldout).items():
            images_dir = source / split / "images"
            if not images_dir.is_dir():
                continue
            (out / target / "images").mkdir(parents=True, exist_ok=True)
            (out / target / "labels").mkdir(parents=True, exist_ok=True)
            for image in sorted(images_dir.iterdir()):
                if image.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                label_file = source / split / "labels" / f"{image.stem}.txt"
                raw = label_file.read_text(encoding="utf-8").splitlines() if label_file.exists() else []
                lines = remap_label_lines(raw, names, mapping, list(class_names))

                stem = f"{source.name}_{image.stem}"
                shutil.copyfile(image, out / target / "images" / f"{stem}{image.suffix}")
                text = "\n".join(lines) + "\n" if lines else ""
                (out / target / "labels" / f"{stem}.txt").write_text(text, encoding="utf-8")
                counts[target] += 1
    return counts


def write_dataset_yaml(out_dir, class_names) -> None:
    out = Path(out_dir)
    lines = [
        f"path: {out.resolve().as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(class_names)]
    (out / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", default=str(root / "data" / "sources"))
    parser.add_argument("--class-map", default=str(root / "class_map.json"))
    parser.add_argument("--out", default=str(root / "data" / "merged"))
    parser.add_argument("--heldout", default="team")
    args = parser.parse_args()

    class_map = json.loads(Path(args.class_map).read_text(encoding="utf-8"))
    counts = merge_sources(args.sources, class_map, args.out, CLASS_NAMES, args.heldout)
    write_dataset_yaml(args.out, CLASS_NAMES)
    print(f"merged images: {counts}")
    if counts["test"] == 0:
        print("WARNING: no held-out test images. Add a test split to the team source before trusting any metric.")


if __name__ == "__main__":
    main()
