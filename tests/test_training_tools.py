import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

from build_dataset import merge_sources, parse_names, remap_label_lines, write_dataset_yaml
from evaluate import evaluate, greedy_match, iou, read_yolo_labels, summarize
from fakes import FakeDetector
from obstacle_detector import CLASS_NAMES, Detection


class TestParseNames(unittest.TestCase):
    def test_inline_list(self):
        self.assertEqual(parse_names("train: x\nnames: ['door', 'window']\nnc: 2"), ["door", "window"])

    def test_index_mapping(self):
        self.assertEqual(parse_names("names:\n  1: window\n  0: door\nnc: 2"), ["door", "window"])

    def test_block_list(self):
        self.assertEqual(parse_names("names:\n- door\n- window\nnc: 2"), ["door", "window"])

    def test_missing_names_raises(self):
        with self.assertRaises(ValueError):
            parse_names("train: x")


class TestRemapLabelLines(unittest.TestCase):
    NAMES = ["door", "window", "wall"]
    MAPPING = {"door": "door", "window": "window"}

    def remap(self, lines):
        return remap_label_lines(lines, self.NAMES, self.MAPPING, CLASS_NAMES)

    def test_box_is_remapped_to_apr_class_ids(self):
        self.assertEqual(self.remap(["1 0.5 0.5 0.2 0.2"]), ["0 0.500000 0.500000 0.200000 0.200000"])

    def test_unmapped_out_of_range_blank_and_malformed_lines_are_dropped(self):
        self.assertEqual(self.remap(["2 0.5 0.5 0.2 0.2", "9 0.5 0.5 0.2 0.2", "", "0 0.5 0.5 0.2"]), [])

    def test_oriented_box_becomes_its_axis_aligned_envelope(self):
        self.assertEqual(
            self.remap(["0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4"]),
            ["1 0.200000 0.300000 0.200000 0.200000"],
        )


def write(path, text=b""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text if isinstance(text, bytes) else text.encode())


class TestMergeSources(unittest.TestCase):
    def test_splits_remaps_and_keeps_team_test_held_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources, out = root / "sources", root / "merged"

            write(sources / "pub" / "data.yaml", "names: ['door', 'window', 'wall']")
            write(sources / "pub" / "train" / "images" / "a.jpg", b"x")
            write(sources / "pub" / "train" / "labels" / "a.txt", "0 0.5 0.5 0.2 0.2\n2 0.1 0.1 0.1 0.1\n")
            write(sources / "pub" / "test" / "images" / "b.jpg", b"x")

            write(sources / "team" / "data.yaml", "names: ['window']")
            write(sources / "team" / "valid" / "images" / "c.jpg", b"x")  # no label file: a clean wall
            write(sources / "team" / "test" / "images" / "d.jpg", b"x")
            write(sources / "team" / "test" / "labels" / "d.txt", "0 0.5 0.5 0.3 0.3\n")

            write(sources / "other" / "data.yaml", "names: ['x']")  # not in the class map

            class_map = {"pub": {"door": "door", "window": "window"}, "team": {"window": "window"}}
            counts = merge_sources(sources, class_map, out, CLASS_NAMES, heldout="team")

            self.assertEqual(counts, {"train": 2, "val": 1, "test": 1})
            self.assertEqual((out / "train" / "labels" / "pub_a.txt").read_text(), "1 0.500000 0.500000 0.200000 0.200000\n")
            self.assertTrue((out / "train" / "images" / "pub_b.jpg").exists())  # public test feeds training
            self.assertFalse((out / "test" / "images" / "pub_b.jpg").exists())
            self.assertEqual((out / "val" / "labels" / "team_c.txt").read_text(), "")
            self.assertTrue((out / "test" / "images" / "team_d.jpg").exists())


class TestWriteDatasetYaml(unittest.TestCase):
    def test_lists_classes_in_index_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_dataset_yaml(Path(tmp), CLASS_NAMES)
            text = (Path(tmp) / "dataset.yaml").read_text()
        self.assertIn("names:\n  0: window\n  1: door", text)
        self.assertIn("test: test/images", text)


class TestMatching(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)

    def test_greedy_match_pairs_highest_scoring_detection_first(self):
        gts = [(0, 0, 10, 10)]
        dets = [(0, 0, 10, 10), (1, 1, 10, 10)]
        self.assertEqual(greedy_match(gts, dets, [0.6, 0.9]), [(0, 1)])

    def test_greedy_match_ignores_low_overlap(self):
        self.assertEqual(greedy_match([(0, 0, 10, 10)], [(50, 50, 60, 60)], [0.9]), [])


class TestEvaluate(unittest.TestCase):
    def make_split(self, tmp):
        images, labels = Path(tmp) / "images", Path(tmp) / "labels"
        images.mkdir()
        labels.mkdir()
        cv2.imwrite(str(images / "wall.png"), np.zeros((100, 100, 3), dtype=np.uint8))
        (labels / "wall.txt").write_text("0 0.5 0.5 0.4 0.4\n")  # a window at (30,30)-(70,70)
        return images, labels

    def test_read_yolo_labels_returns_pixel_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, labels = self.make_split(tmp)
            self.assertEqual(read_yolo_labels(labels / "wall.txt", 100, 100), [(0, 30.0, 30.0, 70.0, 70.0)])
            self.assertEqual(read_yolo_labels(labels / "missing.txt", 100, 100), [])

    def test_correct_detection_counts_as_true_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels = self.make_split(tmp)
            stats = evaluate(FakeDetector([Detection("window", 0.9, 30, 30, 70, 70)]), images, labels, CLASS_NAMES, 0.25)
            summary = summarize(stats)
        self.assertEqual(stats["window"]["tp"], 1)
        self.assertEqual(summary["window"]["recall"], 1.0)
        self.assertEqual(summary["window"]["precision"], 1.0)
        self.assertIsNone(summary["door"]["recall"])

    def test_wrong_class_is_a_miss_for_the_class_but_still_found_by_any_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels = self.make_split(tmp)
            stats = evaluate(FakeDetector([Detection("door", 0.9, 30, 30, 70, 70)]), images, labels, CLASS_NAMES, 0.25)
        self.assertEqual(stats["window"]["fn"], 1)
        self.assertEqual(stats["door"]["fp"], 1)
        self.assertEqual(stats["window"]["found_any"], 1)

    def test_no_detection_is_a_miss_and_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            images, labels = self.make_split(tmp)
            stats = evaluate(FakeDetector([]), images, labels, CLASS_NAMES, 0.25)
        self.assertEqual(stats["window"]["fn"], 1)
        self.assertEqual(stats["window"]["found_any"], 0)


if __name__ == "__main__":
    unittest.main()
