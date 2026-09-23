import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

from build_dataset import merge_sources, parse_names, remap_label_lines, write_dataset_yaml
from obstacle_detector import CLASS_NAMES


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


if __name__ == "__main__":
    unittest.main()
