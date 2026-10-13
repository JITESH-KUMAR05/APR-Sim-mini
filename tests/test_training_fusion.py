import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "training", "gpu"))
sys.path.insert(0, ROOT)

import fuse
import pool

W, D, AC, MP, PI, GR = range(6)
HUMAN_WINDOW = (W, 0.10, 0.10, 0.30, 0.40)


def proposal(cls, score, box):
    return [cls, score, *box]


class TestHumanAndTeacherWindows(unittest.TestCase):
    def test_human_boxes_are_always_kept(self):
        boxes, usable, _ = fuse.fuse_image([HUMAN_WINDOW], [])
        self.assertEqual(boxes, [HUMAN_WINDOW])
        self.assertTrue(usable)

    def test_a_confident_teacher_window_away_from_human_boxes_is_added(self):
        boxes, _, stats = fuse.fuse_image([HUMAN_WINDOW], [proposal(W, 0.5, (0.5, 0.1, 0.7, 0.4))])
        self.assertIn((W, 0.5, 0.1, 0.7, 0.4), boxes)
        self.assertEqual(stats["teacher_wd_added"], 1)

    def test_a_teacher_window_on_a_human_box_is_not_added_twice(self):
        boxes, _, _ = fuse.fuse_image([HUMAN_WINDOW], [proposal(W, 0.9, (0.11, 0.11, 0.30, 0.40))])
        self.assertEqual(boxes, [HUMAN_WINDOW])

    def test_a_weak_or_huge_teacher_window_is_not_added(self):
        boxes, _, _ = fuse.fuse_image([], [proposal(W, 0.2, (0.5, 0.1, 0.7, 0.4)), proposal(W, 0.9, (0.0, 0.0, 0.9, 0.9))])
        self.assertEqual(boxes, [])


class TestRareClasses(unittest.TestCase):
    def test_a_confident_ac_unit_is_added(self):
        boxes, usable, _ = fuse.fuse_image([], [proposal(AC, 0.5, (0.6, 0.6, 0.72, 0.72))])
        self.assertEqual([b[0] for b in boxes], [AC])
        self.assertTrue(usable)

    def test_an_ac_or_meter_that_is_really_a_human_labelled_window_is_dropped(self):
        for cls in (AC, MP):
            with self.subTest(cls=cls):
                boxes, _, _ = fuse.fuse_image([HUMAN_WINDOW], [proposal(cls, 0.9, (0.10, 0.10, 0.30, 0.40))])
                self.assertEqual([b[0] for b in boxes], [W])

    def test_a_grill_may_sit_on_a_window(self):
        boxes, _, _ = fuse.fuse_image([HUMAN_WINDOW], [proposal(GR, 0.6, (0.10, 0.10, 0.30, 0.40))])
        self.assertEqual(sorted(b[0] for b in boxes), [W, GR])

    def test_a_pipe_has_to_be_long_and_thin(self):
        boxes, _, _ = fuse.fuse_image([], [proposal(PI, 0.9, (0.5, 0.1, 0.56, 0.2)), proposal(PI, 0.9, (0.2, 0.1, 0.23, 0.9))])
        self.assertEqual([b[0] for b in boxes], [PI])
        self.assertAlmostEqual(boxes[0][1], 0.2)

    def test_a_photo_with_an_unsure_rare_box_is_left_out_of_training(self):
        box = (0.6, 0.6, 0.72, 0.72)
        boxes, usable, _ = fuse.fuse_image([HUMAN_WINDOW], [proposal(AC, 0.26, box)])  # between 0.24 and 0.30
        self.assertFalse(usable)
        boxes, usable, _ = fuse.fuse_image([HUMAN_WINDOW], [proposal(AC, 0.23, box)])  # below the grey zone
        self.assertTrue(usable)
        self.assertEqual([b[0] for b in boxes], [W])
        boxes, usable, _ = fuse.fuse_image([HUMAN_WINDOW], [proposal(AC, 0.30, box)])  # at the threshold
        self.assertTrue(usable)
        self.assertEqual(sorted(b[0] for b in boxes), [W, AC])

    def test_a_very_low_score_is_just_background(self):
        _, usable, _ = fuse.fuse_image([HUMAN_WINDOW], [proposal(AC, 0.12, (0.6, 0.6, 0.72, 0.72))])
        self.assertTrue(usable)

    def test_a_row_of_misfires_leaves_the_photo_out(self):
        row = [proposal(AC, 0.6, (0.05 + 0.1 * i, 0.1, 0.1 + 0.1 * i, 0.2)) for i in range(9)]
        boxes, usable, stats = fuse.fuse_image([], row)
        self.assertFalse(usable)
        self.assertEqual(boxes, [])
        self.assertEqual(stats["dropped_misfire"], 9)


class TestHumanLabelledRareClasses(unittest.TestCase):
    def test_a_class_people_labelled_in_the_photo_is_not_touched_by_the_teacher(self):
        human = [(AC, 0.1, 0.1, 0.3, 0.3)]
        boxes, usable, stats = fuse.fuse_image(human, [proposal(AC, 0.9, (0.6, 0.6, 0.72, 0.72)), proposal(AC, 0.26, (0.4, 0.4, 0.5, 0.5))])
        self.assertEqual(boxes, human)
        self.assertTrue(usable)  # the unsure teacher box cannot veto a photo whose AC boxes are human-made
        self.assertEqual(stats["teacher_rare_added"], 0)

    def test_other_rare_classes_in_the_same_photo_still_follow_the_teacher_rules(self):
        human = [(AC, 0.1, 0.1, 0.3, 0.3)]
        boxes, _, _ = fuse.fuse_image(human, [proposal(GR, 0.6, (0.6, 0.1, 0.75, 0.3))])
        self.assertEqual(sorted(b[0] for b in boxes), [AC, GR])


class PoolTestCase(unittest.TestCase):
    def use_pool(self, tmp):
        """Point the pool module at a temporary folder for one test and restore it afterwards."""
        for name in ("POOL", "DATA"):
            self.addCleanup(setattr, pool, name, getattr(pool, name))
        pool.POOL = os.path.join(tmp, "pool")
        pool.DATA = os.path.join(tmp, "data")
        os.makedirs(os.path.join(pool.POOL, "images"))
        os.makedirs(os.path.join(pool.POOL, "labels_human"))


class TestImporters(PoolTestCase):
    def test_a_mapping_turns_dataset_names_into_app_classes(self):
        self.assertEqual(pool.parse_mapping("Window=window, air_conditioning=ac_unit"), {"window": 0, "air_conditioning": 2})
        with self.assertRaises(ValueError):
            pool.parse_mapping("ac=aircon")

    def test_class_names_are_read_from_all_three_yaml_layouts(self):
        import tempfile

        layouts = {
            "flow": "train: x\nnames: ['air conditioner', \"grill\"]\nnc: 2\n",
            "block": "names:\n  - air conditioner\n  - grill\nnc: 2\n",
            "numbered": "path: x\nnames:\n  0: air conditioner\n  1: grill\n",
        }
        for name, text in layouts.items():
            with self.subTest(layout=name), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "data.yaml")
                open(path, "w").write(text)
                self.assertEqual(pool.read_names(path), ["air conditioner", "grill"])

    def test_a_yolo_export_is_pooled_remapped_and_split_by_hash(self):
        import cv2
        import numpy as np
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            export = os.path.join(tmp, "export")
            for part in ("train", "test"):
                os.makedirs(os.path.join(export, part, "images"))
                os.makedirs(os.path.join(export, part, "labels"))
            open(os.path.join(export, "data.yaml"), "w").write("names: ['aircon', 'cat', 'bars']\n")
            for part, name, label in (("train", "a", "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n"), ("test", "b", "2 0.5 0.5 0.4 0.4\n"),
                                      ("train", "c", "1 0.5 0.5 0.2 0.2\n")):
                cv2.imwrite(os.path.join(export, part, "images", name + ".jpg"), np.zeros((40, 60, 3), np.uint8))
                open(os.path.join(export, part, "labels", name + ".txt"), "w").write(label)
            self.use_pool(tmp)
            index = {}
            n = pool.from_yolo_export(index, "team", export, {"aircon": 2, "bars": 5})
            self.assertEqual(n, 2)  # photo c has no mapped class and is skipped
            self.assertEqual(set(index), {"team_a", "team_b"})
            self.assertEqual(open(os.path.join(pool.POOL, "labels_human", "team_a.txt")).read().split(), ["2", "0.500000", "0.500000", "0.200000", "0.200000"])
            self.assertEqual(open(os.path.join(pool.POOL, "labels_human", "team_b.txt")).read().split()[0], "5")
            n = pool.from_yolo_export(index, "team", export, {"aircon": 2}, keep_empty=True)
            self.assertEqual(n, 3)  # clean-wall photos are kept on request


class TestCmpImporter(PoolTestCase):
    @staticmethod
    def obj(label_id, name, x1, x2, y1, y2):
        # the real files put every value on its own line
        return "\n".join(["<object>", "<points>", "<x>", str(x1), "</x>", "<x>", str(x2), "</x>", "<y>", str(y1), "</y>",
                          "<y>", str(y2), "</y>", "</points>", "<label>", str(label_id), "</label>", "<labelname>", name,
                          "</labelname>", "<flag>", "1", "</flag>", "</object>", ""])

    @property
    def XML(self):
        return self.obj(2, "facade", 0.0, 1, 0.0, 1) + self.obj(3, "window", 0.1, 0.3, 0.2, 0.5) + self.obj(4, "door", 0.6, 0.8, 0.4, 1.0)

    def test_window_and_door_boxes_are_read_from_the_real_multi_line_xml(self):
        # This once produced empty label files for all 378 photos: the label text sits on its own line.
        import cv2
        import numpy as np
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "base")
            os.makedirs(folder)
            cv2.imwrite(os.path.join(folder, "cmp_b0001.jpg"), np.zeros((40, 60, 3), np.uint8))
            open(os.path.join(folder, "cmp_b0001.xml"), "w").write(self.XML)
            self.use_pool(tmp)
            index = {}
            self.assertEqual(pool.from_cmp(index, folder), 1)
            lines = open(os.path.join(pool.POOL, "labels_human", "cmp_cmp_b0001.txt")).read().splitlines()
            self.assertEqual([line.split()[0] for line in lines], ["0", "1"])  # the facade box is not an obstacle
            cls, cx, cy, w, h = lines[0].split()
            # the <x> pair (0.1 to 0.3) is the vertical range and the <y> pair (0.2 to 0.5) the horizontal range
            self.assertAlmostEqual(float(cx), 0.35, places=4)
            self.assertAlmostEqual(float(cy), 0.2, places=4)
            self.assertAlmostEqual(float(w), 0.3, places=4)
            self.assertAlmostEqual(float(h), 0.2, places=4)


class TestHumanVeto(unittest.TestCase):
    def test_a_teacher_window_that_contains_a_human_box_is_not_added(self):
        human = [(AC, 0.40, 0.40, 0.50, 0.50)]
        boxes, _, _ = fuse.fuse_image(human, [proposal(W, 0.9, (0.1, 0.1, 0.9, 0.9))])
        self.assertEqual(boxes, human)

    def test_a_teacher_box_of_another_class_on_a_human_box_is_dropped(self):
        human = [(AC, 0.40, 0.40, 0.60, 0.60)]
        boxes, usable, _ = fuse.fuse_image(human, [proposal(GR, 0.9, (0.41, 0.41, 0.60, 0.60)), proposal(MP, 0.9, (0.4, 0.4, 0.6, 0.6))])
        self.assertEqual(boxes, human)
        self.assertTrue(usable)


class TestPoolRules(PoolTestCase):
    def test_augmented_copies_of_one_photo_always_share_a_split(self):
        for n in range(40):
            stem = f"wall_{n}"
            splits = {pool.split_of(pool.split_key("team", f"{stem}_jpg.rf.{h:032x}")) for h in range(12)}
            self.assertEqual(len(splits), 1, stem)
        self.assertEqual(pool.split_key("team", "a_jpg.rf.0123abcd"), pool.split_key("team", "a.jpg"[:-4]))

    def test_the_split_does_not_depend_on_which_part_a_photo_was_exported_in(self):
        import cv2
        import numpy as np
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.use_pool(tmp)
            splits = {}
            for part in ("train", "test"):
                export = os.path.join(tmp, "export_" + part)
                os.makedirs(os.path.join(export, part, "images"))
                os.makedirs(os.path.join(export, part, "labels"))
                open(os.path.join(export, "data.yaml"), "w").write("names: ['aircon']\n")
                cv2.imwrite(os.path.join(export, part, "images", "wall7.jpg"), np.zeros((20, 20, 3), np.uint8))
                open(os.path.join(export, part, "labels", "wall7.txt"), "w").write("0 0.5 0.5 0.4 0.4\n")
                index = {}
                pool.from_yolo_export(index, "team", export, {"aircon": 2})
                splits[part] = index["team_wall7"]["split"]
            self.assertEqual(splits["train"], splits["test"])

    def test_read_names_handles_comments_inline_dicts_and_multi_line_lists(self):
        import tempfile

        layouts = {
            "comment": "names:\n  0: window # the opening\n  1: door\n",
            "inline dict": "names: {0: window, 1: door}\n",
            "multi-line flow": "names: ['window',\n  'door']\nnc: 2\n",
        }
        for name, text in layouts.items():
            with self.subTest(layout=name), tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "data.yaml")
                open(path, "w").write(text)
                self.assertEqual(pool.read_names(path), ["window", "door"])

    def test_a_mapping_that_fits_nothing_raises_instead_of_importing_clean_walls(self):
        import cv2
        import numpy as np
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.use_pool(tmp)
            export = os.path.join(tmp, "export")
            os.makedirs(os.path.join(export, "train", "images"))
            os.makedirs(os.path.join(export, "train", "labels"))
            open(os.path.join(export, "data.yaml"), "w").write("names: ['aircon']\n")
            cv2.imwrite(os.path.join(export, "train", "images", "a.jpg"), np.zeros((20, 20, 3), np.uint8))
            for mapping in ({"conditioner": 2}, {"aircon": 2, "typo": 3}):
                with self.subTest(mapping=mapping), self.assertRaisesRegex(ValueError, "does not fit"):
                    pool.from_yolo_export({}, "team", export, mapping, keep_empty=True)

    def test_polygons_become_boxes_and_a_confidence_field_is_ignored(self):
        remap = {0: 2}
        polygon = pool._box_from_fields("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5".split(), remap)
        self.assertEqual(polygon[0], 2)
        self.assertAlmostEqual(polygon[1], 0.3)
        self.assertAlmostEqual(polygon[3], 0.4)
        with_confidence = pool._box_from_fields("0 0.5 0.5 0.2 0.2 0.93".split(), remap)
        self.assertEqual(with_confidence[0], 2)
        for got, expected in zip(with_confidence[1:], (0.5, 0.5, 0.2, 0.2)):
            self.assertAlmostEqual(got, expected)
        self.assertIsNone(pool._box_from_fields("1 0.5 0.5 0.2 0.2".split(), remap))  # a class that is not mapped
        clipped = pool._box_from_fields("0 0.95 0.5 0.3 0.2".split(), remap)
        self.assertAlmostEqual(clipped[1] + clipped[3] / 2, 1.0)

    def test_source_names_are_checked(self):
        for bad in ("oi", "cmp", "Team", "a/b", "", "my team"):
            with self.subTest(source=bad), self.assertRaises(ValueError):
                pool.from_yolo_export({}, bad, "nowhere", {"a": 0})

    def test_a_missing_lvis_download_does_not_purge_the_pool(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.use_pool(tmp)
            open(os.path.join(pool.POOL, "images", "lvis_1.jpg"), "wb").write(b"x")
            open(os.path.join(pool.POOL, "labels_human", "lvis_1.txt"), "w").write("2 0.5 0.5 0.1 0.1\n")
            index = {"lvis_1": {"source": "lvis", "split": "train"}}
            self.assertEqual(pool.from_lvis(index), 0)
            self.assertIn("lvis_1", index)
            self.assertTrue(os.path.exists(os.path.join(pool.POOL, "images", "lvis_1.jpg")))


class TestHelpers(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(fuse.iou((0, 0, 1, 1), (0, 0, 1, 1)), 1.0)
        self.assertEqual(fuse.iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)
        self.assertAlmostEqual(fuse.iou((0, 0, 2, 1), (1, 0, 3, 1)), 1 / 3)

    def test_nms_keeps_the_best_of_overlapping_boxes(self):
        kept = fuse.nms([[0, 0.5, 0, 0, 1, 1], [0, 0.9, 0.05, 0, 1.05, 1], [0, 0.4, 3, 3, 4, 4]])
        self.assertEqual([k[1] for k in kept], [0.9, 0.4])

    def test_the_split_is_stable_and_roughly_80_10_10(self):
        ids = [f"img_{i}" for i in range(3000)]
        splits = [pool.split_of(i) for i in ids]
        self.assertEqual(splits, [pool.split_of(i) for i in ids])
        for name, share in (("train", 0.8), ("val", 0.1), ("test", 0.1)):
            self.assertAlmostEqual(splits.count(name) / len(ids), share, delta=0.03)


if __name__ == "__main__":
    unittest.main()
