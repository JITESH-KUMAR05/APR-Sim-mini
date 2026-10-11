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
