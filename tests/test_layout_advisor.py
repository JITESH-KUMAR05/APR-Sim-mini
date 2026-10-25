import unittest

from geometry_engine import DIRECTIONS, START_CORNERS, GeometryEngine
from layout_advisor import choose_layout, evaluate_layouts

W, H = 4000.0, 2800.0
WINDOW = {"id": "w", "type": "window", "x": 1500.0, "y": 900.0, "w": 900.0, "h": 1100.0}
DOOR = {"id": "d", "type": "door", "x": 3300.0, "y": 0.0, "w": 700.0, "h": 2000.0}
TALL_PIPE = {"id": "p", "type": "pipe", "x": 600.0, "y": 0.0, "w": 120.0, "h": 2800.0}


def options(obstacles, **kw):
    return evaluate_layouts(GeometryEngine(), W, H, 250.0, 12.0, 60.0, obstacles, **kw)


class TestEvaluate(unittest.TestCase):
    def test_every_corner_and_direction_is_scored(self):
        result = options([WINDOW])
        self.assertEqual(len(result), len(START_CORNERS) * len(DIRECTIONS))
        self.assertEqual({(o["start_corner"], o["direction"]) for o in result},
                         {(c, d) for c in START_CORNERS for d in DIRECTIONS})
        for rank, option in enumerate(result, 1):
            self.assertEqual(option["rank"], rank)
            for key in ("regions", "unreached_m2", "transit_m", "total_m", "spray_m", "turns"):
                self.assertIn(key, option)

    def test_the_list_is_sorted_by_unreached_wall_then_non_painting_travel_then_turns(self):
        result = options([WINDOW, DOOR])
        keys = [(o["unreached_m2"], round(o["transit_m"], 1), o["turns"]) for o in result]
        self.assertEqual(keys, sorted(keys))

    def test_a_floor_to_ceiling_pipe_cuts_the_wall_and_the_rover_starts_on_the_bigger_side(self):
        result = options([TALL_PIPE])
        by = {(o["start_corner"], o["direction"]): o for o in result}
        left, right = by[("bottom_left", "horizontal")], by[("bottom_right", "horizontal")]
        self.assertGreater(left["unreached_m2"], right["unreached_m2"])  # the strip left of the pipe is the small side
        self.assertGreater(right["unreached_m2"], 0)
        self.assertTrue(result[0]["start_corner"].endswith("right"))

    def test_a_clear_wall_is_one_region_with_nothing_cut_off(self):
        for option in options([]):
            self.assertEqual((option["regions"], option["unreached_m2"]), (1, 0))
            self.assertAlmostEqual(option["transit_m"] + option["spray_m"], option["total_m"], delta=0.02)

    def test_a_window_in_the_middle_splits_the_wall_into_regions(self):
        for option in options([WINDOW]):
            self.assertGreater(option["regions"], 1)
            self.assertEqual(option["unreached_m2"], 0)

    def test_scoring_does_not_change_the_obstacles(self):
        obstacles = [dict(WINDOW), dict(DOOR)]
        before = [dict(o) for o in obstacles]
        options(obstacles)
        self.assertEqual(obstacles, before)


class TestChoose(unittest.TestCase):
    def test_auto_picks_the_top_option_and_explains_it(self):
        choice = choose_layout(GeometryEngine(), W, H, 250.0, 12.0, 60.0, [TALL_PIPE], "auto", "auto")
        self.assertEqual(choice["mode"], "auto")
        self.assertEqual((choice["start_corner"], choice["direction"]),
                         (choice["options"][0]["start_corner"], choice["options"][0]["direction"]))
        self.assertIn("cut off", choice["reason"])
        self.assertIn("region", choice["reason"])

    def test_a_fixed_corner_limits_the_search_to_that_corner(self):
        choice = choose_layout(GeometryEngine(), W, H, 250.0, 12.0, 60.0, [WINDOW], "top_right", "auto")
        self.assertEqual(choice["start_corner"], "top_right")
        self.assertEqual({o["start_corner"] for o in choice["options"]}, {"top_right"})

    def test_a_fixed_direction_limits_the_search_to_that_direction(self):
        choice = choose_layout(GeometryEngine(), W, H, 250.0, 12.0, 60.0, [WINDOW], "auto", "horizontal")
        self.assertEqual(choice["direction"], "horizontal")
        self.assertEqual({o["direction"] for o in choice["options"]}, {"horizontal"})

    def test_both_fixed_is_manual_and_keeps_the_operators_choice(self):
        choice = choose_layout(GeometryEngine(), W, H, 250.0, 12.0, 60.0, [WINDOW], "bottom_right", "vertical")
        self.assertEqual(choice["mode"], "manual")
        self.assertEqual((choice["start_corner"], choice["direction"]), ("bottom_right", "vertical"))
        self.assertEqual(len(choice["options"]), 1)

    def test_missing_values_mean_the_defaults_not_auto(self):
        choice = choose_layout(GeometryEngine(), W, H, 250.0, 12.0, 60.0, [], None, None)
        self.assertEqual((choice["mode"], choice["start_corner"], choice["direction"]), ("manual", "bottom_left", "horizontal"))

    def test_bad_values_are_rejected(self):
        with self.assertRaises(ValueError):
            choose_layout(GeometryEngine(), W, H, 250.0, 12.0, 60.0, [], "middle", "auto")
        with self.assertRaises(ValueError):
            choose_layout(GeometryEngine(), W, H, 250.0, 12.0, 60.0, [], "auto", "diagonal")

    def test_ties_keep_the_default_corner_first(self):
        choice = choose_layout(GeometryEngine(), W, H, 250.0, 12.0, 60.0, [], "auto", "horizontal")
        self.assertEqual(choice["start_corner"], "bottom_left")


class TestRoverLimits(unittest.TestCase):
    """The firmware refuses a spray MOVE that would run the pump over 30 s (about 7 m at 250 mm/s)."""

    def test_a_tall_wall_gets_horizontal_stripes_because_vertical_ones_are_refused(self):
        choice = choose_layout(GeometryEngine(), 5000.0, 8000.0, 250.0, 12.0, 60.0, [], "auto", "auto")
        self.assertEqual(choice["direction"], "horizontal")
        by = {(o["start_corner"], o["direction"]): o for o in choice["options"]}
        self.assertFalse(by[("bottom_left", "vertical")]["feasible"])
        self.assertTrue(by[("bottom_left", "horizontal")]["feasible"])
        self.assertIn(choice["options"][0]["feasible"], (True,))

    def test_feasible_layouts_sort_before_infeasible_ones(self):
        options = evaluate_layouts(GeometryEngine(), 5000.0, 8000.0, 250.0, 12.0, 60.0, [])
        flags = [o["feasible"] for o in options]
        self.assertEqual(flags, sorted(flags, reverse=True))

    def test_a_slower_rover_has_a_shorter_limit(self):
        options = evaluate_layouts(GeometryEngine(), 4000.0, 2800.0, 250.0, 12.0, 60.0, [], max_speed_mps=0.1)
        self.assertTrue(all(not o["feasible"] for o in options if o["direction"] == "horizontal"))  # 4 m > 2.8 m at 100 mm/s
        self.assertTrue(any(o["feasible"] for o in options if o["direction"] == "vertical"))  # 2.8 m fits

    def test_when_nothing_fits_the_reason_says_so_and_the_best_option_is_still_returned(self):
        choice = choose_layout(GeometryEngine(), 20000.0, 20000.0, 250.0, 12.0, 60.0, [], "auto", "auto")
        self.assertFalse(choice["options"][0]["feasible"])
        self.assertIn("spray limit", choice["reason"])

    def test_ordinary_walls_are_all_feasible(self):
        self.assertTrue(all(o["feasible"] for o in options([WINDOW, DOOR])))


if __name__ == "__main__":
    unittest.main()
