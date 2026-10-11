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
            for key in ("crossings", "transit_m", "total_m", "spray_m", "turns"):
                self.assertIn(key, option)

    def test_the_list_is_sorted_by_crossings_then_non_painting_travel_then_turns(self):
        result = options([WINDOW, DOOR])
        keys = [(o["crossings"], round(o["transit_m"], 1), o["turns"]) for o in result]
        self.assertEqual(keys, sorted(keys))

    def test_a_tall_pipe_makes_vertical_stripes_cross_it_less_than_horizontal_ones(self):
        by = {(o["start_corner"], o["direction"]): o for o in options([TALL_PIPE])}
        self.assertGreater(by[("bottom_left", "horizontal")]["crossings"], by[("bottom_left", "vertical")]["crossings"])
        self.assertEqual(options([TALL_PIPE])[0]["direction"], "vertical")

    def test_a_clear_wall_has_no_crossings(self):
        for option in options([]):
            self.assertEqual(option["crossings"], 0)
            self.assertAlmostEqual(option["transit_m"] + option["spray_m"], option["total_m"], delta=0.02)

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
        self.assertIn("vertical", choice["reason"])
        self.assertIn("crossing", choice["reason"])

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


if __name__ == "__main__":
    unittest.main()
