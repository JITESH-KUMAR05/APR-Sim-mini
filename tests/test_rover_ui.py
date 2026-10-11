import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module
from geometry_engine import START_CORNERS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


class TestRoverUi(unittest.TestCase):
    def test_page_has_the_start_corner_select_and_the_rover_download(self):
        html = app_module.app.test_client().get("/").get_data(as_text=True)
        self.assertIn('id="startCorner"', html)
        for corner in START_CORNERS:
            self.assertIn(f'value="{corner}"', html)
        self.assertIn('data-export="rover"', html)

    def test_script_sends_the_start_corner_on_analyze_and_replan(self):
        js = read("static", "app.js")
        self.assertIn("const startCornerInput = $('startCorner');", js)
        self.assertIn("formData.append('start_corner', startCornerInput.value);", js)
        self.assertIn("start_corner: startCorner,", js)
        self.assertIn("startCorner = currentData.wall.start_corner", js)
        self.assertIn("startCornerInput.addEventListener('change'", js)

    def test_page_and_script_offer_auto_layout_and_show_the_reason(self):
        html = app_module.app.test_client().get("/").get_data(as_text=True)
        self.assertIn('id="stripeDirection"', html)
        self.assertIn('id="layoutAdvice"', html)
        self.assertEqual(html.count('value="auto"'), 2)  # start corner and stripe direction
        js = read("static", "app.js")
        self.assertIn("formData.append('direction', directionInput.value);", js)
        self.assertIn("direction: directionInput.value,", js)
        self.assertIn("function showLayout(layout)", js)
        self.assertIn("showLayout(data.layout);", js)
        self.assertIn("showLayout(body.layout);", js)
        self.assertIn("directionInput.addEventListener('change'", js)
        self.assertNotIn("layoutAdvice.innerHTML = layout", js)  # reasons are inserted as text, never as markup

    def test_start_label_is_no_longer_hardcoded_to_the_origin(self):
        self.assertNotIn("START (0,0)", read("static", "app.js"))


if __name__ == "__main__":
    unittest.main()
