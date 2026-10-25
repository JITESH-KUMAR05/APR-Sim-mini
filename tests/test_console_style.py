import os
import re
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TOKEN_BLOCK = r"(:root|\[data-theme=\"(?:light|dark)\"\])\s*\{[^}]*\}"
NON_COLOUR_TOKENS = ("--radius", "--font-sans", "--font-mono", "--shadow")


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


class TestTheme(unittest.TestCase):
    def test_theme_module_switches_and_remembers_and_reads_tokens(self):
        js = read("static", "theme.js")
        for needle in ("data-theme", "prefers-color-scheme", "localStorage", "apr-theme-change", "getPropertyValue"):
            self.assertIn(needle, js)

    def test_page_loads_the_theme_before_the_app(self):
        html = read("templates", "index.html")
        self.assertLess(html.index("theme.js"), html.index("app.js"))
        self.assertIn('id="themeToggle"', html)


class TestHonestAndTokenised(unittest.TestCase):
    def test_no_untrue_status_remains(self):
        page = read("templates", "index.html") + read("static", "app.js") + read("static", "rover_sim.js")
        for phrase in ("ROS 2 READY", "SYSTEM READY", "TELEMETRY LIVE", "Vacuum Force", "94.7", "94.2%", "CV SEGMENTATION CONFIDENCE"):
            self.assertNotIn(phrase, page, phrase)

    def test_every_colour_in_the_stylesheet_is_a_token(self):
        css = read("static", "style.css")
        self.assertGreaterEqual(len(re.findall(TOKEN_BLOCK, css)), 2)
        outside = re.sub(TOKEN_BLOCK, "", css)
        self.assertEqual(re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", outside), [])

    def test_both_themes_define_every_token(self):
        css = read("static", "style.css")
        dark = re.search(r":root\s*\{([^}]*)\}", css).group(1)
        light = re.search(r"\[data-theme=\"light\"\]\s*\{([^}]*)\}", css).group(1)

        def names(block):
            return set(re.findall(r"(--[a-z-]+)\s*:", block))

        colour_tokens = {n for n in names(dark) if n not in NON_COLOUR_TOKENS}
        self.assertTrue(colour_tokens <= names(light), colour_tokens - names(light))


if __name__ == "__main__":
    unittest.main()
