import os
import re
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Photos still to be supplied. Remove a name from this list once its file is added.
PENDING = {"anand-babu.jpg"}


class TestTeamPhotos(unittest.TestCase):
    def test_every_photo_the_about_page_links_exists_with_the_same_lower_case_name(self):
        # Vercel runs on Linux, where Gopi.jpeg and gopi.jpg are different files.
        html = open(os.path.join(ROOT, "templates", "about.html"), encoding="utf-8").read()
        names = re.findall(r'src="/static/photos/([^"]+)"', html)
        self.assertGreaterEqual(len(names), 5)
        for name in names:
            with self.subTest(name=name):
                if name in PENDING:
                    continue
                self.assertEqual(name, name.lower())
                self.assertTrue(os.path.isfile(os.path.join(ROOT, "static", "photos", name)), name)

    def test_the_photos_are_small_enough_for_a_page_of_tiny_avatars(self):
        folder = os.path.join(ROOT, "static", "photos")
        for name in os.listdir(folder):
            self.assertLess(os.path.getsize(os.path.join(folder, name)), 120 * 1024, name)


if __name__ == "__main__":
    unittest.main()
