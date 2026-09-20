import os
import re
import tomllib
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _normalise(spec):
    name, version = spec.split("==")
    return name.strip().lower().replace("_", "-"), version.strip()


class TestDependencyPins(unittest.TestCase):
    def test_requirements_and_pyproject_match_and_are_exact(self):
        with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as handle:
            requirements = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
            declared = tomllib.load(handle)["project"]["dependencies"]

        for spec in requirements + declared:
            self.assertRegex(spec, r"^[A-Za-z0-9_.\-]+==[0-9][A-Za-z0-9.\-]*$", f"{spec} is not an exact pin")
        self.assertEqual(sorted(map(_normalise, requirements)), sorted(map(_normalise, declared)))

    def test_no_training_frameworks_in_runtime_dependencies(self):
        with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as handle:
            text = handle.read().lower()
        for banned in ("torch", "ultralytics"):
            self.assertIsNone(re.search(rf"^{banned}\b", text, re.M), f"{banned} must not be a runtime dependency")


if __name__ == "__main__":
    unittest.main()
