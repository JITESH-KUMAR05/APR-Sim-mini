"""
Guard for the firmware core's constraints (it must build for an AVR board): no heap, no standard
library containers, no Arduino headers and no float printing. Scans rover/firmware/core after
removing comments.
"""

import os
import re
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORE_DIR = os.path.join(ROOT, "rover", "firmware", "core")

BANNED_HEADERS = "vector string map set list iostream sstream algorithm memory functional array deque queue Arduino"

RULES = [
    ("heap allocation with new", re.compile(r"\bnew\b")),
    ("heap allocation with malloc/calloc/realloc/free", re.compile(r"\bmalloc\b|\bcalloc\b|\brealloc\b|\bfree\b")),
    ("standard library (std::)", re.compile(r"std::")),
    # C++ container headers have no extension (<string> is banned, <string.h> is plain C and fine);
    # anything starting with Arduino is banned.
    ("banned header", re.compile(r"#\s*include\s*<\s*(?:Arduino|(?:%s)\s*>)" % "|".join(BANNED_HEADERS.split()[:-1]))),
    ("float strtod/atof", re.compile(r"\bstrtod\b|\batof\b")),
]
FLOAT_FORMAT = re.compile(r"%[-+ #0-9.]*[fgeFGE]")
STRING_LITERAL = re.compile(r'"(?:\\.|[^"\\\n])*"')


def strip_comments(text):
    """Remove // and /* */ comments but keep the line structure, so line numbers stay right."""
    out = []
    i = 0
    length = len(text)
    while i < length:
        two = text[i:i + 2]
        if text[i] == '"':  # a string literal: copy it whole, comment markers inside it are text
            match = STRING_LITERAL.match(text, i)
            end = match.end() if match else i + 1
            out.append(text[i:end])
            i = end
        elif two == "//":
            while i < length and text[i] != "\n":
                i += 1
        elif two == "/*":
            end = text.find("*/", i + 2)
            end = length if end < 0 else end + 2
            out.append("\n" * text.count("\n", i, end))
            i = end
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def violations(text):
    """Return a list of (line number, rule description, line text) for the rules broken in `text`."""
    found = []
    for number, line in enumerate(strip_comments(text).splitlines(), 1):
        for description, pattern in RULES:
            if pattern.search(line):
                found.append((number, description, line.strip()))
        for literal in STRING_LITERAL.findall(line):
            if FLOAT_FORMAT.search(literal):
                found.append((number, "float printing in a format string", line.strip()))
    return found


class TestScanner(unittest.TestCase):
    def test_a_bad_snippet_is_reported(self):
        bad = "\n".join([
            "#include <vector>",
            "#include <Arduino.h>",
            "int* p = new int[4];",
            "void* q = malloc(8); free(q);",
            "std::string s;",
            'snprintf(buf, 8, "%5.2f", x);',
            'snprintf(buf, 8, "%g", x);',
            "double d = atof(text);",
        ])
        self.assertEqual({number for number, _, _ in violations(bad)}, set(range(1, 9)))

    def test_a_clean_snippet_is_not_reported(self):
        clean = "\n".join([
            "#include <stdint.h>",
            "#include <string.h>",
            "// we never call new, malloc or std::vector here, and \"%f\" is only in a comment",
            "/* free(p); */",
            'snprintf(buf, 8, "%s%lu.%lu %02X %d", sign, a, b, c, d);',
            "int renewal = 1;  // a word containing 'new' is not heap use",
        ])
        self.assertEqual(violations(clean), [])

    def test_comments_are_removed_but_lines_kept(self):
        self.assertEqual(strip_comments("a // x\n/* y\nz */ b\n").count("\n"), 3)


class TestFirmwareCore(unittest.TestCase):
    def test_the_core_follows_the_avr_rules(self):
        names = sorted(name for name in os.listdir(CORE_DIR) if name.endswith((".h", ".cpp")))
        self.assertTrue(names, "no firmware core files were found in " + CORE_DIR)
        problems = []
        for name in names:
            with open(os.path.join(CORE_DIR, name), encoding="utf-8") as handle:
                for number, description, line in violations(handle.read()):
                    problems.append(f"{name}:{number}: {description}: {line}")
        self.assertEqual(problems, [], "\n" + "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
