# Console UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the console page (`/`) as a calm, professional engineering tool with a graphite dark and a warm light theme, true status only, and every existing feature kept.

**Architecture:** Plain HTML, CSS and JavaScript as today, no build step. A small theme module (`static/theme.js`) owns the theme state and gives the drawing code its colours from CSS tokens. `templates/index.html` and `static/style.css` are rewritten for the new layout; `static/app.js` is rewritten around the same API calls and the same behaviours; `static/viewer3d.js` and `static/rover_sim.js` keep their logic and take their colours from the theme. One new endpoint, `GET /api/capabilities`.

**Tech Stack:** Flask, vanilla JS, Three.js (vendored), IBM Plex from Google Fonts with system fallbacks, Playwright (scratchpad only) for browser checks, Python unittest.

**Spec:** `docs/superpowers/specs/2026-10-25-console-ui-redesign-design.md`

## Global Constraints

- Run Python only as `.venv/Scripts/python.exe` from the repo root (never `uv run`, `python3` or `pip`).
- No new Python or JavaScript dependencies in the app. Playwright lives in the session scratchpad, never in the repo.
- Every colour in `static/style.css` comes from the token blocks (`:root`, `[data-theme="light"]`, `[data-theme="dark"]`); drawing code reads colours through `APRTheme.color(name)`.
- Nothing untrue on screen: no "ROS 2 READY", "SYSTEM READY", "TELEMETRY LIVE", "Vacuum Force", fixed confidence numbers, stepper, canned log lines.
- Server text is inserted with `textContent`, never `innerHTML`.
- Keep these behaviours exactly: review token and `save_corrections` sent on replan; keep-corrections tick box; simulator stale-reply guard (`simGeneration`); Random Wall disabled until its analysis finishes; start corner and stripe direction re-plan after a plan exists.
- Commit trailer, own paragraph: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- The About page is not changed, except that it may share the theme switch later (out of scope here).

## File structure

| File | Change | Responsibility |
|---|---|---|
| `app.py` | modify | add `GET /api/capabilities` |
| `static/theme.js` | create | theme state, switch, `APRTheme.color()`, change event |
| `templates/index.html` | rewrite | new layout, same element IDs where the element survives |
| `static/style.css` | rewrite | tokens for both themes, layout, components, responsive |
| `static/app.js` | rewrite | controller: tabs, panels, analyse, replan, review, layout reason, simulator, Random Wall, exports, status bar, log drawer |
| `static/viewer3d.js` | modify | colours from the theme, `applyTheme()` |
| `static/rover_sim.js` | modify | colours from the theme, renders inside the Simulate tab, untrue readouts removed |
| `tests/test_capabilities.py` | create | endpoint tests |
| `tests/test_rover_ui.py` | modify | markup and script checks for the new structure |
| `tests/test_console_style.py` | create | token-only colours, no untrue strings, theme switch present |

---

### Task 1: `GET /api/capabilities`

**Files:** Modify `app.py`; Create `tests/test_capabilities.py`

- [ ] **Step 1: Write the failing test**

```python
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module


class TestCapabilities(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()

    def get(self):
        res = self.client.get("/api/capabilities")
        return res, json.loads(res.data)

    def test_reports_the_detector_and_the_simulator(self):
        res, data = self.get()
        self.assertEqual(res.status_code, 200)
        self.assertIn(data["detector"], ("yolo11n-onnx", "classical-fallback"))
        self.assertIsInstance(data["simulator"], bool)

    def test_the_detector_follows_what_is_loaded(self):
        with mock.patch.object(app_module.geometry_engine, "detector", None):
            self.assertEqual(self.get()[1]["detector"], "classical-fallback")
        with mock.patch.object(app_module.geometry_engine, "detector", object()):
            self.assertEqual(self.get()[1]["detector"], "yolo11n-onnx")

    def test_no_simulator_on_a_deployment_or_without_gpp(self):
        with mock.patch.dict(os.environ, {"VERCEL": "1"}):
            self.assertFalse(self.get()[1]["simulator"])
        with mock.patch("shutil.which", return_value=None):
            self.assertFalse(self.get()[1]["simulator"])
```

- [ ] **Step 2: Run it, expect 404 failures**

Run: `.venv/Scripts/python.exe -m unittest tests.test_capabilities -v`

- [ ] **Step 3: Implement** (in `app.py`, next to the other routes; `shutil` imported at the top)

```python
@app.route("/api/capabilities", methods=["GET"])
def capabilities():
    """What this running app can do, so the page shows only true status."""
    simulator = (
        not os.environ.get("VERCEL")
        and os.path.isdir(os.path.join(BASE_DIR, "rover", "sim"))
        and shutil.which("g++") is not None
    )
    detector = "yolo11n-onnx" if geometry_engine.detector is not None else "classical-fallback"
    return jsonify({"detector": detector, "simulator": bool(simulator)})
```

- [ ] **Step 4: Run the test and the full suite, expect PASS**
- [ ] **Step 5: Commit** `Add the capabilities endpoint for the console status bar`

---

### Task 2: Theme module

**Files:** Create `static/theme.js`; Create `tests/test_console_style.py` (first part)

`theme.js` (complete):

```javascript
/**
 * Theme state for the console: dark or light, first visit follows the operating system,
 * the choice is remembered. Drawing code asks APRTheme.color('accent') for a token's value
 * and listens for 'apr-theme-change' to redraw.
 */
(function () {
    const KEY = 'apr-theme';
    const root = document.documentElement;

    function stored() {
        try { return localStorage.getItem(KEY); } catch (e) { return null; }
    }

    function remember(value) {
        try { localStorage.setItem(KEY, value); } catch (e) { /* storage blocked: still switch */ }
    }

    function preferred() {
        const saved = stored();
        if (saved === 'dark' || saved === 'light') return saved;
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }

    function apply(value) {
        root.setAttribute('data-theme', value);
        document.dispatchEvent(new CustomEvent('apr-theme-change', { detail: { theme: value } }));
    }

    function color(name) {
        return getComputedStyle(root).getPropertyValue('--' + name).trim();
    }

    apply(preferred());

    window.APRTheme = {
        get: () => root.getAttribute('data-theme'),
        set: (value) => { remember(value); apply(value); },
        toggle: () => window.APRTheme.set(window.APRTheme.get() === 'dark' ? 'light' : 'dark'),
        color,
    };
})();
```

It is loaded in `<head>` before the stylesheet's first paint matters, so the page never flashes the wrong theme.

- [ ] **Step 1: Failing test** (`tests/test_console_style.py`)

```python
import os
import re
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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
```

- [ ] **Step 2: Run, expect failure (no theme.js)**; **Step 3:** create `theme.js`; **Step 4:** the second test passes only after Task 3; **Step 5:** commit with Task 3.

---

### Task 3: New page markup and stylesheet

**Files:** Rewrite `templates/index.html` and `static/style.css`; extend `tests/test_console_style.py`; update `tests/test_rover_ui.py`

Markup requirements (all must hold):

- `<html lang="en" data-theme="dark">`, `<title>APR Console</title>`, Google Fonts link for IBM Plex Sans 400/500/600 and IBM Plex Mono 400/500, then `theme.js`, then `style.css`.
- Top bar `<header class="topbar">`: brand "APR Console"; `<nav role="tablist">` with buttons `id="tabPlan"`, `id="tab3d"`, `id="tabSimulate"`, `id="tabExport"` (`role="tab"`, `aria-selected`, `aria-controls`); About link; `<button id="themeToggle">` with sun/moon icon and an accessible label; on narrow screens buttons `id="openProps"` and `id="openInspector"`.
- Properties `<aside id="propsPanel" class="panel props">` with groups Source, Wall, Spray, Detection, Layout. Kept IDs: `sampleResidential`, `sampleCommercial`, `randomWallBtn`, `dropZone`, `fileInput`, `fileName`, `cornersRow`, `cornersBtn`, `cornersStatus`, `width`, `height`, `sprayWidth`, `sprayValue`, `overlap`, `overlapValue`, `safetyBuffer`, `bufferValue`, `sensitivity`, `sensitivityValue`, `startCorner` (with `value="auto"` first), `stripeDirection` (with `value="auto"` first), `generate` (text "Plan wall"), `keepCorrections` (checked, with the "never uploaded" note).
- Stage `<main id="stage">` with four panels `role="tabpanel"`: `id="panelPlan"` (contains `canvas2d`, a floating toolbar with `layerPhoto`, `layerStripes`, `layerTransit`, `layerKeepouts` toggles and `fitBtn`, and a "Why this layout?" button `id="layoutWhy"` that opens `layoutAdvice`), `id="panel3d"` (contains `viewport3d` with `resetViewBtn`, `isoViewBtn`, `topDownViewBtn`, `wireframeBtn`, `togglePathBtn`, `toggleObstaclesBtn`), `id="panelSimulate"` (two sections: "Preview sweep" with `simCanvas`, `simStartBtn`, `simPauseBtn`, `simResetBtn`, speed buttons `.sim-speed-btn`, readouts `simPosX`, `simPosY`, `simProgressPct`, `simProgressBar`, `simWpProgress`, `simAreaCovered`, `simPaintVolume`, `simElapsedTime`, `simRoverStatus`; and "Physics replay" with `simRoverBtn` and `simReport`), `id="panelExport"` (groups CAD / Rover / Manifest of `[data-export]` buttons with a one-line description each, `exportCaution`, then the waypoint table with `telemetryBody`).
- Inspector `<aside id="inspector" class="panel inspector">`: readouts `statusWallBoundary`, `paintVolValue`, `statusSprayDist`, `cycleTimeValue`, `areaValue`, `activeObstacleCount`, `layoutValue` (passes), `cellsValue` (waypoints), `distanceValue`; detector note `detectorBanner`; review section `reviewPanel` with `reviewCount`, `reviewList`; obstacle list `obstacleList`.
- Status bar `<footer class="statusbar">`: `statusDetector`, `statusSimulator`, `statusWall`, `statusLayout`, `statusLast`, button `logToggle` that opens `logDrawer` containing `log`.
- Dialogs kept: `cornerModal` and its IDs (`cornerTitle`, `cornerHint`, `cornerCanvas`, `cornerUndoBtn`, `cornerClearBtn`, `cornerCancelBtn`, `cornerApplyBtn`).
- Removed: `simModalOverlay`, `openSimBtn`, `closeSimBtn`, `simPlayBtn`, `confidenceValue`, `confidenceBar`, `throughputValue`, `simVacuumForce`, `simPosZ`, `simSolenoidState`, `simTerminal`, `tabTelemetry`, the intro section, the bottom timeline, the status badge.
- Footer line under the status bar is not repeated; the grant reference sits in the About link's title and on the About page.

Stylesheet requirements:

- Token blocks exactly as in the spec table (dark on `:root` and `[data-theme="dark"]`, light on `[data-theme="light"]`), plus `--radius: 5px`, `--font-sans`, `--font-mono`, `--focus` (accent at 60%).
- Grid: `topbar` 44 px, `statusbar` 28 px, columns `280px minmax(0,1fr) 300px`; `<1100px` the inspector becomes an off-canvas drawer; `<720px` both side panels are drawers, the stage is full width, tabs scroll horizontally inside the top bar only.
- Components: `.group` (heading 12 px semibold muted), `.row` (label left, control right), `.field`, `select`, `input[type=range]` (slim track, accent thumb), `.btn`, `.btn-primary` (accent), `.btn-quiet`, `.chip` (samples), `.segmented` (speed buttons), `.readout` (label + mono value), `.toolbar` (floating, panel colour, hairline), `.drawer`, `.dialog`, `.review-item`, `.export-group`, `table.waypoints`.
- No `box-shadow` except dialogs and drawers (one soft shadow token `--shadow`), no gradients, no glow, no uppercase transforms except none.
- `@media (prefers-reduced-motion: reduce)` removes transitions.

- [ ] **Step 1: Failing tests** (append to `tests/test_console_style.py`)

```python
class TestHonestAndTokenised(unittest.TestCase):
    def test_no_untrue_status_remains(self):
        page = read("templates", "index.html") + read("static", "app.js") + read("static", "rover_sim.js")
        for phrase in ("ROS 2 READY", "SYSTEM READY", "TELEMETRY LIVE", "Vacuum Force", "94.7", "94.2%", "CV SEGMENTATION CONFIDENCE"):
            self.assertNotIn(phrase, page, phrase)

    def test_every_colour_in_the_stylesheet_is_a_token(self):
        css = read("static", "style.css")
        token_blocks = re.findall(r"(:root|\[data-theme=\"(?:light|dark)\"\])\s*\{[^}]*\}", css)
        self.assertGreaterEqual(len(token_blocks), 2)
        outside = re.sub(r"(:root|\[data-theme=\"(?:light|dark)\"\])\s*\{[^}]*\}", "", css)
        self.assertEqual(re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", outside), [])

    def test_both_themes_define_every_token(self):
        css = read("static", "style.css")
        dark = re.search(r":root\s*\{([^}]*)\}", css).group(1)
        light = re.search(r"\[data-theme=\"light\"\]\s*\{([^}]*)\}", css).group(1)
        names = lambda block: set(re.findall(r"(--[a-z-]+)\s*:", block))
        colour_tokens = {n for n in names(dark) if n not in ("--radius", "--font-sans", "--font-mono", "--shadow")}
        self.assertTrue(colour_tokens <= names(light), colour_tokens - names(light))
```

`tests/test_rover_ui.py`: keep every behavioural check, change the selectors that moved (`tab2d` becomes `tabPlan`; `simReport` is in the Simulate panel; `layoutAdvice` opens from `layoutWhy`), add `id="themeToggle"` and `id="statusDetector"` checks.

- [ ] **Step 2:** run, expect failures; **Step 3:** write the markup and the stylesheet; **Step 4:** tests pass except the script ones that Task 4 covers; **Step 5:** commit `Rebuild the console layout and stylesheet with two themes`.

---

### Task 4: Controller rewrite (`static/app.js`)

**Files:** Rewrite `static/app.js`; update `tests/test_rover_ui.py`

Behaviour list to implement and check, each against the existing code it replaces:

1. Tabs: Plan, 3D, Simulate, Export with roving `tabindex`, arrow-key switching, `aria-selected`, panel `hidden`; 3D resizes the viewer when shown; Plan redraws.
2. Samples, Random Wall (button stays disabled until `runAnalysis()` settles, file named `random_wall_<seed>.jpg`), upload by click and drop, corners dialog (unchanged logic from today's `app.js`).
3. Plan wall: `runAnalysis()` posts the same form fields as today (`width_mm`, `height_mm`, `spray_width_mm`, `overlap_pct`, `safety_buffer_mm`, `sensitivity`, `start_corner`, `direction`, `file` or `sample_id`, `corners`), stores `review_token`, returns its promise.
4. Replan: same JSON body as today including `review_token`, `save_corrections: keepCorrectionsInput.checked`, `start_corner`, `direction`; start corner and stripe direction changes replan after a plan exists; review actions replan with the select values; stale-reply guard via `clearSimResult()` and `simGeneration`.
5. Inspector: readouts, obstacle list (every obstacle with its type and size), review items first with Keep / Change type / Remove buttons.
6. Layout reason: `showLayout(layout)` fills `layoutAdvice` by `textContent`; "Why this layout?" toggles it.
7. Physics replay: same fetch to `/api/simulate_rover`, `plain()` escaping in log lines, report in `simReport`, true and reported path overlay on the Plan canvas (switches to Plan to show it), 503 updates `statusSimulator`.
8. Status bar: on load `GET /api/capabilities` fills `statusDetector` and `statusSimulator`; after each plan `statusWall`, `statusLayout`, `statusLast` ("Planned 12:04 · 74 waypoints").
9. Activity log: `appendLog(text, kind)` builds the line with `textContent` only, newest last, kept in the drawer; errors also show inline under the Plan button.
10. Theme: `themeToggle` calls `APRTheme.toggle()`; on `apr-theme-change` the Plan canvas, the 3D view and the sweep preview redraw.
11. 2D canvas: same drawing as today with colours from `APRTheme.color()`, layer toggles respected, fit-to-panel sizing with device pixel ratio, wall drawn in proportion.
12. Drawers on narrow screens: `openProps`, `openInspector`, Escape and backdrop click close them.

- [ ] **Step 1:** extend `tests/test_rover_ui.py` with checks for 1, 3, 4, 7, 8, 9, 10 (strings such as `fetch('/api/capabilities'`, `review_token: currentData.review_token,`, `save_corrections: keepCorrectionsInput.checked,`, `if (generation !== simGeneration)`, `return runAnalysis();`, `APRTheme.toggle()`, `addEventListener('apr-theme-change'`, `.textContent =` in `appendLog`, and that `innerHTML` is never assigned from response data: no `innerHTML = ` followed by a template literal containing `body.` or `data.`);
- [ ] **Step 2:** run, expect failures; **Step 3:** write `app.js`; **Step 4:** tests pass; **Step 5:** commit `Rewrite the console controller for the new layout`.

---

### Task 5: Drawing colours from the theme; the sweep preview in its tab

**Files:** Modify `static/viewer3d.js`, `static/rover_sim.js`

- `viewer3d.js`: a `palette()` helper reads `bg`, `canvas`, `line`, `accent`, `keep`, `review`, `text`, `muted` through `APRTheme.color()` and converts with `new THREE.Color(value)`; scene background, fog, grid, wall, path and obstacle materials use it; `applyTheme()` updates materials and background without rebuilding geometry; the cyan fill light becomes neutral.
- `rover_sim.js`: same palette helper for every fill and stroke; draws into `simCanvas` inside `panelSimulate` and sizes to its container; readouts limited to position X/Y, progress, waypoint index, area covered, paint used, elapsed time, state; removed: vacuum force, Z standoff, solenoid badge, canned terminal lines; listens for `apr-theme-change`.
- [ ] Tests: `test_console_style.py` gains `test_drawing_code_reads_colours_from_the_theme` (no `0x[0-9a-f]{6}` in `viewer3d.js` except `0xffffff` lights, no `#[0-9a-f]{6}` in `rover_sim.js`, both contain `APRTheme.color`).
- [ ] Commit `Draw the 3D view and the sweep preview in theme colours`.

---

### Task 6: Browser verification and contrast

**Files:** scratchpad scripts only (`ui_verify.js`, `contrast.py`), screenshots in the scratchpad

- [ ] `contrast.py` parses the two token blocks and prints the contrast ratio of `text`/`bg`, `text`/`panel`, `muted`/`panel`, `muted`/`bg`, accent text on `bg`, and button text on `accent`; every body-text pair at least 4.5, muted at least 4.5 on panel, large or UI text at least 3. Adjust tokens until it passes.
- [ ] `ui_verify.js` (Playwright): for each theme and each width (1440, 1024, 390): load `/`, wait for the first plan, run Random Wall, change stripe direction, open "Why this layout?", switch through all tabs, run Physics replay (when the status bar says available), toggle the theme, open and close the log drawer; collect console errors and page errors (must be zero); assert no horizontal scroll (`document.documentElement.scrollWidth <= innerWidth`); save a screenshot of each tab per theme and width.
- [ ] Look at every screenshot; fix what looks wrong; repeat.
- [ ] Run the full suite: `.venv/Scripts/python.exe -m unittest discover -s tests -p "test_*.py"`.

---

### Task 7: Review and finish

- [ ] Show the lead the screenshots (both themes, three widths) and collect changes.
- [ ] Independent code review of the branch (spec compliance, accessibility, no untrue status, behaviour parity).
- [ ] Fix findings, re-run Task 6, merge locally with `--no-ff`, do not push.
