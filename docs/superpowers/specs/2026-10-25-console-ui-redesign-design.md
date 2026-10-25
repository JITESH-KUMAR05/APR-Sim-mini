# Console UI redesign

Date: 2026-10-25
Status: draft for review. Direction, themes and page structure were chosen by the project lead in the brainstorming session on 2026-10-23 and 2026-10-25.

## Why

The console (`templates/index.html`, `static/style.css`, `static/app.js`) looks like a generated "AI dashboard": neon teal on near-black, monospace capitals everywhere, glowing pills, cards inside cards, and decoration that is not true (a "SYSTEM READY · ROS 2 READY" badge, a fixed "94.7% CV confidence", a six-step pipeline stepper, a fake telemetry terminal, "Vacuum force 320 N", "TELEMETRY LIVE"). The lead wants it to look professional and advanced, like a real engineering tool, and to switch between a dark and a light theme.

## Decisions made

| Question | Decision |
|---|---|
| Who it is for | Operators who plan a wall, and demos to the MSME panel. The About page keeps its own look. |
| Direction | B, "instrument": a calm professional tool in the style of CAD software. |
| Themes | Graphite dark and warm light, one switch, first visit follows the operating system, the choice is remembered. |
| Untrue decoration | Removed. Status shows only facts the app actually knows. |
| Build approach | Rebuild the page structure in the existing plain HTML/CSS/JS. No framework, no build step, the Vercel deploy is unchanged. |

## 1. Page structure

```
┌ top bar: APR Console │ Plan  3D  Simulate  Export │            About · ☀/☾ ┐
├ properties (left) ─┬ canvas (centre, one per tab) ─────────┬ inspector (right) ┤
│ Source             │                                        │ Paint needed      │
│ Wall               │                                        │ Spray path        │
│ Spray              │                                        │ Time at speed     │
│ Detection          │                                        │ Area              │
│ Layout             │                                        │ Obstacles         │
│ [Plan wall]        │                                        │  (review first)   │
│ keep corrections ☐ │                                        │                   │
├ status bar: model loaded · simulator available · 4000 × 2800 mm · auto: horizontal, bottom-left · planned 12:04 ┤
└──────────────────────────────────────────────────────────────────────────────────┘
```

- **Top bar:** product name "APR Console"; tabs Plan, 3D, Simulate, Export; on the right the About link and the theme switch.
- **Properties panel (left, always visible):**
  - Source: Residential and Commercial samples, Random Wall, Upload (click or drop), "Mark corners" (opens the existing corner picker).
  - Wall: width and height (mm).
  - Spray: nozzle width, overlap, keep-out buffer.
  - Detection: sensitivity.
  - Layout: start corner and stripe direction (Auto first).
  - "Plan wall" button, then the keep-corrections tick box in small print.
- **Canvas, one per tab:**
  - **Plan:** the 2D wall map (the main view). A small floating toolbar with zoom-to-fit and layer toggles (photo, stripes, transit, keep-out zones). A "Why this layout?" link opens the scored layouts table.
  - **3D:** the existing Three.js twin, view buttons Front / Iso / Top, layer toggles Path and Keep-outs, Wireframe.
  - **Simulate:** two clearly separate tools. "Preview sweep" is the existing browser animation, labelled as a visual preview, with play, pause, reset and speed. "Physics replay" runs the C++ rover simulator, draws the true and the reported path, and shows its report. On a deployment without the simulator it says so in one line.
  - **Export:** downloads grouped as CAD (DXF, OBJ, MTL, ZIP), Rover (mission JSON, waypoints CSV) and Manifest (mission JSON), with what each file is for; then the waypoint table.
- **Inspector (right):** the numbers that matter (paint needed, spray path, time at speed, paintable area, keep-out count) and the obstacle list. Items that need review come first, each with Keep, Change type and Remove. This replaces the separate review panel.
- **Status bar (bottom), true facts only:** which detector is loaded (trained model or basic edge detection), simulator available or not, wall size, the chosen layout, the last action with its time. Clicking it opens the activity log as a drawer, closed by default.
- **Removed:** the intro banner and its stepper, the "SYSTEM READY · ROS 2 READY" badge, the CV confidence bar, the pipeline timeline, the "Mission Telemetry & Engine Log" card, the grant badge in the header (the grant reference stays in the footer line and on the About page), the full-screen simulation modal (its content moves into the Simulate tab), and every made-up readout in it (vacuum force, "telemetry live", the canned log lines).
- **Narrow screens:** below 1100 px the inspector becomes a drawer opened from the top bar. Below 720 px both side panels are drawers and the canvas fills the screen. No horizontal page scroll at any width.

## 2. Visual system

- **Type:** IBM Plex Sans for the interface, IBM Plex Mono only for numbers with units. Sizes 12, 13, 15 and 20 px. Sentence case; no letter-spaced capitals. Fonts load from Google Fonts with system fallbacks, so the app still works offline.
- **Colour tokens**, defined on `:root` for both themes and used everywhere:

| Token | Dark | Light (warm) |
|---|---|---|
| `--bg` | #121315 | #f4f4f2 |
| `--chrome` (top bar, tool strip, status bar) | #18191c | #ffffff |
| `--panel` | #16171a | #fafaf9 |
| `--canvas` | #0f1012 | #ecebe8 |
| `--field` | #0e0f11 | #ffffff |
| `--line` | #2a2c31 | #dcdbd6 |
| `--text` | #e6e7ea | #1b1c1e |
| `--muted` | #8a8d94 | #626469 |
| `--accent` (paint, primary button, active tab only) | #e8a33d | #a65a08 |
| `--keep` (keep-out) | #e25c50 | #c23b2f |
| `--review` (needs review) | #d9b45a | #a15c00 |
| `--ok` (model loaded dot only) | #6cbf7b | #2f8f46 |

  Values are a starting point; the implementation checks each text pair for at least 4.5:1 contrast and adjusts.
- **Shape:** 4 to 6 px corners, 1 px hairlines, no shadows or glows inside the tool, no nested cards. Panels are separated by lines.
- **Controls:** compact rows, label left and value right. Overlap, buffer and sensitivity keep a slim slider with the value beside it. One style for selects, checkboxes and buttons; one primary button per panel.
- **Drawings (2D canvas, 3D view, Simulate):** read their colours from the theme tokens. Wall: flat surface with a thin outline. Stripes: solid accent. Transit: thin dashed muted line. Keep-out: outline plus a light fill. Unverified: dashed review colour. Labels: small plain text. Switching theme redraws them.
- **Motion:** 120 ms fades for tabs and drawers only, none under `prefers-reduced-motion`.
- **Theme switch:** `data-theme="dark|light"` on `<html>`; first visit follows `prefers-color-scheme`; the choice is kept in `localStorage` (wrapped so a blocked storage never breaks the page).

## 3. Behaviour

Everything the console does today keeps working, through the same API calls:

- Analyse (samples, upload with optional corners, Random Wall), replan after review, auto layout with its reason and table, keep-corrections tick box and review token, rover simulator replay, exports, the 3D twin, the sweep animation.
- Element IDs used by `app.js`, `viewer3d.js`, `rover_sim.js` and `corner_picker.js` are kept where the element still exists; where an element moves, the script changes with it. The stale-reply guards (simulator generation counter, Random Wall disabled until its analysis finishes) stay.
- Planning is explicit: changing a parameter does not re-plan by itself, except start corner and stripe direction after a plan exists (as today). "Plan wall" is the single primary action.
- Errors from the server appear inline next to the control that caused them and in the activity log, as text (never as HTML).
- The status bar's facts come from a new `GET /api/capabilities` that answers `{"detector": "yolo11n-onnx" | "classical-fallback", "simulator": true | false}` (the simulator counts as available when the app is not a deployment, the rover sources are present and `g++` is on the path; the first replay may still find it blocked, and then the status bar updates from the 503 answer), from the analyse response, and from the last action.
- Keyboard: tabs reachable and switchable with arrow keys, every control focusable with a visible focus ring, Escape closes drawers and dialogs.

## 4. Testing

- **Existing tests stay green.** `tests/test_rover_ui.py` checks markup and script strings; those checks are updated to the new structure, and every check that guards a behaviour (review token sent, keep-corrections box, stale-reply guard, reasons inserted as text) is kept.
- **New string tests:** the theme switch exists and writes `data-theme`; no untrue badges remain (no "ROS 2 READY", "TELEMETRY LIVE", "Vacuum Force", fixed confidence); every CSS colour in the console comes from a token (a test scans `style.css` for raw hex colours outside the token blocks).
- **Real browser check with Playwright** (installed in the session scratchpad, not a project dependency): load the page in both themes at 1440, 1024 and 390 px wide; run a sample analysis, Random Wall, a review change, a layout change, the physics replay, a tab tour; fail on any console error; save screenshots for a visual review.
- **Contrast check:** a small script computes the contrast of the text token pairs in both themes.

## Out of scope

- The About page (only its link and theme continuity).
- New features: the console only gets what it already does, presented well.
- A JavaScript framework or a build step.

## Risks

- **Script coupling.** `app.js` (about 1,000 lines) and `rover_sim.js` (about 800) are tied to the current markup. Mitigated by keeping IDs and by the Playwright run after each step.
- **Canvas colours** are hard-coded in three scripts today. Moving them to tokens touches every draw call; done once, with a single helper that reads the current theme.
- **Taste.** "Not AI-looking" is a judgement. The Playwright screenshots in both themes are reviewed with the lead before merging.
