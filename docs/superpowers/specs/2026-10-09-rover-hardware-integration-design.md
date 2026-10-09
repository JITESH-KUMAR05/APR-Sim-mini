# Rover hardware integration design

Date: 2026-10-09
Status: draft for review

## Context

The software side of APR is done: the web app takes a wall photo or DXF, detects obstacles, plans a boustrophedon coverage path and exports DXF, OBJ, mission JSON and waypoints CSV (`docs/superpowers/specs/2026-09-20-real-world-obstacle-detection-design.md`, 13 tasks, merged to `main`). That spec said "No rover hardware exists yet" and deferred ROS 2, SLAM and closed-loop navigation to a later phase.

This spec starts that phase. The team has begun planning the first physical rover. The goal is a rover that runs a mission from the app on its own, with the laptop acting only as a control console.

Inputs read for this design: the team's frozen architecture (`APR_Final_Review_Document.pdf`), requirements (`ARP-REQURIMENTS_updated.docx`), use-case document, the existing exporters (`cad_exporter.py`) and planner (`geometry_engine.py`), and two hand-drawn sheets from the project lead.

## Decisions made

| Question | Decision |
|---|---|
| What is built first | A small demo prototype, not the full V1 (15 kg vacuum climber with Jetson) |
| Where it works | Stage A on the **floor** (nothing can fall), then Stage B on the **90° wall** |
| Autonomy | Mission is loaded onto the rover and runs onboard. The laptop is a console (start, pause, stop, e-stop, monitor). |
| Paint method | Pump + valve + spray nozzle, switched on and off by the controller |
| First milestone | Floor painting plus laptop console (start, stop, live position) |
| Control architecture | **Raspberry Pi (brain) + Arduino (hardware brain)**, built from the Arduino upward |
| Parts in hand | A controller board only (Arduino or Pi acceptable). Motors, driver, pump, valve, nozzle, power and chassis are still to be sourced. |
| Demo wall | 8 ft x 8 ft (2438.4 mm square, 5.95 m2), single coat |
| Paint supply | 1 L onboard tank recommended. External paint line (umbilical) stays a supported alternative to cut weight. |
| Power | External supply over a tether is the leading idea. Onboard battery sizing is deferred. The "65,000" figure on the sheet was discussion only and is ignored. |
| Start position | Chosen by the operator per wall and conditions (see section 2). No trained model decides it. |
| Hand-drawn sheets | Rough ideation. They informed the design but are not requirements. |

## Non-goals

- Vertical wall adhesion, tracks, payload and fall-arrest tether design. That is Plan P4, a separate spec with its own safety review.
- ROS 2, Nav2, SLAM Toolbox and Jetson. They remain the direction of the frozen V1 architecture. This design keeps the Arduino firmware and protocol so a Jetson can replace the Pi later.
- Proving the 50 um dry film thickness. The floor demo uses a low-pressure pump, so it proves path accuracy and valve timing only.
- Live onboard perception. YOLO stays in the laptop planning stage.
- Remote control through the Vercel deployment (see Links).

## 1. Layers and ownership

```
LAPTOP (local copy of this web app)   RASPBERRY PI "brain"              ARDUINO "hardware brain"
- photo/DXF -> obstacles              - stores the mission file          - motor speed control (encoders)
- plan + compile mission        --->  - runs it step by step       --->  - IMU heading
- Rover console: upload, start,       - telemetry to laptop              - pump + valve on/off
  pause, stop, e-stop, live position  - later: camera, YOLO             - edge sensors / bumpers
                              WiFi                              USB serial - e-stop relay + watchdog
```

Invariants:

1. Only the Arduino enforces safety. If it receives nothing from the Pi for 500 ms it stops the motors, closes the valve and stops the pump.
2. A physical e-stop button and relay cut motor and pump power, independent of all code. The laptop e-stop button is a second path, not the safety.
3. The laptop never drives motors. It uploads a mission and sends start, pause, stop and e-stop. This matches the frozen rule that the dashboard sits above the control loop.
4. AI stays in planning. YOLO output feeds the obstacle map and never a motor or valve command.
5. The Arduino firmware and serial protocol are the permanent part. A laptop on USB stands in for the Pi until the Pi exists.

## 2. Mission file and planner changes

Current exports are absolute waypoints (x, y, spray flag) in a wall frame with origin bottom-left, and the planner starts at the top row. A rover cannot execute that directly.

1. **Operator-chosen start.** `plan_coverage_path` gains `start_corner` (any of the four wall corners, default bottom-left, the rover's (0,0)) and `row_order` (bottom-up or top-down, default bottom-up). The operator picks them in the app per wall and conditions, places the rover there, and the planner builds the path from that start. Top-down avoids drips on fresh paint on a vertical wall, which is one reason the choice is per job. No trained model decides the start: that keeps the frozen rule that AI never decides motion. A deterministic "suggest a start" heuristic (for example fewest transit metres) can be added later and would still only suggest.
2. **Rover mission export, `apr-rover-mission/1` (JSON).** A new module `rover_mission.py` compiles waypoints into primitives:
   - `MOVE <mm>` with a spray flag
   - `TURN <deg>` (on the spot, skid-steer or differential)
   - `EDGE_ALIGN` at row ends, to re-zero one coordinate against the wall edge
   - Each step carries the expected (x, y) after it, for comparing planned and real position. The file has a header (wall size, spray width, row pitch, speed, paint estimate) and a checksum.
3. **`rover_profile.json`.** Wheel diameter, ticks per revolution, wheelbase, max speed, pump maximum continuous run time. The compiler and firmware both read it.
4. **Honest parameters.** The exports use the request's real spray width, overlap and speed. Today the manifest hardcodes 250 mm and 0.25 m/s.
5. Existing DXF, OBJ, CSV and JSON exports are unchanged. The 2D map gets a rover-path overlay. `static/rover_sim.js` plays the compiled primitives so the preview matches what the rover receives.

For the 8 x 8 ft wall the current planner gives 12 rows, row pitch 220 mm (250 mm spray width, 12% overlap), 31.45 m of path and about 2:47 cycle time at 0.25 m/s.

## 3. Links, rover states and positioning

### Links

- **Pi to Arduino:** USB serial, line-based ASCII with a CRC per line. The Pi sends one step at a time (`MOVE`, `TURN`, `SPRAY`, `STOP`). The Arduino replies `ACK`, `DONE` or `FAULT` and streams `POSE x y heading` about 10 times a second. The exact grammar is written in `docs/rover-protocol.md` during P2.
- **Laptop to Pi:** WiFi. The Pi exposes `POST /mission` (verifies the checksum, stores the file), `/start`, `/pause`, `/stop`, `/estop`, and a live telemetry stream. The app gets a Rover console page for this.
- **The Rover console works only from the local copy of the app** (`uv run python app.py` on the rover's WiFi). The Vercel deployment cannot reach a rover on a LAN, and browsers block an https page from calling a local http device. Vercel keeps the planning side only.

### Arduino states

BOOT, IDLE (all outputs off), ARMED, RUNNING, PAUSED, DONE.

- ARMED requires the e-stop released and a live heartbeat.
- Any fault or e-stop goes to a latched FAULT with all outputs off. Only a person can clear it.
- The pump runs only in RUNNING with the spray flag set, and has a maximum continuous run time.

### Positioning

Pose is wheel-encoder distance plus gyro heading, starting at (0,0). Drift is the largest technical risk: a 1 degree heading error over a 2.4 m row is about 40 mm sideways, against a 220 mm row pitch.

- Each row ends with `EDGE_ALIGN`: the rover touches the wall edge with a bumper switch (or distance sensor) and resets that coordinate.
- On the floor the painted stripes are the measurement.
- Stage A targets, proposed and to be confirmed after Stage 1: row-to-row error within 30 mm, end position within 50 mm, heading within 2 degrees over a row.

## 4. First-pass hardware sizing (Stage A, floor)

| Part | Choice | Reason |
|---|---|---|
| Drive | 2 geared DC motors with encoders plus a caster | Turns on the spot. 0.25 m/s on an 80 mm wheel is about 60 RPM at the wheel. |
| Motor driver | Dual H-bridge rated at least 2x motor stall current | L298N-class boards waste too much power |
| Heading | Gyro or IMU (MPU6050 class, or BNO055) | Corrects heading drift |
| Edge sensing | 2 micro-switch bumpers, distance sensor optional | For `EDGE_ALIGN` |
| Spray | 12 V diaphragm pump + solenoid valve + flat-fan nozzle | The app plans 0.73 L over about 117 s of spraying, about 0.35 to 0.4 L/min at the nozzle |
| Tank | 1 L | 0.5 L covers about two thirds of the wall at the app's film target |
| Power | Leading idea: external 12 V supply over a tether, with fuse and a 5 V buck converter for the Pi. Fallback: 3S Li-ion pack, 5 Ah. | Peak draw about 5 to 6 A, full run about 3 minutes, so either works on the floor. On the wall a tether adds a cable to manage (P4). |
| Pi | Zero 2 W for minimum weight, or Pi 4/5 if a camera is wanted later | WiFi and Python are enough for Stage A |

These are sizing guidance, not a finished parts list. A diaphragm pump gives low-pressure spray, not true airless, so film thickness validation is out of scope here.

## 5. Plans, stages and testing

### Plan split

| Plan | Content | Hardware needed |
|---|---|---|
| P1 | Planner start corner and row order, `rover_profile.json`, `rover_mission.py`, new export, simulator playback, honest export parameters | None. Can start immediately. |
| P2 | Arduino firmware and serial protocol: safety state machine, motion, pump and valve, `docs/rover-protocol.md` | Bench only |
| P3 | Pi agent and the Rover console in the app | Pi |
| P4 | Vertical wall: adhesion, tracks, fall-arrest tether, payload. Separate spec and safety review. | Yes |

### Hardware stages and exit criteria

- **Stage 0, bench (wheels off the ground):** `MOVE 1000` and `TURN 90` land within 2% and 2 degrees. E-stop and watchdog cut all outputs in under 0.5 s (the requirement in the team's requirements document).
- **Stage 1, floor, dry:** the taped 8 x 8 ft area is covered and the drift targets are met.
- **Stage 2, floor, water:** continuous stripes with no gaps. Spray on/off timing is calibrated for pump and valve delay.
- **Stage 3, Pi plus laptop console:** an autonomous run started from the laptop with live position, pause, stop and e-stop working. This is the first milestone.

### Testing without hardware

- Firmware parser, state machine and motion maths are written independent of the board and unit-tested on the PC.
- A software fake rover lets the Pi agent and the console be tested end to end.
- `docs/rover-protocol.md` is the single source of truth. Both ends carry conformance tests against it.
- `rover_mission.py` is tested with the existing `unittest` setup: for an 8 x 8 ft wall with no obstacles the compiled steps must reproduce the planner's waypoints within 1 mm when replayed.

### Planned repository layout

```
rover_mission.py             compiler, next to geometry_engine.py
rover_profile.json           shared rover constants
rover/firmware/              Arduino C++ (PlatformIO), core logic host-testable
rover/agent/                 Raspberry Pi Python agent
docs/rover-protocol.md       protocol, written in P2
tests/test_rover_mission.py  and further test files per plan
```

## Risks

- **Drift** is the largest risk. Mitigated by `EDGE_ALIGN` and measured in Stage 1.
- **Pump and valve delay** shifts where stripes start and stop. Calibrated in Stage 2 as a spray lead distance in millimetres.
- **Vertical wall** is not designed here. The requirements document already lists the open items: vacuum pump within the Rs 15 lakh budget, battery sizing for pump plus motors together, steering, a numeric fall-detection threshold, and IP54 against wet paint.
- **Paint estimate gap.** The app estimates 0.73 L for the 8 x 8 ft wall, the project lead's sheet says about 0.4 to 0.5 L. Both fit once the film thickness is stated. The tank is sized from the app's figure.

## Resolved notes and deferred items

Answers from the project lead on the hand-drawn sheets (2026-10-09):

- The "65,000 -> KW / VA" and "65V" notes were discussion, not requirements. Power is to be supplied externally over a tether for now. Battery sizing is deferred.
- Wall adhesion is not fixed. The idea under consideration is a bio-inspired design with multiple suction pads, like a lizard or gecko. It belongs to P4 and does not affect P1 to P3.
- The remaining sketches (the box with a cable, the unit pointing to a server, the second circle after "Arduino") were rough ideation and are not requirements.
- Start position and painting direction depend on the wall and its conditions. The operator chooses them per job (section 2). The system does not learn them.
