# Rover firmware and protocol design (P2)

Date: 2026-10-09
Status: draft for review

## Context

P1 is merged: the app compiles a wall plan into a rover mission file (`apr-rover-mission/1`: `MOVE` / `TURN` / `EDGE_ALIGN` steps with expected poses and a checksum, file contract in the `rover_mission.py` docstring). Section 1 of `docs/superpowers/specs/2026-10-09-rover-hardware-integration-design.md` defines the control architecture: a Raspberry Pi "brain" and an Arduino "hardware brain" that alone enforces safety. This spec is plan P2 of that design: the Arduino firmware and the serial protocol.

No rover parts have been bought, so P2 is everything that can be built and tested without them. The hardware drivers and bench bring-up are P2b.

## Decisions made

| Question | Decision |
|---|---|
| Board | Arduino Mega 2560 (recommended: pins, 8 KB RAM, spare serial ports). Not bought yet. |
| Toolchain | PlatformIO in VS Code for the board build. Plain `g++` (MinGW 6.3 is installed) for the PC build. |
| Build and test approach | One portable C++ core, compiled for the board and for the PC. The PC build is a simulator, `rover_sim`, that Python tests drive. |
| P2 scope | Protocol, safety state machine, motion control, simulator, host tool, tests. Hardware drivers and bring-up are P2b, after the parts arrive. |
| Who sequences a mission | The host. `run_mission.py` stands in for the Pi (spec section 1, invariant 5). The Pi agent replaces it in P3 with the same protocol. |

## Non-goals

- Real drivers for the motor driver, encoders, IMU, pump, valve and edge sensors, `main.cpp`, gain tuning, and an AVR build. All P2b.
- The Pi agent and the Rover console (P3).
- Wall adhesion and anything vertical (P4). Learned policies (P5).
- Splitting spray passes longer than the pump limit in the mission compiler (the firmware rejects them, see Safety).

## 1. Layout and run flow

```
rover/
  firmware/
    core/            portable C++11, no Arduino code, no heap, no standard containers
      protocol       line framing, CRC-8, parse and format
      safety         state machine, watchdog, e-stop latch, fault codes
      motion         MOVE / TURN / EDGE executor (closed loop)
      hal.h          interfaces: clock, motors, encoders, gyro, pump and valve, edge sensors, e-stop input
      rover          ties it together: handle_line(), tick(now_ms)
      config.h       gains, limits, timings (tuned in P2b)
    src/main.cpp     Mega glue only (pins, Serial, millis). P2b.
    platformio.ini   board environment. P2b.
  sim/
    plant            differential-drive model: wheel slip, encoder quantisation, gyro bias and noise, valve delay
    sim_main.cpp     `rover_sim`: the same core on the PC, protocol on stdin and stdout
  tools/
    run_mission.py   host tool: streams a mission file to a rover (serial port or rover_sim)
docs/rover-protocol.md   the protocol, single source of truth (written in P2)
tests/test_rover_*.py    run with the existing unittest suite, skip themselves if g++ is missing
```

`rover/sim/build/` is added to `.gitignore`.

Run flow, as `run_mission.py` performs it:

1. Load the mission, check its checksum with `rover_mission.verify_checksum`.
2. Send `HELLO` (the rover answers with its protocol version and maximum speed), refuse if the mission speed is above it.
3. Send `WALL` (mission wall size), `SPEED` (mission `speed_mps` in mm/s), `POSE` (mission `start_pose`), then `ARM`.
4. For each mission step send `MOVE id mm spray`, `TURN id deg` or `EDGE id axis value` (the step's `i` is the id) and wait for `DONE` before the next. Send `PING` while a step runs so the watchdog stays fed.
5. The rover streams `POSE x y heading` about 10 times a second. The host compares it with the step's expected pose and reports the difference.

The core never calls Arduino functions. It only uses `hal.h`: in P2 the simulator implements it, in P2b the real drivers do.

## 2. Protocol

One ASCII line per message: `WORDS args*CC\n`, where `CC` is a CRC-8 (polynomial 0x07, initial value 0) in two hex digits (upper case when written, either case accepted) over everything before the `*`. Maximum 80 characters. USB serial at 115200 baud. A line with a bad CRC or length is never executed and gets `NAK 0 BADCRC`. The complete protocol is in `docs/rover-protocol.md`, which is the source of truth.

| Host to rover | Meaning |
|---|---|
| `HELLO` | rover replies `HELLO fw=<version> proto=1 max_speed_mm_s=<n> pump_max_s=<n>` |
| `WALL w h` | wall size in mm, sets the bounds for the safety check |
| `SPEED v` | mm/s, rejected (`NAK`) above the firmware maximum |
| `POSE x y heading` | placed pose in mm and degrees (the mission's `start_pose`) |
| `ARM` | IDLE to ARMED, only if `HELLO`, `WALL`, `SPEED`, `POSE` were received, the e-stop is released and the heartbeat is live |
| `MOVE id mm spray` | drive `mm` along the current heading, spray 0 or 1 |
| `TURN id deg` | turn on the spot, positive is counter-clockwise |
| `EDGE id axis value` | creep into the wall edge until both edge sensors touch it, then set that coordinate to `value` and square the heading to the nominal one |
| `PAUSE` / `RESUME` | stop and resume the running step |
| `STOP` | abort the mission, outputs off, back to IDLE |
| `PING` | heartbeat, rover replies `PONG` |
| `CLEAR` | clear a latched fault (see Safety) |

| Rover to host | Meaning |
|---|---|
| `ACK id` | step accepted |
| `NAK id reason` | rejected, nothing executed (`BADCRC`, `UNKNOWN`, `BADSTATE`, `BADARG`, `TOO_LONG`, `SPEED`) |
| `DONE id x y heading` | step finished, with the pose after it (after the reset for `EDGE`) |
| `POSE x y heading` | about 10 Hz while a step runs |
| `STATE name` | on every state change |
| `FAULT code` | entering FAULT, once |

## 3. Safety

All of this is enforced inside the core, never by the host.

- **States:** BOOT, IDLE, ARMED, RUNNING, PAUSED, and a latched FAULT reachable from every state. There is no DONE state: the host sequences the steps and sends `STOP` at the end.
- **Outputs:** motors, pump and valve are off in every state except RUNNING during a step. The pump and valve are on only during a `MOVE` with spray 1.
- **Watchdog:** nothing valid received for 500 ms while ARMED, RUNNING or PAUSED gives `FAULT WATCHDOG`.
- **E-stop input:** `FAULT ESTOP` on the next 10 ms tick. The hardware relay also cuts motor and pump power, independent of any code (spec section 1, invariant 2).
- **Clearing:** `CLEAR` is accepted only in FAULT and only with the e-stop released. It goes to IDLE and `WALL`, `SPEED`, `POSE` and `ARM` must be sent again. The console (P3) makes a person trigger it.

| Fault | Condition |
|---|---|
| `STALL` | commanded wheel speed of at least 20 mm/s and less than 1 mm of wheel motion in a 1 s window |
| `STEP_TIMEOUT` | a step runs longer than twice its expected time plus 2 s |
| `OUT_OF_BOUNDS` | estimated pose more than 50 mm outside the `WALL` rectangle |
| `HEADING` | heading error above 15 degrees during a `MOVE` or an `EDGE` |
| `EDGE_NOT_FOUND` | no edge contact within 150 mm plus 5% of the wall size along the search axis |
| `EDGE_MISMATCH` | the edge contact is more than 150 mm beyond the search distance from the estimate, or squaring needs more than 10 degrees of heading correction |
| `PUMP_TIMEOUT` | pump on longer than 30 s (the firmware's own limit, reported in `HELLO` as `pump_max_s`; the profile field `pump_max_continuous_s` in `rover_profile.json` must not exceed it and the host tool checks that) |

A command that does not fit the current state is a `NAK BADSTATE`, not a fault. A `MOVE` longer than `max(wall width, wall height) + 100 mm`, or a spray `MOVE` that would run the pump longer than 30 s at the set speed (travel time plus a 2 s allowance and the valve-open delay), is a `NAK TOO_LONG`. The host tool reports it and stops, because splitting passes in the compiler is future work.

## 4. Motion control

All parameters live in `config.h`. P2 gets them working in the simulator, P2b tunes them on the rover.

- **Pose:** distance from the mean of the wheel encoders, heading from the gyro, updated at a fixed 100 Hz control tick.
- **`MOVE`:** trapezoidal speed profile, a speed loop per wheel from its encoder, and heading hold (the gyro heading error trims the left/right speed difference). It stops on the encoder distance. For a spray move the valve opens first and an early-close distance is applied at the end. Both valve delays are constants that stay 0 until calibrated in Stage 2 of the integration design.
- **`TURN`:** gyro closed loop with a ramped turn rate, stopping within half a degree of the target.
- **`EDGE`:** creep at `edge_speed_mm_s` until both edge sensors trigger (at most 150 mm plus 5% of the wall size along the axis), stop, set the coordinate on `axis` to `value`, square the heading to the nominal one and report the corrected pose.
- **Nominal heading:** the rover keeps a nominal heading, the `POSE` heading plus every commanded `TURN`. `MOVE` and `EDGE` hold it with a P+I loop on the gyro, and `TURN` aims at it, so the half-degree stopping tolerance of a turn does not add up over a mission.

## 5. Simulator

`rover_sim` is the core built with `g++` and a simulated plant: differential-drive kinematics, per-wheel slip, encoder quantisation, gyro bias and noise, valve delay, and an edge sensor that triggers when the simulated rover reaches the wall boundary. Options: `--start`, `--wall`, `--slip`, `--seed`, `--gyro-bias`, `--valve-delay` and `--motor-scale` (weak motors, for the timeout faults). The simulator commands `@STALL` and `@ESTOP` block the wheels or press the e-stop at any moment. Squaring against an edge the rover drives into is modelled, because without it gyro drift ruins row spacing (see Findings). The Python host tool and, in P3, the Pi agent and console are tested against the same program, so there is one implementation of the safety logic.

## 6. Tests

All in `tests/test_rover_*.py`, run by the existing `uv run python -m unittest discover -s tests -p "test_*.py"`. They build `rover_sim` with `g++` into `rover/sim/build/` and skip with a clear message if `g++` is not found or the binary cannot run.

1. **Protocol:** golden lines, CRC, malformed and oversized lines, every `NAK` reason.
2. **Safety:** each fault, the watchdog, e-stop, the clear-and-re-arm flow, and a check that motors, pump and valve are off in every state except RUNNING.
3. **Motion accuracy:** `MOVE 1000` within 2%, `TURN 90` within 2 degrees, and a full compiled 8 x 8 ft mission with 3% wheel slip and a gyro bias of 0.01 degrees per second: x re-zeroed within 1 mm at every row end, row-to-row spacing error under 30 mm, and an absolute y error under 4% of the vertical distance driven (about 65 mm measured). These are the Stage 0 and Stage 1 targets of the integration design, in simulation, except the absolute y position (see Findings).
4. **End to end:** `run_mission.py` against `rover_sim` with a mission produced by the app's compiler.

The core uses fixed-width integer types, no heap and no standard containers, and builds with `-std=c++11 -Wall -Wextra -Werror -fno-exceptions -fno-rtti`. That keeps it ready for the Mega's 8 KB of RAM, but only an AVR build in P2b proves the memory fits.

## 7. P2 and P2b

- **P2 (now, no parts):** `docs/rover-protocol.md`, the core, `config.h`, the simulator, `run_mission.py`, the tests, and the `pump_max_continuous_s` profile field.
- **P2b (when the parts arrive):** the `hal.h` implementations for the real motor driver, encoders, gyro, pump, valve, edge sensors and e-stop input, `main.cpp`, the PlatformIO build for the Mega, gain tuning, and the Stage 0 bench criteria measured on the real board: `MOVE 1000` and `TURN 90` within 2% and 2 degrees, and the e-stop and watchdog cutting all outputs in under 0.5 s.

## Findings while building P2

- **Gyro drift ruins row spacing unless the heading is reset.** With a gyro bias of 0.01 degrees per second the true heading drifts about 1 degree over a 200 s mission, and rows then lean in alternating directions: the row-to-row spacing error reached 90 to 180 mm, far over the 30 mm target. Pushing into the wall edge with both edge sensors squares the rover to the edge, so `EDGE` now resets the heading as well as the coordinate. That needs two edge sensors (the hardware sizing table already lists two bumper switches), and the real rover must actually square up against the edge. With it the spacing error is under 20 mm at 3% slip.
- **Nothing re-references y.** The edge contacts re-zero x at every row end, but y is only dead-reckoned from the encoders, so its error is about the wheel slip times the vertical distance driven: about 65 mm at 3% slip on an 8 ft wall, 100 mm at 5%. The integration design's "end position within 50 mm" target cannot be met in y without a y reference. Options for P2b or later: a distance sensor to the floor or ceiling, an `EDGE` on the y axis at the end of the last row, or learning the slip factor online from how far past the planned point each row's edge contact happens (a deterministic estimator, and a good first learning-from-logs candidate).
- **A fixed 150 mm edge search is too short on long rows.** At 3% to 5% slip a 4 m row ends 120 to 200 mm short of the edge. The search distance is therefore 150 mm plus 5% of the wall size along the axis, and a contact further than that distance plus 150 mm from the estimate is `FAULT EDGE_MISMATCH` instead of a silent pose rewrite.
- **The tests must tolerate Windows Application Control.** On this machine a freshly built executable is blocked at random, so `tests/rover_support.py` rebuilds and retries, and the tests skip if no build ever runs. A compile error still fails the tests.
- **The host must survive noise and starve nothing.** `run_mission.py` skips lines with a bad CRC, refuses a heartbeat slower than 250 ms, reports `FAULT ESTOP` ahead of the `NAK BADSTATE` that follows it, and sends a best-effort `STOP` when it gives up.

## Risks

- **Simulator fidelity.** Real wheel slip, gyro drift and valve delay will differ from the model. The simulator proves the logic and the protocol, not the tuned numbers. Tuning is P2b.
- **Memory on the board.** Unknown until an AVR build exists. The core avoids heap and containers, and `config.h` keeps all buffer sizes visible.
- **Old compiler.** The installed MinGW is GCC 6.3, so the core stays at C++11.
- **Executables can be blocked.** This machine has had a Windows Application Control policy block a Python executable. If it blocks `rover_sim`, the tests skip rather than fail, and the protocol cannot be exercised on that machine.

## Assumptions to confirm in review

- 100 Hz control tick, 10 Hz pose stream, 115200 baud.
- CRC-8 polynomial 0x07 and the 80-character line limit.
- Watchdog 500 ms, stall 1 s, bounds margin 50 mm, heading limit 15 degrees, edge search 150 mm plus 5% of the wall, pump limit 30 s.
- All the limits above are starting values in `config.h` and the profile, and are expected to change after the bench tests.
