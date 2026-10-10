# Rover serial protocol

Version 1 (`proto=1`). This is the single source of truth for what the host (the Raspberry Pi agent, or `rover/tools/run_mission.py` in the simulator) and the Arduino firmware say to each other. The firmware lives in `rover/firmware/core/`, the host side in `rover/tools/rover_link.py`. Where this text and the code disagree, the tests in `tests/test_rover_*.py` decide and this document is the bug.

Nothing here has run on a real board yet. The numbers in section 7 are simulator starting values, not tuned values.

## 1. Framing

One ASCII line per message:

```
WORDS args*CC\n
```

- `CC` is a CRC-8 (polynomial 0x07, initial value 0, no reflection, no final XOR) over every byte before the `*`, written as two hex digits. The rover writes upper case and accepts either case.
- At most 80 characters per line, counting the `*CC` and not counting the line end. A longer line is treated like a bad CRC.
- Words and arguments are separated by spaces.
- Distances are millimetres and angles are degrees, written as decimal text with at most 3 fractional digits, rounded to one decimal (`1500`, `1500.0` and `1500.04` are all accepted; the rover reports one decimal, for example `1500.0`). Internally everything is tenths.
- Ids (`id`) are whole numbers from 1 up. In the mission file they are the step's `i`.
- Transport: USB serial at 115200 baud on the real board (P2b), stdin and stdout of `rover_sim` in the simulator.

Examples: `PING*1F`, `MOVE 3 1500.0 1*98`, `EDGE 4 x 2438.4*3C`, `TURN 5 90.0*ED`. `rover_link.frame()` builds a line and `rover_link.parse_line()` checks one.

## 2. Host to rover

| Line | Meaning | Accepted in |
|---|---|---|
| `HELLO` | handshake; the rover answers `HELLO fw=<version> proto=1 max_speed_mm_s=<n> pump_max_s=<n>` | any state |
| `WALL w h` | wall width and height in mm, both above 0 | IDLE |
| `SPEED v` | mm/s, from 25.0 to 250.0 inclusive | IDLE |
| `POSE x y heading` | where the operator placed the rover (nozzle centre) and the direction it faces | IDLE |
| `ARM` | IDLE to ARMED, only after `HELLO`, `WALL`, `SPEED` and `POSE`, with the e-stop released | IDLE |
| `MOVE id mm spray` | drive `mm` (above 0) along the nominal heading; `spray` is `0` or `1` | ARMED |
| `TURN id deg` | turn on the spot to the nominal heading plus `deg`; positive is counter-clockwise; from -180.0 to 180.0 | ARMED |
| `EDGE id axis value` | creep into the wall edge ahead (`axis` is `x` or `y`), then set that coordinate to `value` and square the heading | ARMED |
| `PAUSE` | stop motion, pump and valve; keep the step | RUNNING |
| `RESUME` | continue the paused step | PAUSED |
| `STOP` | abort the step, outputs off, setup forgotten, to IDLE | any state except FAULT |
| `PING` | heartbeat; the rover answers `PONG` | any state |
| `CLEAR` | clear a latched fault, only with the e-stop released | FAULT |

Rules the rover enforces on arguments:

- `WALL` and `POSE` must agree: a pose more than 50 mm outside the wall rectangle (inclusive margin), in either order of sending, gets `NAK 0 BADARG`. A refused `POSE` also forgets an earlier accepted pose, so `ARM` fails until a good `POSE` arrives.
- `EDGE` must face the edge: the nominal heading must point within about 45 degrees (cosine at least 0.7) along the axis, and `value` must be the wall edge the rover faces (`0` or the wall size on that axis) within 1 mm. Otherwise `NAK id BADARG`.
- `MOVE` is `NAK id TOO_LONG` if `mm` is more than `max(wall width, wall height) + 100`, or if a spray move would run the pump too long: travel time at the set speed, plus a 2 s allowance for the speed ramp and the creep, plus the valve-open delay, above 30 s.
- A command that does not fit the state is `NAK <id> BADSTATE` and not a fault. One exception: `MOVE`, `TURN`, `EDGE` (in ARMED) and `RESUME` (in PAUSED) with the e-stop pressed answer `FAULT ESTOP`, `STATE FAULT` and then the NAK (`NAK <id> BADSTATE` for a step, `NAK 0 BADSTATE` for `RESUME`).
- Number syntax: an optional `-`, at least one digit before the point, then optionally a point and 1 to 3 digits. No `+`, no exponent, and `7.` and `.5` are refused. The whole part is at most 1,000,000, and so are ids (1 to 1,000,000). The `POSE` heading has no range check.
- `STOP` in IDLE answers `ACK 0` with no `STATE` line, and still forgets the setup.

`HELLO` is a one-time handshake and is not forgotten by `STOP`, a fault or `CLEAR`. `WALL`, `SPEED`, `POSE` and `ARM` have to be sent again after any of them.

## 3. Rover to host

| Line | Meaning |
|---|---|
| `ACK id` | accepted. `id` is the step id for `MOVE`/`TURN`/`EDGE`, and `0` for every other command. |
| `NAK id reason` | refused, nothing executed. `id` is the step id when one could be read, otherwise `0`. Reasons: `BADCRC`, `UNKNOWN`, `BADSTATE`, `BADARG`, `TOO_LONG`, `SPEED`. |
| `DONE id x y heading` | the step finished, with the pose after it (after the reset, for `EDGE`). |
| `POSE x y heading` | the current pose estimate, every 100 ms while a step is RUNNING. |
| `STATE name` | on every state change: `BOOT`, `IDLE`, `ARMED`, `RUNNING`, `PAUSED`, `FAULT`. |
| `FAULT code` | entering FAULT, once. Codes in section 5. |
| `PONG` | answer to `PING`. |
| `HELLO ...` | answer to `HELLO`. |

A line with a bad CRC, a bad length, or a missing or malformed `*CC` gets `NAK 0 BADCRC` and does not feed the watchdog. A well-framed line with an unknown word gets `NAK 0 UNKNOWN`. A known word with the wrong number or shape of arguments gets `NAK <id> BADARG` (`SPEED` of 0 or less is `NAK 0 BADARG`; `SPEED` above 250.0 or below 25.0 is `NAK 0 SPEED`). Any line with a correct length and CRC feeds the watchdog, even one the rover answers with a NAK.

The host must treat a line that fails its CRC as noise and skip it (`run_mission.py` does).

Normal order for one step: `ACK id`, `STATE RUNNING`, a stream of `POSE`, `DONE id x y heading`, `STATE ARMED`.

E-stop pressed between ticks, then a command arrives: `FAULT ESTOP`, `STATE FAULT`, `NAK id BADSTATE`, in that order. The host must read the `FAULT` and not report the `NAK` alone (`run_mission.py` reports the fault).

## 4. States

```
BOOT -> IDLE -> ARMED -> RUNNING -> ARMED     (a step finishes: DONE)
                         RUNNING <-> PAUSED   (PAUSE / RESUME)
any state -> FAULT (latched) -> CLEAR -> IDLE
STOP: ARMED, RUNNING or PAUSED -> IDLE (in IDLE it is only acknowledged)
```

- There is no DONE state. The host sequences the steps and sends `STOP` at the end of the mission.
- Motors, pump and valve are off in every state except RUNNING.
- The pump and valve are on only during a `MOVE` with spray 1. They are off while PAUSED.
- **Watchdog:** no line with a correct length and CRC (including one the rover NAKs, and `PING`) for 500 ms while ARMED, RUNNING or PAUSED gives `FAULT WATCHDOG`. The fault comes on the first 10 ms tick after the 500 ms have passed, so about 510 ms after the last line. A host must therefore send a line at least every 250 ms (`run_mission.py` pings every 100 ms and refuses anything slower than 250 ms).
- **E-stop input:** `FAULT ESTOP` on the next tick, in every state. The hardware relay also cuts motor and pump power independent of any code.
- **FAULT** is latched. The only ways out are `CLEAR` (with the e-stop released), which goes to IDLE with the setup forgotten, or a power cycle. `STOP` is refused in FAULT. `PING` still answers.

## 5. Faults

| Code | Condition |
|---|---|
| `WATCHDOG` | no line with a correct length and CRC for 500 ms while ARMED, RUNNING or PAUSED |
| `ESTOP` | the e-stop input is pressed |
| `STALL` | commanded wheel speed of at least 20 mm/s and less than 1 mm of wheel motion in a 1 s window (fixed windows, so it is reported 1 to 2 s after the wheels stop) |
| `STEP_TIMEOUT` | a step runs unpaused for longer than twice its expected time plus 2 s (`MOVE`: travel at the set speed, plus the valve-open delay; `TURN`: at 60 deg/s; `EDGE`: the search distance at the 30 mm/s creep speed) |
| `OUT_OF_BOUNDS` | the pose estimate is more than 50 mm outside the `WALL` rectangle, in any state once `WALL` and `POSE` are known (so also in IDLE, ARMED and PAUSED if the rover is carried; not checked during `EDGE`, which has to touch the edge) |
| `HEADING` | heading error above 15 degrees during a `MOVE` or an `EDGE` |
| `EDGE_NOT_FOUND` | no edge contact within the search distance: 150 mm plus 5% of the wall size along the search axis (wheel slip over a long row grows with the row) |
| `EDGE_MISMATCH` | the contact is more than 150 mm beyond the search distance from where the estimate said, or squaring to the edge needs a heading correction above 10 degrees |
| `PUMP_TIMEOUT` | the pump has been on for more than 30 s without a break |

## 6. Motion

- **Pose:** distance from the mean of the wheel encoders, heading from the gyro, on a 10 ms control tick. The pose is the nozzle centre in the wall frame (origin bottom-left, x right, y up, heading counter-clockwise from +x).
- **Nominal heading:** the rover keeps a nominal heading, the `POSE` heading plus every commanded `TURN`. `MOVE` and `EDGE` hold it with a P+I loop on the gyro, and `TURN` aims at it, so the half-degree stopping tolerance of a turn does not add up over a mission.
- **`MOVE`:** trapezoid speed profile, a speed loop per wheel, stops within 1 mm of the encoder distance. A spray move opens the pump and valve first, waits the valve-open delay (0 until calibrated) and closes both at the end.
- **`EDGE`:** creep at 30 mm/s until the edge sensors touch the wall edge, stop, set the coordinate on `axis` to `value`, square the heading to the nominal one and report the corrected pose. Pushing into the edge with both sensors is what squares the rover, so the real rover needs two edge sensors.

## 7. Limits (starting values in `rover/firmware/core/config.h`)

| Name | Value |
|---|---|
| Protocol version | 1 |
| Maximum / minimum SPEED | 250 / 25 mm/s |
| Watchdog | 500 ms |
| Pose stream | 100 ms |
| Line length | 80 characters |
| Bounds margin | 50 mm |
| Heading fault | 15 degrees |
| Stall | 1 s, 1 mm, command at least 20 mm/s |
| Pump limit | 30 s |
| EDGE search | 150 mm + 5% of the wall |
| EDGE mismatch | search + 150 mm, or 10 degrees |
| MOVE length limit | longest wall side + 100 mm |

Plan P2b replaces the geometry and the gains with the real rover's, and the profile field `pump_max_continuous_s` in `rover_profile.json` (30.0) must not exceed the firmware's `pump_max_s` from `HELLO` (`run_mission.py` checks).

## 8. Known gaps

- Nothing re-references y. The edge contacts re-zero x at every row end, but y is dead-reckoned from the encoders, so its error is about the wheel slip times the vertical distance driven (about 65 mm at 3% slip on an 8 ft wall).
- The offset between the edge sensors and the nozzle centre is not in the protocol yet (P2b).
- The firmware has not been built for the Arduino; no AVR build exists until P2b.
