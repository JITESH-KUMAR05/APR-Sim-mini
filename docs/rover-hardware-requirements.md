# APR rover: hardware requirements for the team

Status: draft for the team to review before anything is bought. Date: 2026-10-11.

This file lists what the team needs to buy and decide so that the software that already exists (planner, mission compiler, firmware core, protocol, simulator) runs on a real rover. It is written for the people who will source, wire and test the parts, not for the people who write the code.

How to read it:

- **Tier 1** is the floor demo prototype: a small rover that paints an 8 x 8 ft taped area on the floor. This is what the software is ready for. Buy this first.
- **Tier 2** is the vertical wall climber. It is **not designed yet**. That section lists what has to be decided and calculated before any money is spent. It is not a parts list.
- Every number marked **(sim)** is a starting value the firmware uses today, found in `rover/firmware/core/config.h`. It was only tested in the simulator and will be tuned on the real rover. If a part cannot meet it, tell the software team and the number changes.
- Brand and part names are examples of the class of part, to make searching easier. Check price, stock and datasheet yourself. Nothing here has been checked against a supplier.

## 1. What the software assumes about the rover

| Item | Value the software uses | Where it comes from |
|---|---|---|
| Drive | two driven wheels (differential drive), one passive caster | firmware motion model |
| Wheel diameter | 80 mm (sim) | `kWheelDiameterMm` |
| Wheel base (centre to centre) | 180 mm (sim) | `kWheelbaseMm` |
| Encoder resolution | 1000 counts per wheel revolution (sim) | `kTicksPerWheelRev` |
| Top speed | 250 mm/s, set per mission; minimum 25 mm/s | protocol `SPEED` |
| Top wheel speed at full power | 300 mm/s (sim) | `kMaxWheelSpeedMmS` |
| Control loop | every 10 ms (100 Hz) | `kTickMs` |
| Heartbeat | the Pi must send a line at least every 250 ms, rover stops after 500 ms | protocol |
| Edge contact | two sensors that both touch the wall edge so the rover squares up against it | `EDGE` command |
| Spray | pump and valve switched on and off by the Arduino, only while moving with spray on | `MOVE ... 1` |
| Wall frame | origin bottom-left, x right, y up. The position of the nozzle centre is what is tracked. | mission file |

The wheel and encoder numbers are free to change. What matters: wheel diameter, wheel base and counts per revolution must be measured on the real rover and put into `config.h`.

## 2. Tier 1: floor demo prototype, parts list

### 2.1 Controller and compute

| Part | Requirement | Qty | Why |
|---|---|---|---|
| Arduino Mega 2560 (or a compatible clone) | 5 V logic, at least 4 external-interrupt pins (2, 3, 18, 19, 20, 21), 3 spare PWM pins, 8 KB RAM | 1 (+1 spare) | runs the safety firmware. The team already owns a controller board: check whether it is a Mega. An Uno has only 2 interrupt pins and 2 KB RAM and is **not** enough. |
| Raspberry Pi 4 (2 GB or more) or Pi 5 | Wi-Fi, USB, runs Python 3.11 or newer | 1 | stores the mission, streams steps to the Arduino, talks to the laptop. A Pi Zero 2 W is enough for the floor demo but cannot run a camera model later. |
| microSD card, 32 GB A1/A2 class | | 2 | one spare |
| USB cable Pi to Arduino, short, with ferrite | data cable, not charge-only | 2 | serial link at 115200 baud |
| 5 V buck converter, 5 A | input 12 V, stable under motor noise | 1 | powers the Pi from the 12 V rail (the Pi must not draw from the Arduino) |
| Laptop (the existing one) | the web app runs locally on it | - | the console works only from the local app, not from the Vercel deployment |

### 2.2 Drive

| Part | Requirement | Qty | Why |
|---|---|---|---|
| Gear motors with quadrature encoders | 12 V, wheel speed about 60 RPM at the wheel for 250 mm/s on an 80 mm wheel, torque at least 3 kg-cm per motor with margin, encoder at least 500 counts per wheel revolution after the gearbox (more is better), Hall-effect type | 2 | the same model for both. Check the encoder counts after the gearbox, not at the motor shaft. |
| Wheels | 80 mm, rubber tyre, hub that fits the motor shaft | 2 | |
| Caster | ball caster or swivel caster, low friction | 1 | |
| Motor driver | dual channel, at least 10 A continuous per channel (more than 2x the motor stall current), PWM up to 20 kHz, logic input 3.3 V or 5 V, over-current protection | 1 (+1 spare) | the L298N-class boards waste power as heat and are not recommended |
| Chassis | rigid plate (3 to 5 mm aluminium, acrylic or plywood), room for tank, pump, battery or tether, boards, bumpers | 1 | keep the nozzle at the centre line between the wheels so the tracked point is the nozzle centre |

### 2.3 Sensing

| Part | Requirement | Qty | Why |
|---|---|---|---|
| Gyro / IMU | z-axis gyro, range at least +-250 deg/s, low bias drift, I2C. BNO055 or ICM-20948 class preferred. A bare MPU6050 works but drifts more and needs calibration at every start. | 1 (+1 spare) | heading. The simulator tests used 0.01 deg/s bias; a worse gyro means more drift between edge contacts. |
| Edge sensors | micro-switch bumpers with lever, mounted left and right at the front, so both touch when the rover pushes square into the edge | 2 (+2 spare) | the `EDGE` command: row end re-zero of x and heading squaring. Measure the distance from each switch to the nozzle centre and give it to the software team. |
| Position reference for y | see section 4 | 1 set | nothing re-references y today |
| E-stop input to the Arduino | one extra contact from the e-stop button | 1 | lets the firmware report an e-stop as a fault |

### 2.4 Spray system

| Part | Requirement | Qty | Why |
|---|---|---|---|
| Diaphragm pump | 12 V, 0.4 to 1 L/min, self-priming, rated for water-based paint | 1 (+1 spare) | the app plans about 0.35 to 0.4 L/min at the nozzle; check against the real nozzle pressure |
| Solenoid valve | 12 V, normally closed, rated for paint (not a drinking water valve with a tiny orifice) | 1 (+1 spare) | closes by default when power is lost |
| Nozzle | flat-fan, 20 to 40 cm fan width at the working distance, with a strainer | 2 | width target from the requirements document |
| Tank | 1 L for the demo (the app estimates 0.73 L for 8 x 8 ft at its film target), with a lid and outlet filter | 1 | 5 L is the V1 requirement |
| Pressure regulator or bypass loop | adjustable | 1 | a diaphragm pump gives low pressure spray, not airless |
| Check valve, tubing, fittings, quick connectors | food-grade silicone or PU, matching the pump port size | set | |
| Flow sensor | in-line, inlet size matching the tubing (YF-S401 class) | 1 | optional but recommended: a `no flow while spraying` fault catches a blocked nozzle or an empty tank |
| Tank level sensor | float switch | 1 | optional: stops a run before it paints with an empty tank |
| Drip tray or sealed chassis cover | | 1 | paint on the electronics ends the demo |

The spray fan width and the row spacing in the planner must match: the existing app uses 250 mm spacing with a configurable overlap. Whoever picks the nozzle must measure the real stripe width at the working distance and tell the software team.

### 2.5 Power and protection

| Part | Requirement | Qty | Why |
|---|---|---|---|
| 12 V DC supply | at least 15 A (180 W), regulated, with a long cable to the rover (tether, 5 m or more) | 1 | leading idea for the demo: external supply. Peak draw about 5 to 6 A (motors plus pump plus valve). |
| Alternative: battery | 3S Li-ion pack, 5 Ah or more, with a BMS | 1 | fallback, if no tether. A full run is about 3 minutes on the floor. |
| Main fuse | 10 A, in-line, near the supply | 1 (+ spares) | |
| E-stop button | red mushroom head, twist to release, normally closed contact, with latching | 1 | physical cut. See the next row. |
| Relay or contactor | at least 20 A DC contact rating, cuts the 12 V feed to motors and pump | 1 | the e-stop drops this relay. It must not rely on any code. |
| Wiring | 18 AWG or thicker for motor and pump power, twisted pairs for encoders | set | keep motor wires away from encoder and IMU wires |
| Common ground and star-point wiring | | | one reference point for logic ground |
| Decoupling capacitors and flyback diodes | for motors, pump and valve | set | the valve and pump coils make voltage spikes |
| Terminal blocks, connectors, heat shrink, cable ties | | set | |
| Indicator LEDs or a small buzzer | arm / run / fault | 3 | operator feedback |

### 2.6 Test environment

| Item | Requirement |
|---|---|
| Test surface | flat floor, at least 3 x 3 m free, hard and smooth, or a plywood sheet |
| Test wall | 8 x 8 ft (2438 x 2438 mm) area marked with masking tape; the wall edge must be a hard edge the rover can touch (a board on edge, a skirting board, a taped-on strip) |
| Measuring | steel tape 5 m, laser distance meter, protractor or phone level, stopwatch |
| Paint for tests | water first, then a thin water-based emulsion. A paper roll on the floor shows the stripes. |
| Camera | phone on a tripod, top view, to record every run |
| Spares | one spare of motor, driver, IMU, pump and valve (these are the parts that break) |

## 3. Tier 1 acceptance tests (what "working" means)

| Stage | Test | Pass condition |
|---|---|---|
| 0, bench, wheels off the ground | `MOVE 1000` and `TURN 90` | within 2% and 2 degrees |
| 0 | e-stop, and the heartbeat removed | all outputs off in under 0.5 s. The firmware watchdog fires about 510 ms after the last line: decide with the team how this is measured, or lower the watchdog. |
| 1, floor, dry | the taped 8 x 8 ft area | row-to-row spacing error within 30 mm, heading within 2 degrees over a row, x within 5 mm at each row end |
| 2, floor, water | spray the same path | continuous stripes with no gaps; pump and valve delay calibrated into the firmware |
| 3, Pi plus laptop | start, pause, stop and e-stop from the laptop console | an autonomous run with live position |

## 4. The one open problem: position along y

The firmware knows x well: every row ends at the wall edge and the rover touches it, so x is re-zeroed. **Nothing re-references y.** The error in y grows with wheel slip times the distance driven: about 65 mm at 3% slip on an 8 ft wall (simulator result). The target in the integration design (end position within 50 mm) is not reachable without help.

The team should pick one:

| Option | Part | Pros | Cons |
|---|---|---|---|
| A. Optical flow sensor facing down | PMW3901 or PAA5100 class module, 1 to 3 cm from the floor | independent of wheel slip, small | needs a textured surface and the right height; useless on a smooth wall |
| B. Distance sensor along y | time-of-flight sensor with 4 m range (VL53L1X class), or a laser rangefinder module, pointing at a fixed end board | direct y reading | a flat target must exist; sunlight and dark surfaces disturb it |
| C. Final y edge contact | a third bumper or the same two bumpers on a y edge | no new part, y re-zeroed once | only at the end of the last row, not during the run |
| D. Software slip estimate | none | free, from edge contacts | still an estimate, tested only in simulation |

Recommended for the demo: C plus D, and A as the upgrade if y still drifts too much. B for the vertical wall (Tier 2).

## 5. Tier 1 compute and software environment (not hardware, but it must be ready)

- Laptop: Windows 11, Python via `uv`, the project's local Flask app. For the image model: an NVIDIA GPU helps (training was done on a laptop RTX 3050); the running app uses the CPU through ONNX.
- Pi: Raspberry Pi OS Lite 64-bit, Python 3.11 or newer, Wi-Fi on the same network as the laptop, SSH enabled, a fixed IP or a hostname.
- Router: any, as long as the laptop and the Pi are on the same network. A phone hotspot works for demos.
- Arduino toolchain: PlatformIO in VS Code for the Mega.

## 6. Tier 1 compute for camera and learning (later, optional)

| Part | Requirement | Why |
|---|---|---|
| Camera | Raspberry Pi Camera Module 3 (wide), or a USB webcam | the rover looking at its own wall: stripe check, obstacles it was not told about |
| Pi 5 (4 GB or 8 GB) with active cooler | | on-rover inference of a small model |
| Optional accelerator | Pi AI Kit class module, or a Jetson Orin Nano | the frozen V1 design names a Jetson. The Arduino firmware and protocol stay the same if the Pi is replaced. |

This is only needed once the floor demo works and run logs exist. The learned decision-making (plan P5) starts after that.

## 7. Tier 2: vertical wall climber (decisions needed, not a parts list)

The requirements document (WBS 1.2) fixes the targets: speed 0.1 to 0.5 m/s, load at least 15 kg, adhesion at least 2x the weight, turning radius 0.5 m or less, tank 5 L or more, battery 4 hours, e-stop within 0.5 s, fall detection, IP54, budget Rs 15 lakh. The prototype in section 2 does not test any of the vertical points. Before buying anything for it, the team has to answer:

1. **Adhesion method.** Vacuum with one large seal, a multi-pad array (the lizard idea), or something else. Suction needs a pump that holds a pressure difference **continuously** while the rover moves; a leak on a rough or painted wall is the main risk.
2. **How much force.** Adhesion "2x the weight" is a force *into* the wall. A rover held by friction slides unless friction carries the weight. Rough first calculation, to be replaced by a measurement: with a 15 kg rover (147 N), safety factor 2, and a friction coefficient of 0.5 between the seal and a painted wall, the force pressing the rover on to the wall must be about 147 x 2 / 0.5 = 590 N. At a pressure difference of 20 kPa that needs a sealed area of about 0.03 m2 (about 300 cm2). Measure the friction coefficient on the real wall before trusting any of these numbers.
3. **Locomotion.** Tracks, wheels with suction, or legs. The requirements document says tracks and a 0.5 m turning radius, which needs a concrete steering mechanism (skid-steer or articulated).
4. **Power.** Vacuum pump, spray pump and drive motors together draw far more than the floor demo's 5 to 6 A. A battery sized for 4 hours at 24 V or 48 V needs the sum of all loads, not each alone. A tether for power and paint is the alternative and also needs a fall-arrest rope.
5. **Fall detection.** A pressure sensor on the vacuum line with a number that triggers the alarm (in kPa), plus an independent mechanical fall-arrest tether and winch. Software never replaces this.
6. **Environment.** IP54 against splash and dust, while the rover itself gets sprayed with wet paint: confirm IP54 is enough around the electronics or use a higher rating.
7. **Position on a vertical wall.** Wheel encoders slip more on a wall. A distance sensor to the floor or ceiling (option B above), and an edge contact at the top and sides, are needed, and probably a camera that sees the wall.
8. **Safety review.** A written review by someone who is not on the team, before the first test off the floor. Tests happen at low height first.

## 8. Suggested order of buying

1. Confirm what the team's controller board is (Mega or not). If not, buy a Mega 2560.
2. Buy the section 2 parts for the bench test first (controller, driver, two motors with encoders, IMU, supply, fuse, e-stop, relay). That is enough for Stage 0.
3. Add chassis, caster, bumpers and wheels for Stage 1.
4. Add the Pi and the spray system for Stage 2 and 3.
5. Decide the y reference (section 4) before Stage 1 ends.
6. Start the Tier 2 decisions in section 7 in parallel, because they have the longest lead time.

## 9. What the team should send back to the software team

- The exact motor, encoder, driver and IMU model chosen, with datasheets.
- Measured wheel diameter, wheel base, encoder counts per wheel revolution.
- The distance from each edge bumper to the nozzle centre line, and from the nozzle to the wheel axis.
- The measured stripe width of the chosen nozzle at the working distance, and the pump flow at the working pressure.
- The measured pump and valve delay (switch on to paint, switch off to no paint).
- Pin map of the wiring to the Arduino (the software team will provide the pin list; use the interrupt pins for the encoders).
- Which y reference option (section 4) was chosen.
