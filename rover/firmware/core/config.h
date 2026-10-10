#pragma once
#include <stdint.h>

// Tunable constants. The geometry values are initial numbers for the simulator;
// plan P2b replaces them with the real rover's numbers and tunes the gains on the bench.
namespace cfg {

const char kFirmwareVersion[] = "0.1";
const int kProtocolVersion = 1;

// Geometry
const float kWheelDiameterMm = 80.0f;
const float kTicksPerWheelRev = 1000.0f;
const float kWheelbaseMm = 180.0f;
const float kMmPerTick = 3.14159265f * kWheelDiameterMm / kTicksPerWheelRev;
const float kMaxWheelSpeedMmS = 300.0f;  // wheel surface speed at duty 1.0

// Timing
const uint32_t kTickMs = 10;               // control loop period
const uint32_t kPoseStreamMs = 100;        // POSE line period while a step runs
const uint32_t kWatchdogMs = 500;          // no valid line for this long: FAULT WATCHDOG

// Speed limits and profile
const int32_t kMaxSpeedMmS = 250;          // largest accepted SPEED
const float kAccelMmS2 = 150.0f;
const float kMinMoveSpeedMmS = 25.0f;      // creep speed at the end of a MOVE
const float kStopToleranceMm = 1.0f;
const float kEdgeSpeedMmS = 30.0f;
const float kEdgeSearchMm = 150.0f;

// Turning
const float kTurnRateDegS = 60.0f;
const float kTurnAccelDegS2 = 240.0f;
const float kTurnMinRateDegS = 8.0f;
const float kTurnToleranceDeg = 0.5f;

// Control gains
const float kWheelKp = 0.0020f;            // duty per mm/s of speed error
const float kWheelKi = 0.0100f;            // duty per mm/s*s of speed error
const float kSpeedFilter = 0.3f;           // EMA weight of the newest wheel speed sample
const float kHeadingKp = 3.0f;             // mm/s of wheel speed difference per degree of heading error
const float kHeadingKi = 1.5f;             // mm/s per degree-second of accumulated heading error
const float kHeadingMaxIntegralMmS = 20.0f;
const float kHeadingMaxTrimMmS = 40.0f;

// Safety limits
const float kBoundsMarginMm = 50.0f;       // OUT_OF_BOUNDS beyond the wall rectangle by more than this
const float kHeadingFaultDeg = 15.0f;
const uint32_t kStallMs = 1000;
const float kStallMinMotionMm = 1.0f;      // wheel motion needed within kStallMs
const float kStallMinCommandMmS = 20.0f;
const uint32_t kPumpMaxMs = 30000;
const int kPumpMaxSeconds = 30;
const float kMoveMarginMm = 100.0f;        // MOVE longer than max(wall) + this is TOO_LONG

// Spray timing, calibrated in Stage 2 of the integration design
const uint32_t kValveOpenDelayMs = 0;      // wait after opening the valve before moving
const float kValveCloseLeadMm = 0.0f;      // close the valve this far before the end of a spray MOVE

}  // namespace cfg
