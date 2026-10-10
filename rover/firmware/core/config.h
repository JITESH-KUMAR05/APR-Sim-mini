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
const float kMinSpeedMmS = 25.0f;          // smallest accepted SPEED: below it the stall detector (20 mm/s) is blind
const float kAccelMmS2 = 150.0f;
const float kMinMoveSpeedMmS = 25.0f;      // creep speed at the end of a MOVE
const float kStopToleranceMm = 1.0f;
const float kEdgeSpeedMmS = 30.0f;
const float kEdgeSearchBaseMm = 150.0f;        // EDGE gives up after this plus kEdgeSearchWallFraction of the wall
const float kEdgeSearchWallFraction = 0.05f;   // along the search axis (wheel slip over a long row grows with the row)

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
const uint32_t kPumpAllowanceMs = 2000;    // a spray MOVE is TOO_LONG if its travel time plus this exceeds kPumpMaxMs
                                           // (covers the speed ramp and the creep at the end; kValveOpenDelayMs is added separately)

// EDGE requests and the correction an edge contact may apply
const float kEdgeFacingCos = 0.7f;               // the rover must face the edge this squarely (cosine of the angle)
const float kEdgeAxisTolMm = 1.0f;               // the requested edge value may differ this much from the wall edge
const float kEdgeMismatchExtraMm = 150.0f;       // a contact more than the search distance plus this from the estimate: FAULT EDGE_MISMATCH
const float kEdgeMaxHeadingCorrectionDeg = 10.0f;  // larger heading correction at contact: FAULT EDGE_MISMATCH

// Spray timing, calibrated in Stage 2 of the integration design
const uint32_t kValveOpenDelayMs = 0;      // wait after opening the valve before moving
const float kValveCloseLeadMm = 0.0f;      // close the valve this far before the end of a spray MOVE

}  // namespace cfg
