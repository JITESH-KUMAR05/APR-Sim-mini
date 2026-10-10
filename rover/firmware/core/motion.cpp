#include "motion.h"

#include <math.h>

#include "config.h"

namespace {
const float kDt = cfg::kTickMs / 1000.0f;
const float kDegToRad = 3.14159265f / 180.0f;

float clampf(float value, float low, float high) {
  return value < low ? low : (value > high ? high : value);
}
}  // namespace

Motion::Motion(Hal& hal)
    : hal_(hal), kind_(KIND_NONE), fault_(FAULT_NONE), paused_(false),
      x_(0), y_(0), heading_(0), gyroOffset_(0), lastLeftTicks_(0), lastRightTicks_(0), poseKnown_(false),
      wallW_(0), wallH_(0), wallKnown_(false), speed_(100.0f),
      leftMeasured_(0), rightMeasured_(0), leftIntegral_(0), rightIntegral_(0),
      targetMm_(0), traveled_(0), vCmd_(0), holdHeading_(0), targetHeading_(0), rateCmd_(0),
      nominalHeading_(0), headingIntegral_(0),
      edgeAxis_('x'), edgeValue_(0), spray_(false), sprayOpen_(false), settledTicks_(0), waitUntilMs_(0),
      runMs_(0), timeoutMs_(0), pumpOnSinceMs_(0), pumpOn_(false), stallWindowMs_(0), stallMotionMm_(0) {}

void Motion::setWall(float widthMm, float heightMm) {
  wallW_ = widthMm;
  wallH_ = heightMm;
  wallKnown_ = true;
}

void Motion::setSpeed(float mmPerSecond) { speed_ = mmPerSecond; }

void Motion::forgetSetup() {
  wallKnown_ = false;
  poseKnown_ = false;
}

void Motion::setPose(float xMm, float yMm, float headingDeg) {
  x_ = xMm;
  y_ = yMm;
  heading_ = headingDeg;
  nominalHeading_ = headingDeg;
  gyroOffset_ = hal_.gyroHeadingDeg() - headingDeg;
  lastLeftTicks_ = hal_.encoderTicksLeft();
  lastRightTicks_ = hal_.encoderTicksRight();
  leftMeasured_ = rightMeasured_ = 0;
  poseKnown_ = true;
}

float Motion::headingDeg() const {
  float h = fmodf(heading_, 360.0f);
  if (h > 180.0f) h -= 360.0f;
  if (h <= -180.0f) h += 360.0f;
  return h;
}

void Motion::updatePose(float& leftMm, float& rightMm) {
  int32_t leftTicks = hal_.encoderTicksLeft();
  int32_t rightTicks = hal_.encoderTicksRight();
  leftMm = (leftTicks - lastLeftTicks_) * cfg::kMmPerTick;
  rightMm = (rightTicks - lastRightTicks_) * cfg::kMmPerTick;
  lastLeftTicks_ = leftTicks;
  lastRightTicks_ = rightTicks;

  leftMeasured_ += cfg::kSpeedFilter * (leftMm / kDt - leftMeasured_);
  rightMeasured_ += cfg::kSpeedFilter * (rightMm / kDt - rightMeasured_);

  heading_ = hal_.gyroHeadingDeg() - gyroOffset_;
  float mean = (leftMm + rightMm) * 0.5f;
  x_ += mean * cosf(heading_ * kDegToRad);
  y_ += mean * sinf(heading_ * kDegToRad);
}

float Motion::wheelDuty(float target, float measured, float& integral) {
  float error = target - measured;
  integral = clampf(integral + cfg::kWheelKi * error * kDt, -0.4f, 0.4f);
  return clampf(target / cfg::kMaxWheelSpeedMmS + cfg::kWheelKp * error + integral, -1.0f, 1.0f);
}

void Motion::drive(float leftMmS, float rightMmS) {
  float left = wheelDuty(leftMmS, leftMeasured_, leftIntegral_);
  float right = wheelDuty(rightMmS, rightMeasured_, rightIntegral_);
  hal_.setMotorDuty(left, right);
}

void Motion::outputsOff() {
  hal_.setMotorDuty(0, 0);
  hal_.setPump(false);
  hal_.setValve(false);
  pumpOn_ = false;
  leftIntegral_ = rightIntegral_ = 0;
}

void Motion::closeSpray() {
  hal_.setValve(false);
  hal_.setPump(false);
  pumpOn_ = false;
  sprayOpen_ = false;
}

void Motion::abort() {
  kind_ = KIND_NONE;
  paused_ = false;
  spray_ = false;
  sprayOpen_ = false;
  outputsOff();
}

Motion::Event Motion::raise(Fault fault) {
  fault_ = fault;
  abort();
  return EV_FAULT;
}

Motion::Event Motion::finish() {
  kind_ = KIND_NONE;
  paused_ = false;
  spray_ = false;
  sprayOpen_ = false;
  outputsOff();
  return EV_DONE;
}

static void resetStall(uint32_t& windowMs, float& motionMm) {
  windowMs = 0;
  motionMm = 0;
}

void Motion::startMove(float mm, bool spray) {
  kind_ = KIND_MOVE;
  paused_ = false;
  targetMm_ = mm;
  traveled_ = 0;
  vCmd_ = 0;
  holdHeading_ = nominalHeading_;
  headingIntegral_ = 0;
  spray_ = spray;
  sprayOpen_ = false;
  runMs_ = 0;
  uint32_t now = hal_.nowMs();
  waitUntilMs_ = now;
  if (spray) {
    hal_.setPump(true);
    hal_.setValve(true);
    pumpOn_ = true;
    pumpOnSinceMs_ = now;
    sprayOpen_ = true;
    waitUntilMs_ = now + cfg::kValveOpenDelayMs;
  }
  float timeout = 2.0f * (mm / speed_) * 1000.0f + 2000.0f + (float)cfg::kValveOpenDelayMs;
  if (timeout > 4.0e9f) timeout = 4.0e9f;  // keep the float to uint32 conversion in range
  timeoutMs_ = (uint32_t)timeout;
  resetStall(stallWindowMs_, stallMotionMm_);
}

void Motion::startTurn(float degrees) {
  kind_ = KIND_TURN;
  paused_ = false;
  nominalHeading_ += degrees;
  targetHeading_ = nominalHeading_;
  rateCmd_ = 0;
  settledTicks_ = 0;
  runMs_ = 0;
  float expectedMs = fabsf(degrees) / cfg::kTurnRateDegS * 1000.0f;
  timeoutMs_ = (uint32_t)(2.0f * expectedMs) + 2000;
  resetStall(stallWindowMs_, stallMotionMm_);
}

void Motion::startEdge(char axis, float valueMm) {
  kind_ = KIND_EDGE;
  paused_ = false;
  edgeAxis_ = axis;
  edgeValue_ = valueMm;
  traveled_ = 0;
  holdHeading_ = nominalHeading_;
  headingIntegral_ = 0;
  runMs_ = 0;
  timeoutMs_ = (uint32_t)(2.0f * (cfg::kEdgeSearchMm / cfg::kEdgeSpeedMmS) * 1000.0f) + 2000;
  resetStall(stallWindowMs_, stallMotionMm_);
}

bool Motion::edgeRequestValid(char axis, float valueMm) const {
  float angle = nominalHeading_ * kDegToRad;
  float along = axis == 'x' ? cosf(angle) : sinf(angle);
  if (fabsf(along) < cfg::kEdgeFacingCos) return false;
  float edge = along > 0 ? (axis == 'x' ? wallW_ : wallH_) : 0.0f;
  return fabsf(valueMm - edge) <= cfg::kEdgeAxisTolMm;
}

void Motion::pause() {
  if (kind_ == KIND_NONE || paused_) return;
  paused_ = true;
  hal_.setPump(false);
  hal_.setValve(false);
  pumpOn_ = false;
}

void Motion::resume() {
  if (!paused_) return;
  paused_ = false;
  vCmd_ = 0;
  rateCmd_ = 0;
  resetStall(stallWindowMs_, stallMotionMm_);
  if (spray_ && sprayOpen_) {
    uint32_t now = hal_.nowMs();
    hal_.setPump(true);
    hal_.setValve(true);
    pumpOn_ = true;
    pumpOnSinceMs_ = now;
    waitUntilMs_ = now + cfg::kValveOpenDelayMs;
  }
}

float Motion::headingTrim(float errorDeg) {
  headingIntegral_ = clampf(headingIntegral_ + cfg::kHeadingKi * errorDeg * kDt,
                            -cfg::kHeadingMaxIntegralMmS, cfg::kHeadingMaxIntegralMmS);
  return clampf(cfg::kHeadingKp * errorDeg + headingIntegral_, -cfg::kHeadingMaxTrimMmS, cfg::kHeadingMaxTrimMmS);
}

bool Motion::checkStall(float commandMmS, float wheelMotionMm) {
  if (commandMmS < cfg::kStallMinCommandMmS) {
    resetStall(stallWindowMs_, stallMotionMm_);
    return false;
  }
  stallWindowMs_ += cfg::kTickMs;
  stallMotionMm_ += wheelMotionMm;
  if (stallWindowMs_ < cfg::kStallMs) return false;
  bool stalled = stallMotionMm_ < cfg::kStallMinMotionMm;
  resetStall(stallWindowMs_, stallMotionMm_);
  return stalled;
}

Motion::Event Motion::tick() {
  float leftMm, rightMm;
  updatePose(leftMm, rightMm);
  float mean = (leftMm + rightMm) * 0.5f;
  float wheelMotion = (fabsf(leftMm) + fabsf(rightMm)) * 0.5f;
  uint32_t now = hal_.nowMs();

  if (pumpOn_ && now - pumpOnSinceMs_ > cfg::kPumpMaxMs) return raise(FAULT_PUMP_TIMEOUT);

  if (kind_ != KIND_EDGE && wallKnown_ && poseKnown_) {
    float m = cfg::kBoundsMarginMm;
    if (x_ < -m || x_ > wallW_ + m || y_ < -m || y_ > wallH_ + m) return raise(FAULT_OUT_OF_BOUNDS);
  }

  if (kind_ == KIND_NONE || paused_) {
    hal_.setMotorDuty(0, 0);
    leftIntegral_ = rightIntegral_ = 0;
    return EV_NONE;
  }

  runMs_ += cfg::kTickMs;
  if (runMs_ > timeoutMs_) return raise(FAULT_STEP_TIMEOUT);

  if (kind_ == KIND_MOVE) {
    if ((int32_t)(now - waitUntilMs_) < 0) {
      drive(0, 0);
      return EV_NONE;
    }
    traveled_ += mean;
    float remaining = targetMm_ - traveled_;
    if (remaining <= cfg::kStopToleranceMm) {
      closeSpray();
      return finish();
    }
    if (sprayOpen_ && remaining <= cfg::kValveCloseLeadMm) {
      hal_.setValve(false);
      hal_.setPump(false);
      pumpOn_ = false;
      sprayOpen_ = false;
    }
    float profile = fminf(speed_, sqrtf(2.0f * cfg::kAccelMmS2 * remaining) + cfg::kMinMoveSpeedMmS);
    vCmd_ = fminf(vCmd_ + cfg::kAccelMmS2 * kDt, profile);
    float error = holdHeading_ - heading_;
    if (fabsf(error) > cfg::kHeadingFaultDeg) return raise(FAULT_HEADING);
    float trim = headingTrim(error);
    drive(vCmd_ - trim, vCmd_ + trim);
    if (checkStall(vCmd_, wheelMotion)) return raise(FAULT_STALL);
    return EV_NONE;
  }

  if (kind_ == KIND_TURN) {
    float error = targetHeading_ - heading_;
    if (fabsf(error) <= cfg::kTurnToleranceDeg) {
      if (++settledTicks_ >= 3) return finish();
      drive(0, 0);
      return EV_NONE;
    }
    settledTicks_ = 0;
    float profile = fminf(cfg::kTurnRateDegS, sqrtf(2.0f * cfg::kTurnAccelDegS2 * fabsf(error)) + cfg::kTurnMinRateDegS);
    float magnitude = fminf(fabsf(rateCmd_) + cfg::kTurnAccelDegS2 * kDt, profile);
    rateCmd_ = (error > 0 ? 1.0f : -1.0f) * magnitude;
    float wheel = rateCmd_ * kDegToRad * cfg::kWheelbaseMm * 0.5f;
    drive(-wheel, wheel);
    if (checkStall(fabsf(wheel), wheelMotion)) return raise(FAULT_STALL);
    return EV_NONE;
  }

  // KIND_EDGE
  if (hal_.edgeContact()) {
    // A contact far from where the estimate said, or one that turns the rover a lot, means the
    // rover is not where it thinks it is: report it instead of silently rewriting the pose.
    float estimate = edgeAxis_ == 'x' ? x_ : y_;
    if (fabsf(estimate - edgeValue_) > cfg::kEdgeMaxCorrectionMm ||
        fabsf(heading_ - nominalHeading_) > cfg::kEdgeMaxHeadingCorrectionDeg) {
      return raise(FAULT_EDGE_MISMATCH);
    }
    if (edgeAxis_ == 'x') x_ = edgeValue_; else y_ = edgeValue_;
    // Both bumpers touching means the rover is square to the edge: the heading is the nominal one.
    gyroOffset_ = hal_.gyroHeadingDeg() - nominalHeading_;
    heading_ = nominalHeading_;
    return finish();
  }
  traveled_ += mean;
  if (traveled_ > cfg::kEdgeSearchMm) return raise(FAULT_EDGE_NOT_FOUND);
  float error = holdHeading_ - heading_;
  if (fabsf(error) > cfg::kHeadingFaultDeg) return raise(FAULT_HEADING);
  float trim = headingTrim(error);
  drive(cfg::kEdgeSpeedMmS - trim, cfg::kEdgeSpeedMmS + trim);
  if (checkStall(cfg::kEdgeSpeedMmS, wheelMotion)) return raise(FAULT_STALL);
  return EV_NONE;
}
