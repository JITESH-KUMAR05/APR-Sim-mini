#include "plant.h"

#include <math.h>

#include "../firmware/core/config.h"

namespace {
const float kLagSeconds = 0.04f;
const float kDegToRad = 3.14159265f / 180.0f;
}  // namespace

Plant::Plant(const PlantOptions& options)
    : options_(options), rng_(options.seed ? options.seed : 1), slipLeft_(0), slipRight_(0), nowMs_(0),
      dutyLeft_(0), dutyRight_(0), wheelLeft_(0), wheelRight_(0), ticksLeft_(0), ticksRight_(0),
      x_(options.startXMm), y_(options.startYMm), headingDeg_(options.startHeadingDeg), gyroBiasDeg_(0),
      pump_(false), valve_(false), estop_(false), stalled_(false), valveOpenedAtMs_(0),
      valveEffective_(false), paintedMm_(0) {
  slipLeft_ = options.slip * (0.75f + 0.5f * nextUniform());
  slipRight_ = options.slip * (0.75f + 0.5f * nextUniform());
}

float Plant::nextUniform() {
  rng_ = rng_ * 1664525u + 1013904223u;
  return (float)(rng_ >> 8) / 16777216.0f;
}

uint32_t Plant::nowMs() { return nowMs_; }
void Plant::setMotorDuty(float left, float right) { dutyLeft_ = left; dutyRight_ = right; }
int32_t Plant::encoderTicksLeft() { return (int32_t)floor(ticksLeft_); }
int32_t Plant::encoderTicksRight() { return (int32_t)floor(ticksRight_); }
float Plant::gyroHeadingDeg() { return headingDeg_ + gyroBiasDeg_; }
bool Plant::estopPressed() { return estop_; }

void Plant::setPump(bool on) { pump_ = on; }

void Plant::setValve(bool open) {
  if (open && !valve_) valveOpenedAtMs_ = nowMs_;
  valve_ = open;
}

bool Plant::edgeContact() {
  const float eps = 0.01f;
  return x_ <= eps || x_ >= options_.wallWidthMm - eps || y_ <= eps || y_ >= options_.wallHeightMm - eps;
}

void Plant::step(uint32_t dtMs) {
  float dt = dtMs / 1000.0f;
  float targetLeft = dutyLeft_ * cfg::kMaxWheelSpeedMmS * options_.motorScale;
  float targetRight = dutyRight_ * cfg::kMaxWheelSpeedMmS * options_.motorScale;
  if (estop_ || stalled_) targetLeft = targetRight = 0;
  float alpha = dt / (kLagSeconds + dt);
  wheelLeft_ += alpha * (targetLeft - wheelLeft_);
  wheelRight_ += alpha * (targetRight - wheelRight_);
  if (stalled_) wheelLeft_ = wheelRight_ = 0;

  ticksLeft_ += wheelLeft_ * dt / cfg::kMmPerTick;
  ticksRight_ += wheelRight_ * dt / cfg::kMmPerTick;

  float groundLeft = wheelLeft_ * (1.0f - slipLeft_);
  float groundRight = wheelRight_ * (1.0f - slipRight_);
  float speed = (groundLeft + groundRight) * 0.5f;
  float omega = (groundRight - groundLeft) / cfg::kWheelbaseMm;  // rad/s, counter-clockwise positive

  headingDeg_ += omega / kDegToRad * dt;
  float newX = x_ + speed * cosf(headingDeg_ * kDegToRad) * dt;
  float newY = y_ + speed * sinf(headingDeg_ * kDegToRad) * dt;
  bool hitX = false, hitY = false;
  if (newX < 0) { newX = 0; hitX = true; }
  if (newX > options_.wallWidthMm) { newX = options_.wallWidthMm; hitX = true; }
  if (newY < 0) { newY = 0; hitY = true; }
  if (newY > options_.wallHeightMm) { newY = options_.wallHeightMm; hitY = true; }
  // Pushing into an edge with both bumpers squares the rover to it.
  // (only when the rover is really driving into that edge, not sliding along it or turning on the spot)
  bool driving = fabsf(speed) > 15.0f;
  if (driving && hitX && fabsf(cosf(headingDeg_ * kDegToRad)) > 0.7f) headingDeg_ = 180.0f * roundf(headingDeg_ / 180.0f);
  if (driving && hitY && fabsf(sinf(headingDeg_ * kDegToRad)) > 0.7f) headingDeg_ = 90.0f + 180.0f * roundf((headingDeg_ - 90.0f) / 180.0f);
  float moved = sqrtf((newX - x_) * (newX - x_) + (newY - y_) * (newY - y_));
  x_ = newX;
  y_ = newY;
  gyroBiasDeg_ += options_.gyroBiasDegPerS * dt;

  nowMs_ += dtMs;
  valveEffective_ = valve_ && nowMs_ - valveOpenedAtMs_ >= options_.valveDelayMs;
  if (pump_ && valveEffective_) paintedMm_ += moved;
}
