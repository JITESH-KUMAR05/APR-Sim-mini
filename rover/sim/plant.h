#pragma once
#include <stdint.h>

#include "../firmware/core/hal.h"

// A differential-drive rover on a flat wall rectangle, standing in for the real hardware.
// Wheels follow the duty command with a first-order lag and lose a fixed fraction of
// their travel to slip (the encoders count wheel rotation, the ground does not move that far).
// The gyro reports the true heading plus a constant bias drift. The rover cannot leave the
// rectangle: at the boundary the edge sensor reports contact.
struct PlantOptions {
  float slip;            // fraction of wheel travel lost, 0.03 means 3%
  uint32_t seed;
  float wallWidthMm;
  float wallHeightMm;
  float startXMm, startYMm, startHeadingDeg;
  float gyroBiasDegPerS;
  uint32_t valveDelayMs;
  float motorScale;      // 1.0 normal; below 1 the motors reach only that fraction of the commanded speed
  PlantOptions()
      : slip(0.0f), seed(1), wallWidthMm(2438.4f), wallHeightMm(2438.4f),
        startXMm(0.0f), startYMm(125.0f), startHeadingDeg(0.0f), gyroBiasDegPerS(0.0f), valveDelayMs(0), motorScale(1.0f) {}
};

class Plant : public Hal {
 public:
  explicit Plant(const PlantOptions& options);

  // Hal
  uint32_t nowMs();
  void setMotorDuty(float left, float right);
  int32_t encoderTicksLeft();
  int32_t encoderTicksRight();
  float gyroHeadingDeg();
  void setPump(bool on);
  void setValve(bool open);
  bool edgeContact();
  bool estopPressed();

  // Simulation
  void step(uint32_t dtMs);
  void setEstop(bool pressed) { estop_ = pressed; }
  void setStall(bool blocked) { stalled_ = blocked; }

  float trueX() const { return x_; }
  float trueY() const { return y_; }
  float trueHeadingDeg() const { return headingDeg_; }
  bool pumpOn() const { return pump_; }
  bool valveOpen() const { return valve_; }
  float dutyLeft() const { return dutyLeft_; }
  float dutyRight() const { return dutyRight_; }
  float paintedMm() const { return paintedMm_; }

 private:
  float nextUniform();

  PlantOptions options_;
  uint32_t rng_;
  float slipLeft_, slipRight_;
  uint32_t nowMs_;
  float dutyLeft_, dutyRight_;
  float wheelLeft_, wheelRight_;      // wheel surface speeds, mm/s
  double ticksLeft_, ticksRight_;
  float x_, y_, headingDeg_, gyroBiasDeg_;
  bool pump_, valve_, estop_, stalled_;
  uint32_t valveOpenedAtMs_;
  bool valveEffective_;
  float paintedMm_;
};
