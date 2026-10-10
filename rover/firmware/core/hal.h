#pragma once
#include <stdint.h>

// Everything the firmware core needs from the outside world. The simulator implements
// Hal (rover/sim/plant.h); plan P2b implements it for the real board. The core never
// calls Arduino functions.
class Hal {
 public:
  virtual uint32_t nowMs() = 0;
  virtual void setMotorDuty(float left, float right) = 0;  // -1..1
  virtual int32_t encoderTicksLeft() = 0;                  // cumulative, signed
  virtual int32_t encoderTicksRight() = 0;
  virtual float gyroHeadingDeg() = 0;                      // integrated yaw, counter-clockwise positive, continuous
  virtual void setPump(bool on) = 0;
  virtual void setValve(bool open) = 0;
  // True only when BOTH front edge sensors touch the edge the rover is driving into. A rover that
  // touches with one corner only is not yet square to the edge, so that is not contact.
  virtual bool edgeContact() = 0;
  virtual bool estopPressed() = 0;
};

// Where the core writes protocol lines (no trailing newline).
class Sink {
 public:
  virtual void writeLine(const char* line) = 0;
};
