#pragma once
#include <stdint.h>

#include "hal.h"
#include "safety.h"

// Executes one step at a time (MOVE, TURN or EDGE) with closed loops on the encoders and the
// gyro, and keeps the pose estimate. All distances are millimetres, angles degrees.
class Motion {
 public:
  enum Event { EV_NONE, EV_DONE, EV_FAULT };

  explicit Motion(Hal& hal);

  void setWall(float widthMm, float heightMm);
  void setSpeed(float mmPerSecond);
  void setPose(float xMm, float yMm, float headingDeg);
  void forgetSetup();  // wall and pose are unknown again (after STOP, CLEAR or a fault)

  void startMove(float mm, bool spray);
  void startTurn(float degrees);
  void startEdge(char axis, float valueMm);
  // True if an EDGE request makes sense: the rover faces the wall edge named by axis and value.
  bool edgeRequestValid(char axis, float valueMm) const;

  void pause();
  void resume();
  void abort();       // drop the step and switch everything off
  void outputsOff();  // motors off, pump off, valve closed

  Event tick();       // call every cfg::kTickMs
  Fault fault() const { return fault_; }
  void clearFault() { fault_ = FAULT_NONE; }

  bool busy() const { return kind_ != KIND_NONE; }
  float x() const { return x_; }
  float y() const { return y_; }
  float headingDeg() const;  // normalised to (-180, 180]

 private:
  enum Kind { KIND_NONE, KIND_MOVE, KIND_TURN, KIND_EDGE };

  void updatePose(float& leftMm, float& rightMm);
  float wheelDuty(float target, float measured, float& integral);
  void drive(float leftMmS, float rightMmS);
  Event raise(Fault fault);
  Event finish();
  bool checkStall(float commandMmS, float wheelMotionMm);
  void closeSpray();
  float headingTrim(float errorDeg);

  Hal& hal_;
  Kind kind_;
  Fault fault_;
  bool paused_;

  // pose
  float x_, y_, heading_, gyroOffset_;
  int32_t lastLeftTicks_, lastRightTicks_;
  bool poseKnown_;
  float wallW_, wallH_;
  bool wallKnown_;
  float speed_;

  // wheel loops
  float leftMeasured_, rightMeasured_;
  float leftIntegral_, rightIntegral_;

  // current step
  float targetMm_, traveled_, vCmd_, holdHeading_;
  float targetHeading_, rateCmd_;
  float nominalHeading_, headingIntegral_;  // nominal = pose heading plus every commanded turn
  char edgeAxis_;
  float edgeValue_;
  float edgeSearchMm_;
  bool spray_, sprayOpen_;
  int settledTicks_;
  uint32_t waitUntilMs_;
  uint32_t runMs_, timeoutMs_;
  uint32_t pumpOnSinceMs_;
  bool pumpOn_;
  uint32_t stallWindowMs_;
  float stallMotionMm_;
};
