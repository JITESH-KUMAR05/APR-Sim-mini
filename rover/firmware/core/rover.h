#pragma once
#include <stdint.h>

#include "hal.h"
#include "motion.h"
#include "protocol.h"
#include "safety.h"

// The whole firmware core: protocol handling, the safety state machine and the step executor.
// The caller feeds it received lines and calls tick() every cfg::kTickMs.
class Rover {
 public:
  Rover(Hal& hal, Sink& sink);

  void begin();                       // BOOT, then IDLE
  void handleLine(const char* line);  // one received line, without the newline
  void tick();

  State state() const { return state_; }
  Fault fault() const { return fault_; }

 private:
  void emit(const char* payload);
  void emitState();
  void emitNak(int32_t id, const char* reason);
  void emitAck(int32_t id);
  void emitPose(const char* word, int32_t id, bool withId);
  void setState(State state);
  void enterFault(Fault fault);
  void clearSetup();
  void startStep(const proto::Message& message);

  Hal& hal_;
  Sink& sink_;
  Motion motion_;
  State state_;
  Fault fault_;

  bool helloSeen_, wallSet_, speedSet_, poseSet_;
  int32_t wallWTenths_, wallHTenths_;
  int32_t speedTenths_;
  int32_t currentId_;
  uint32_t lastRxMs_;
  uint32_t lastPoseMs_;
};
