#include "rover.h"

#include <stdio.h>
#include <string.h>

#include "config.h"

namespace {
float fromTenths(int32_t tenths) { return tenths / 10.0f; }

int32_t toTenths(float value) {
  return (int32_t)(value * 10.0f + (value >= 0 ? 0.5f : -0.5f));
}
}  // namespace

Rover::Rover(Hal& hal, Sink& sink)
    : hal_(hal), sink_(sink), motion_(hal), state_(STATE_BOOT), fault_(FAULT_NONE),
      helloSeen_(false), wallSet_(false), speedSet_(false), poseSet_(false),
      wallWTenths_(0), wallHTenths_(0), poseXTenths_(0), poseYTenths_(0), speedTenths_(0), currentId_(0), lastRxMs_(0), lastPoseMs_(0) {}

void Rover::emit(const char* payload) {
  char line[proto::kMaxLine + 8];
  if (proto::formatLine(line, sizeof(line), payload)) sink_.writeLine(line);
}

void Rover::emitState() {
  char payload[24];
  snprintf(payload, sizeof(payload), "STATE %s", stateName(state_));
  emit(payload);
}

void Rover::emitAck(int32_t id) {
  char payload[24];
  snprintf(payload, sizeof(payload), "ACK %ld", (long)id);
  emit(payload);
}

void Rover::emitNak(int32_t id, const char* reason) {
  char payload[40];
  snprintf(payload, sizeof(payload), "NAK %ld %s", (long)id, reason);
  emit(payload);
}

void Rover::emitPose(const char* word, int32_t id, bool withId) {
  char x[16], y[16], h[16];
  proto::formatTenths(x, sizeof(x), toTenths(motion_.x()));
  proto::formatTenths(y, sizeof(y), toTenths(motion_.y()));
  proto::formatTenths(h, sizeof(h), toTenths(motion_.headingDeg()));
  char payload[64];
  if (withId) {
    snprintf(payload, sizeof(payload), "%s %ld %s %s %s", word, (long)id, x, y, h);
  } else {
    snprintf(payload, sizeof(payload), "%s %s %s %s", word, x, y, h);
  }
  emit(payload);
}

void Rover::setState(State state) {
  state_ = state;
  emitState();
}

void Rover::clearSetup() {
  wallSet_ = speedSet_ = poseSet_ = false;  // HELLO is a one-time handshake, not part of the setup
  motion_.forgetSetup();  // a stale wall or pose estimate must not fault again in IDLE
}

// The pose may be outside the wall rectangle by at most cfg::kBoundsMarginMm (inclusive), in tenths.
bool Rover::poseInsideWall(int32_t xTenths, int32_t yTenths, int32_t wTenths, int32_t hTenths) const {
  int32_t marginTenths = (int32_t)(cfg::kBoundsMarginMm * 10.0f);
  return xTenths >= -marginTenths && xTenths <= wTenths + marginTenths &&
         yTenths >= -marginTenths && yTenths <= hTenths + marginTenths;
}

void Rover::enterFault(Fault fault) {
  if (state_ == STATE_FAULT) return;
  motion_.abort();
  fault_ = fault;
  clearSetup();
  char payload[40];
  snprintf(payload, sizeof(payload), "FAULT %s", faultName(fault));
  emit(payload);
  setState(STATE_FAULT);
}

void Rover::begin() {
  setState(STATE_BOOT);
  motion_.outputsOff();
  lastRxMs_ = hal_.nowMs();
  setState(STATE_IDLE);
}

void Rover::startStep(const proto::Message& message) {
  currentId_ = message.id;
  float speedMmS = fromTenths(speedTenths_);
  motion_.setSpeed(speedMmS);
  switch (message.cmd) {
    case proto::CMD_MOVE: motion_.startMove(fromTenths(message.a), message.b == 1); break;
    case proto::CMD_TURN: motion_.startTurn(fromTenths(message.a)); break;
    default: motion_.startEdge(message.axis, fromTenths(message.a)); break;
  }
  emitAck(message.id);
  setState(STATE_RUNNING);
}

void Rover::handleLine(const char* line) {
  proto::Message message;
  int32_t id = 0;
  proto::ParseResult result = proto::parse(line, message, id);
  if (result == proto::PARSE_BADCRC) {
    emitNak(0, "BADCRC");
    return;
  }
  lastRxMs_ = hal_.nowMs();
  if (result == proto::PARSE_UNKNOWN) {
    emitNak(0, "UNKNOWN");
    return;
  }
  if (result == proto::PARSE_BADARG) {
    emitNak(id, "BADARG");
    return;
  }

  switch (message.cmd) {
    case proto::CMD_HELLO: {
      helloSeen_ = true;
      char payload[72];
      snprintf(payload, sizeof(payload), "HELLO fw=%s proto=%d max_speed_mm_s=%ld pump_max_s=%d",
               cfg::kFirmwareVersion, cfg::kProtocolVersion, (long)cfg::kMaxSpeedMmS, cfg::kPumpMaxSeconds);
      emit(payload);
      return;
    }
    case proto::CMD_PING:
      emit("PONG");
      return;
    case proto::CMD_WALL:
    case proto::CMD_SPEED:
    case proto::CMD_POSE:
      if (state_ != STATE_IDLE) {
        emitNak(0, "BADSTATE");
        return;
      }
      if (message.cmd == proto::CMD_WALL) {
        if (poseSet_ && !poseInsideWall(poseXTenths_, poseYTenths_, message.a, message.b)) {
          emitNak(0, "BADARG");
          return;
        }
        wallWTenths_ = message.a;
        wallHTenths_ = message.b;
        motion_.setWall(fromTenths(message.a), fromTenths(message.b));
        wallSet_ = true;
      } else if (message.cmd == proto::CMD_SPEED) {
        if (message.a > cfg::kMaxSpeedMmS * 10 || message.a < (int32_t)(cfg::kMinSpeedMmS * 10)) {
          emitNak(0, "SPEED");
          return;
        }
        speedTenths_ = message.a;
        speedSet_ = true;
      } else {
        if (wallSet_ && !poseInsideWall(message.a, message.b, wallWTenths_, wallHTenths_)) {
          poseSet_ = false;  // a host that ignores this NAK must not be able to ARM on an older pose
          emitNak(0, "BADARG");
          return;
        }
        poseXTenths_ = message.a;
        poseYTenths_ = message.b;
        motion_.setPose(fromTenths(message.a), fromTenths(message.b), fromTenths(message.c));
        poseSet_ = true;
      }
      emitAck(0);
      return;
    case proto::CMD_ARM:
      if (state_ != STATE_IDLE || !helloSeen_ || !wallSet_ || !speedSet_ || !poseSet_ || hal_.estopPressed()) {
        emitNak(0, "BADSTATE");
        return;
      }
      emitAck(0);
      setState(STATE_ARMED);
      return;
    case proto::CMD_MOVE:
    case proto::CMD_TURN:
    case proto::CMD_EDGE:
      if (state_ != STATE_ARMED) {
        emitNak(message.id, "BADSTATE");
        return;
      }
      if (hal_.estopPressed()) {  // pressed since the last tick: do not start anything
        enterFault(FAULT_ESTOP);
        emitNak(message.id, "BADSTATE");
        return;
      }
      if (message.cmd == proto::CMD_MOVE) {
        int32_t longestTenths = wallWTenths_ > wallHTenths_ ? wallWTenths_ : wallHTenths_;
        int32_t marginTenths = (int32_t)(cfg::kMoveMarginMm * 10.0f);
        float seconds = fromTenths(message.a) / fromTenths(speedTenths_);
        // The pump time of a spray MOVE is its travel time, the valve-open delay, and an allowance for the ramp and the creep.
        if (message.a > longestTenths + marginTenths ||
            (message.b == 1 &&
             seconds * 1000.0f + (float)cfg::kPumpAllowanceMs + (float)cfg::kValveOpenDelayMs > (float)cfg::kPumpMaxMs)) {
          emitNak(message.id, "TOO_LONG");
          return;
        }
      } else if (message.cmd == proto::CMD_EDGE && !motion_.edgeRequestValid(message.axis, fromTenths(message.a))) {
        emitNak(message.id, "BADARG");
        return;
      }
      startStep(message);
      return;
    case proto::CMD_PAUSE:
      if (state_ != STATE_RUNNING) {
        emitNak(0, "BADSTATE");
        return;
      }
      motion_.pause();
      emitAck(0);
      setState(STATE_PAUSED);
      return;
    case proto::CMD_RESUME:
      if (state_ != STATE_PAUSED) {
        emitNak(0, "BADSTATE");
        return;
      }
      if (hal_.estopPressed()) {
        enterFault(FAULT_ESTOP);
        emitNak(0, "BADSTATE");
        return;
      }
      motion_.resume();
      emitAck(0);
      setState(STATE_RUNNING);
      return;
    case proto::CMD_STOP:
      if (state_ == STATE_FAULT) {
        emitNak(0, "BADSTATE");
        return;
      }
      motion_.abort();
      clearSetup();
      emitAck(0);
      if (state_ != STATE_IDLE) setState(STATE_IDLE);
      return;
    case proto::CMD_CLEAR:
      if (state_ != STATE_FAULT || hal_.estopPressed()) {
        emitNak(0, "BADSTATE");
        return;
      }
      motion_.clearFault();
      fault_ = FAULT_NONE;
      clearSetup();
      emitAck(0);
      setState(STATE_IDLE);
      return;
  }
}

void Rover::tick() {
  uint32_t now = hal_.nowMs();

  if (hal_.estopPressed() && state_ != STATE_FAULT) enterFault(FAULT_ESTOP);

  bool watched = state_ == STATE_ARMED || state_ == STATE_RUNNING || state_ == STATE_PAUSED;
  if (watched && now - lastRxMs_ > cfg::kWatchdogMs) enterFault(FAULT_WATCHDOG);

  Motion::Event event = motion_.tick();
  if (event == Motion::EV_FAULT) {
    enterFault(motion_.fault());
  } else if (event == Motion::EV_DONE && state_ == STATE_RUNNING) {
    emitPose("DONE", currentId_, true);
    setState(STATE_ARMED);
  }

  if (state_ != STATE_RUNNING) motion_.outputsOff();

  if (state_ == STATE_RUNNING && now - lastPoseMs_ >= cfg::kPoseStreamMs) {
    lastPoseMs_ = now;
    emitPose("POSE", 0, false);
  }
}
