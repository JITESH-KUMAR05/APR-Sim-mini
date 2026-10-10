#pragma once

// Rover states and fault codes, see docs/rover-protocol.md.
enum State { STATE_BOOT, STATE_IDLE, STATE_ARMED, STATE_RUNNING, STATE_PAUSED, STATE_FAULT };

enum Fault {
  FAULT_NONE,
  FAULT_WATCHDOG,
  FAULT_ESTOP,
  FAULT_STALL,
  FAULT_STEP_TIMEOUT,
  FAULT_OUT_OF_BOUNDS,
  FAULT_HEADING,
  FAULT_EDGE_NOT_FOUND,
  FAULT_PUMP_TIMEOUT,
  FAULT_EDGE_MISMATCH
};

inline const char* stateName(State state) {
  switch (state) {
    case STATE_BOOT: return "BOOT";
    case STATE_IDLE: return "IDLE";
    case STATE_ARMED: return "ARMED";
    case STATE_RUNNING: return "RUNNING";
    case STATE_PAUSED: return "PAUSED";
    case STATE_FAULT: return "FAULT";
  }
  return "?";
}

inline const char* faultName(Fault fault) {
  switch (fault) {
    case FAULT_NONE: return "NONE";
    case FAULT_WATCHDOG: return "WATCHDOG";
    case FAULT_ESTOP: return "ESTOP";
    case FAULT_STALL: return "STALL";
    case FAULT_STEP_TIMEOUT: return "STEP_TIMEOUT";
    case FAULT_OUT_OF_BOUNDS: return "OUT_OF_BOUNDS";
    case FAULT_HEADING: return "HEADING";
    case FAULT_EDGE_NOT_FOUND: return "EDGE_NOT_FOUND";
    case FAULT_PUMP_TIMEOUT: return "PUMP_TIMEOUT";
    case FAULT_EDGE_MISMATCH: return "EDGE_MISMATCH";
  }
  return "?";
}
