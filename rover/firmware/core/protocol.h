#pragma once
#include <stddef.h>
#include <stdint.h>

// Line protocol, see docs/rover-protocol.md. All quantities are fixed point: millimetres and
// degrees in tenths, so the core needs no float printing (which AVR's snprintf lacks).
namespace proto {

const size_t kMaxLine = 80;

enum Cmd {
  CMD_HELLO, CMD_WALL, CMD_SPEED, CMD_POSE, CMD_ARM, CMD_MOVE, CMD_TURN, CMD_EDGE,
  CMD_PAUSE, CMD_RESUME, CMD_STOP, CMD_PING, CMD_CLEAR
};

enum ParseResult { PARSE_OK, PARSE_BADCRC, PARSE_BADARG, PARSE_UNKNOWN };

struct Message {
  Cmd cmd;
  int32_t id;     // step id, 0 for commands without one
  int32_t a;      // first number: mm, deg, width, speed or x, in tenths (spray flag for MOVE is in b)
  int32_t b;
  int32_t c;
  char axis;      // 'x' or 'y' for EDGE
};

uint8_t crc8(const char* data, size_t length);  // polynomial 0x07, initial value 0

// Parse one received line (without newline). On PARSE_OK `out` is filled. `idOut` is the step
// id when it could be read, else 0, for the NAK reply.
ParseResult parse(const char* line, Message& out, int32_t& idOut);

// "-90", "2438.4", "0.25": optional sign, digits, optional fraction of up to 3 digits, rounded
// to tenths. Returns false for anything else.
bool parseTenths(const char* text, int32_t& out);

// Write "payload*CC" into buf. Returns false if it does not fit.
bool formatLine(char* buf, size_t capacity, const char* payload);

// Write a tenths value as decimal text ("-90.0", "2438.4"), always NUL-terminated when capacity > 0.
// Returns the number of characters actually written (at most capacity - 1), 0 on error.
size_t formatTenths(char* buf, size_t capacity, int32_t tenths);

}  // namespace proto
