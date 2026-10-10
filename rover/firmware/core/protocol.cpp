#include "protocol.h"

#include <stdio.h>
#include <string.h>

namespace proto {

uint8_t crc8(const char* data, size_t length) {
  uint8_t crc = 0;
  for (size_t i = 0; i < length; i++) {
    crc ^= (uint8_t)data[i];
    for (int bit = 0; bit < 8; bit++) {
      crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
    }
  }
  return crc;
}

bool parseTenths(const char* text, int32_t& out) {
  if (text == NULL || *text == '\0') return false;
  bool negative = false;
  if (*text == '-') {
    negative = true;
    text++;
  }
  if (*text < '0' || *text > '9') return false;
  int32_t whole = 0;
  while (*text >= '0' && *text <= '9') {
    whole = whole * 10 + (*text - '0');
    if (whole > 1000000) return false;
    text++;
  }
  int32_t tenths = 0;
  if (*text == '.') {
    text++;
    int digits = 0;
    int32_t fraction = 0;  // up to 3 digits, scaled to thousandths
    while (*text >= '0' && *text <= '9') {
      if (digits == 3) return false;
      fraction = fraction * 10 + (*text - '0');
      digits++;
      text++;
    }
    if (digits == 0) return false;
    while (digits < 3) {
      fraction *= 10;
      digits++;
    }
    tenths = (fraction + 50) / 100;  // round half up on the magnitude
  }
  if (*text != '\0') return false;
  int32_t value = whole * 10 + tenths;
  out = negative ? -value : value;
  return true;
}

static bool parseUnsigned(const char* text, int32_t& out) {
  if (text == NULL || *text == '\0') return false;
  int32_t value = 0;
  while (*text) {
    if (*text < '0' || *text > '9') return false;
    value = value * 10 + (*text - '0');
    if (value > 1000000) return false;
    text++;
  }
  out = value;
  return true;
}

static int hexValue(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

static const int kMaxTokens = 5;

ParseResult parse(const char* line, Message& out, int32_t& idOut) {
  idOut = 0;
  size_t length = strlen(line);
  if (length < 4 || length > kMaxLine) return PARSE_BADCRC;
  if (line[length - 3] != '*') return PARSE_BADCRC;
  int high = hexValue(line[length - 2]);
  int low = hexValue(line[length - 1]);
  if (high < 0 || low < 0) return PARSE_BADCRC;
  if (crc8(line, length - 3) != (uint8_t)(high * 16 + low)) return PARSE_BADCRC;

  char payload[kMaxLine + 1];
  size_t payloadLength = length - 3;
  memcpy(payload, line, payloadLength);
  payload[payloadLength] = '\0';

  char* tokens[kMaxTokens + 1];
  int count = 0;
  char* cursor = payload;
  while (*cursor) {
    while (*cursor == ' ') cursor++;
    if (*cursor == '\0') break;
    if (count == kMaxTokens + 1) return PARSE_BADARG;
    tokens[count++] = cursor;
    while (*cursor && *cursor != ' ') cursor++;
    if (*cursor) *cursor++ = '\0';
  }
  if (count == 0) return PARSE_UNKNOWN;

  out.id = 0;
  out.a = out.b = out.c = 0;
  out.axis = 0;
  const char* word = tokens[0];

  if (strcmp(word, "MOVE") == 0 || strcmp(word, "TURN") == 0 || strcmp(word, "EDGE") == 0) {
    int32_t id;
    if (count >= 2 && parseUnsigned(tokens[1], id)) idOut = id;
  }

  struct Shape { const char* word; Cmd cmd; int args; };
  static const Shape shapes[] = {
    {"HELLO", CMD_HELLO, 0}, {"WALL", CMD_WALL, 2}, {"SPEED", CMD_SPEED, 1}, {"POSE", CMD_POSE, 3},
    {"ARM", CMD_ARM, 0}, {"MOVE", CMD_MOVE, 3}, {"TURN", CMD_TURN, 2}, {"EDGE", CMD_EDGE, 3},
    {"PAUSE", CMD_PAUSE, 0}, {"RESUME", CMD_RESUME, 0}, {"STOP", CMD_STOP, 0}, {"PING", CMD_PING, 0},
    {"CLEAR", CMD_CLEAR, 0},
  };
  const Shape* shape = NULL;
  for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
    if (strcmp(word, shapes[i].word) == 0) shape = &shapes[i];
  }
  if (shape == NULL) return PARSE_UNKNOWN;
  if (count - 1 != shape->args) return PARSE_BADARG;
  out.cmd = shape->cmd;

  switch (shape->cmd) {
    case CMD_WALL:
      if (!parseTenths(tokens[1], out.a) || !parseTenths(tokens[2], out.b)) return PARSE_BADARG;
      if (out.a <= 0 || out.b <= 0) return PARSE_BADARG;
      break;
    case CMD_SPEED:
      if (!parseTenths(tokens[1], out.a) || out.a <= 0) return PARSE_BADARG;
      break;
    case CMD_POSE:
      if (!parseTenths(tokens[1], out.a) || !parseTenths(tokens[2], out.b) || !parseTenths(tokens[3], out.c)) {
        return PARSE_BADARG;
      }
      break;
    case CMD_MOVE:
      if (!parseUnsigned(tokens[1], out.id) || out.id < 1) return PARSE_BADARG;
      if (!parseTenths(tokens[2], out.a) || out.a <= 0) return PARSE_BADARG;
      if (strcmp(tokens[3], "0") != 0 && strcmp(tokens[3], "1") != 0) return PARSE_BADARG;
      out.b = tokens[3][0] - '0';
      break;
    case CMD_TURN:
      if (!parseUnsigned(tokens[1], out.id) || out.id < 1) return PARSE_BADARG;
      if (!parseTenths(tokens[2], out.a)) return PARSE_BADARG;
      if (out.a < -1800 || out.a > 1800) return PARSE_BADARG;
      break;
    case CMD_EDGE:
      if (!parseUnsigned(tokens[1], out.id) || out.id < 1) return PARSE_BADARG;
      if (strcmp(tokens[2], "x") != 0 && strcmp(tokens[2], "y") != 0) return PARSE_BADARG;
      out.axis = tokens[2][0];
      if (!parseTenths(tokens[3], out.a)) return PARSE_BADARG;
      break;
    default:
      break;
  }
  return PARSE_OK;
}

bool formatLine(char* buf, size_t capacity, const char* payload) {
  size_t length = strlen(payload);
  if (length + 3 > kMaxLine || length + 4 > capacity) return false;
  memcpy(buf, payload, length);
  uint8_t crc = crc8(payload, length);
  snprintf(buf + length, capacity - length, "*%02X", (unsigned)crc);
  return true;
}

size_t formatTenths(char* buf, size_t capacity, int32_t tenths) {
  bool negative = tenths < 0;
  int32_t magnitude = negative ? -tenths : tenths;
  int written = snprintf(buf, capacity, "%s%ld.%ld", negative ? "-" : "", (long)(magnitude / 10), (long)(magnitude % 10));
  return written < 0 ? 0 : (size_t)written;
}

}  // namespace proto
