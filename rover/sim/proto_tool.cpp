// proto_tool: lets the Python tests exercise the firmware's protocol module directly.
// One command per stdin line, one answer line each:
//   CRC <text>        -> CRC <two hex digits>
//   PARSE <line>      -> OK <word> id=.. a=.. b=.. c=.. axis=..   |  BADCRC  |  BADARG id=..  |  UNKNOWN
//   TENTHS <text>     -> OK <integer tenths>  |  BAD
//   FORMAT <payload>  -> the payload with "*CC" appended (or TOOLONG)
//   FMT <integer>     -> a tenths value as text
//   QUIT
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../firmware/core/protocol.h"

static const char* wordOf(proto::Cmd cmd) {
  switch (cmd) {
    case proto::CMD_HELLO: return "HELLO";
    case proto::CMD_WALL: return "WALL";
    case proto::CMD_SPEED: return "SPEED";
    case proto::CMD_POSE: return "POSE";
    case proto::CMD_ARM: return "ARM";
    case proto::CMD_MOVE: return "MOVE";
    case proto::CMD_TURN: return "TURN";
    case proto::CMD_EDGE: return "EDGE";
    case proto::CMD_PAUSE: return "PAUSE";
    case proto::CMD_RESUME: return "RESUME";
    case proto::CMD_STOP: return "STOP";
    case proto::CMD_PING: return "PING";
    case proto::CMD_CLEAR: return "CLEAR";
  }
  return "?";
}

int main() {
  char line[256];
  while (fgets(line, sizeof(line), stdin)) {
    size_t length = strlen(line);
    while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r')) line[--length] = '\0';

    if (strcmp(line, "QUIT") == 0) return 0;
    if (strncmp(line, "CRC ", 4) == 0) {
      printf("CRC %02X\n", (unsigned)proto::crc8(line + 4, strlen(line + 4)));
    } else if (strncmp(line, "PARSE ", 6) == 0) {
      proto::Message message;
      int32_t id = 0;
      switch (proto::parse(line + 6, message, id)) {
        case proto::PARSE_OK:
          printf("OK %s id=%ld a=%ld b=%ld c=%ld axis=%c\n", wordOf(message.cmd), (long)message.id, (long)message.a,
                 (long)message.b, (long)message.c, message.axis ? message.axis : '-');
          break;
        case proto::PARSE_BADCRC: printf("BADCRC\n"); break;
        case proto::PARSE_BADARG: printf("BADARG id=%ld\n", (long)id); break;
        case proto::PARSE_UNKNOWN: printf("UNKNOWN\n"); break;
      }
    } else if (strncmp(line, "TENTHS ", 7) == 0) {
      int32_t value;
      if (proto::parseTenths(line + 7, value)) printf("OK %ld\n", (long)value); else printf("BAD\n");
    } else if (strncmp(line, "FORMAT ", 7) == 0) {
      char out[proto::kMaxLine + 8];
      if (proto::formatLine(out, sizeof(out), line + 7)) printf("%s\n", out); else printf("TOOLONG\n");
    } else if (strncmp(line, "FMT ", 4) == 0) {
      char out[24];
      proto::formatTenths(out, sizeof(out), (int32_t)atol(line + 4));
      printf("%s\n", out);
    } else {
      printf("?\n");
    }
    fflush(stdout);
  }
  return 0;
}
