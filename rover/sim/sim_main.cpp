// rover_sim: the firmware core on the PC, against a simulated rover.
//
// stdin: protocol lines (they go to the core) and simulator commands that start with '@':
//   @ADVANCE ms   run the simulation for ms milliseconds (multiples of 10)
//   @TRUTH        print the true state of the simulated rover
//   @ESTOP 0|1    release or press the e-stop
//   @STALL 0|1    block or free the wheels
// stdout: every line the core writes, then "@END" after each processed input line.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../firmware/core/config.h"
#include "../firmware/core/rover.h"
#include "plant.h"

class StdoutSink : public Sink {
 public:
  void writeLine(const char* line) { printf("%s\n", line); }
};

static bool parseOptions(int argc, char** argv, PlantOptions& options) {
  for (int i = 1; i < argc; i++) {
    const char* flag = argv[i];
    if (strcmp(flag, "--slip") == 0 && i + 1 < argc) options.slip = (float)atof(argv[++i]);
    else if (strcmp(flag, "--seed") == 0 && i + 1 < argc) options.seed = (uint32_t)atol(argv[++i]);
    else if (strcmp(flag, "--gyro-bias") == 0 && i + 1 < argc) options.gyroBiasDegPerS = (float)atof(argv[++i]);
    else if (strcmp(flag, "--valve-delay") == 0 && i + 1 < argc) options.valveDelayMs = (uint32_t)atol(argv[++i]);
    else if (strcmp(flag, "--motor-scale") == 0 && i + 1 < argc) options.motorScale = (float)atof(argv[++i]);
    else if (strcmp(flag, "--wall") == 0 && i + 2 < argc) {
      options.wallWidthMm = (float)atof(argv[++i]);
      options.wallHeightMm = (float)atof(argv[++i]);
    } else if (strcmp(flag, "--start") == 0 && i + 3 < argc) {
      options.startXMm = (float)atof(argv[++i]);
      options.startYMm = (float)atof(argv[++i]);
      options.startHeadingDeg = (float)atof(argv[++i]);
    } else {
      fprintf(stderr, "unknown or incomplete option: %s\n", flag);
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  PlantOptions options;
  if (!parseOptions(argc, argv, options)) return 2;

  Plant plant(options);
  StdoutSink sink;
  Rover rover(plant, sink);
  rover.begin();
  printf("@END\n");
  fflush(stdout);

  char line[256];
  while (fgets(line, sizeof(line), stdin)) {
    size_t length = strlen(line);
    // A line longer than the buffer must not be split into two commands: discard the rest of it and
    // give the core an empty line (which it answers with NAK 0 BADCRC).
    bool tooLong = length == sizeof(line) - 1 && line[length - 1] != '\n';
    if (tooLong) {
      int c;
      while ((c = getchar()) != EOF && c != '\n') {
      }
      line[0] = '\0';
      length = 0;
    }
    while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r')) line[--length] = '\0';

    if (line[0] == '@') {
      if (strncmp(line, "@ADVANCE ", 9) == 0) {
        long ms = atol(line + 9);
        for (long elapsed = 0; elapsed + (long)cfg::kTickMs <= ms; elapsed += cfg::kTickMs) {
          plant.step(cfg::kTickMs);
          rover.tick();
        }
      } else if (strcmp(line, "@TRUTH") == 0) {
        printf("TRUTH %.3f %.3f %.3f %d %d %.3f %.3f %.3f %lu\n", plant.trueX(), plant.trueY(),
               plant.trueHeadingDeg(), plant.pumpOn() ? 1 : 0, plant.valveOpen() ? 1 : 0, plant.dutyLeft(),
               plant.dutyRight(), plant.paintedMm(), (unsigned long)plant.nowMs());
      } else if (strncmp(line, "@ESTOP ", 7) == 0) {
        plant.setEstop(atoi(line + 7) != 0);
      } else if (strncmp(line, "@STALL ", 7) == 0) {
        plant.setStall(atoi(line + 7) != 0);
      } else if (strcmp(line, "@QUIT") == 0) {
        return 0;
      } else {
        printf("@ERR unknown simulator command\n");
      }
    } else if (length > 0 || tooLong) {
      rover.handleLine(line);
    }
    printf("@END\n");
    fflush(stdout);
  }
  return 0;
}
