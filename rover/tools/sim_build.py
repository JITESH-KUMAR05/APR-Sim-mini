"""
Build the C++ rover simulator (and other small firmware test programs) with g++ and make sure the
result can run.

Windows Application Control on some machines blocks a freshly built executable at random
("An Application Control policy has blocked this file"), even for an identical source. Each build
gets a new binary, so build_program() simply tries again, up to a limit. If no build ever runs, or
g++ is missing, it raises SimulatorUnavailable. A compile error is a real failure: it raises
AssertionError and is never retried.
"""

import os
import shutil
import subprocess
from typing import Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CORE_DIR = os.path.join(ROOT, "rover", "firmware", "core")
SIM_DIR = os.path.join(ROOT, "rover", "sim")
BUILD_DIR = os.path.join(SIM_DIR, "build")
FLAGS = ["-std=c++11", "-Wall", "-Wextra", "-Werror", "-fno-exceptions", "-fno-rtti", "-O2"]

_binaries: Dict[str, str] = {}
_blocked: Dict[str, str] = {}


class SimulatorUnavailable(Exception):
    """The simulator cannot be built or launched on this machine (no g++, or the OS blocks it)."""


def _source_files() -> List[str]:
    paths = []
    for folder in (CORE_DIR, SIM_DIR):
        for name in sorted(os.listdir(folder)):
            if name.endswith((".h", ".cpp")):
                paths.append(os.path.join(folder, name))
    return paths


def _runs(binary: str, probe: str) -> bool:
    """
    True if the program launched and exited cleanly on `probe`; False if the operating system
    refused to launch it (an OSError, which is what Application Control produces). A program that
    launches but crashes, exits non-zero or hangs is broken, not blocked: that raises AssertionError.
    """
    try:
        result = subprocess.run([binary], input=probe, capture_output=True, text=True, timeout=20)
    except OSError:
        return False
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        raise AssertionError(f"{binary} launched but did not finish within {error.timeout} s; stderr: {stderr!r}")
    if result.returncode != 0:
        raise AssertionError(f"{binary} launched but exited with return code {result.returncode}; stderr: {result.stderr!r}")
    return True


def build_program(name: str, sources: List[str], probe: str, attempts: int = 12) -> str:
    """Return the path of a runnable program built from `sources`, or raise SimulatorUnavailable."""
    if name in _binaries:
        return _binaries[name]
    if name in _blocked:
        raise SimulatorUnavailable(_blocked[name])
    gpp = shutil.which("g++")
    if gpp is None:
        raise SimulatorUnavailable("g++ was not found; install MinGW or GCC to build the rover simulator")

    os.makedirs(BUILD_DIR, exist_ok=True)
    binary = os.path.join(BUILD_DIR, name + (".exe" if os.name == "nt" else ""))
    newest = max(os.path.getmtime(path) for path in _source_files())
    if os.path.exists(binary) and os.path.getmtime(binary) > newest and _runs(binary, probe):
        _binaries[name] = binary
        return binary

    for _ in range(attempts):
        if os.path.exists(binary):
            try:
                os.remove(binary)
            except OSError:
                pass  # a locked old executable: g++ will say so if it cannot overwrite it
        result = subprocess.run([gpp] + FLAGS + sources + ["-o", binary], capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(f"the rover firmware does not compile:\n{result.stderr}")
        if _runs(binary, probe):
            _binaries[name] = binary
            return binary
    message = f"{name} was blocked by the operating system after {attempts} builds"
    _blocked[name] = message  # later calls fail at once instead of paying for every build again
    raise SimulatorUnavailable(message)


def simulator_binary() -> str:
    core = [os.path.join(CORE_DIR, f) for f in ("protocol.cpp", "motion.cpp", "rover.cpp")]
    sim = [os.path.join(SIM_DIR, f) for f in ("plant.cpp", "sim_main.cpp")]
    return build_program("rover_sim", core + sim, "@QUIT\n")
