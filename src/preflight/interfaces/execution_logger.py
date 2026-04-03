from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import TextIO

# ── ANSI colour helpers ───────────────────────────────────────────────────────
_R = "\033[0m"  # reset
_B = "\033[1m"  # bold
_D = "\033[2m"  # dim

_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_BLUE = "\033[94m"
_MAGENTA = "\033[95m"
_CYAN = "\033[96m"
_WHITE = "\033[97m"


def _c(*codes: str) -> str:
    return "".join(codes)


def _colorize_detail(line: str) -> str:
    """Return the ANSI-coloured version of a stage_detail log line."""
    lo = line.lower()

    # ── agent section headers ─────────────────────────────────────────────
    if "─" in line:
        if "RESPONSE" in line:
            if "service_agent" in line:
                return _c(_B, _CYAN) + line + _R
            if "rhel_expert" in line:
                return _c(_B, _BLUE) + line + _R
            if "synthesizer" in line:
                return _c(_B, _MAGENTA) + line + _R
            return _c(_B, _CYAN) + line + _R
        if "PROMPT" in line:
            return _c(_D) + line + _R

    # ── failures / rollbacks ──────────────────────────────────────────────
    if re.search(
        r"(apply failed|rollback|failed_validation|health gate failed"
        r"|validation failed|apply_mode.*failed|CRITICAL)",
        lo,
    ):
        return _c(_RED) + line + _R

    # ── evaluation decisions ──────────────────────────────────────────────
    if "decision=accepted" in lo or "decision: accepted" in lo:
        return _c(_B, _GREEN) + line + _R
    if re.search(r"decision=(rejected|reject)\b", lo):
        return _c(_RED) + line + _R
    if "decision=inconclusive" in lo:
        return _c(_YELLOW) + line + _R
    if "decision=promising" in lo:
        return _c(_CYAN) + line + _R

    # ── best config update ────────────────────────────────────────────────
    if "best config updated" in lo:
        return _c(_B, _GREEN) + line + _R

    # ── apply / hypothesis ────────────────────────────────────────────────
    if lo.startswith("apply:") or lo.startswith("apply ("):
        return _c(_YELLOW) + line + _R
    if lo.startswith("hypothesis:"):
        return _c(_CYAN) + line + _R

    # ── benchmark workload lines ──────────────────────────────────────────
    if "rps=" in lo and "latency_ms=" in lo:
        return _c(_BLUE) + line + _R

    # ── runtime telemetry signals ─────────────────────────────────────────
    if "↑increasing" in line or "drops detected" in lo or "time_squeeze" in lo:
        return _c(_YELLOW) + line + _R

    return line


class ExecutionLogger:
    def stage_start(self, name: str) -> None:
        """Log that a stage has started."""

    def stage_detail(self, stage: str, message: str) -> None:
        """Log an informational message for a stage."""

    def command(self, stage: str, command: str) -> None:
        """Log a command before execution."""

    def stage_end(self, name: str) -> None:
        """Log that a stage has completed."""

    def artifact_written(self, stage: str, path: str) -> None:
        """Log that a stage artifact was written."""

    def debug_enabled(self) -> bool:
        """Return whether debug-only logs should be emitted."""
        return False


class NullExecutionLogger(ExecutionLogger):
    pass


@dataclass
class VerboseExecutionLogger(ExecutionLogger):
    stream: TextIO = sys.stderr

    def stage_start(self, name: str) -> None:
        self._write(f"+++ {name.upper()} +++")

    def stage_detail(self, stage: str, message: str) -> None:
        for line in message.splitlines() or ("",):
            self._write(f"[{stage}] {line}")

    def command(self, stage: str, command: str) -> None:
        _ = stage
        _ = command

    def stage_end(self, name: str) -> None:
        self._write(f"--- {name.upper()} complete ---")

    def artifact_written(self, stage: str, path: str) -> None:
        self._write(f"[{stage}] artifact -> {path}")

    def _write(self, message: str) -> None:
        self.stream.write(f"{message}\n")


@dataclass
class DebugExecutionLogger(VerboseExecutionLogger):
    def debug_enabled(self) -> bool:
        return True

    def command(self, stage: str, command: str) -> None:
        self._write(f"[{stage}] $ {command}")


@dataclass
class ColorExecutionLogger(VerboseExecutionLogger):
    """VerboseExecutionLogger with ANSI colour highlights.

    Colours applied per-line based on content:
    - Agent response headers  → bold cyan / blue / magenta
    - Agent prompt headers    → dim
    - Failures / rollbacks    → bright red
    - ACCEPTED decisions      → bold green
    - REJECTED decisions      → bright red
    - INCONCLUSIVE            → yellow
    - PROMISING               → cyan
    - Apply: / Hypothesis:    → yellow / cyan
    - Benchmark workload rps  → blue
    - Telemetry pressure      → yellow
    - Best config updated     → bold green
    """

    debug: bool = False
    color: bool = field(default_factory=lambda: sys.stderr.isatty())

    def debug_enabled(self) -> bool:
        return self.debug

    def command(self, stage: str, command: str) -> None:
        if self.debug:
            line = f"[{stage}] $ {command}"
            self._write(_c(_D) + line + _R if self.color else line)

    def stage_start(self, name: str) -> None:
        line = f"+++ {name.upper()} +++"
        self._write(_c(_B, _WHITE) + line + _R if self.color else line)

    def stage_end(self, name: str) -> None:
        line = f"--- {name.upper()} complete ---"
        self._write(_c(_B, _WHITE) + line + _R if self.color else line)

    def stage_detail(self, stage: str, message: str) -> None:
        for raw in message.splitlines() or ("",):
            prefix = f"[{stage}] "
            if self.color:
                self._write(prefix + _colorize_detail(raw))
            else:
                self._write(prefix + raw)
