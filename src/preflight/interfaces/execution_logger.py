from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO


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
        self._write(f"[{stage}] $ {command}")

    def stage_end(self, name: str) -> None:
        self._write(f"--- {name.upper()} complete ---")

    def artifact_written(self, stage: str, path: str) -> None:
        self._write(f"[{stage}] artifact -> {path}")

    def _write(self, message: str) -> None:
        self.stream.write(f"{message}\n")
