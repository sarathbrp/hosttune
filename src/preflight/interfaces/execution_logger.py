from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import TextIO

_hosttune_logger = logging.getLogger("hosttune")


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
class StdlibExecutionLogger(ExecutionLogger):
    """ExecutionLogger backed by Python stdlib logging.

    Routes all output through the 'hosttune' logger so log level,
    format, and handlers are controlled by logging.basicConfig()
    at startup (configured from config.yaml log_level).

    Level mapping:
      stage_start / stage_end / stage_detail / artifact_written -> INFO
      command -> DEBUG
    """

    logger: logging.Logger = field(default_factory=lambda: _hosttune_logger)

    def stage_start(self, name: str) -> None:
        self.logger.info("+++ %s +++", name.upper())

    def stage_detail(self, stage: str, message: str) -> None:
        for line in message.splitlines() or ("",):
            self.logger.info("[%s] %s", stage, line)

    def command(self, stage: str, command: str) -> None:
        self.logger.debug("[%s] $ %s", stage, command)

    def stage_end(self, name: str) -> None:
        self.logger.info("--- %s complete ---", name.upper())

    def artifact_written(self, stage: str, path: str) -> None:
        self.logger.info("[%s] artifact -> %s", stage, path)

    def debug_enabled(self) -> bool:
        return self.logger.isEnabledFor(logging.DEBUG)


# Legacy aliases kept for backward compatibility with any external callers.
VerboseExecutionLogger = StdlibExecutionLogger
DebugExecutionLogger = StdlibExecutionLogger
ColorExecutionLogger = StdlibExecutionLogger
