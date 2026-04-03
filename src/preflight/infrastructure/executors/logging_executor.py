from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import CommandExecutor, CommandResult
from preflight.interfaces.execution_logger import ExecutionLogger


@dataclass
class LoggingCommandExecutor:
    inner: CommandExecutor
    logger: ExecutionLogger
    stage_name: str

    def run(self, command: str) -> CommandResult:
        self.logger.command(self.stage_name, command)
        return self.inner.run(command)
