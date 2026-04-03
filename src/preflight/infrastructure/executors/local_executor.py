from __future__ import annotations

import subprocess

from preflight.domain.models import CommandResult
from preflight.infrastructure.executors.base import BaseCommandExecutor


class LocalCommandExecutor(BaseCommandExecutor):
    def run(self, command: str) -> CommandResult:
        completed = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
        )
        return CommandResult(
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout.strip(),
            stderr=completed.stderr.strip(),
        )
