from __future__ import annotations

import subprocess

from preflight.domain.models import CommandResult, SshTargetConfig
from preflight.infrastructure.executors.base import BaseCommandExecutor


class SshCommandExecutor(BaseCommandExecutor):
    def __init__(self, target: SshTargetConfig) -> None:
        self._target = target

    def run(self, command: str) -> CommandResult:
        ssh_command = [
            "ssh",
            "-i",
            str(self._target.private_key_path),
            "-o",
            f"ConnectTimeout={self._target.connect_timeout_seconds}",
            "-p",
            str(self._target.port),
            f"{self._target.user}@{self._target.host}",
            command,
        ]
        completed = subprocess.run(
            ssh_command,
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
