from __future__ import annotations

import io
from pathlib import Path
from subprocess import CompletedProcess

from preflight.domain.models import CommandResult, SshTargetConfig
from preflight.infrastructure.executors.logging_executor import LoggingCommandExecutor
from preflight.infrastructure.executors.local_executor import LocalCommandExecutor
from preflight.infrastructure.executors.ssh_executor import SshCommandExecutor
from preflight.interfaces.execution_logger import DebugExecutionLogger, VerboseExecutionLogger


def test_local_executor_wraps_subprocess_result(monkeypatch) -> None:
    def fake_run(*args, **kwargs) -> CompletedProcess[str]:
        assert args[0] == ["/bin/sh", "-lc", "hostname"]
        return CompletedProcess(args="hostname", returncode=0, stdout="node-a\n", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = LocalCommandExecutor().run("hostname")

    assert result.exit_code == 0
    assert result.stdout == "node-a"


def test_ssh_executor_builds_expected_command(monkeypatch) -> None:
    target = SshTargetConfig(
        host="10.0.0.5",
        user="ec2-user",
        private_key_path=Path("/tmp/id_rsa"),
        port=2222,
        connect_timeout_seconds=9,
    )

    def fake_run(*args, **kwargs) -> CompletedProcess[str]:
        command = args[0]
        assert command[:6] == [
            "ssh",
            "-i",
            "/tmp/id_rsa",
            "-o",
            "ConnectTimeout=9",
            "-p",
        ]
        assert command[6:] == ["2222", "ec2-user@10.0.0.5", "uname -r"]
        assert kwargs["capture_output"] is True
        return CompletedProcess(args=command, returncode=0, stdout="5.14.0\n", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = SshCommandExecutor(target).run("uname -r")

    assert result.stdout == "5.14.0"


def test_logging_executor_hides_commands_in_verbose_mode() -> None:
    class FakeExecutor:
        def run(self, command: str) -> CommandResult:
            return CommandResult(command=command, exit_code=0, stdout="ok", stderr="")

    stream = io.StringIO()
    executor = LoggingCommandExecutor(
        inner=FakeExecutor(),
        logger=VerboseExecutionLogger(stream=stream),
        stage_name="baseline",
    )

    result = executor.run("hostname")

    assert result.stdout == "ok"
    assert stream.getvalue() == ""


def test_logging_executor_logs_commands_in_debug_mode() -> None:
    class FakeExecutor:
        def run(self, command: str) -> CommandResult:
            return CommandResult(command=command, exit_code=0, stdout="ok", stderr="")

    stream = io.StringIO()
    executor = LoggingCommandExecutor(
        inner=FakeExecutor(),
        logger=DebugExecutionLogger(stream=stream),
        stage_name="baseline",
    )

    result = executor.run("hostname")

    assert result.stdout == "ok"
    assert "[baseline] $ hostname" in stream.getvalue()
