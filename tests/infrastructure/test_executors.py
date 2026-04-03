from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

from preflight.domain.models import SshTargetConfig
from preflight.infrastructure.executors.local_executor import LocalCommandExecutor
from preflight.infrastructure.executors.ssh_executor import SshCommandExecutor


def test_local_executor_wraps_subprocess_result(monkeypatch) -> None:
    def fake_run(*args, **kwargs) -> CompletedProcess[str]:
        assert args[0] == "hostname"
        assert kwargs["shell"] is True
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
