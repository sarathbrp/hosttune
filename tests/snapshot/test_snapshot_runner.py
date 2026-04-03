from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight.domain.models import CommandResult
from snapshot.application.snapshot_runner import SnapshotRunner

from tests.onboard.test_service_definition_validator import build_valid_definition


class FakeExecutor:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def run(self, command: str) -> CommandResult:
        self.commands.append(command)
        return CommandResult(command=command, exit_code=0, stdout="ok", stderr="")


def test_snapshot_runner_captures_service_state() -> None:
    service = ServiceDefinitionValidator().validate(build_valid_definition())
    executor = FakeExecutor()

    result = SnapshotRunner().run(service, executor)

    assert result.service_name == "nginx"
    assert "/etc/nginx/nginx.conf" in result.captured_paths
    assert any(command.startswith("mkdir -p /var/tmp/hosttune") for command in executor.commands)
    assert result.snapshot_directory == "/var/tmp/hosttune"
