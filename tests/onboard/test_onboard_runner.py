from pathlib import Path

import pytest

from onboard.application.onboard_runner import OnboardRunner
from onboard.infrastructure.service_compatibility_evaluator import ServiceCompatibilityEvaluator
from onboard.infrastructure.service_definition_loader import ServiceDefinitionLoader
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight.domain.models import CommandResult

from tests.onboard.test_service_compatibility_evaluator import build_preflight_snapshot
from tests.onboard.test_service_definition_validator import build_valid_definition


class FakeExecutor:
    def run(self, command: str) -> CommandResult:
        return CommandResult(command=command, exit_code=0, stdout="", stderr="")


def test_onboard_runner_returns_onboard_result(tmp_path: Path) -> None:
    registry = tmp_path / "service-monitor"
    registry.mkdir()
    definition = registry / "nginx.yaml"
    import yaml

    definition.write_text(yaml.safe_dump(build_valid_definition()), encoding="utf-8")
    runner = OnboardRunner(
        loader=ServiceDefinitionLoader(registry_path=registry),
        validator=ServiceDefinitionValidator(),
        evaluator=ServiceCompatibilityEvaluator(),
    )

    result = runner.run("nginx", build_preflight_snapshot(), FakeExecutor())

    assert result.service_name == "nginx"
    assert result.service.identity.service_name == "nginx"


def test_onboard_runner_rejects_incompatible_service(tmp_path: Path) -> None:
    registry = tmp_path / "service-monitor"
    registry.mkdir()
    definition = registry / "nginx.yaml"
    import yaml

    data = build_valid_definition()
    data["identity"]["config_paths"] = ["/missing/path"]  # type: ignore[index]
    definition.write_text(yaml.safe_dump(data), encoding="utf-8")
    runner = OnboardRunner(
        loader=ServiceDefinitionLoader(registry_path=registry),
        validator=ServiceDefinitionValidator(),
        evaluator=ServiceCompatibilityEvaluator(),
    )

    class MissingPathExecutor(FakeExecutor):
        def run(self, command: str) -> CommandResult:
            exit_code = 1 if "test -e /missing/path" in command else 0
            return CommandResult(command=command, exit_code=exit_code, stdout="", stderr="")

    with pytest.raises(ValueError, match="Onboard compatibility check failed"):
        runner.run("nginx", build_preflight_snapshot(), MissingPathExecutor())
