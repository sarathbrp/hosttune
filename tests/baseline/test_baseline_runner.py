import json
from pathlib import Path

from baseline.application.baseline_runner import BaselineRunner
from baseline.domain.models import BenchmarkConfig
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight.domain.models import CommandResult, LocalTargetConfig, SshTargetConfig

from tests.onboard.test_service_definition_validator import build_valid_definition


class FakeExecutor:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def run(self, command: str) -> CommandResult:
        self.commands.append(command)
        if "benchmark.sh" in command:
            return CommandResult(command=command, exit_code=0, stdout="ok", stderr="")
        if "compare-results.sh" in command:
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="homepage improved by 3%",
                stderr="",
            )
        payload = {
            "results": {
                "requests": {"per_sec": 4567.8, "total": 12345},
                "latency": {"avg": 6.7},
            }
        }
        return CommandResult(command=command, exit_code=0, stdout=json.dumps(payload), stderr="")


def test_baseline_runner_uses_service_benchmark_hints() -> None:
    service = ServiceDefinitionValidator().validate(build_valid_definition())
    executor = FakeExecutor()
    config = BenchmarkConfig(
        runner_target=LocalTargetConfig(),
        contestant_name="hosttune",
        script_path="/root/hackathon-tools/benchmark.sh",
        results_directory="/root/hackathon-results",
        workloads=("homepage", "small"),
        compare_script_path="/root/hackathon-tools/compare-results.sh",
    )

    result = BaselineRunner(benchmark_config=config).run(
        service,
        benchmark_executor=executor,
        dut_target=SshTargetConfig(
            host="10.1.90.178",
            user="root",
            private_key_path=Path("/tmp/id_rsa"),
        ),
    )

    assert result.service_name == "nginx"
    assert result.benchmark_target == "10.1.90.178"
    assert len(result.workload_results) == 2
    assert result.workload_results[0].requests_per_second == 4567.8
    assert result.expected_variance == 0.05
    assert result.warmup_seconds == 10
    assert result.comparison_output == "homepage improved by 3%"
