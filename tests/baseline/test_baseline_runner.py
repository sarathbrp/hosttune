import json
import io
from pathlib import Path

from baseline.application.baseline_runner import BaselineRunner
from baseline.domain.models import BenchmarkConfig
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight.domain.models import CommandResult, LocalTargetConfig, SshTargetConfig
from preflight.interfaces.execution_logger import VerboseExecutionLogger

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


class PayloadExecutor:
    def __init__(self, payloads: dict[str, dict[str, object]]) -> None:
        self._payloads = payloads

    def run(self, command: str) -> CommandResult:
        if "benchmark.sh" in command:
            return CommandResult(command=command, exit_code=0, stdout="ok", stderr="")
        for workload_name, payload in self._payloads.items():
            if f"_{workload_name}.json" in command:
                return CommandResult(
                    command=command,
                    exit_code=0,
                    stdout=json.dumps(payload),
                    stderr="",
                )
        return CommandResult(command=command, exit_code=1, stdout="", stderr="missing payload")


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


def test_baseline_runner_parses_duration_units() -> None:
    runner = BaselineRunner(
        benchmark_config=BenchmarkConfig(
            runner_target=LocalTargetConfig(),
            contestant_name="hosttune",
            script_path="/root/hackathon-tools/benchmark.sh",
            results_directory="/root/hackathon-results",
            workloads=("homepage",),
            compare_script_path=None,
        )
    )

    assert runner._parse_duration_ms("2.16ms") == 2.16
    assert runner._parse_duration_ms("500us") == 0.5
    assert runner._parse_duration_ms("1.5s") == 1500.0


def test_baseline_runner_parses_multiple_workload_payload_shapes() -> None:
    runner = BaselineRunner(
        benchmark_config=BenchmarkConfig(
            runner_target=LocalTargetConfig(),
            contestant_name="hosttune",
            script_path="/root/hackathon-tools/benchmark.sh",
            results_directory="/root/hackathon-results",
            workloads=("homepage", "mixed", "large"),
            compare_script_path=None,
        )
    )
    payloads = {
        "homepage": {
            "results": {
                "requests": {"per_sec": 876010.04, "total": 26368044},
                "latency": {"avg": "2.16ms"},
            }
        },
        "mixed": {
            "results": {
                "requests": {"per_sec": 125000.0, "total": 3750000},
                "latency": {"avg": "850us"},
            }
        },
        "large": {
            "results": {
                "requests": {"per_sec": 5021.5, "total": 150645},
                "latency": {"avg": "1.25s"},
            }
        },
    }
    executor = PayloadExecutor(payloads)

    service = ServiceDefinitionValidator().validate(build_valid_definition())
    result = runner.run(
        service=service,
        benchmark_executor=executor,
        dut_target=SshTargetConfig(
            host="10.1.90.178",
            user="root",
            private_key_path=Path("/tmp/id_rsa"),
        ),
    )

    assert result.workload_results[0].average_latency_ms == 2.16
    assert result.workload_results[1].average_latency_ms == 0.85
    assert result.workload_results[2].average_latency_ms == 1250.0


def test_baseline_runner_logs_combined_workload_summary() -> None:
    stream = io.StringIO()
    runner = BaselineRunner(
        benchmark_config=BenchmarkConfig(
            runner_target=LocalTargetConfig(),
            contestant_name="hosttune",
            script_path="/root/hackathon-tools/benchmark.sh",
            results_directory="/root/hackathon-results",
            workloads=("homepage", "mixed"),
            compare_script_path=None,
        ),
        logger=VerboseExecutionLogger(stream=stream),
    )
    payloads = {
        "homepage": {
            "results": {
                "requests": {"per_sec": 876010.04, "total": 26368044},
                "latency": {"avg": "2.16ms"},
            }
        },
        "mixed": {
            "results": {
                "requests": {"per_sec": 125000.0, "total": 3750000},
                "latency": {"avg": "850us"},
            }
        },
    }
    executor = PayloadExecutor(payloads)
    service = ServiceDefinitionValidator().validate(build_valid_definition())

    runner.run(
        service=service,
        benchmark_executor=executor,
        dut_target=SshTargetConfig(
            host="10.1.90.178",
            user="root",
            private_key_path=Path("/tmp/id_rsa"),
        ),
    )

    logged = stream.getvalue()
    assert "[baseline] Workload summary:" in logged
    assert "homepage" in logged
    assert "mixed" in logged
    assert "latency_ms=" in logged
