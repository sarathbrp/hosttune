from baseline.application.baseline_runner import BaselineRunner
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight.domain.models import BenchmarkResult

from tests.onboard.test_service_definition_validator import build_valid_definition


class FakeBenchmarkRunner:
    def run(self, executor):  # type: ignore[no-untyped-def]
        _ = executor
        return BenchmarkResult(
            command="printf '1.0'",
            exit_code=0,
            primary_metric_name="score",
            primary_metric_value=1.0,
            raw_output="1.0",
        )


def test_baseline_runner_uses_service_benchmark_hints() -> None:
    service = ServiceDefinitionValidator().validate(build_valid_definition())

    result = BaselineRunner(benchmark_runner=FakeBenchmarkRunner()).run(service, executor=object())

    assert result.service_name == "nginx"
    assert result.expected_variance == 0.05
    assert result.warmup_seconds == 10
