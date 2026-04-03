import json

from onboard.domain.models import ApplyMode
from preflight.domain.models import CommandResult
from tune.application.benchmark_executor import TuneBenchmarkExecutor
from tune.domain.apply_models import AppliedChange
from tune.domain.hypothesis_models import CandidateSource, TunePhase, TuningHypothesis
from tune.domain.tuning_layer import tuning_layer_for_parameter_key
from tune.domain.validation_models import ValidationCheck, ValidationResult

from tests.tune.test_candidate_catalog_builder import build_tune_context


class BenchmarkExecutorDouble:
    def __init__(self, payloads_by_run: list[dict[str, dict[str, object]]]) -> None:
        self._payloads_by_run = payloads_by_run
        self._current_run = -1

    def run(self, command: str) -> CommandResult:
        if "benchmark.sh" in command:
            self._current_run += 1
            return CommandResult(command=command, exit_code=0, stdout="ok", stderr="")
        for workload_name, payload in self._payloads_by_run[self._current_run].items():
            if f"_{workload_name}.json" in command:
                return CommandResult(
                    command=command,
                    exit_code=0,
                    stdout=json.dumps(payload),
                    stderr="",
                )
        return CommandResult(command=command, exit_code=1, stdout="", stderr="missing payload")


class FailingBenchmarkExecutorDouble:
    def run(self, command: str) -> CommandResult:
        if "benchmark.sh" in command:
            return CommandResult(
                command=command,
                exit_code=2,
                stdout="=== Hackathon Performance Benchmark ===\nTarget unreachable",
                stderr="curl: (7) Failed to connect",
            )
        return CommandResult(command=command, exit_code=1, stdout="", stderr="unexpected")


def build_validation_result() -> ValidationResult:
    hypothesis = TuningHypothesis(
        phase=TunePhase.WIDE_SWEEP,
        parameter_key="sysctl.net.core.somaxconn",
        parameter_name="net.core.somaxconn",
        domain="kernel_sysctl",
        tuning_layer=tuning_layer_for_parameter_key("sysctl.net.core.somaxconn"),
        proposed_value="65535",
        source=CandidateSource.SERVICE_SYSCTL,
        apply_mode=ApplyMode.RELOAD,
        rationale="Increase backlog.",
    )
    applied_change = AppliedChange(
        hypothesis=hypothesis,
        target_path="net.core.somaxconn",
        previous_value="4096",
        applied_value="65535",
        apply_mode=ApplyMode.RELOAD,
        apply_command="sysctl -w net.core.somaxconn=65535",
        rollback_command="sysctl -w net.core.somaxconn=4096",
    )
    return ValidationResult(
        applied_change=applied_change,
        healthy=True,
        checks=(ValidationCheck(name="health_probe", passed=True, detail="ok"),),
    )


def test_tune_benchmark_executor_aggregates_median_results() -> None:
    context = build_tune_context()
    executor = BenchmarkExecutorDouble(
        [
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 1040.0, "total": 10400},
                        "latency": {"avg": "2.0ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 918.0, "total": 9180},
                        "latency": {"avg": "1.5ms"},
                    }
                },
            },
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 1060.0, "total": 10600},
                        "latency": {"avg": "1.9ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 920.0, "total": 9200},
                        "latency": {"avg": "1.4ms"},
                    }
                },
            },
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 1050.0, "total": 10500},
                        "latency": {"avg": "2.1ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 922.0, "total": 9220},
                        "latency": {"avg": "1.6ms"},
                    }
                },
            },
        ]
    )

    result = TuneBenchmarkExecutor(run_count=1, sleeper=lambda _seconds: None).run(
        context=context,
        iteration_number=1,
        validation_result=build_validation_result(),
        benchmark_executor=executor,
    )

    assert result.stable is True
    assert result.run_count == 1
    assert result.workload_summaries[0].median_requests_per_second == 1040.0
    assert result.workload_summaries[1].median_requests_per_second == 918.0
    assert "hosttune_nosession_iter001_run01" in result.benchmark_command


def test_tune_benchmark_executor_flags_unstable_variance() -> None:
    context = build_tune_context()
    executor = BenchmarkExecutorDouble(
        [
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 1000.0, "total": 10000},
                        "latency": {"avg": "2.0ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 800.0, "total": 8000},
                        "latency": {"avg": "1.5ms"},
                    }
                },
            },
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 1400.0, "total": 14000},
                        "latency": {"avg": "1.9ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 1200.0, "total": 12000},
                        "latency": {"avg": "1.4ms"},
                    }
                },
            },
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 900.0, "total": 9000},
                        "latency": {"avg": "2.1ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 700.0, "total": 7000},
                        "latency": {"avg": "1.6ms"},
                    }
                },
            },
        ]
    )

    result = TuneBenchmarkExecutor(run_count=1, sleeper=lambda _seconds: None).run(
        context=context,
        iteration_number=2,
        validation_result=build_validation_result(),
        benchmark_executor=executor,
    )

    assert result.stable is True
    assert all(summary.stable is True for summary in result.workload_summaries)


def test_tune_benchmark_executor_respects_cooling_period() -> None:
    context = build_tune_context()
    executor = BenchmarkExecutorDouble(
        [
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 1040.0, "total": 10400},
                        "latency": {"avg": "2.0ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 918.0, "total": 9180},
                        "latency": {"avg": "1.5ms"},
                    }
                },
            },
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 1060.0, "total": 10600},
                        "latency": {"avg": "1.9ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 920.0, "total": 9200},
                        "latency": {"avg": "1.4ms"},
                    }
                },
            },
            {
                "homepage": {
                    "results": {
                        "requests": {"per_sec": 1050.0, "total": 10500},
                        "latency": {"avg": "2.1ms"},
                    }
                },
                "small": {
                    "results": {
                        "requests": {"per_sec": 922.0, "total": 9220},
                        "latency": {"avg": "1.6ms"},
                    }
                },
            },
        ]
    )
    sleep_calls: list[float] = []

    TuneBenchmarkExecutor(
        run_count=3,
        sleeper=lambda seconds: sleep_calls.append(seconds),
    ).run(
        context=context,
        iteration_number=1,
        validation_result=build_validation_result(),
        benchmark_executor=executor,
    )

    assert sleep_calls == [30, 30]


def test_tune_benchmark_executor_surfaces_exit_code_and_output() -> None:
    context = build_tune_context()

    try:
        TuneBenchmarkExecutor(run_count=1).run(
            context=context,
            iteration_number=1,
            validation_result=build_validation_result(),
            benchmark_executor=FailingBenchmarkExecutorDouble(),
        )
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("Expected benchmark failure to raise ValueError")

    assert "exit_code=2" in message
    assert "=== Hackathon Performance Benchmark ===" in message
    assert "Failed to connect" in message
