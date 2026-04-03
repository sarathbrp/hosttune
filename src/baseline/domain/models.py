from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import TargetConfig


@dataclass(frozen=True)
class BenchmarkConfig:
    runner_target: TargetConfig
    contestant_name: str
    script_path: str
    results_directory: str
    workloads: tuple[str, ...]
    compare_script_path: str | None
    cooling_period_seconds: int = 30


@dataclass(frozen=True)
class WorkloadBenchmarkResult:
    workload_name: str
    result_path: str
    requests_per_second: float
    total_requests: int
    average_latency_ms: float


@dataclass(frozen=True)
class BaselineResult:
    service_name: str
    benchmark_command: str
    benchmark_target: str
    workload_results: tuple[WorkloadBenchmarkResult, ...]
    expected_variance: float
    warmup_seconds: int
    guardrail_metrics: tuple[str, ...]
    comparison_output: str | None
