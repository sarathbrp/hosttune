from __future__ import annotations

from dataclasses import dataclass

from tune.domain.validation_models import ValidationResult


@dataclass(frozen=True)
class BenchmarkSample:
    run_index: int
    requests_per_second: float
    total_requests: int
    average_latency_ms: float


@dataclass(frozen=True)
class BenchmarkWorkloadSummary:
    workload_name: str
    samples: tuple[BenchmarkSample, ...]
    median_requests_per_second: float
    median_total_requests: int
    median_latency_ms: float
    relative_variance: float
    stable: bool


@dataclass(frozen=True)
class TuneBenchmarkResult:
    validation_result: ValidationResult | None
    benchmark_command: str
    run_count: int
    stable: bool
    variance_threshold: float
    workload_summaries: tuple[BenchmarkWorkloadSummary, ...]
