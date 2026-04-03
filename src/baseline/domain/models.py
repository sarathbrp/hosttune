from __future__ import annotations

from dataclasses import dataclass

from preflight.domain.models import BenchmarkResult


@dataclass(frozen=True)
class BaselineResult:
    service_name: str
    benchmark_command: str
    benchmark_result: BenchmarkResult
    expected_variance: float
    warmup_seconds: int
    guardrail_metrics: tuple[str, ...]
