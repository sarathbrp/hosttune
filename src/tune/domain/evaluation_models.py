from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tune.domain.benchmark_models import TuneBenchmarkResult


class EvaluationDecision(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class WorkloadEvaluation:
    workload_name: str
    baseline_requests_per_second: float
    current_requests_per_second: float
    relative_change: float
    above_noise_floor: bool


@dataclass(frozen=True)
class EvaluationResult:
    benchmark_result: TuneBenchmarkResult
    decision: EvaluationDecision
    summary: str
    primary_metric: str
    variance_threshold: float
    guardrails_held: bool
    drift_detected: bool
    workload_evaluations: tuple[WorkloadEvaluation, ...]
    missing_guardrails: tuple[str, ...]
