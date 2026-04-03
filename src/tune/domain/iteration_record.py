from __future__ import annotations

from dataclasses import dataclass

from tune.domain.apply_models import AppliedChange
from tune.domain.benchmark_models import TuneBenchmarkResult
from tune.domain.evaluation_models import AttributionVerificationResult, EvaluationResult
from tune.domain.hypothesis_models import TunePhase, TuningHypothesis
from tune.domain.validation_models import ValidationResult


@dataclass(frozen=True)
class TuneIterationRecord:
    iteration_number: int
    phase: TunePhase
    hypothesis: TuningHypothesis
    applied_change: AppliedChange | None
    validation_result: ValidationResult | None
    benchmark_result: TuneBenchmarkResult | None
    evaluation_result: EvaluationResult | None
    attribution_verification: AttributionVerificationResult | None
    active_parameter_keys: tuple[str, ...]
    started_at_utc: str
    completed_at_utc: str
    duration_seconds: float
