from __future__ import annotations

from dataclasses import dataclass

from tune.domain.benchmark_models import BenchmarkWorkloadSummary, TuneBenchmarkResult
from tune.domain.evaluation_models import (
    EvaluationDecision,
    EvaluationResult,
    WorkloadEvaluation,
)
from tune.domain.tune_context import TuneContext


@dataclass
class ResultEvaluator:
    def evaluate(
        self,
        context: TuneContext,
        benchmark_result: TuneBenchmarkResult,
    ) -> EvaluationResult:
        baseline_by_workload = {
            workload.workload_name: workload for workload in context.baseline.workload_results
        }
        workload_evaluations = tuple(
            self._evaluate_workload(
                summary=summary,
                baseline_requests_per_second=baseline_by_workload[
                    summary.workload_name
                ].requests_per_second,
                variance_threshold=context.baseline.expected_variance,
            )
            for summary in benchmark_result.workload_summaries
            if summary.workload_name in baseline_by_workload
        )
        missing_guardrails = self._find_missing_guardrails(context)
        guardrails_held = len(missing_guardrails) == 0
        drift_detected = not benchmark_result.stable
        decision = self._decide(
            benchmark_result=benchmark_result,
            workload_evaluations=workload_evaluations,
            missing_guardrails=missing_guardrails,
        )
        summary = self._build_summary(decision, benchmark_result.stable, workload_evaluations)
        return EvaluationResult(
            benchmark_result=benchmark_result,
            decision=decision,
            summary=summary,
            primary_metric=context.onboard.service.benchmark_hints.primary_metric,
            variance_threshold=context.baseline.expected_variance,
            guardrails_held=guardrails_held,
            drift_detected=drift_detected,
            workload_evaluations=workload_evaluations,
            missing_guardrails=missing_guardrails,
        )

    def _evaluate_workload(
        self,
        summary: BenchmarkWorkloadSummary,
        baseline_requests_per_second: float,
        variance_threshold: float,
    ) -> WorkloadEvaluation:
        relative_change = 0.0
        if baseline_requests_per_second > 0.0:
            relative_change = (
                summary.median_requests_per_second - baseline_requests_per_second
            ) / baseline_requests_per_second
        return WorkloadEvaluation(
            workload_name=summary.workload_name,
            baseline_requests_per_second=baseline_requests_per_second,
            current_requests_per_second=summary.median_requests_per_second,
            relative_change=relative_change,
            above_noise_floor=abs(relative_change) > variance_threshold,
        )

    def _find_missing_guardrails(self, context: TuneContext) -> tuple[str, ...]:
        supported_guardrails = {"p95_latency", "error_rate"}
        return tuple(
            guardrail
            for guardrail in context.onboard.service.benchmark_hints.guardrail_metrics
            if guardrail not in supported_guardrails
        )

    def _decide(
        self,
        benchmark_result: TuneBenchmarkResult,
        workload_evaluations: tuple[WorkloadEvaluation, ...],
        missing_guardrails: tuple[str, ...],
    ) -> EvaluationDecision:
        if not benchmark_result.stable:
            return EvaluationDecision.INCONCLUSIVE
        if missing_guardrails:
            return EvaluationDecision.INCONCLUSIVE
        if not workload_evaluations:
            return EvaluationDecision.INCONCLUSIVE

        average_change = sum(item.relative_change for item in workload_evaluations) / len(
            workload_evaluations
        )
        if average_change > benchmark_result.variance_threshold:
            return EvaluationDecision.ACCEPT
        if average_change < -benchmark_result.variance_threshold:
            return EvaluationDecision.REJECT
        return EvaluationDecision.INCONCLUSIVE

    def _build_summary(
        self,
        decision: EvaluationDecision,
        stable: bool,
        workload_evaluations: tuple[WorkloadEvaluation, ...],
    ) -> str:
        average_change = 0.0
        if workload_evaluations:
            average_change = sum(item.relative_change for item in workload_evaluations) / len(
                workload_evaluations
            )
        return (
            f"decision={decision.value}; "
            f"stable={stable}; "
            f"average_relative_change={average_change:.4f}"
        )
