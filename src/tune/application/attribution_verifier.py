from __future__ import annotations

import logging
from dataclasses import dataclass

from preflight.domain.models import CommandExecutor
from tune.application.benchmark_executor import TuneBenchmarkExecutor
from tune.application.health_validator import HealthValidator
from tune.domain.apply_models import AppliedChange
from tune.domain.benchmark_models import BenchmarkWorkloadSummary, TuneBenchmarkResult
from tune.domain.evaluation_models import AttributionVerificationResult
from tune.domain.tune_context import TuneContext


@dataclass
class AttributionVerifier:
    benchmark_executor: TuneBenchmarkExecutor
    health_validator: HealthValidator

    def verify(
        self,
        context: TuneContext,
        iteration_number: int,
        applied_change: AppliedChange,
        accepted_benchmark_result: TuneBenchmarkResult,
        target_executor: CommandExecutor,
        benchmark_runner_executor: CommandExecutor,
    ) -> AttributionVerificationResult:
        rollback_result = target_executor.run(applied_change.rollback_command)
        if rollback_result.exit_code != 0:
            return AttributionVerificationResult(
                verified=False,
                summary=(
                    "attribution rollback failed: "
                    f"{rollback_result.stderr or rollback_result.stdout}"
                ),
                reverted_benchmark_result=None,
                average_drop=0.0,
            )

        baseline_checks = self.health_validator.validate_baseline(context, target_executor)
        failed_checks = [check for check in baseline_checks if not check.passed]
        if failed_checks:
            detail = ", ".join(f"{check.name}: {check.detail}" for check in failed_checks)
            return AttributionVerificationResult(
                verified=False,
                summary=f"verification aborted after rollback: {detail}",
                reverted_benchmark_result=None,
                average_drop=0.0,
            )

        reverted_benchmark_result = self.benchmark_executor.run(
            context=context,
            iteration_number=iteration_number,
            validation_result=None,
            benchmark_executor=benchmark_runner_executor,
            label="verify",
            telemetry_executor=target_executor,
        )
        (
            average_drop,
            max_drop,
            compared_workloads,
            material_gain_workloads,
        ) = self._calculate_average_drop(
            context=context,
            accepted_benchmark_result=accepted_benchmark_result,
            reverted_benchmark_result=reverted_benchmark_result,
        )
        # When reverting the change actually *improved* performance (negative drop),
        # the change was harmful — mark as not verified so the engine rolls it back.
        # When average_drop is near zero and there are already large active changes,
        # the primary hypothesis likely contributed little on top of cumulative gains;
        # treat as unverified so it can be cleanly rolled back without masking the
        # real signal. This avoids spurious INCONCLUSIVE loops.
        verified = (
            average_drop > context.effective_variance_threshold
            or max_drop > context.effective_variance_threshold
        )
        if verified:
            reapply_result = target_executor.run(applied_change.apply_command)
            if reapply_result.exit_code != 0:
                return AttributionVerificationResult(
                    verified=False,
                    summary=(
                        f"average_drop={average_drop:.4f}; "
                        f"max_drop={max_drop:.4f}; "
                        f"threshold={context.effective_variance_threshold:.4f}; "
                        f"compared_workloads={compared_workloads}; "
                        f"material_gain_workloads={material_gain_workloads}; "
                        "attribution reapply failed: "
                        f"{reapply_result.stderr or reapply_result.stdout}"
                    ),
                    reverted_benchmark_result=reverted_benchmark_result,
                    average_drop=average_drop,
                )
        return AttributionVerificationResult(
            verified=verified,
            summary=(
                f"average_drop={average_drop:.4f}; "
                f"max_drop={max_drop:.4f}; "
                f"threshold={context.effective_variance_threshold:.4f}; "
                f"compared_workloads={compared_workloads}; "
                f"material_gain_workloads={material_gain_workloads}; "
                f"verified={verified}"
            ),
            reverted_benchmark_result=reverted_benchmark_result,
            average_drop=average_drop,
        )

    def _calculate_average_drop(
        self,
        context: TuneContext,
        accepted_benchmark_result: TuneBenchmarkResult,
        reverted_benchmark_result: TuneBenchmarkResult,
    ) -> tuple[float, float, int, int]:
        baseline_by_name = {
            workload.workload_name: workload.requests_per_second
            for workload in context.baseline.workload_results
        }
        reverted_by_name = {
            item.workload_name: item for item in reverted_benchmark_result.workload_summaries
        }
        material_gain_workloads = [
            accepted_summary
            for accepted_summary in accepted_benchmark_result.workload_summaries
            if self._is_material_gain(
                accepted_summary=accepted_summary,
                baseline_rps=baseline_by_name.get(accepted_summary.workload_name),
                variance_threshold=context.effective_variance_threshold,
            )
        ]
        selected_workloads = material_gain_workloads or list(
            accepted_benchmark_result.workload_summaries
        )
        drops = self._calculate_drops(selected_workloads, reverted_by_name)
        if not drops and material_gain_workloads:
            logging.getLogger(__name__).warning(
                "No matching reverted workloads for material-gain verification set; "
                "falling back to all matched workloads"
            )
            selected_workloads = list(accepted_benchmark_result.workload_summaries)
            drops = self._calculate_drops(selected_workloads, reverted_by_name)
        if not drops:
            logging.getLogger(__name__).warning(
                "No matching workloads between accepted and reverted benchmarks; "
                "average_drop defaults to 0.0"
            )
            return 0.0, 0.0, 0, len(material_gain_workloads)
        return (
            sum(drops) / len(drops),
            max(drops),
            len(drops),
            len(material_gain_workloads),
        )

    def _calculate_drops(
        self,
        accepted_workloads: list[BenchmarkWorkloadSummary],
        reverted_by_name: dict[str, BenchmarkWorkloadSummary],
    ) -> list[float]:
        drops: list[float] = []
        for accepted_summary in accepted_workloads:
            reverted_summary = reverted_by_name.get(accepted_summary.workload_name)
            if reverted_summary is None:
                continue
            accepted_rps = accepted_summary.median_requests_per_second
            reverted_rps = reverted_summary.median_requests_per_second
            if accepted_rps <= 0.0:
                continue
            drops.append((accepted_rps - reverted_rps) / accepted_rps)
        return drops

    def _is_material_gain(
        self,
        accepted_summary: BenchmarkWorkloadSummary,
        baseline_rps: float | None,
        variance_threshold: float,
    ) -> bool:
        if baseline_rps is None or baseline_rps <= 0.0:
            return False
        relative_gain = (accepted_summary.median_requests_per_second - baseline_rps) / baseline_rps
        return relative_gain > variance_threshold
