from __future__ import annotations

import logging
from dataclasses import dataclass

from preflight.domain.models import CommandExecutor
from tune.application.benchmark_executor import TuneBenchmarkExecutor
from tune.application.health_validator import HealthValidator
from tune.domain.apply_models import AppliedChange
from tune.domain.benchmark_models import TuneBenchmarkResult
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
        average_drop = self._calculate_average_drop(
            accepted_benchmark_result=accepted_benchmark_result,
            reverted_benchmark_result=reverted_benchmark_result,
        )
        verified = average_drop > context.effective_variance_threshold
        if verified:
            reapply_result = target_executor.run(applied_change.apply_command)
            if reapply_result.exit_code != 0:
                return AttributionVerificationResult(
                    verified=False,
                    summary=(
                        f"average_drop={average_drop:.4f}; "
                        f"threshold={context.effective_variance_threshold:.4f}; "
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
                f"threshold={context.effective_variance_threshold:.4f}; "
                f"verified={verified}"
            ),
            reverted_benchmark_result=reverted_benchmark_result,
            average_drop=average_drop,
        )

    def _calculate_average_drop(
        self,
        accepted_benchmark_result: TuneBenchmarkResult,
        reverted_benchmark_result: TuneBenchmarkResult,
    ) -> float:
        reverted_by_name = {
            item.workload_name: item for item in reverted_benchmark_result.workload_summaries
        }
        drops: list[float] = []
        for accepted_summary in accepted_benchmark_result.workload_summaries:
            reverted_summary = reverted_by_name.get(accepted_summary.workload_name)
            if reverted_summary is None:
                continue
            accepted_rps = accepted_summary.median_requests_per_second
            reverted_rps = reverted_summary.median_requests_per_second
            if accepted_rps <= 0.0:
                continue
            drops.append((accepted_rps - reverted_rps) / accepted_rps)
        if not drops:
            logging.getLogger(__name__).warning(
                "No matching workloads between accepted and reverted benchmarks; "
                "average_drop defaults to 0.0"
            )
            return 0.0
        return sum(drops) / len(drops)
