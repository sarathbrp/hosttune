from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from time import sleep
from typing import Any, cast

from preflight.domain.models import CommandExecutor
from preflight.interfaces.execution_logger import ExecutionLogger, NullExecutionLogger
from tune.domain.benchmark_models import (
    BenchmarkSample,
    BenchmarkWorkloadSummary,
    TuneBenchmarkResult,
)
from tune.domain.tune_context import TuneContext
from tune.domain.validation_models import ValidationResult


@dataclass
class TuneBenchmarkExecutor:
    run_count: int = 1
    logger: ExecutionLogger = NullExecutionLogger()
    sleeper: Callable[[float], None] = sleep

    def run(
        self,
        context: TuneContext,
        iteration_number: int,
        validation_result: ValidationResult,
        benchmark_executor: CommandExecutor,
    ) -> TuneBenchmarkResult:
        benchmark_target = context.baseline.benchmark_target
        benchmark_command = ""
        workload_samples: dict[str, list[BenchmarkSample]] = {
            workload_name: [] for workload_name in context.benchmark_config.workloads
        }

        for run_index in range(1, self.run_count + 1):
            contestant_name = self._build_contestant_name(context, iteration_number, run_index)
            benchmark_command = self._build_benchmark_command(
                context=context,
                benchmark_target=benchmark_target,
                contestant_name=contestant_name,
            )
            self.logger.stage_detail(
                "tune",
                (
                    f"Benchmark run {run_index}/{self.run_count} started against "
                    f"{benchmark_target} as {contestant_name}"
                ),
            )
            run_result = benchmark_executor.run(benchmark_command)
            if run_result.exit_code != 0:
                msg = f"Benchmark command failed: {run_result.stderr or run_result.stdout}"
                raise ValueError(msg)
            for workload_name in context.benchmark_config.workloads:
                sample = self._load_workload_sample(
                    benchmark_executor=benchmark_executor,
                    context=context,
                    contestant_name=contestant_name,
                    workload_name=workload_name,
                    run_index=run_index,
                )
                workload_samples[workload_name].append(sample)
            self.logger.stage_detail(
                "tune",
                self._build_run_summary(run_index, workload_samples),
            )
            if run_index < self.run_count and context.benchmark_config.cooling_period_seconds > 0:
                self.logger.stage_detail(
                    "tune",
                    (
                        "Cooling period: "
                        f"sleeping {context.benchmark_config.cooling_period_seconds}s "
                        "before next benchmark run"
                    ),
                )
                self.sleeper(context.benchmark_config.cooling_period_seconds)

        workload_summaries = tuple(
            self._summarize_workload(
                workload_name=workload_name,
                samples=tuple(samples),
                variance_threshold=context.baseline.expected_variance,
            )
            for workload_name, samples in workload_samples.items()
        )
        stable = all(summary.stable for summary in workload_summaries)
        return TuneBenchmarkResult(
            validation_result=validation_result,
            benchmark_command=benchmark_command,
            run_count=self.run_count,
            stable=stable,
            variance_threshold=context.baseline.expected_variance,
            workload_summaries=workload_summaries,
        )

    def _build_benchmark_command(
        self,
        context: TuneContext,
        benchmark_target: str,
        contestant_name: str,
    ) -> str:
        config = context.benchmark_config
        env_assignments = [
            f"TARGET_HOST={shlex.quote(benchmark_target)}",
            f"RESULTS_DIR={shlex.quote(config.results_directory)}",
        ]
        benchmark_script = shlex.quote(config.script_path)
        return " ".join((*env_assignments, benchmark_script, shlex.quote(contestant_name)))

    def _load_workload_sample(
        self,
        benchmark_executor: CommandExecutor,
        context: TuneContext,
        contestant_name: str,
        workload_name: str,
        run_index: int,
    ) -> BenchmarkSample:
        result_path = (
            f"{context.benchmark_config.results_directory}/"
            f"{contestant_name}_{workload_name}.json"
        )
        command = f"cat {shlex.quote(result_path)}"
        command_result = benchmark_executor.run(command)
        if command_result.exit_code != 0:
            msg = f"Failed to load benchmark result {result_path}: {command_result.stderr}"
            raise ValueError(msg)
        payload = json.loads(command_result.stdout)
        result_data = cast(dict[str, Any], payload.get("results", {}))
        requests = cast(dict[str, Any], result_data.get("requests", {}))
        latency = cast(dict[str, Any], result_data.get("latency", {}))
        return BenchmarkSample(
            run_index=run_index,
            requests_per_second=float(requests.get("per_sec", 0.0)),
            total_requests=int(requests.get("total", 0)),
            average_latency_ms=self._parse_duration_ms(latency.get("avg", 0.0)),
        )

    def _summarize_workload(
        self,
        workload_name: str,
        samples: tuple[BenchmarkSample, ...],
        variance_threshold: float,
    ) -> BenchmarkWorkloadSummary:
        request_values = [sample.requests_per_second for sample in samples]
        total_values = [sample.total_requests for sample in samples]
        latency_values = [sample.average_latency_ms for sample in samples]
        request_median = float(median(request_values))
        total_median = int(median(total_values))
        latency_median = float(median(latency_values))
        relative_variance = self._calculate_relative_variance(request_values, request_median)
        return BenchmarkWorkloadSummary(
            workload_name=workload_name,
            samples=samples,
            median_requests_per_second=request_median,
            median_total_requests=total_median,
            median_latency_ms=latency_median,
            relative_variance=relative_variance,
            stable=relative_variance <= variance_threshold,
        )

    def _calculate_relative_variance(
        self,
        request_values: list[float],
        request_median: float,
    ) -> float:
        if request_median == 0.0:
            return 0.0
        spread = max(request_values) - min(request_values)
        return spread / request_median

    def _parse_duration_ms(self, value: object) -> float:
        if isinstance(value, int | float):
            return float(value)
        if not isinstance(value, str):
            return 0.0
        normalized = value.strip().lower()
        if normalized.endswith("ms"):
            return float(normalized.removesuffix("ms"))
        if normalized.endswith("us"):
            return float(normalized.removesuffix("us")) / 1000.0
        if normalized.endswith("s"):
            return float(normalized.removesuffix("s")) * 1000.0
        return float(normalized)

    def _build_contestant_name(
        self,
        context: TuneContext,
        iteration_number: int,
        run_index: int,
    ) -> str:
        session_id = "nosession"
        if context.artifacts is not None:
            session_id = context.artifacts.session_id
        base_name = context.benchmark_config.contestant_name
        return f"{base_name}_{session_id}_iter{iteration_number:03d}_run{run_index:02d}"

    def _build_run_summary(
        self,
        run_index: int,
        workload_samples: dict[str, list[BenchmarkSample]],
    ) -> str:
        lines = [f"Benchmark run {run_index}/{self.run_count} summary:"]
        for workload_name, samples in workload_samples.items():
            sample = samples[-1]
            lines.append(
                f"{workload_name:<10} "
                f"rps={sample.requests_per_second:>10.2f} "
                f"total={sample.total_requests:>10} "
                f"latency_ms={sample.average_latency_ms:>8.2f}"
            )
        return "\n".join(lines)
