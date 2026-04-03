from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any, cast

from baseline.domain.models import (
    BaselineResult,
    BenchmarkConfig,
    WorkloadBenchmarkResult,
)
from onboard.domain.models import ServiceDefinition
from preflight.domain.models import CommandExecutor, TargetConfig
from preflight.interfaces.execution_logger import ExecutionLogger, NullExecutionLogger


@dataclass
class BaselineRunner:
    benchmark_config: BenchmarkConfig
    logger: ExecutionLogger = NullExecutionLogger()

    def run(
        self,
        service: ServiceDefinition,
        benchmark_executor: CommandExecutor,
        dut_target: TargetConfig,
    ) -> BaselineResult:
        benchmark_target = self._resolve_target_host(dut_target)
        self.logger.stage_detail("baseline", f"Resolved DUT target: {benchmark_target}")
        benchmark_command = self._build_benchmark_command(benchmark_target)
        self.logger.stage_detail("baseline", "Running benchmark workloads")
        benchmark_run = benchmark_executor.run(benchmark_command)
        if benchmark_run.exit_code != 0:
            msg = f"Benchmark command failed: {benchmark_run.stderr or benchmark_run.stdout}"
            raise ValueError(msg)

        self.logger.stage_detail("baseline", "Loading workload result files")
        workload_results = tuple(
            self._load_workload_result(benchmark_executor, workload_name)
            for workload_name in self.benchmark_config.workloads
        )
        self.logger.stage_detail(
            "baseline",
            self._format_workload_summary(workload_results),
        )
        self.logger.stage_detail("baseline", "Running baseline comparison")
        comparison_output = self._run_comparison(benchmark_executor)
        return BaselineResult(
            service_name=service.identity.service_name,
            benchmark_command=benchmark_command,
            benchmark_target=benchmark_target,
            workload_results=workload_results,
            expected_variance=service.benchmark_hints.expected_variance,
            warmup_seconds=service.benchmark_hints.warmup_seconds,
            guardrail_metrics=service.benchmark_hints.guardrail_metrics,
            comparison_output=comparison_output,
        )

    def _resolve_target_host(self, dut_target: TargetConfig) -> str:
        if hasattr(dut_target, "host"):
            return str(dut_target.host)
        return "127.0.0.1"

    def _build_benchmark_command(self, benchmark_target: str) -> str:
        env_assignments = [
            f"TARGET_HOST={shlex.quote(benchmark_target)}",
            f"RESULTS_DIR={shlex.quote(self.benchmark_config.results_directory)}",
        ]
        benchmark_script = shlex.quote(self.benchmark_config.script_path)
        contestant_name = shlex.quote(self.benchmark_config.contestant_name)
        return " ".join((*env_assignments, benchmark_script, contestant_name))

    def _load_workload_result(
        self,
        benchmark_executor: CommandExecutor,
        workload_name: str,
    ) -> WorkloadBenchmarkResult:
        result_path = (
            f"{self.benchmark_config.results_directory}/"
            f"{self.benchmark_config.contestant_name}_{workload_name}.json"
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

        return WorkloadBenchmarkResult(
            workload_name=workload_name,
            result_path=result_path,
            requests_per_second=float(requests.get("per_sec", 0.0)),
            total_requests=int(requests.get("total", 0)),
            average_latency_ms=self._parse_duration_ms(latency.get("avg", 0.0)),
        )

    def _run_comparison(self, benchmark_executor: CommandExecutor) -> str | None:
        compare_script = self.benchmark_config.compare_script_path
        if compare_script is None:
            return None

        command = (
            f"{shlex.quote(compare_script)} "
            f"{shlex.quote(self.benchmark_config.contestant_name)}"
        )
        command_result = benchmark_executor.run(command)
        if command_result.exit_code != 0:
            return command_result.stderr or command_result.stdout
        return command_result.stdout

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

    def _format_workload_summary(
        self,
        workload_results: tuple[WorkloadBenchmarkResult, ...],
    ) -> str:
        header = "Workload summary:"
        rows = [
            (
                f"{result.workload_name:<10} "
                f"rps={result.requests_per_second:>10.2f} "
                f"total={result.total_requests:>10} "
                f"latency_ms={result.average_latency_ms:>8.2f}"
            )
            for result in workload_results
        ]
        return "\n".join((header, *rows))
