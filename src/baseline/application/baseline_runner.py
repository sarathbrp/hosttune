from __future__ import annotations

from dataclasses import dataclass

from baseline.domain.models import BaselineResult
from onboard.domain.models import ServiceDefinition
from preflight.domain.models import BenchmarkRunner, CommandExecutor


@dataclass
class BaselineRunner:
    benchmark_runner: BenchmarkRunner

    def run(self, service: ServiceDefinition, executor: CommandExecutor) -> BaselineResult:
        result = self.benchmark_runner.run(executor)
        return BaselineResult(
            service_name=service.identity.service_name,
            benchmark_command=result.command,
            benchmark_result=result,
            expected_variance=service.benchmark_hints.expected_variance,
            warmup_seconds=service.benchmark_hints.warmup_seconds,
            guardrail_metrics=service.benchmark_hints.guardrail_metrics,
        )
