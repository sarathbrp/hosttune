from __future__ import annotations

from dataclasses import dataclass

from onboard.domain.models import OnboardResult
from onboard.infrastructure.service_compatibility_evaluator import ServiceCompatibilityEvaluator
from onboard.infrastructure.service_definition_loader import ServiceDefinitionLoader
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight.domain.models import CommandExecutor, DiscoverySnapshot
from preflight.interfaces.execution_logger import ExecutionLogger, NullExecutionLogger


@dataclass
class OnboardRunner:
    loader: ServiceDefinitionLoader
    validator: ServiceDefinitionValidator
    evaluator: ServiceCompatibilityEvaluator
    logger: ExecutionLogger = NullExecutionLogger()

    def run(
        self,
        service_name: str,
        preflight: DiscoverySnapshot,
        executor: CommandExecutor,
    ) -> OnboardResult:
        self.logger.stage_detail("onboard", f"Loading service driver: {service_name}")
        raw_definition = self.loader.load(service_name)
        self.logger.stage_detail("onboard", "Validating service driver schema")
        service_definition = self.validator.validate(raw_definition)
        self.logger.stage_detail("onboard", "Evaluating runtime compatibility")
        compatibility = self.evaluator.evaluate(preflight, service_definition, executor)
        self.logger.stage_detail(
            "onboard",
            f"Compatibility findings: {len(compatibility.findings)}",
        )
        if not compatibility.compatible:
            messages = [
                finding.message
                for finding in compatibility.findings
                if finding.severity.value == "error"
            ]
            msg = "Onboard compatibility check failed: " + "; ".join(messages)
            raise ValueError(msg)
        return OnboardResult(
            service_name=service_name,
            service=service_definition,
            compatibility=compatibility,
        )
