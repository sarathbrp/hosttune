from __future__ import annotations

from dataclasses import dataclass

from onboard.domain.models import OnboardResult
from onboard.infrastructure.service_compatibility_evaluator import ServiceCompatibilityEvaluator
from onboard.infrastructure.service_definition_loader import ServiceDefinitionLoader
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight.domain.models import CommandExecutor, DiscoverySnapshot


@dataclass
class OnboardRunner:
    loader: ServiceDefinitionLoader
    validator: ServiceDefinitionValidator
    evaluator: ServiceCompatibilityEvaluator

    def run(
        self,
        service_name: str,
        preflight: DiscoverySnapshot,
        executor: CommandExecutor,
    ) -> OnboardResult:
        raw_definition = self.loader.load(service_name)
        service_definition = self.validator.validate(raw_definition)
        compatibility = self.evaluator.evaluate(preflight, service_definition, executor)
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
