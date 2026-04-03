from __future__ import annotations

from dataclasses import dataclass

from onboard.domain.models import ApplyMode, DirectiveValueType
from tune.domain.hypothesis_models import CandidateParameter, CandidateSource
from tune.domain.tune_context import TuneContext


@dataclass
class CandidateCatalogBuilder:
    def build(self, context: TuneContext) -> tuple[CandidateParameter, ...]:
        candidates: list[CandidateParameter] = []
        candidates.extend(self._build_service_directive_candidates(context))
        candidates.extend(self._build_service_sysctl_candidates(context))
        return tuple(candidates)

    def _build_service_directive_candidates(
        self,
        context: TuneContext,
    ) -> list[CandidateParameter]:
        candidates: list[CandidateParameter] = []
        for (
            directive_name,
            constraint,
        ) in context.onboard.service.tunable_surface.allowed_directives.items():
            candidates.append(
                CandidateParameter(
                    parameter_key=f"service.directive.{directive_name}",
                    domain="service_config",
                    parameter_name=directive_name,
                    source=CandidateSource.SERVICE_DIRECTIVE,
                    value_type=constraint.value_type,
                    apply_mode=constraint.apply_mode,
                    allowed_values=constraint.allowed_values,
                    min_value=constraint.min_value,
                    max_value=constraint.max_value,
                    rationale_hint=(
                        f"Allowed nginx directive from service plugin for "
                        f"{context.onboard.service_name}"
                    ),
                )
            )
        return candidates

    def _build_service_sysctl_candidates(
        self,
        context: TuneContext,
    ) -> list[CandidateParameter]:
        kernel_tuning_available = self._capability_available(
            context,
            "kernel_sysctl_tuning",
        )
        if not kernel_tuning_available:
            return []

        candidates: list[CandidateParameter] = []
        for sysctl_name in context.onboard.service.tunable_surface.relevant_sysctls:
            candidates.append(
                CandidateParameter(
                    parameter_key=f"sysctl.{sysctl_name}",
                    domain="kernel_sysctl",
                    parameter_name=sysctl_name,
                    source=CandidateSource.SERVICE_SYSCTL,
                    value_type=DirectiveValueType.STRING,
                    apply_mode=self._resolve_sysctl_apply_mode(context),
                    allowed_values=(),
                    min_value=None,
                    max_value=None,
                    rationale_hint=(
                        f"Relevant sysctl for service {context.onboard.service_name} "
                        "and platform capability map"
                    ),
                )
            )
        return candidates

    def _capability_available(self, context: TuneContext, flag_name: str) -> bool:
        return any(
            flag.name == flag_name and flag.available
            for flag in context.preflight.capability_map.flags
        )

    def _resolve_sysctl_apply_mode(self, context: TuneContext) -> ApplyMode:
        restart_mode = context.onboard.service.restart.change_categories.get("kernel_network")
        if restart_mode is None:
            return ApplyMode.RELOAD
        return restart_mode
