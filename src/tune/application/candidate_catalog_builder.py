from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

from onboard.domain.models import ApplyMode, DirectiveValueType
from preflight.domain.models import CommandExecutor
from tune.domain.hypothesis_models import CandidateParameter, CandidateSource
from tune.domain.tune_context import TuneContext


@dataclass
class CandidateCatalogBuilder:
    def build(
        self,
        context: TuneContext,
        executor: CommandExecutor | None = None,
    ) -> tuple[CandidateParameter, ...]:
        candidates: list[CandidateParameter] = []
        candidates.extend(self._build_service_directive_candidates(context, executor))
        candidates.extend(self._build_service_sysctl_candidates(context, executor))
        return tuple(candidates)

    def _build_service_directive_candidates(
        self,
        context: TuneContext,
        executor: CommandExecutor | None,
    ) -> list[CandidateParameter]:
        candidates: list[CandidateParameter] = []
        for (
            directive_name,
            constraint,
        ) in context.onboard.service.tunable_surface.allowed_directives.items():
            if constraint.apply_mode is ApplyMode.REBOOT:
                continue
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
                    current_value=self._read_service_directive_current_value(
                        context=context,
                        directive_name=directive_name,
                        executor=executor,
                    ),
                )
            )
        return candidates

    def _build_service_sysctl_candidates(
        self,
        context: TuneContext,
        executor: CommandExecutor | None,
    ) -> list[CandidateParameter]:
        kernel_tuning_available = self._capability_available(
            context,
            "kernel_sysctl_tuning",
        )
        if not kernel_tuning_available:
            return []

        candidates: list[CandidateParameter] = []
        for sysctl_name in context.onboard.service.tunable_surface.relevant_sysctls:
            apply_mode = self._resolve_sysctl_apply_mode(context)
            if apply_mode is ApplyMode.REBOOT:
                continue
            candidates.append(
                CandidateParameter(
                    parameter_key=f"sysctl.{sysctl_name}",
                    domain="kernel_sysctl",
                    parameter_name=sysctl_name,
                    source=CandidateSource.SERVICE_SYSCTL,
                    value_type=DirectiveValueType.STRING,
                    apply_mode=apply_mode,
                    allowed_values=(),
                    min_value=None,
                    max_value=None,
                    rationale_hint=(
                        f"Relevant sysctl for service {context.onboard.service_name} "
                        "and platform capability map"
                    ),
                    current_value=self._read_sysctl_current_value(sysctl_name, executor),
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

    def _read_service_directive_current_value(
        self,
        context: TuneContext,
        directive_name: str,
        executor: CommandExecutor | None,
    ) -> str | None:
        if executor is None:
            return None
        config_path = self._resolve_config_path(context)
        command = (
            f"grep -E '^\\s*{re.escape(directive_name)}\\s+' "
            f"{shlex.quote(config_path)} | tail -n 1"
        )
        result = executor.run(command)
        if result.exit_code != 0 or result.stdout == "":
            return None
        line = result.stdout.strip().rstrip(";")
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            return None
        return parts[1].strip()

    def _read_sysctl_current_value(
        self,
        sysctl_name: str,
        executor: CommandExecutor | None,
    ) -> str | None:
        if executor is None:
            return None
        result = executor.run(f"sysctl -n {shlex.quote(sysctl_name)}")
        if result.exit_code != 0:
            return None
        return result.stdout.strip()

    def _resolve_config_path(self, context: TuneContext) -> str:
        for path in context.onboard.service.identity.config_paths:
            if path.endswith(".conf"):
                return path
        msg = "No concrete nginx config file path is available for directive candidate."
        raise ValueError(msg)
