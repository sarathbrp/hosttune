from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass

from onboard.domain.models import ApplyMode, DirectiveValueType, PriorityTier
from preflight.domain.models import CommandExecutor
from tune.application.apply_coordinator import PrlimitApplier
from tune.domain.hypothesis_models import CandidateParameter, CandidateSource
from tune.domain.tune_context import TuneContext
from tune.domain.tuning_layer import resolve_tuning_layer

_SUPPORTED_RUNTIME_PRLIMIT_NAMES = frozenset({"nofile_soft"})


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
        candidates.extend(self._build_network_ring_candidates(context))
        candidates.extend(self._build_runtime_prlimit_candidates(context, executor))
        return tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    self._priority_rank(candidate.priority_tier),
                    candidate.domain,
                    candidate.parameter_key,
                ),
            )
        )

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
            domain = "runtime" if directive_name == "worker_rlimit_nofile" else "service_config"
            parameter_key = f"service.directive.{directive_name}"
            candidates.append(
                CandidateParameter(
                    parameter_key=parameter_key,
                    domain=domain,
                    tuning_layer=resolve_tuning_layer(parameter_key, constraint.tuning_layer),
                    parameter_name=directive_name,
                    source=CandidateSource.SERVICE_DIRECTIVE,
                    value_type=constraint.value_type,
                    apply_mode=constraint.apply_mode,
                    priority_tier=constraint.priority_tier,
                    allowed_values=constraint.allowed_values,
                    forbidden_values=constraint.forbidden_values,
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
        for sysctl_entry in context.onboard.service.tunable_surface.relevant_sysctls:
            sysctl_name = sysctl_entry.name
            apply_mode = self._resolve_sysctl_apply_mode(context)
            if apply_mode is ApplyMode.REBOOT:
                continue
            pkey = f"sysctl.{sysctl_name}"
            candidates.append(
                CandidateParameter(
                    parameter_key=pkey,
                    domain="kernel_sysctl",
                    tuning_layer=resolve_tuning_layer(pkey, sysctl_entry.tuning_layer),
                    parameter_name=sysctl_name,
                    source=CandidateSource.SERVICE_SYSCTL,
                    value_type=DirectiveValueType.STRING,
                    apply_mode=apply_mode,
                    priority_tier=sysctl_entry.priority_tier,
                    allowed_values=(),
                    forbidden_values=(),
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

    def _build_network_ring_candidates(
        self,
        context: TuneContext,
    ) -> list[CandidateParameter]:
        if not self._capability_available(context, "network_ring_buffer_tuning"):
            return []

        candidates: list[CandidateParameter] = []
        interface_name = context.preflight.network.interface_name
        ring_tier = context.onboard.service.tunable_surface.network_ring_priority_tier
        ring_layer_override = context.onboard.service.tunable_surface.network_ring_tuning_layer
        for parameter_name, current_value, max_value in (
            (
                "rx",
                context.preflight.network.rx_ring_current,
                context.preflight.network.rx_ring_max,
            ),
            (
                "tx",
                context.preflight.network.tx_ring_current,
                context.preflight.network.tx_ring_max,
            ),
        ):
            if max_value <= current_value:
                continue
            pkey = f"network.ring.{parameter_name}"
            candidates.append(
                CandidateParameter(
                    parameter_key=pkey,
                    domain="network",
                    tuning_layer=resolve_tuning_layer(pkey, ring_layer_override),
                    parameter_name=parameter_name,
                    source=CandidateSource.PLATFORM_CAPABILITY,
                    value_type=DirectiveValueType.INTEGER,
                    apply_mode=ApplyMode.RELOAD,
                    priority_tier=ring_tier,
                    allowed_values=(),
                    forbidden_values=(),
                    min_value=current_value,
                    max_value=max_value,
                    rationale_hint=(
                        f"NIC ring buffer tuning on {interface_name} "
                        f"for {parameter_name} from {current_value} to {max_value}"
                    ),
                    current_value=str(current_value),
                )
            )
        return candidates

    def _build_runtime_prlimit_candidates(
        self,
        context: TuneContext,
        executor: CommandExecutor | None,
    ) -> list[CandidateParameter]:
        if not self._capability_available(context, "runtime_prlimit_tuning"):
            return []
        pid_file = context.onboard.service.snapshot.process_state.pid_file
        if not pid_file:
            return []
        candidates: list[CandidateParameter] = []
        for (
            limit_name,
            constraint,
        ) in context.onboard.service.tunable_surface.runtime_limits.items():
            if limit_name not in _SUPPORTED_RUNTIME_PRLIMIT_NAMES:
                continue
            if constraint.apply_mode is ApplyMode.REBOOT:
                continue
            parameter_key = f"runtime.prlimit.{limit_name}"
            candidates.append(
                CandidateParameter(
                    parameter_key=parameter_key,
                    domain="runtime",
                    tuning_layer=resolve_tuning_layer(parameter_key, constraint.tuning_layer),
                    parameter_name=limit_name,
                    source=CandidateSource.RUNTIME_PRLIMIT,
                    value_type=constraint.value_type,
                    apply_mode=constraint.apply_mode,
                    priority_tier=constraint.priority_tier,
                    allowed_values=constraint.allowed_values,
                    forbidden_values=constraint.forbidden_values,
                    min_value=constraint.min_value,
                    max_value=constraint.max_value,
                    rationale_hint=(
                        f"Runtime prlimit for {limit_name} on service "
                        f"{context.onboard.service_name} main PID (pid_file={pid_file})"
                    ),
                    current_value=self._read_current_nofile_soft(executor, context),
                )
            )
        return candidates

    def _read_current_nofile_soft(
        self,
        executor: CommandExecutor | None,
        context: TuneContext,
    ) -> str | None:
        if executor is None:
            return None
        try:
            return PrlimitApplier.current_nofile_soft(executor, context)
        except ValueError:
            logging.getLogger(__name__).warning(
                "Failed to read current NOFILE soft limit; no-op check will be skipped"
            )
            return None

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

    def _priority_rank(self, tier: PriorityTier) -> int:
        order = {
            PriorityTier.HIGH: 0,
            PriorityTier.MEDIUM: 1,
            PriorityTier.LOW: 2,
        }
        return order[tier]
