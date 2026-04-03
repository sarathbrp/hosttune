from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass

from onboard.domain.models import ApplyMode, DirectiveValueType, PriorityTier
from preflight.domain.models import CommandExecutor
from tune.application.apply_coordinator import PrlimitApplier, SystemdUnitLimitApplier
from tune.application.candidate_value_reads import (
    read_network_ring_catalog_current,
    read_service_directive_catalog_current,
    read_sysctl_catalog_current,
)
from tune.domain.hypothesis_models import (
    CandidateAvailability,
    CandidateParameter,
    CandidateSource,
)
from tune.domain.tune_context import TuneContext
from tune.domain.tuning_layer import resolve_tuning_layer

_SUPPORTED_RUNTIME_PRLIMIT_NAMES = frozenset({"nofile_soft"})
_SUPPORTED_SYSTEMD_UNIT_LIMIT_NAMES = frozenset({"limit_nofile", "limit_nproc"})


@dataclass
class CandidateCatalogBuilder:
    """Build selectable rows with live probes and preflight fallbacks.

    See `candidate_value_reads` for read semantics. Runtime prlimit and
    systemd limits stay live-only. Broad context in preflight/snapshot.
    """

    def build(
        self,
        context: TuneContext,
        executor: CommandExecutor | None = None,
    ) -> tuple[CandidateParameter, ...]:
        _log = logging.getLogger(__name__)
        candidates: list[CandidateParameter] = []
        builders = [
            ("service_directives", self._build_service_directive_candidates),
            ("service_sysctls", self._build_service_sysctl_candidates),
            ("network_rings", self._build_network_ring_candidates),
            ("runtime_prlimit", self._build_runtime_prlimit_candidates),
            ("systemd_unit_limits", self._build_systemd_unit_limit_candidates),
            ("host_profile", self._build_host_profile_candidates),
        ]
        for builder_name, builder in builders:
            try:
                rows = builder(context, executor)
                candidates.extend(rows)
            except Exception as exc:
                _log.warning(
                    "Catalog builder '%s' failed and returned 0 candidates: %s",
                    builder_name,
                    exc,
                    exc_info=True,
                )
        # Warn when expected service directives are missing (helps diagnose Issues).
        if context.onboard.service.tunable_surface.allowed_directives and not any(
            c.source.value == "service_directive" for c in candidates
        ):
            _log.warning(
                "No service directive candidates built despite %d allowed_directives — "
                "check executor connectivity and config_path resolution.",
                len(context.onboard.service.tunable_surface.allowed_directives),
            )
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
                    current_value=read_service_directive_catalog_current(
                        executor=executor,
                        config_path=self._resolve_config_path(context),
                        directive_name=directive_name,
                        runtime_state_output=context.snapshot.runtime_state_output,
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
            pkey = f"sysctl.{sysctl_name}"
            if apply_mode is ApplyMode.REBOOT:
                availability = CandidateAvailability.DEFERRED
                hint = (
                    f"Relevant sysctl for {context.onboard.service_name}; "
                    "policy marks kernel_network as reboot — deferred to reboot_batch phase "
                    "when engagement allows reboot."
                )
            else:
                availability = CandidateAvailability.ACTIVE
                hint = (
                    f"Relevant sysctl for service {context.onboard.service_name} "
                    "and platform capability map"
                )
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
                    rationale_hint=hint,
                    current_value=read_sysctl_catalog_current(
                        sysctl_name,
                        executor,
                        context.preflight.kernel.sysctl_profile,
                    ),
                    availability=availability,
                )
            )
        return candidates

    def _build_network_ring_candidates(
        self,
        context: TuneContext,
        executor: CommandExecutor | None,
    ) -> list[CandidateParameter]:
        if not self._capability_available(context, "network_ring_buffer_tuning"):
            return []

        candidates: list[CandidateParameter] = []
        interface_name = context.preflight.network.interface_name
        ring_tier = context.onboard.service.tunable_surface.network_ring_priority_tier
        ring_layer_override = context.onboard.service.tunable_surface.network_ring_tuning_layer
        for parameter_name, preflight_current, max_value in (
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
            if max_value <= preflight_current:
                continue
            catalog_current = read_network_ring_catalog_current(
                parameter_name,
                interface_name,
                executor,
                preflight_current,
            )
            try:
                current_int = int(catalog_current)
            except ValueError:
                current_int = preflight_current
                catalog_current = str(preflight_current)
            if max_value <= current_int:
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
                    min_value=current_int,
                    max_value=max_value,
                    rationale_hint=(
                        f"NIC ring buffer tuning on {interface_name} "
                        f"for {parameter_name} from {current_int} to {max_value}"
                    ),
                    current_value=catalog_current,
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

    def _build_systemd_unit_limit_candidates(
        self,
        context: TuneContext,
        executor: CommandExecutor | None,
    ) -> list[CandidateParameter]:
        if not self._capability_available(context, "systemd_unit_limit_tuning"):
            return []
        candidates: list[CandidateParameter] = []
        for (
            limit_name,
            constraint,
        ) in context.onboard.service.tunable_surface.systemd_unit_limits.items():
            if limit_name not in _SUPPORTED_SYSTEMD_UNIT_LIMIT_NAMES:
                continue
            if constraint.apply_mode is ApplyMode.REBOOT:
                continue
            parameter_key = f"systemd.unit.{limit_name}"
            candidates.append(
                CandidateParameter(
                    parameter_key=parameter_key,
                    domain="runtime",
                    tuning_layer=resolve_tuning_layer(parameter_key, constraint.tuning_layer),
                    parameter_name=limit_name,
                    source=CandidateSource.SYSTEMD_UNIT_LIMIT,
                    value_type=constraint.value_type,
                    apply_mode=constraint.apply_mode,
                    priority_tier=constraint.priority_tier,
                    allowed_values=constraint.allowed_values,
                    forbidden_values=constraint.forbidden_values,
                    min_value=constraint.min_value,
                    max_value=constraint.max_value,
                    rationale_hint=(
                        f"systemd unit limit {limit_name} on "
                        f"{context.onboard.service.identity.systemd_unit_name}"
                    ),
                    current_value=self._read_systemd_unit_limit_current(
                        executor, context, limit_name
                    ),
                )
            )
        return candidates

    def _read_systemd_unit_limit_current(
        self,
        executor: CommandExecutor | None,
        context: TuneContext,
        limit_name: str,
    ) -> str | None:
        if executor is None:
            return None
        prop = SystemdUnitLimitApplier.property_name(limit_name)
        unit = context.onboard.service.identity.systemd_unit_name
        try:
            return SystemdUnitLimitApplier.read_property_value(executor, unit, prop)
        except ValueError:
            logging.getLogger(__name__).warning(
                "Failed to read systemd unit limit %s; no-op check may be skipped",
                limit_name,
            )
            return None

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

    def _build_host_profile_candidates(
        self,
        context: TuneContext,
        executor: CommandExecutor | None,
    ) -> list[CandidateParameter]:
        """Generate candidates from the optional host profile."""
        if context.host_profile is None:
            return []
        surface = context.host_profile.tunable_surface
        existing_keys = set()  # deduplicate against service-level sysctl candidates
        candidates: list[CandidateParameter] = []

        # ── NIC queue expansion ──────────────────────────────────────────────
        if surface.network_queues is not None:
            nq = surface.network_queues
            pkey = "network.queue.combined"
            current_queues = context.preflight.network.combined_queues
            # Resolve max: 0 = use logical_cores; always cap against the NIC's
            # actual hardware preset maximum read live from ethtool -l.
            yaml_max = nq.max_combined or context.preflight.cpu.logical_cores
            hardware_max = yaml_max  # will be overridden if executor is available
            current_val: str | None = str(current_queues) if executor else None
            if executor:
                iface = shlex.quote(context.preflight.network.interface_name)
                ethtool_result = executor.run(
                    f"ethtool -l {iface} 2>/dev/null | "
                    "awk '/Pre-set maximums/{found=1} found && /Combined/{print $2; exit}'"
                )
                if ethtool_result.exit_code == 0 and ethtool_result.stdout.strip().isdigit():
                    hardware_max = min(yaml_max, int(ethtool_result.stdout.strip()))
                current_result = executor.run(
                    f"ethtool -l {iface} 2>/dev/null | "
                    "awk '/Current hardware settings/{found=1} found && /Combined/{print $2; exit}'"
                )
                if current_result.exit_code == 0 and current_result.stdout.strip().isdigit():
                    current_val = current_result.stdout.strip()
            if hardware_max > current_queues:
                candidates.append(
                    CandidateParameter(
                        parameter_key=pkey,
                        domain="network",
                        tuning_layer=resolve_tuning_layer(pkey, None),
                        parameter_name="combined",
                        source=CandidateSource.HOST_NIC_QUEUE,
                        value_type=DirectiveValueType.INTEGER,
                        apply_mode=nq.apply_mode,
                        priority_tier=nq.priority_tier,
                        allowed_values=(),
                        forbidden_values=(),
                        min_value=nq.min_combined,
                        max_value=hardware_max,
                        rationale_hint=(
                            f"NIC hardware max combined queues={hardware_max}; "
                            f"currently {current_queues} — expand to parallelize packet processing"
                        ),
                        current_value=current_val,
                    )
                )

        # ── CPU governor ─────────────────────────────────────────────────────
        if surface.cpu_governor is not None:
            cg = surface.cpu_governor
            pkey = "platform.cpu_governor.scaling_governor"
            current_gov: str | None = None
            if executor:
                result = executor.run(
                    "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null"
                )
                value = result.stdout.strip()
                current_gov = value if result.exit_code == 0 and value else None
            candidates.append(
                CandidateParameter(
                    parameter_key=pkey,
                    domain="platform",
                    tuning_layer=resolve_tuning_layer(pkey, None),
                    parameter_name="scaling_governor",
                    source=CandidateSource.HOST_CPU_GOVERNOR,
                    value_type=DirectiveValueType.ENUM,
                    apply_mode=cg.apply_mode,
                    priority_tier=cg.priority_tier,
                    allowed_values=cg.allowed_governors,
                    forbidden_values=cg.forbidden_governors,
                    min_value=None,
                    max_value=None,
                    rationale_hint=(
                        f"CPU governor controls frequency scaling; "
                        f"preferred={cg.preferred_governor}"
                    ),
                    current_value=current_gov,
                )
            )

        # ── Host-level sysctls (deduplicate against service sysctl candidates) ──
        service_sysctl_keys = {
            f"sysctl.{entry.name}"
            for entry in context.onboard.service.tunable_surface.relevant_sysctls
        }
        for entry in surface.host_sysctls:
            pkey = f"sysctl.{entry.name}"
            if pkey in service_sysctl_keys or pkey in existing_keys:
                continue  # already in catalog from service YAML
            existing_keys.add(pkey)
            current_val = read_sysctl_catalog_current(
                entry.name, executor, context.preflight.kernel.sysctl_profile
            )
            candidates.append(
                CandidateParameter(
                    parameter_key=pkey,
                    domain="kernel_sysctl",
                    tuning_layer=resolve_tuning_layer(pkey, None),
                    parameter_name=entry.name,
                    source=CandidateSource.HOST_SYSCTL,
                    value_type=DirectiveValueType.STRING,
                    apply_mode=self._resolve_sysctl_apply_mode(context),
                    priority_tier=entry.priority_tier,
                    allowed_values=(),
                    forbidden_values=(),
                    min_value=None,
                    max_value=None,
                    rationale_hint=entry.rationale_hint,
                    current_value=current_val,
                )
            )

        return candidates
