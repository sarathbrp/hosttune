"""
Curated text for LLM hypothesis prompts.

Full captures live in preflight JSONL, snapshot artifacts, and the candidate catalog; this layer
only emits bounded summaries and digests for the model.
"""

from __future__ import annotations

from preflight.domain.kernel_sysctl_profile import format_sysctl_profile_compact
from preflight.domain.models import DiscoverySnapshot
from tune.application.benchmark_runtime_telemetry import (
    format_runtime_telemetry_digest,
    truncate_for_prompt,
)
from tune.application.snapshot_prompt_digest import format_snapshot_digest_for_prompt
from tune.domain.hypothesis_context import HypothesisContext
from tune.domain.hypothesis_models import (
    CandidateParameter,
    CandidateSource,
    HypothesisRecord,
    TunePhase,
)
from tune.domain.triage_models import TriageResult
from tune.domain.tune_context import TuneContext

_LIMIT_SOURCES = frozenset({CandidateSource.RUNTIME_PRLIMIT, CandidateSource.SYSTEMD_UNIT_LIMIT})

# sysctl line in prompts: smaller than console/preflight one-liners.
_LLM_SYSCTL_PROFILE_MAX_CHARS = 560
# Candidate rationale hints can be long; cap for the prompt only.
_LLM_CANDIDATE_HINT_MAX_CHARS = 140


PROCESS_STATE_PROMPT_KEYS: tuple[str, ...] = (
    "pid_file",
    "worker_processes",
    "open_connections",
    "worker_process_hint",
)


def hypothesis_prompt_layer_preamble() -> list[str]:
    return [
        "Context policy: you receive curated digests plus selected raw snippets "
        "(no full raw file dumps). "
        "Structured host facts come from preflight discovery; tunable knobs and measured "
        "`current` values come from the selectable candidate list (catalog). "
        "Snapshot and telemetry sections are truncated. Triage autofix runs before the LLM path; "
        "if you are seeing this prompt, no deterministic autofix was applied. "
        "Propose exactly one parameter that appears under 'Selectable candidates'.",
    ]


def format_preflight_digest_lines(preflight: DiscoverySnapshot) -> list[str]:
    net = preflight.network
    stor = preflight.storage
    irq = preflight.irq
    cgroup = preflight.cgroup
    return [
        f"- platform={preflight.platform_summary}",
        (
            "- cpu="
            f"{preflight.cpu.logical_cores} logical cores; "
            f"numa_nodes={preflight.cpu.numa_nodes}; "
            f"hyperthreading={preflight.cpu.hyperthreading_enabled}"
        ),
        (
            "- memory="
            f"swap_kib={preflight.memory.swap_total_kib}; "
            f"hugepages_total={preflight.memory.hugepages_total}; "
            f"thp={preflight.memory.transparent_hugepages_mode}"
        ),
        (
            "- kernel="
            f"selinux={preflight.kernel.selinux_mode}; "
            f"tuned={preflight.kernel.tuned_profile}; "
            f"sysctl_writable={preflight.kernel.sysctl_writable}"
        ),
        (
            "- kernel_sysctl_profile="
            + format_sysctl_profile_compact(
                preflight.kernel.sysctl_profile,
                max_chars=_LLM_SYSCTL_PROFILE_MAX_CHARS,
            )
            + " (preflight read of network/vm sysctls; YAML lists the tunable subset)"
        ),
        (
            "- network="
            f"{net.interface_name}; driver={net.driver_name}; "
            f"rings_rx={net.rx_ring_current}/{net.rx_ring_max}; "
            f"rings_tx={net.tx_ring_current}/{net.tx_ring_max}; "
            f"queues={net.combined_queues}"
        ),
        (
            "- irq="
            f"irqbalance_active={irq.irqbalance_active}; "
            f"nic_irq_cpus={irq.nic_irq_cpu_summary}"
        ),
        (
            "- cgroup="
            f"version={cgroup.cgroup_version}; "
            f"cpu_ctrl={cgroup.cpu_controller_available}; "
            f"mem_ctrl={cgroup.memory_controller_available}"
        ),
        (
            "- storage="
            f"{stor.device_name}; type={stor.device_type}; "
            f"scheduler={stor.scheduler}; "
            f"readahead_kb={stor.readahead_kb if stor.readahead_kb >= 0 else 'unknown'}"
        ),
    ]


def format_contract_digest_lines(tune_context: TuneContext) -> list[str]:
    svc = tune_context.onboard.service
    surface = svc.tunable_surface
    hints = svc.benchmark_hints
    allowed_directives = ", ".join(sorted(surface.allowed_directives))
    relevant_sysctls = ", ".join(
        f"{entry.name}({entry.priority_tier.value})" for entry in surface.relevant_sysctls
    )
    runtime_limit_names = ", ".join(sorted(surface.runtime_limits))
    systemd_unit_limit_names = ", ".join(sorted(surface.systemd_unit_limits))
    cgroup_control_names = ", ".join(sorted(surface.cgroup_resource_controls))
    guardrails = ", ".join(hints.guardrail_metrics)
    interference = ", ".join(hints.interference_sources)
    return [
        f"- service={tune_context.onboard.service_name}",
        f"- allowed_directives={allowed_directives or 'none'}",
        f"- relevant_sysctls={relevant_sysctls or 'none'}",
        f"- runtime_limits={runtime_limit_names or 'none'}",
        f"- systemd_unit_limits={systemd_unit_limit_names or 'none'}",
        f"- cgroup_resource_controls={cgroup_control_names or 'none'} (cgroup v2; no applier yet)",
        f"- health_probe={svc.health_check.probe_type.value}",
        f"- primary_metric={hints.primary_metric}",
        f"- guardrails={guardrails or 'none'}",
        f"- interference_sources={interference or 'none'}",
    ]


def format_baseline_digest_lines(tune_context: TuneContext) -> list[str]:
    b = tune_context.baseline
    workload_lines = [
        (
            f"- {w.workload_name}: rps={w.requests_per_second:.2f}; "
            f"latency_ms={w.average_latency_ms:.2f}; total={w.total_requests}"
        )
        for w in b.workload_results
    ]
    return [
        f"- target={b.benchmark_target}",
        f"- expected_variance={b.expected_variance:.2%}",
        f"- warmup_seconds={b.warmup_seconds}",
        "Workloads:",
        *(workload_lines or ["- none"]),
    ]


def format_limit_baseline_lines(candidates: tuple[CandidateParameter, ...]) -> list[str]:
    """Current values for prlimit and systemd unit limits across all catalog rows.

    Shown before the selectable candidate list so the LLM can detect no-op proposals
    (e.g. limit_nofile already at max) without scanning every candidate line.
    Includes deferred candidates so the full limit picture is visible in every phase.
    """
    lines = [
        f"- {c.parameter_key}: current={c.current_value or 'unknown'}; "
        f"max={c.max_value}; priority={c.priority_tier.value}"
        for c in candidates
        if c.source in _LIMIT_SOURCES
    ]
    return lines or ["- (no limit candidates in catalog)"]


def format_triage_lines(result: TriageResult) -> list[str]:
    autofix = (
        f"- autofix_action={result.autofix_action.parameter_key} -> "
        f"{result.autofix_action.proposed_value}; "
        f"reason={result.autofix_action.reason}"
        if result.autofix_action is not None
        else "- autofix_action=none"
    )
    recommendation = (
        f"- recommended_action={result.recommended_action.parameter_key} -> "
        f"{result.recommended_action.proposed_value}; "
        f"reason={result.recommended_action.reason}"
        if result.recommended_action is not None
        else "- recommended_action=none"
    )
    triggered_lines = [
        f"- {rule.section}:{rule.rule_id}; outcome={rule.outcome}; detail={rule.detail}"
        for rule in result.triggered_rules
    ]
    return [
        autofix,
        recommendation,
        f"- safe_candidate_subset={', '.join(result.safe_candidate_subset) or 'none'}",
        f"- suppressed_candidates={', '.join(result.suppressed_candidates) or 'none'}",
        f"- reboot_required_flags={', '.join(result.reboot_required_flags) or 'none'}",
        f"- escalation_reason={result.escalation_reason}",
        f"- rule_summary={result.non_triggered_summary}",
        "Triggered rules:",
        *(triggered_lines or ["- none"]),
    ]


def format_runtime_config_snippet(snapshot: str | None) -> str:
    if not snapshot:
        return "(runtime config unavailable)"
    lines = snapshot.splitlines()
    interesting_patterns = (
        "worker_processes",
        "worker_connections",
        "worker_rlimit_nofile",
        "access_log",
        "keepalive_",
        "sendfile",
        "net.core.",
        "listen",
    )
    selected = [
        line.rstrip() for line in lines if any(pattern in line for pattern in interesting_patterns)
    ]
    if not selected:
        selected = [line.rstrip() for line in lines[:16]]
    return truncate_for_prompt("\n".join(selected[:16]), 1200)


def format_service_yaml_reference_snippet(tune_context: TuneContext) -> str:
    directives = sorted(tune_context.onboard.service.tunable_surface.allowed_directives)
    sysctls = [
        entry.name for entry in tune_context.onboard.service.tunable_surface.relevant_sysctls
    ]
    runtime_limits = sorted(tune_context.onboard.service.tunable_surface.runtime_limits)
    systemd_limits = sorted(tune_context.onboard.service.tunable_surface.systemd_unit_limits)
    snippet = "\n".join(
        (
            f"identity.service_name: {tune_context.onboard.service_name}",
            f"identity.systemd_unit_name: "
            f"{tune_context.onboard.service.identity.systemd_unit_name}",
            f"tunable_surface.allowed_directives: {directives}",
            f"tunable_surface.relevant_sysctls: {sysctls}",
            f"tunable_surface.runtime_limits: {runtime_limits}",
            f"tunable_surface.systemd_unit_limits: {systemd_limits}",
            f"benchmark_hints.primary_metric: "
            f"{tune_context.onboard.service.benchmark_hints.primary_metric}",
            f"benchmark_hints.interference_sources: "
            f"{list(tune_context.onboard.service.benchmark_hints.interference_sources)}",
        )
    )
    return truncate_for_prompt(snippet, 1000)


_TELEMETRY_MAX_SECTION = 420

_PHASE_OBJECTIVES: dict[TunePhase, str] = {
    TunePhase.WIDE_SWEEP: "Explore broadly across domains with maximum diversity.",
    TunePhase.DOMAIN_FOCUS: "Focus on domains that have shown positive signal.",
    TunePhase.INTERACTION: "Explore interactions between promising parameters.",
    TunePhase.BOUNDARY_PUSH: "Push promising parameters toward safe limits.",
    TunePhase.EXPLOIT: "Refine around the current best configuration.",
    TunePhase.REBOOT_BATCH: "Apply reboot-required parameters as a batch.",
}


def _format_history_lines(history: tuple[HypothesisRecord, ...]) -> list[str]:
    _hist_eval_max = 180
    return [
        (
            f"- iteration={r.iteration_number}; phase={r.phase.value}; "
            f"parameter_key={r.hypothesis.parameter_key}; value={r.hypothesis.proposed_value}; "
            f"status={r.status.value}; "
            f"evaluation={truncate_for_prompt(r.evaluation_summary or '', _hist_eval_max)}"
        )
        for r in history
    ] or ["- none"]


def format_host_profile_digest_lines(context: HypothesisContext) -> list[str]:
    """Compact host profile summary for the rhel_expert prompt."""
    host_profile = context.tune_context.host_profile
    if host_profile is None:
        return ["- host_profile=(not configured)"]
    identity = host_profile.identity
    surface = host_profile.tunable_surface
    variant = identity.variant or "bare-metal"
    lines: list[str] = [
        f"- host_profile={identity.name} "
        f"(platform={identity.platform} {identity.version}, variant={variant})",
    ]
    if surface.network_queues is not None:
        nq = surface.network_queues
        lines.append(
            f"- host_network_queues: min={nq.min_combined} "
            f"max={'ncpus' if nq.max_combined == 0 else nq.max_combined}; "
            f"irq_affinity={nq.allow_irq_affinity}"
        )
    else:
        lines.append("- host_network_queues: not applicable (VM variant)")
    if surface.cpu_governor is not None:
        cg = surface.cpu_governor
        lines.append(
            f"- host_cpu_governor: preferred={cg.preferred_governor}; "
            f"allowed={list(cg.allowed_governors)}"
        )
    host_sysctl_names = [s.name for s in surface.host_sysctls]
    lines.append(f"- host_sysctls: {host_sysctl_names or 'none'}")
    return lines


def format_hybrid_hypothesis_prompt(
    context: HypothesisContext,
    triage: TriageResult,
) -> str:
    tune_context = context.tune_context
    phase_obj = _PHASE_OBJECTIVES.get(context.phase, "")
    candidate_lines = [format_candidate_line_for_llm(c) for c in context.candidates]
    deferred_lines = [format_candidate_line_for_llm(c) for c in context.deferred_candidates]
    capability_lines = [
        f"- {flag.name}: {flag.detail}"
        for flag in tune_context.preflight.capability_map.flags
        if flag.available
    ]
    telemetry_body = (
        context.last_benchmark_runtime_telemetry_digest
        if context.last_benchmark_runtime_telemetry_digest
        else format_runtime_telemetry_digest((), max_chars_per_section=_TELEMETRY_MAX_SECTION)
    )
    sections = [
        "You are the single hybrid hypothesizer for HostTune.",
        "A deterministic rule-based triage layer has already inspected the host and "
        "service context. Use triage signal as a hard priority input, then reason "
        "across service, runtime, kernel, "
        "network, and platform layers to choose exactly one change.",
        "Return strict JSON with keys: "
        '{"parameter_key": "...", "proposed_value": "...", "tuning_layer": "...", '
        '"apply_mode": "...", "rationale": "...", '
        '"expected_benchmark_impact": "...", "rollback_plan": "..."}',
        f"Current phase: {context.phase.value}",
        f"Phase objective: {phase_obj}",
        f"Iteration: {context.iteration_number}",
        *hypothesis_prompt_layer_preamble(),
        "Rule-based triage result:",
        *format_triage_lines(triage),
        "Host facts:",
        *format_preflight_digest_lines(tune_context.preflight),
        "Host profile:",
        *format_host_profile_digest_lines(context),
        "Available tunable surfaces:",
        *(capability_lines or ["- none"]),
        "Service contract:",
        *format_contract_digest_lines(tune_context),
        "Limit baselines:",
        *format_limit_baseline_lines(context.candidates + context.deferred_candidates),
        "Snapshot digest:",
        format_snapshot_digest_for_prompt(tune_context.snapshot),
        "Selected runtime config snippet:",
        format_runtime_config_snippet(tune_context.snapshot.runtime_state_output),
        "Selected service YAML reference snippet:",
        format_service_yaml_reference_snippet(tune_context),
        "Baseline workload results:",
        *format_baseline_digest_lines(tune_context),
        "Last benchmark runtime telemetry:",
        telemetry_body,
        "Current tune state:",
        f"- active_changes={', '.join(context.active_parameter_keys) or 'none'}",
        (
            f"- best_config="
            f"{', '.join(f'{k}={v}' for k, v in context.best_parameter_values) or 'none'}"
        ),
        "Prior history:",
        *_format_history_lines(context.history),
        "Selectable candidates:",
        *(candidate_lines or ["- none"]),
        "Deferred candidates (visibility only):",
        *(deferred_lines or ["- none"]),
        "Output rules:",
        "- choose exactly one selectable candidate",
        "- triage autofix is already resolved before this prompt; "
        "do not simulate or re-propose autofix logic",
        "- if recommended_action is present, treat it as the default choice and "
        "only override it when broader context makes it clearly unsafe or lower-value",
        "- use triggered signal rules as supporting context, not as direct "
        "parameter proposals unless they map to an actual selectable candidate",
        "- do not select suppressed candidates",
        "- do not select reboot-only candidates outside reboot_batch",
        "- do not invent unsupported knobs mentioned only in signal text",
        "- expected_benchmark_impact should predict primary metric movement concisely",
        "- rollback_plan should be human-readable and specific to the chosen knob",
    ]
    return "\n".join(sections)


def format_candidate_line_for_llm(candidate: CandidateParameter) -> str:
    """Compact candidate row; rationale hint truncated for prompt size."""
    hint = truncate_for_prompt(candidate.rationale_hint, _LLM_CANDIDATE_HINT_MAX_CHARS)
    return (
        f"- key={candidate.parameter_key}; domain={candidate.domain}; "
        f"tuning_layer={candidate.tuning_layer.value}; "
        f"availability={candidate.availability.value}; "
        f"parameter={candidate.parameter_name}; source={candidate.source.value}; "
        f"apply_mode={candidate.apply_mode.value}; priority={candidate.priority_tier.value}; "
        f"value_type={candidate.value_type.value}; "
        f"min={candidate.min_value}; max={candidate.max_value}; "
        f"allowed={candidate.allowed_values}; current={candidate.current_value}; "
        f"hint={hint}"
    )
