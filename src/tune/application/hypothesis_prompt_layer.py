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
from tune.domain.tune_context import TuneContext
from tune.domain.tuning_layer import TuningLayer

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
        "Context policy: you receive curated digests only (no raw file dumps). "
        "Structured host facts come from preflight discovery; tunable knobs and measured "
        "`current` values come from the selectable candidate list (catalog). "
        "Snapshot and telemetry sections are truncated. Propose only parameters that appear "
        "under 'Selectable candidates'.",
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


_SERVICE_LAYERS = frozenset({TuningLayer.SERVICE, TuningLayer.RUNTIME})
_RHEL_LAYERS = frozenset({TuningLayer.KERNEL, TuningLayer.NETWORK})
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


def format_service_expert_prompt(context: HypothesisContext) -> str:
    """Prompt for the service configuration expert (service/runtime layer candidates only)."""
    tune_context = context.tune_context
    phase_obj = _PHASE_OBJECTIVES.get(context.phase, "")
    service_candidates = [c for c in context.candidates if c.tuning_layer in _SERVICE_LAYERS]
    service_deferred = [c for c in context.deferred_candidates if c.tuning_layer in _SERVICE_LAYERS]
    limit_candidates = context.candidates + context.deferred_candidates
    candidate_lines = [format_candidate_line_for_llm(c) for c in service_candidates]
    deferred_lines = [format_candidate_line_for_llm(c) for c in service_deferred]
    sections = [
        "You are the service configuration expert for HostTune.",
        "Your domain: service-level and runtime parameters "
        "(application directives, fd limits, process limits, keepalive settings).",
        "Select ONE candidate from the service/runtime list below.",
        "Return strict JSON: "
        '{"parameter_key": "...", "proposed_value": "...", "rationale": "...", '
        '"confidence": "high|medium|low"}',
        f"Current phase: {context.phase.value}",
        f"Phase objective: {phase_obj}",
        f"Iteration: {context.iteration_number}",
        "Service contract (your domain):",
        *format_contract_digest_lines(tune_context),
        "Limit baselines (current prlimit/systemd-unit values):",
        *format_limit_baseline_lines(limit_candidates),
        "Snapshot digest:",
        format_snapshot_digest_for_prompt(tune_context.snapshot),
        "Baseline workload results:",
        *format_baseline_digest_lines(tune_context),
        "Prior history:",
        *_format_history_lines(context.history),
        "Service/runtime candidates (YOUR DOMAIN — pick from this list only):",
        *(candidate_lines or ["- none"]),
        "Deferred service/runtime candidates (reboot-required):",
        *(deferred_lines or ["- none"]),
    ]
    return "\n".join(sections)


def format_rhel_expert_prompt(context: HypothesisContext) -> str:
    """Prompt for the RHEL system tuning expert (kernel/network layer candidates only)."""
    tune_context = context.tune_context
    phase_obj = _PHASE_OBJECTIVES.get(context.phase, "")
    rhel_candidates = [c for c in context.candidates if c.tuning_layer in _RHEL_LAYERS]
    rhel_deferred = [c for c in context.deferred_candidates if c.tuning_layer in _RHEL_LAYERS]
    candidate_lines = [format_candidate_line_for_llm(c) for c in rhel_candidates]
    deferred_lines = [format_candidate_line_for_llm(c) for c in rhel_deferred]
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
        "You are the RHEL system tuning expert for HostTune.",
        "Your domain: kernel parameters, network stack, IRQ affinity, cgroup, and storage.",
        "Select ONE candidate from the kernel/network list below.",
        "Return strict JSON: "
        '{"parameter_key": "...", "proposed_value": "...", "rationale": "...", '
        '"confidence": "high|medium|low"}',
        f"Current phase: {context.phase.value}",
        f"Phase objective: {phase_obj}",
        f"Iteration: {context.iteration_number}",
        "Host facts (your domain):",
        *format_preflight_digest_lines(tune_context.preflight),
        "Available tunable surfaces (capability flags):",
        *(capability_lines or ["- none"]),
        "Last benchmark runtime telemetry (ss -s, softnet_stat, ethtool -S; truncated):",
        telemetry_body,
        "Prior history:",
        *_format_history_lines(context.history),
        "Kernel/network candidates (YOUR DOMAIN — pick from this list only):",
        *(candidate_lines or ["- none"]),
        "Deferred kernel/network candidates (reboot-required):",
        *(deferred_lines or ["- none"]),
    ]
    return "\n".join(sections)


def format_debate_planner_prompt(
    context: HypothesisContext,
    expert_recommendations: list[str],
    full_prompt: str,
) -> str:
    """Prompt for the debate planner that synthesizes both expert recommendations."""
    phase_obj = _PHASE_OBJECTIVES.get(context.phase, "")
    candidate_lines = [format_candidate_line_for_llm(c) for c in context.candidates]
    deferred_lines = [format_candidate_line_for_llm(c) for c in context.deferred_candidates]
    best_config = ", ".join(f"{k}={v}" for k, v in context.best_parameter_values) or "none"
    active_changes = ", ".join(context.active_parameter_keys) or "none"
    labeled_recommendations = [
        f"[{label}]: {rec}"
        for label, rec in zip(
            ["service_agent", "rhel_expert"], expert_recommendations, strict=False
        )
    ]
    sections = [
        "You are the tuning decision planner for HostTune.",
        "Two domain experts have made recommendations for the next tuning action.",
        "Apply BOTH when they are from different tuning layers "
        "(e.g. service directive + kernel sysctl) — orthogonal changes combine safely.",
        "Apply only ONE if: both experts picked the same layer, one returned null/error, "
        "or there is a resource conflict.",
        "All parameter_key values MUST appear in the selectable candidates list below.",
        "Return a JSON ARRAY (even for a single item): "
        '[{"parameter_key": "...", "proposed_value": "...", "rationale": "..."}, ...]',
        f"Current phase: {context.phase.value}",
        f"Phase objective: {phase_obj}",
        f"Iteration: {context.iteration_number}",
        "Current tune state:",
        f"- active_changes={active_changes}",
        f"- best_config={best_config}",
        "Expert recommendations:",
        *labeled_recommendations,
        "Selectable candidates (ALL domains — final answer must use one of these keys):",
        *(candidate_lines or ["- none"]),
        "Deferred candidates (reboot-required; selectable only in reboot_batch phase):",
        *(deferred_lines or ["- none"]),
        "Prior history:",
        *_format_history_lines(context.history),
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
