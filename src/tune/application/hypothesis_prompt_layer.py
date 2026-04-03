"""
Curated text for LLM hypothesis prompts.

Full captures live in preflight JSONL, snapshot artifacts, and the candidate catalog; this layer
only emits bounded summaries and digests for the model.
"""

from __future__ import annotations

from preflight.domain.kernel_sysctl_profile import format_sysctl_profile_compact
from preflight.domain.models import DiscoverySnapshot
from tune.application.benchmark_runtime_telemetry import truncate_for_prompt
from tune.domain.hypothesis_models import CandidateParameter, CandidateSource
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
