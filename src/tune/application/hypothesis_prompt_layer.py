"""
Curated text for LLM hypothesis prompts.

Full captures live in preflight JSONL, snapshot artifacts, and the candidate catalog; this layer
only emits bounded summaries and digests for the model.
"""

from __future__ import annotations

import re

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
        f"- cgroup_resource_controls={cgroup_control_names or 'none'}",
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
    alternate_recommendations = (
        "- alternate_recommendations="
        + ", ".join(
            f"{item.parameter_key} -> {item.proposed_value}"
            for item in result.alternate_recommendations
        )
        if result.alternate_recommendations
        else "- alternate_recommendations=none"
    )
    triggered_lines = [
        f"- {rule.section}:{rule.rule_id}; outcome={rule.outcome}; detail={rule.detail}"
        for rule in result.triggered_rules
    ]
    return [
        autofix,
        recommendation,
        alternate_recommendations,
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
        "open_file_cache",
        "sendfile",
        "gzip",
        "tcp_nopush",
        "limit_rate",
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


def format_current_performance_lines(
    context: "HypothesisContext",
    tune_context: TuneContext,
) -> list[str]:
    """Applied params + current RPS vs baseline so LLM sees what's been done and what changed."""
    if not context.current_workload_rps:
        return ["- no benchmark yet"]
    baseline_by_name = {
        w.workload_name: w.requests_per_second
        for w in tune_context.baseline.workload_results
    }
    param_summary = ", ".join(
        f"{k}={v}" for k, v in context.best_parameter_values
    ) or "none"
    lines = [f"- applied={param_summary}"]
    for workload_name, current_rps in context.current_workload_rps:
        baseline_rps = baseline_by_name.get(workload_name)
        if baseline_rps and baseline_rps > 0:
            pct = (current_rps - baseline_rps) / baseline_rps * 100
            sign = "+" if pct >= 0 else ""
            lines.append(
                f"- {workload_name}: rps={current_rps:,.0f} ({sign}{pct:.1f}% vs baseline)"
            )
        else:
            lines.append(f"- {workload_name}: rps={current_rps:,.0f}")
    return lines


def format_kb_best_rps_lines(context: "HypothesisContext", tune_context: TuneContext) -> list[str]:
    """All-time best RPS per workload from KB — the target to beat or match."""
    if not context.kb_best_workload_rps:
        return ["- no prior sessions"]
    baseline_by_name = {
        w.workload_name: w.requests_per_second
        for w in tune_context.baseline.workload_results
    }
    current_by_name = dict(context.current_workload_rps)
    lines = []
    for workload_name, best_rps in context.kb_best_workload_rps:
        current = current_by_name.get(workload_name, baseline_by_name.get(workload_name, 0))
        if best_rps > 0 and current > 0:
            pct_achieved = current / best_rps * 100
            gap = "✓ matched" if pct_achieved >= 95 else f"{pct_achieved:.0f}% achieved"
            lines.append(f"- {workload_name}: best={best_rps:,.0f} rps [{gap}]")
        else:
            lines.append(f"- {workload_name}: best={best_rps:,.0f} rps")
    return lines


def format_environment_blockers_lines(
    context: "HypothesisContext",
    tune_context: TuneContext,
) -> list[str]:
    """Surface known environment constraints/blockers so the LLM targets root causes.

    Covers: IRQ affinity, I/O scheduler, readahead, cgroup CPU/memory throttling,
    systemd file-descriptor and process limits.
    """
    lines: list[str] = []
    preflight = tune_context.preflight

    # ── IRQ affinity ─────────────────────────────────────────────────────────
    irq = getattr(preflight, "irq", None)
    if irq is not None:
        if not irq.irqbalance_active:
            summary = irq.nic_irq_cpu_summary or "unknown"
            cpus = set(summary.replace("-", ",").split(","))
            if len(cpus) == 1 and "unknown" not in summary:
                lines.append(
                    f"BLOCKER — IRQ pinned to single CPU ({summary}): irqbalance is stopped, "
                    "all NIC interrupts land on one core → softirq saturation. "
                    "Fix: restart irqbalance OR spread via /proc/irq/*/smp_affinity. "
                    "Candidate: platform.cpu_governor / irq_affinity_tuning."
                )
            else:
                lines.append(
                    f"IRQ: irqbalance stopped, NIC IRQs on CPUs [{summary}] — "
                    "manual affinity in effect; verify spread is adequate."
                )
        else:
            lines.append("IRQ: irqbalance active — NIC interrupts auto-distributed across CPUs (healthy).")

    # ── Storage I/O scheduler + readahead ────────────────────────────────────
    storage = getattr(preflight, "storage", None)
    if storage is not None:
        scheduler = getattr(storage, "scheduler", "unknown")
        readahead_kb = getattr(storage, "readahead_kb", 0)
        if scheduler and scheduler not in ("none", "noop", "unknown"):
            lines.append(
                f"BLOCKER — I/O scheduler={scheduler!r}: adds latency vs 'none' (NVMe passthrough). "
                "Fix: echo none > /sys/block/<dev>/queue/scheduler. "
                "Candidate: storage_scheduler_tuning."
            )
        else:
            lines.append(f"I/O scheduler={scheduler!r} (healthy for NVMe).")
        if readahead_kb > 0 and readahead_kb < 64:
            lines.append(
                f"BLOCKER — readahead={readahead_kb}KB (very low): each file read triggers "
                "extra I/O ops instead of prefetching. Fix: blockdev --setra 256 <dev>."
            )
        elif readahead_kb > 0:
            lines.append(f"Readahead={readahead_kb}KB.")

    # ── Cgroup + systemd resource caps (from candidate current values) ────────
    cgroup_keys = {
        "systemd.cgroup.cpu_quota_percent": ("CPUQuota", "%", 100),
        "systemd.cgroup.memory_max_mib": ("MemoryMax", "MiB", 4096),
    }
    unit_keys = {
        "systemd.unit.limit_nofile": ("LimitNOFILE", 65535),
        "systemd.unit.limit_nproc": ("LimitNPROC", 1024),
    }
    # Use full_candidates (unfiltered catalog) so blockers are visible in all
    # phases including EXPLOIT where phase-filtered candidates may omit them.
    candidate_map = {
        c.parameter_key: c
        for c in (context.full_candidates if context.full_candidates else context.candidates)
    }

    for key, (label, unit, threshold) in cgroup_keys.items():
        c = candidate_map.get(key)
        if c and c.current_value:
            try:
                val = float(c.current_value)
                if val < threshold:
                    lines.append(
                        f"BLOCKER — {label}={val}{unit}: severely throttled. "
                        f"Fix: raise via systemd.cgroup.{key.split('.')[-1]}. "
                        f"Candidate: {key} (current={val}{unit})."
                    )
            except ValueError:
                pass

    for key, (label, threshold) in unit_keys.items():
        c = candidate_map.get(key)
        if c and c.current_value:
            try:
                val = int(c.current_value)
                if val < threshold:
                    lines.append(
                        f"BLOCKER — {label}={val}: caps connections/processes. "
                        f"Fix: raise via {key}. Candidate active."
                    )
            except ValueError:
                pass

    return [f"- {l}" for l in lines] if lines else ["- no environment blockers detected"]


def format_working_hypothesis_lines(digest: str) -> list[str]:
    """Derive targeted hypothesis lines from the telemetry digest.

    Covers good/bad/edge-case scenarios across all telemetry sources:
    ss-s, softnet_stat, ethtool-S, sockstat, vmstat.
    """
    if not digest or "No runtime telemetry" in digest:
        return ["- no telemetry yet — first iteration or benchmark not completed"]

    hypotheses: list[str] = []

    # ── softnet_stat: time_squeeze (softirq budget) ──────────────────────────
    squeeze_match = re.search(r"time_squeeze total:\s*([\d,]+)", digest)
    drops_match = re.search(r"TOTAL drops:\s*([\d,]+)", digest)
    squeeze = int(squeeze_match.group(1).replace(",", "")) if squeeze_match else 0
    drops = int(drops_match.group(1).replace(",", "")) if drops_match else 0

    if squeeze > 50_000:
        hypotheses.append(
            f"CRITICAL softirq exhaustion (time_squeeze={squeeze:,}): "
            "kernel cannot drain NIC fast enough → target netdev_max_backlog, "
            "network.queue.combined (more NIC queues), network.ring.rx/tx"
        )
    elif squeeze > 5_000:
        hypotheses.append(
            f"Moderate softirq pressure (time_squeeze={squeeze:,}): "
            "consider raising netdev_max_backlog or expanding NIC queue count"
        )
    else:
        hypotheses.append(
            "Softirq healthy (no time_squeeze): kernel packet processing not the bottleneck"
        )

    if drops > 0:
        hypotheses.append(
            f"Softnet drops detected ({drops:,}): receive queue overflowing — "
            "raise netdev_max_backlog or NIC ring buffer (network.ring.rx)"
        )

    # ── ethtool -S: NIC-level drops/errors ───────────────────────────────────
    if "no NIC-level drops or errors" in digest:
        hypotheses.append("NIC layer clean: no rx_discards/errors — bottleneck is above the NIC")
    elif "NIC errors/drops" in digest:
        nic_err = re.search(r"NIC errors/drops:\s*([\d,]+)", digest)
        n = nic_err.group(1) if nic_err else "?"
        hypotheses.append(
            f"NIC drops detected ({n}): ring buffer too small → "
            "priority target: network.ring.rx and network.ring.tx (ethtool -G)"
        )

    # edge case: drops at kernel level but NOT at NIC → queue tuning, not ring buffer
    if drops > 0 and "no NIC-level drops" in digest:
        hypotheses.append(
            "Kernel drops WITHOUT NIC drops: packet loss is in the kernel queue, "
            "NOT the NIC ring → focus on netdev_max_backlog, not ring buffers"
        )

    # ── ss -s: TCP connection states ─────────────────────────────────────────
    estab_match = re.search(r"tcp_established:.*?max=(\d+)", digest)
    tw_match = re.search(r"tcp_timewait:.*?end=(\d+)", digest)
    tw_increasing = "↑increasing" in digest
    estab = int(estab_match.group(1)) if estab_match else 0
    tw_end = int(tw_match.group(1)) if tw_match else 0

    if estab > 5000:
        hypotheses.append(
            f"High concurrent connections (max_estab={estab:,}): "
            "worker_connections may be the ceiling — verify worker_connections >= estab_max"
        )
    elif estab < 100 and squeeze > 1000:
        hypotheses.append(
            "Low established connections despite softirq pressure: "
            "connections being rejected at listen backlog — somaxconn or tcp_max_syn_backlog too low"
        )

    if tw_increasing:
        hypotheses.append(
            f"TIME_WAIT accumulating (end={tw_end:,}, increasing): "
            "port exhaustion risk → ip_local_port_range (widen), tcp_tw_reuse=1, "
            "keepalive_requests (reduce churn)"
        )
    elif tw_end > 10_000:
        hypotheses.append(
            f"High TIME_WAIT (end={tw_end:,}, stable): "
            "large pool but not growing — keepalive_requests/keepalive_timeout "
            "could reduce churn further"
        )

    # ── sockstat: socket memory + timewait pressure ───────────────────────────
    sock_inuse = re.search(r"TCP_inuse=(\d+)", digest)
    sock_tw = re.search(r"tw=(\d+)", digest)
    sock_mem = re.search(r"mem=([\d.]+)MiB", digest)
    if sock_inuse:
        inuse = int(sock_inuse.group(1))
        if inuse > 10_000:
            hypotheses.append(
                f"Very high active sockets (TCP_inuse={inuse:,}): "
                "keepalive_timeout may be too long — connections accumulating"
            )
        elif inuse < 50:
            hypotheses.append(
                f"Very few active sockets (TCP_inuse={inuse}): "
                "benchmark not reaching nginx or connections being dropped early — "
                "check somaxconn and tcp_max_syn_backlog"
            )
    if sock_mem:
        mem = float(sock_mem.group(1))
        if mem > 512:
            hypotheses.append(
                f"TCP socket memory pressure ({mem:.0f}MiB): "
                "rmem_max/wmem_max may be over-allocated — "
                "or high connection count consuming buffer space"
            )
        else:
            hypotheses.append(f"TCP memory healthy ({mem:.0f}MiB): socket buffers not the bottleneck")

    # ── vmstat: CPU utilization + context switches ────────────────────────────
    vm_match = re.search(
        r"vmstat.*?cpu_us=(\d+)%\s+sy=(\d+)%\s+id=(\d+)%\s+wa=(\d+)%\s+cs=([\d,]+)/s",
        digest,
    )
    if vm_match:
        cpu_us = int(vm_match.group(1))
        cpu_sy = int(vm_match.group(2))
        cpu_id = int(vm_match.group(3))
        cpu_wa = int(vm_match.group(4))
        cs = int(vm_match.group(5).replace(",", ""))

        if cpu_id > 50:
            hypotheses.append(
                f"CPU has headroom (idle={cpu_id}%): "
                "can safely scale workers — consider worker_processes increase"
            )
        elif cpu_id < 10:
            hypotheses.append(
                f"CPU saturated (idle={cpu_id}%): "
                "reduce per-request overhead → access_log=off, gzip=off, "
                "sendfile=on — do NOT add more workers"
            )
        elif 10 <= cpu_id <= 30:
            hypotheses.append(
                f"CPU mostly loaded (idle={cpu_id}%): "
                "marginal headroom — optimize efficiency before scaling workers"
            )

        if cpu_sy > 20:
            hypotheses.append(
                f"High system CPU (sy={cpu_sy}%): excessive syscall overhead → "
                "sendfile=on (zero-copy), tcp_nopush=on, multi_accept=on"
            )

        if cpu_wa > 5:
            hypotheses.append(
                f"I/O wait detected (wa={cpu_wa}%): disk bottleneck → "
                "open_file_cache (cache fd/metadata), aio=threads (async I/O)"
            )

        if cs > 5_000_000:
            hypotheses.append(
                f"Very high context switches ({cs:,}/s): "
                "CPU ping-pong between cores → worker_cpu_affinity to pin workers"
            )
        elif cs > 1_000_000:
            hypotheses.append(
                f"Elevated context switches ({cs:,}/s): "
                "worker_cpu_affinity may help reduce scheduling overhead"
            )

    # ── cgroup CPU throttling ─────────────────────────────────────────────────
    cg_throttle = re.search(r"throttle_ratio=([\d.]+)%", digest)
    cg_healthy = "cgroup CPU throttle: not throttled" in digest
    cg_unavailable = "cgroup CPU" not in digest

    if cg_throttle:
        ratio = float(cg_throttle.group(1))
        if ratio >= 50:
            hypotheses.append(
                f"CRITICAL cgroup CPU throttling (throttle_ratio={ratio:.1f}%): "
                "nginx is paused more than half the time — CPUQuota IS the primary bottleneck. "
                "vmstat shows high idle because idle is system-wide across all cores; "
                "nginx is CPU-starved at the cgroup level. "
                "Fix: raise systemd.cgroup.cpu_quota_percent to 400+ "
                "(400% = 4 full cores on this 112-core host). "
                "Priority: fix this BEFORE any other parameter."
            )
        elif ratio >= 20:
            hypotheses.append(
                f"Significant cgroup CPU throttling (throttle_ratio={ratio:.1f}%): "
                "CPUQuota is meaningfully limiting nginx — ~{ratio:.0f}% of benchmark time "
                "nginx processes were paused. Raise systemd.cgroup.cpu_quota_percent."
            )
        elif ratio >= 5:
            hypotheses.append(
                f"Moderate cgroup CPU throttling (throttle_ratio={ratio:.1f}%): "
                "CPUQuota is a contributing factor but not the primary bottleneck."
            )
        else:
            hypotheses.append(
                f"Minor cgroup throttling (throttle_ratio={ratio:.1f}%) — CPUQuota not the bottleneck."
            )
    elif cg_healthy:
        hypotheses.append(
            "cgroup CPU: not throttled — CPUQuota is not limiting nginx throughput."
        )
    elif cg_unavailable:
        hypotheses.append(
            "cgroup CPU stats unavailable: check systemd.cgroup.cpu_quota_percent "
            "candidate for current CPUQuota value — if < 100, it IS the bottleneck "
            "even though vmstat shows high idle (idle is system-wide, not per-service)."
        )

    # ── edge case: all signals healthy ───────────────────────────────────────
    all_healthy = (
        squeeze == 0
        and drops == 0
        and "no NIC-level drops" in digest
        and (not vm_match or int(vm_match.group(3)) > 30)
    )
    if all_healthy:
        hypotheses.append(
            "All telemetry signals healthy: bottleneck is likely in nginx config "
            "(worker_connections ceiling, keepalive_requests, open_file_cache) "
            "or upstream network — try application-layer parameters"
        )

    return [f"- {h}" for h in hypotheses] if hypotheses else ["- insufficient telemetry signal"]


def format_prior_run_memory(tune_context: TuneContext) -> str:
    artifacts = tune_context.artifacts
    knowledge_base = tune_context.knowledge_base
    if artifacts is None or knowledge_base is None:
        return "- none"
    summary = knowledge_base.summarize_similar_runs(
        service_name=tune_context.onboard.service_name,
        cpu_logical_cores=tune_context.preflight.cpu.logical_cores,
        numa_nodes=tune_context.preflight.cpu.numa_nodes,
        platform_summary=tune_context.preflight.platform_summary,
        nic_driver=tune_context.preflight.network.driver_name,
        exclude_run_id=artifacts.session_id,
        limit=3,
    )
    return truncate_for_prompt(summary, 420) if summary else "- none"


_TELEMETRY_MAX_SECTION = 420

_PHASE_OBJECTIVES: dict[TunePhase, str] = {
    TunePhase.KNOWLEDGE_DRIVEN: "Apply KB-validated parameters with highest confidence first.",
    TunePhase.WIDE_SWEEP: "Explore broadly across domains with maximum diversity.",
    TunePhase.DOMAIN_FOCUS: "Focus on domains that have shown positive signal.",
    TunePhase.INTERACTION: "Explore interactions between promising parameters.",
    TunePhase.BOUNDARY_PUSH: "Push promising parameters toward safe limits.",
    TunePhase.EXPLOIT: "Refine around the current best configuration.",
    TunePhase.REBOOT_BATCH: "Apply reboot-required parameters as a batch.",
    TunePhase.RESOLVE: "Apply deterministic bottom-up fixes from dependency graph.",
    TunePhase.OPTIMIZE: (
        "Layers 1-4 resolved by dependency graph. "
        "Fine-tune remaining unresolved parameters."
    ),
}


def format_layer_status_lines(
    layer_statuses: tuple[tuple[str, str], ...],
) -> list[str]:
    """Format dependency layer statuses for the LLM prompt."""
    if not layer_statuses:
        return ["- (no dependency graph active)"]
    icons = {"ok": "OK", "fixed": "FIXED", "llm_deferred": "NEEDS_LLM", "rolled_back": "ROLLED_BACK"}
    return [
        f"- {name}: {icons.get(status, status)}"
        for name, status in layer_statuses
    ]


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


def format_compact_history_lines(history: tuple[HypothesisRecord, ...]) -> list[str]:
    if not history:
        return ["- none"]
    if len(history) <= 4:
        return _format_history_lines(history)

    older = history[:-3]
    recent = history[-3:]
    accepted = sum(1 for item in older if item.status.value == "accepted")
    promising = sum(1 for item in older if item.status.value == "promising")
    inconclusive = sum(1 for item in older if item.status.value == "inconclusive")
    rejected_like = sum(
        1
        for item in older
        if item.status.value in {"rejected", "rejected_pre_apply", "failed_validation"}
    )
    positive_parameters = sorted(
        {
            item.hypothesis.parameter_key
            for item in older
            if item.status.value in {"accepted", "promising"}
        }
    )
    return [
        (
            f"- older_history_summary=count={len(older)}; accepted={accepted}; "
            f"promising={promising}; inconclusive={inconclusive}; "
            f"rejected_like={rejected_like}; "
            f"positive_parameters={', '.join(positive_parameters) or 'none'}"
        ),
        "- recent_history:",
        *_format_history_lines(recent),
    ]


def format_blocked_prior_pairs(
    history: tuple[HypothesisRecord, ...],
    prior_blocked_pairs: tuple[tuple[str, str], ...] = (),
) -> list[str]:
    seen: list[str] = []
    added: set[tuple[str, str]] = set()
    # KB-blocked pairs from prior similar runs (shown first).
    for key, value in prior_blocked_pairs:
        pair = (key, value)
        if pair in added:
            continue
        added.add(pair)
        seen.append(f"{key}={value} (failed in prior run)")
    # Current-session history pairs.
    for item in history:
        pair = (item.hypothesis.parameter_key, item.hypothesis.proposed_value)
        if pair in added:
            continue
        added.add(pair)
        seen.append(f"{pair[0]}={pair[1]}")
    if not seen:
        return ["- none"]
    if len(seen) <= 8:
        return [f"- {value}" for value in seen]
    return [
        *(f"- {value}" for value in seen[:8]),
        f"- ... ({len(seen) - 8} more blocked pair(s))",
    ]


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


def discover_unmodeled_directives(
    runtime_state_output: str | None,
    modeled_directives: set[str],
    *,
    max_results: int = 10,
) -> list[str]:
    """Parse nginx -T output and surface directives not in the YAML model.

    Returns prompt lines describing candidate gaps the operator may want
    to add to the service YAML for future tuning.
    """
    if not runtime_state_output or not runtime_state_output.strip():
        return ["- (no runtime state available for directive discovery)"]
    import re

    # Match top-level directives: `name value;` (not nested in location/upstream).
    directive_pattern = re.compile(r"^\s{0,4}(\w+)\s+[^;]+;", re.MULTILINE)
    found: set[str] = set()
    for match in directive_pattern.finditer(runtime_state_output):
        name = match.group(1)
        # Skip structural keywords and common non-tunable directives.
        if name in {
            "server",
            "location",
            "upstream",
            "include",
            "listen",
            "root",
            "index",
            "error_page",
            "return",
            "proxy_pass",
            "try_files",
            "server_name",
            "charset",
            "default_type",
            "log_format",
            "pid",
            "user",
            "error_log",
            "types",
            "mime",
        }:
            continue
        found.add(name)
    unmodeled = sorted(found - modeled_directives)
    if not unmodeled:
        return ["- (all detected directives are already modeled)"]
    lines = [
        f"- {name} (found in runtime config but not in service YAML)"
        for name in unmodeled[:max_results]
    ]
    if len(unmodeled) > max_results:
        lines.append(f"- ... and {len(unmodeled) - max_results} more")
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
        "Prior similar run memory:",
        format_prior_run_memory(tune_context),
        "Baseline workload results:",
        *format_baseline_digest_lines(tune_context),
        "Last benchmark runtime telemetry:",
        telemetry_body,
        "Working hypothesis (derived from telemetry):",
        *format_working_hypothesis_lines(telemetry_body),
        "Current tune state:",
        f"- active_changes={', '.join(context.active_parameter_keys) or 'none'}",
        (
            f"- best_config="
            f"{', '.join(f'{k}={v}' for k, v in context.best_parameter_values) or 'none'}"
        ),
        "Prior history:",
        *format_compact_history_lines(context.history),
        "Blocked prior parameter/value pairs:",
        *format_blocked_prior_pairs(context.history, context.prior_blocked_pairs),
        "Unmodeled directives (found in runtime config but not in YAML):",
        *discover_unmodeled_directives(
            tune_context.snapshot.runtime_state_output,
            set(tune_context.onboard.service.tunable_surface.allowed_directives),
        ),
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
        "- alternate_recommendations are deterministic fallback options; prefer them only when "
        "the primary recommended_action is clearly inferior in the current context",
        "- use triggered signal rules as supporting context, not as direct "
        "parameter proposals unless they map to an actual selectable candidate",
        "- the service YAML reference may mention supported knobs that are not selectable in "
        "this iteration; only choose from 'Selectable candidates'",
        "- do not repeat a parameter/value pair that already appears under "
        "'Blocked prior parameter/value pairs'",
        "- do not select suppressed candidates",
        "- do not select reboot-only candidates outside reboot_batch",
        "- do not invent unsupported knobs mentioned only in signal text",
        "- expected_benchmark_impact should predict primary metric movement concisely",
        "- rollback_plan should be human-readable and specific to the chosen knob",
    ]
    return "\n".join(sections)


_PERFORMANCE_SEMANTICS: dict[str, str] = {
    # ── Hard throttles / caps ────────────────────────────────────────────
    "service.directive.limit_rate": (
        "THROTTLE: per-connection bandwidth cap; any non-zero value "
        "(e.g. 5m=5MB/s) is a hard throughput limiter. Set 0 to disable."
    ),
    "service.directive.limit_rate_after": (
        "THROTTLE: bandwidth cap kicks in after this many bytes per response."
    ),
    "systemd.unit.cpu_quota_percent": (
        "THROTTLE: values <100% impose a hard CPU ceiling via cgroup; "
        "the service is forcibly throttled regardless of available cores."
    ),
    "systemd.unit.memory_max_mib": (
        "HARD CEILING: exceeded = OOM kill by the kernel; "
        "this is a capacity wall, not a performance tuning knob."
    ),
    "sysctl.net.core.somaxconn": (
        "BACKLOG CAP: TCP listen queue; low values silently drop SYN "
        "packets under load. Raise to match worker_connections."
    ),
    "sysctl.net.core.netdev_max_backlog": (
        "PACKET DROP WALL: per-CPU NIC RX queue; exceeded = silent packet "
        "drops before the kernel even sees the connection."
    ),
    "sysctl.net.ipv4.ip_local_port_range": (
        "PORT EXHAUSTION: narrow range limits concurrent outbound connections; "
        "exhaustion = 'Cannot assign requested address' errors."
    ),
    # ── I/O and overhead ─────────────────────────────────────────────────
    "service.directive.access_log": (
        "I/O OVERHEAD: disk writes per request; 'off' eliminates logging I/O."
    ),
    "service.directive.open_file_cache": (
        "CACHE: caches file descriptors and metadata; " "'off' means re-open on every request."
    ),
    # ── Counterintuitive tradeoffs ───────────────────────────────────────
    "service.directive.gzip": (
        "CPU-BANDWIDTH TRADEOFF: 'on' trades CPU cycles for smaller "
        "responses; beneficial on slow links, overhead on fast local networks."
    ),
    "service.directive.tcp_nopush": (
        "LATENCY TRADEOFF: 'on' delays sending to pack full frames; "
        "reduces syscalls but adds micro-latency per response."
    ),
    "service.directive.keepalive_requests": (
        "CONNECTION REUSE: higher = fewer TCP handshakes but longer-lived "
        "connections consuming memory. Balance with worker_connections."
    ),
    "service.directive.worker_connections": (
        "PER-WORKER LIMIT: this is per worker process, not global. "
        "Effective max = worker_processes * worker_connections."
    ),
    "service.directive.aio": (
        "ASYNC I/O MODE: 'threads' = userspace thread pool (recommended); "
        "'on' = kernel AIO (Linux only, limited). 'off' = synchronous."
    ),
    "sysctl.vm.swappiness": (
        "MEMORY PRESSURE: higher values swap more aggressively, "
        "adding latency to memory-bound workloads."
    ),
    # ── Shadow limits ────────────────────────────────────────────────────
    "systemd.unit.limit_nofile": (
        "SHADOW LIMIT: effective fd limit = MIN(systemd LimitNOFILE, "
        "prlimit nofile_soft). Both must be raised together."
    ),
    "runtime.prlimit.nofile_soft": (
        "SHADOW LIMIT: effective fd limit = MIN(systemd LimitNOFILE, "
        "prlimit nofile_soft). Both must be raised together."
    ),
}


def _format_compressed_triage_lines(result: TriageResult) -> list[str]:
    """Triage lines with signal-only rules and safe_candidate_subset stripped."""
    autofix = (
        f"- autofix={result.autofix_action.parameter_key} -> "
        f"{result.autofix_action.proposed_value}; {result.autofix_action.reason}"
        if result.autofix_action is not None
        else "- autofix=none"
    )
    recommendation = (
        f"- recommend={result.recommended_action.parameter_key} -> "
        f"{result.recommended_action.proposed_value}; "
        f"{result.recommended_action.reason}"
        if result.recommended_action is not None
        else "- recommend=none"
    )
    alternates = (
        "- alternates="
        + ", ".join(
            f"{item.parameter_key} -> {item.proposed_value}"
            for item in result.alternate_recommendations
        )
        if result.alternate_recommendations
        else "- alternates=none"
    )
    # Only include recommend/autofix rules, skip signals.
    actionable_rules = [
        f"- {rule.section}:{rule.rule_id}; {rule.detail}"
        for rule in result.triggered_rules
        if rule.outcome in ("recommend", "autofix")
    ]
    lines = [autofix, recommendation, alternates]
    if result.suppressed_candidates:
        lines.append(f"- suppressed={', '.join(result.suppressed_candidates)}")
    if actionable_rules:
        lines.append("Actionable rules:")
        lines.extend(actionable_rules)
    return lines


def _format_compressed_candidate(candidate: CandidateParameter) -> str:
    """Compact candidate: key, current, priority, constraints, hint + semantics."""
    hint = truncate_for_prompt(candidate.rationale_hint, _LLM_CANDIDATE_HINT_MAX_CHARS)
    semantic = _PERFORMANCE_SEMANTICS.get(candidate.parameter_key, "")
    semantic_tag = f" \u26a0 {semantic}" if semantic else ""
    parts = [f"- {candidate.parameter_key}"]
    parts.append(f"current={candidate.current_value}")
    parts.append(f"priority={candidate.priority_tier.value}")
    parts.append(f"layer={candidate.tuning_layer.value}")
    parts.append(f"mode={candidate.apply_mode.value}")
    if candidate.value_type.value != "string":
        parts.append(f"type={candidate.value_type.value}")
    if candidate.min_value is not None or candidate.max_value is not None:
        parts.append(f"range=[{candidate.min_value},{candidate.max_value}]")
    if candidate.allowed_values:
        parts.append(f"allowed={candidate.allowed_values}")
    parts.append(f"hint={hint}{semantic_tag}")
    return "; ".join(parts)


def format_compressed_hypothesis_prompt(
    context: HypothesisContext,
    triage: TriageResult,
) -> str:
    """Token-optimized prompt: strips static/redundant sections, compresses candidates."""
    tune_context = context.tune_context
    phase_obj = _PHASE_OBJECTIVES.get(context.phase, "")
    candidate_lines = [_format_compressed_candidate(c) for c in context.candidates]
    telemetry_body = (
        context.last_benchmark_runtime_telemetry_digest
        if context.last_benchmark_runtime_telemetry_digest
        else format_runtime_telemetry_digest((), max_chars_per_section=_TELEMETRY_MAX_SECTION)
    )
    # Compact host summary: single line instead of full breakdown.
    cpu = tune_context.preflight.cpu
    net = tune_context.preflight.network
    host_summary = (
        f"- host={tune_context.preflight.platform_summary}; "
        f"cpu={cpu.logical_cores}c/{cpu.numa_nodes}n; "
        f"nic={net.interface_name}/{net.driver_name}; "
        f"kernel_sysctl_writable={tune_context.preflight.kernel.sysctl_writable}"
    )
    sections = [
        "You are the hybrid hypothesizer for HostTune. "
        "Return strict JSON: "
        '{"parameter_key", "proposed_value", "tuning_layer", '
        '"apply_mode", "rationale", "expected_benchmark_impact", '
        '"rollback_plan"}',
        f"Phase: {context.phase.value} — {phase_obj}",
        f"Iteration: {context.iteration_number}",
        "Triage:",
        *_format_compressed_triage_lines(triage),
        "Host:",
        host_summary,
        "Environment blockers (known constraints that cap performance — fix these first):",
        *format_environment_blockers_lines(context, tune_context),
        "Contract:",
        f"- service={tune_context.onboard.service_name}; "
        f"metric={tune_context.onboard.service.benchmark_hints.primary_metric}",
        "Runtime config:",
        format_runtime_config_snippet(tune_context.snapshot.runtime_state_output),
        "Prior runs:",
        format_prior_run_memory(tune_context),
        "Baseline:",
        *format_baseline_digest_lines(tune_context),
        "Current performance (after applied changes):",
        *format_current_performance_lines(context, tune_context),
        "Historical best (KB top-1, same service — target to match or beat):",
        *format_kb_best_rps_lines(context, tune_context),
        "Telemetry:",
        telemetry_body,
        "Working hypothesis (derived from telemetry — validate and act on these signals):",
        *format_working_hypothesis_lines(telemetry_body),
        "State:",
        f"- active={', '.join(context.active_parameter_keys) or 'none'}",
        (f"- best=" f"{', '.join(f'{k}={v}' for k, v in context.best_parameter_values) or 'none'}"),
        *(
            ["Dependency layers (bottom-up):", *format_layer_status_lines(context.layer_statuses)]
            if context.layer_statuses
            else []
        ),
        "History:",
        *format_compact_history_lines(context.history),
        "Blocked pairs:",
        *format_blocked_prior_pairs(context.history, context.prior_blocked_pairs),
        "Candidates:",
        *(candidate_lines or ["- none"]),
        "Deferred candidates (reboot_batch only):",
        *(
            [f"- {c.parameter_key}; current={c.current_value}" for c in context.deferred_candidates]
            or ["- none"]
        ),
        "Rules: pick one candidate; follow triage recommend; "
        "no blocked pairs; no suppressed; "
        "predict metric impact; give rollback plan.",
    ]
    return "\n".join(sections)


def format_candidate_line_for_llm(candidate: CandidateParameter) -> str:
    """Compact candidate row; rationale hint truncated for prompt size."""
    hint = truncate_for_prompt(candidate.rationale_hint, _LLM_CANDIDATE_HINT_MAX_CHARS)
    semantic = _PERFORMANCE_SEMANTICS.get(candidate.parameter_key, "")
    semantic_tag = f" ⚠ {semantic}" if semantic else ""
    return (
        f"- key={candidate.parameter_key}; domain={candidate.domain}; "
        f"tuning_layer={candidate.tuning_layer.value}; "
        f"availability={candidate.availability.value}; "
        f"parameter={candidate.parameter_name}; source={candidate.source.value}; "
        f"apply_mode={candidate.apply_mode.value}; priority={candidate.priority_tier.value}; "
        f"value_type={candidate.value_type.value}; "
        f"min={candidate.min_value}; max={candidate.max_value}; "
        f"allowed={candidate.allowed_values}; current={candidate.current_value}; "
        f"current_source={candidate.current_value_source}; "
        f"hint={hint}{semantic_tag}"
    )
