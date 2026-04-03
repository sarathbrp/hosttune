from __future__ import annotations

import json
import logging
import re
import shlex
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from preflight.domain.models import CommandExecutor
from tune.domain.benchmark_models import BenchmarkTelemetrySample

_log = logging.getLogger(__name__)

# Key NIC counters extracted from ethtool -S for the digest.
_KEY_ETHTOOL_COUNTERS = (
    "rx_ucast_packets",
    "rx_discards",
    "rx_errors",
    "tx_ucast_packets",
    "tx_errors",
    "tx_discards",
)


def truncate_for_prompt(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n... [truncated, {len(text) - max_chars} chars omitted]"


def _parse_ss_tcp(ss_output: str) -> dict[str, int | None]:
    """Extract TCP connection counts from 'ss -s' output."""
    result: dict[str, int | None] = {"estab": None, "closed": None, "timewait": None}
    match = re.search(
        r"TCP:\s+\d+\s+\(estab\s+(\d+),\s+closed\s+(\d+),\s+orphaned\s+\d+,\s+timewait\s+(\d+)\)",
        ss_output,
    )
    if match:
        result["estab"] = int(match.group(1))
        result["closed"] = int(match.group(2))
        result["timewait"] = int(match.group(3))
    return result


def _parse_softnet_line(line: str) -> list[int]:
    """Parse a single softnet_stat hex line into a list of integers."""
    try:
        return [int(field, 16) for field in line.strip().split()]
    except ValueError:
        return []


def _parse_ethtool_counters(ethtool_output: str) -> dict[str, int]:
    """Extract key NIC counters from ethtool -S output."""
    result: dict[str, int] = {}
    for line in ethtool_output.splitlines():
        for key in _KEY_ETHTOOL_COUNTERS:
            match = re.search(rf"\b{re.escape(key)}:\s+(\d+)", line)
            if match:
                result[key] = int(match.group(1))
    return result


def format_runtime_telemetry_digest(
    samples: tuple[BenchmarkTelemetrySample, ...],
    *,
    max_chars_per_section: int = 1200,  # kept for API compatibility
) -> str:
    """Return a compact statistical digest of all telemetry samples for the LLM prompt.

    Instead of dumping every raw sample (which bloats the prompt), this aggregates:
    - ss -s: min/max/trend for TCP connection states across all samples
    - softnet_stat: per-CPU drop and time-squeeze deltas (first→last sample)
    - ethtool -S: key NIC counter deltas (first→last sample)

    Raw samples are saved separately to JSON by save_telemetry_samples_json().
    """
    if not samples:
        return (
            "No runtime telemetry samples yet (first iteration, or telemetry disabled / "
            "benchmark host unreachable)."
        )

    n = len(samples)
    lines: list[str] = [f"Runtime telemetry: {n} sample(s) during benchmark load."]

    # ── ss -s: TCP connection state trends ───────────────────────────────────
    ss_stats = [_parse_ss_tcp(s.ss_s) for s in samples if s.ss_s]
    if ss_stats:
        estabs = [s["estab"] for s in ss_stats if s["estab"] is not None]
        timewaits = [s["timewait"] for s in ss_stats if s["timewait"] is not None]
        closeds = [s["closed"] for s in ss_stats if s["closed"] is not None]
        lines.append("ss -s summary (TCP during load):")
        if estabs:
            lines.append(f"  tcp_established: min={min(estabs)} max={max(estabs)} end={estabs[-1]}")
        if timewaits:
            trend = "↑increasing" if timewaits[-1] > timewaits[0] else "→stable"
            note = (
                " — port reuse pressure, consider ip_local_port_range"
                if timewaits[-1] > timewaits[0]
                else ""
            )
            lines.append(
                f"  tcp_timewait: min={min(timewaits)} max={max(timewaits)} end={timewaits[-1]}"
                f" {trend}{note}"
            )
        if closeds:
            lines.append(f"  tcp_closed: min={min(closeds)} max={max(closeds)} end={closeds[-1]}")

    # ── softnet_stat: drop + time_squeeze deltas (first→last) ────────────────
    first_soft = samples[0].softnet_stat
    last_soft = samples[-1].softnet_stat
    if first_soft and last_soft:
        first_lines = [ln for ln in first_soft.splitlines() if ln.strip()]
        last_lines = [ln for ln in last_soft.splitlines() if ln.strip()]
        n_cpus = min(len(first_lines), len(last_lines))
        if n_cpus:
            lines.append(f"softnet_stat deltas first→last ({n_cpus} CPU(s)):")
            total_drops = 0
            total_squeeze = 0
            max_squeeze = 0
            max_squeeze_cpu = 0
            for cpu_id in range(n_cpus):
                f_vals = _parse_softnet_line(first_lines[cpu_id])
                l_vals = _parse_softnet_line(last_lines[cpu_id])
                if len(f_vals) >= 3 and len(l_vals) >= 3:
                    dropped = max(0, l_vals[1] - f_vals[1])
                    squeeze = max(0, l_vals[2] - f_vals[2])
                    total_drops += dropped
                    total_squeeze += squeeze
                    if squeeze > max_squeeze:
                        max_squeeze = squeeze
                        max_squeeze_cpu = cpu_id
            if total_drops == 0:
                lines.append("  → no softnet drops (receive queue healthy)")
            else:
                lines.append(
                    f"  → TOTAL drops: {total_drops} across {n_cpus} CPUs"
                    " (ring buffer or netdev_max_backlog pressure)"
                )
            if total_squeeze == 0:
                lines.append("  → no time_squeeze (softirq budget not exhausted)")
            else:
                lines.append(
                    f"  → time_squeeze total: {total_squeeze}"
                    f" (worst: cpu{max_squeeze_cpu} +{max_squeeze})"
                    " — kernel softirq budget exhausted, consider sysctl tuning"
                )

    # ── ethtool -S: key NIC counter deltas ───────────────────────────────────
    first_et = samples[0].ethtool_s
    last_et = samples[-1].ethtool_s
    if first_et and last_et:
        first_c = _parse_ethtool_counters(first_et)
        last_c = _parse_ethtool_counters(last_et)
        if first_c and last_c:
            lines.append("ethtool NIC counter deltas (first→last sample):")
            for key in _KEY_ETHTOOL_COUNTERS:
                if key in first_c and key in last_c:
                    delta = last_c[key] - first_c[key]
                    lines.append(f"  {key}: +{delta}")
            error_delta = sum(
                max(0, last_c.get(k, 0) - first_c.get(k, 0))
                for k in ("rx_discards", "rx_errors", "tx_errors", "tx_discards")
            )
            if error_delta == 0:
                lines.append("  → no NIC-level drops or errors")
            else:
                lines.append(
                    f"  → NIC errors/drops: {error_delta} total"
                    " — consider ring buffer tuning (ethtool -G)"
                )

    return "\n".join(lines)


def save_telemetry_samples_json(
    samples: tuple[BenchmarkTelemetrySample, ...],
    path: Path,
) -> None:
    """Persist raw telemetry samples to JSON for post-processing and debugging."""
    try:
        data = [
            {
                "sequence": s.sequence,
                "ss_s": s.ss_s,
                "softnet_stat": s.softnet_stat,
                "ethtool_s": s.ethtool_s,
                "errors": list(s.errors),
            }
            for s in samples
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        _log.warning("Failed to save telemetry JSON to %s: %s", path, exc)


@dataclass
class BenchmarkRuntimeTelemetryCollector:
    """Runs lightweight host probes for accept-queue / softnet / NIC drop hints."""

    network_interface: str

    def capture_sample(self, executor: CommandExecutor, sequence: int) -> BenchmarkTelemetrySample:
        errors: list[str] = []
        r_ss = executor.run("ss -s")
        ss_s = r_ss.stdout if r_ss.exit_code == 0 else ""
        if r_ss.exit_code != 0:
            errors.append(f"ss -s: exit={r_ss.exit_code} stderr={r_ss.stderr.strip()!r}")

        r_soft = executor.run("cat /proc/net/softnet_stat")
        softnet = r_soft.stdout if r_soft.exit_code == 0 else ""
        if r_soft.exit_code != 0:
            errors.append(f"softnet_stat: exit={r_soft.exit_code} stderr={r_soft.stderr.strip()!r}")

        ethtool_s = ""
        iface = self.network_interface.strip()
        if iface:
            cmd = f"ethtool -S {shlex.quote(iface)}"
            r_et = executor.run(cmd)
            ethtool_s = r_et.stdout if r_et.exit_code == 0 else ""
            if r_et.exit_code != 0:
                errors.append(f"ethtool -S: exit={r_et.exit_code} stderr={r_et.stderr.strip()!r}")
        else:
            errors.append("preflight network interface name empty; ethtool skipped")

        return BenchmarkTelemetrySample(
            sequence=sequence,
            ss_s=ss_s,
            softnet_stat=softnet,
            ethtool_s=ethtool_s,
            errors=tuple(errors),
        )


def collect_telemetry_during_blocking_command(
    *,
    collector: BenchmarkRuntimeTelemetryCollector,
    telemetry_executor: CommandExecutor,
    sample_interval_seconds: float,
    blocking_call: Callable[[], None],
) -> tuple[BenchmarkTelemetrySample, ...]:
    """
    Sample telemetry on `telemetry_executor` on a fixed interval while `blocking_call()` runs.

    The first sample is taken immediately; then every `sample_interval_seconds` until the
    blocking call finishes.
    """
    samples: list[BenchmarkTelemetrySample] = []
    lock = threading.Lock()
    stop = threading.Event()

    def worker() -> None:
        seq = 0
        while not stop.is_set():
            try:
                sample = collector.capture_sample(telemetry_executor, seq)
            except Exception as exc:
                sample = BenchmarkTelemetrySample(
                    sequence=seq,
                    ss_s="",
                    softnet_stat="",
                    ethtool_s="",
                    errors=(f"telemetry capture failed: {exc}",),
                )
            seq += 1
            with lock:
                samples.append(sample)
            if stop.wait(timeout=sample_interval_seconds):
                break

    thread = threading.Thread(target=worker, name="benchmark-telemetry", daemon=True)
    thread.start()
    try:
        blocking_call()
    finally:
        stop.set()
        thread.join(timeout=sample_interval_seconds + 5.0)
        if thread.is_alive():
            import logging

            logging.getLogger(__name__).warning(
                "Telemetry thread did not stop within timeout; it may leak resources."
            )

    with lock:
        return tuple(samples)
