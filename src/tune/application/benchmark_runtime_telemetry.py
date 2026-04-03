from __future__ import annotations

import shlex
import threading
from collections.abc import Callable
from dataclasses import dataclass

from preflight.domain.models import CommandExecutor
from tune.domain.benchmark_models import BenchmarkTelemetrySample


def truncate_for_prompt(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n... [truncated, {len(text) - max_chars} chars omitted]"


def format_runtime_telemetry_digest(
    samples: tuple[BenchmarkTelemetrySample, ...],
    *,
    max_chars_per_section: int = 1200,
) -> str:
    if not samples:
        return (
            "No runtime telemetry samples yet (first iteration, or telemetry disabled / "
            "benchmark host unreachable)."
        )
    blocks: list[str] = []
    failed = "[collection failed]"
    for sample in samples:
        err = f"\nerrors: {list(sample.errors)}" if sample.errors else ""
        blocks.append(
            f"Sample #{sample.sequence} (captured during benchmark load)\n"
            f"ss -s:\n{truncate_for_prompt(sample.ss_s, max_chars_per_section) or failed}\n"
            f"/proc/net/softnet_stat:\n"
            f"{truncate_for_prompt(sample.softnet_stat, max_chars_per_section) or failed}\n"
            f"ethtool -S (NIC counters):\n"
            f"{truncate_for_prompt(sample.ethtool_s, max_chars_per_section) or failed}"
            f"{err}"
        )
    return "\n\n".join(blocks)


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
