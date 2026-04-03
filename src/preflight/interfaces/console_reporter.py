from __future__ import annotations

import json
import re
from dataclasses import asdict

from baseline.domain.models import BaselineResult
from onboard.domain.models import OnboardResult
from preflight.domain.models import DiscoverySnapshot
from snapshot.domain.models import SnapshotResult

ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


class ConsoleReporter:
    def render_json(self, snapshot: DiscoverySnapshot) -> str:
        return json.dumps(asdict(snapshot), indent=2, default=str)

    def render(self, snapshot: DiscoverySnapshot) -> str:
        return self.render_json(snapshot)

    def render_runtime(
        self,
        preflight: DiscoverySnapshot,
        onboard: OnboardResult | None,
        snapshot: SnapshotResult | None,
        baseline: BaselineResult | None,
    ) -> str:
        sections = [
            self._render_preflight(preflight),
            self._render_onboard(onboard),
            self._render_snapshot(snapshot),
            self._render_baseline(baseline),
        ]
        return "\n\n".join(section for section in sections if section)

    def _render_preflight(self, preflight: DiscoverySnapshot) -> str:
        capabilities = ", ".join(
            flag.name for flag in preflight.capability_map.flags if flag.available
        )
        return "\n".join(
            (
                "Preflight",
                f"  Platform: {preflight.platform_summary}",
                (
                    f"  CPU: {preflight.cpu.logical_cores} logical cores, "
                    f"{preflight.cpu.numa_nodes} NUMA nodes"
                ),
                (
                    f"  Memory: {preflight.memory.total_memory_kib} KiB RAM, "
                    f"{preflight.memory.swap_total_kib} KiB swap"
                ),
                (
                    f"  Network: {preflight.network.interface_name} "
                    f"({preflight.network.driver_name})"
                ),
                (
                    f"  Storage: {preflight.storage.device_name} "
                    f"({preflight.storage.device_type}, {preflight.storage.scheduler})"
                ),
                f"  Tunables: {capabilities}",
            )
        )

    def _render_onboard(self, onboard: OnboardResult | None) -> str:
        if onboard is None:
            return ""
        findings = len(onboard.compatibility.findings)
        directives = ", ".join(sorted(onboard.service.tunable_surface.allowed_directives))
        return "\n".join(
            (
                "Onboard",
                f"  Service: {onboard.service_name}",
                f"  Unit: {onboard.service.identity.systemd_unit_name}",
                f"  Compatible: {onboard.compatibility.compatible}",
                f"  Findings: {findings}",
                f"  Allowed directives: {directives}",
            )
        )

    def _render_snapshot(self, snapshot: SnapshotResult | None) -> str:
        if snapshot is None:
            return ""
        captured = ", ".join(snapshot.captured_paths)
        return "\n".join(
            (
                "Snapshot",
                f"  Directory: {snapshot.snapshot_directory}",
                f"  Captured paths: {captured}",
                f"  Restore steps: {len(snapshot.restore_sequence)}",
            )
        )

    def _render_baseline(self, baseline: BaselineResult | None) -> str:
        if baseline is None:
            return ""
        lines = [
            "Baseline",
            f"  Target: {baseline.benchmark_target}",
            (
                f"  Expected variance: {baseline.expected_variance:.2%} | "
                f"Warmup: {baseline.warmup_seconds}s"
            ),
            "  Workloads:",
            "    name       rps         total        latency_ms",
        ]
        lines.extend(
            (
                f"    {result.workload_name:<10} "
                f"{result.requests_per_second:>10.2f} "
                f"{result.total_requests:>12} "
                f"{result.average_latency_ms:>12.2f}"
            )
            for result in baseline.workload_results
        )
        comparison_table = self._render_comparison_table(baseline.comparison_output)
        if comparison_table:
            lines.extend(("", "Comparison", *comparison_table))
        return "\n".join(lines)

    def _render_comparison_table(self, comparison_output: str | None) -> list[str]:
        if comparison_output is None:
            return []

        clean_output = ANSI_PATTERN.sub("", comparison_output)
        rows: list[tuple[str, str, str, str, str]] = []
        for line in clean_output.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("Workload", "---", "Legend:", "For detailed")):
                continue
            if "|" not in stripped:
                continue
            parts = [part.strip() for part in stripped.split("|")]
            if len(parts) != 5 or parts[0] == "Workload":
                continue
            rows.append((parts[0], parts[1], parts[2], parts[3], parts[4]))

        if not rows:
            return []

        rendered = ["  workload   baseline_rps   current_rps   change    status"]
        rendered.extend(
            f"  {name:<10} {baseline_rps:>12} {current_rps:>12} {change:>8} {status:>9}"
            for name, baseline_rps, current_rps, change, status in rows
        )
        return rendered
