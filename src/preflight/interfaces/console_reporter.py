from __future__ import annotations

import json
import re
from dataclasses import asdict

from prettytable import PrettyTable

from baseline.domain.models import BaselineResult
from onboard.domain.models import OnboardResult
from preflight.domain.kernel_sysctl_profile import format_sysctl_profile_compact
from preflight.domain.models import DiscoverySnapshot
from snapshot.domain.models import SnapshotResult
from tune.domain.hypothesis_models import HypothesisStatus
from tune.domain.tune_state import TuneState

ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


class ConsoleReporter:
    def _truncate_cell(self, value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return f"{value[: limit - 3]}..."

    def _render_pretty_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        *,
        align: str = "l",
    ) -> list[str]:
        table = PrettyTable()
        table.field_names = headers
        table.align = align
        for row in rows:
            table.add_row(row)
        return [f"  {line}" for line in str(table).splitlines()]

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
        tune: TuneState | None,
    ) -> str:
        sections = [
            self._render_preflight(preflight),
            self._render_onboard(onboard),
            self._render_snapshot(snapshot),
            self._render_baseline(baseline),
            self._render_tune(tune, baseline),
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
                    "  Kernel sysctl profile: "
                    + format_sysctl_profile_compact(preflight.kernel.sysctl_profile, max_chars=400)
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
        workload_rows = [
            [
                result.workload_name,
                f"{result.requests_per_second:.2f}",
                str(result.total_requests),
                f"{result.average_latency_ms:.2f}",
            ]
            for result in baseline.workload_results
        ]
        lines = [
            "Baseline",
            f"  Target: {baseline.benchmark_target}",
            (
                f"  Expected variance: {baseline.expected_variance:.2%} | "
                f"Warmup: {baseline.warmup_seconds}s"
            ),
            "  Workloads:",
        ]
        lines.extend(
            self._render_pretty_table(
                ["name", "rps", "total", "latency_ms"],
                workload_rows,
            )
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

        return self._render_pretty_table(
            ["workload", "baseline_rps", "current_rps", "change", "status"],
            [
                [name, baseline_rps, current_rps, change, status]
                for name, baseline_rps, current_rps, change, status in rows
            ],
        )

    def _render_best_iteration_table(
        self,
        tune: TuneState,
        baseline: BaselineResult | None,
    ) -> list[str]:
        if tune.best_configuration is None or baseline is None:
            return []
        best_record = next(
            (
                record
                for record in tune.iteration_records
                if record.iteration_number == tune.best_configuration.iteration_number
            ),
            None,
        )
        if best_record is None or best_record.benchmark_result is None:
            return []

        baseline_by_name = {
            workload.workload_name: workload.requests_per_second
            for workload in baseline.workload_results
        }
        rows: list[tuple[str, str, str, str]] = []
        for summary in best_record.benchmark_result.workload_summaries:
            baseline_rps = baseline_by_name.get(summary.workload_name)
            if baseline_rps is None or baseline_rps <= 0.0:
                continue
            best_rps = summary.median_requests_per_second
            relative_change = (best_rps - baseline_rps) / baseline_rps
            rows.append(
                (
                    summary.workload_name,
                    f"{baseline_rps:.2f}",
                    f"{best_rps:.2f}",
                    f"{relative_change:.1%}",
                )
            )
        if not rows:
            return []
        rendered = [
            f"  Best iteration: {tune.best_configuration.iteration_number}",
            "  Best comparison",
        ]
        rendered.extend(
            self._render_pretty_table(
                ["workload", "baseline_rps", "best_rps", "change"],
                [
                    [name, baseline_rps, best_rps, change]
                    for name, baseline_rps, best_rps, change in rows
                ],
            )
        )
        return rendered

    def _render_iteration_history_table(
        self,
        tune: TuneState,
        baseline: BaselineResult | None,
    ) -> list[str]:
        if not tune.iteration_records:
            return []

        baseline_summary = "n/a"
        if baseline is not None and baseline.workload_results:
            baseline_summary = "; ".join(
                f"{item.workload_name}={item.requests_per_second:.2f}"
                for item in baseline.workload_results
            )
        history_by_iteration = {
            item.iteration_number: item.status.value
            for item in tune.history
            if hasattr(item, "iteration_number")
        }
        rows: list[list[str]] = []
        for record in tune.iteration_records:
            benchmark_summary = "n/a"
            if record.benchmark_result is not None and record.benchmark_result.workload_summaries:
                benchmark_summary = "; ".join(
                    f"{item.workload_name}={item.median_requests_per_second:.2f}"
                    for item in record.benchmark_result.workload_summaries
                )
            usage_display = "0"
            if record.hypothesis.model_usage is not None:
                usage_display = str(record.hypothesis.model_usage.total_tokens)
            status = history_by_iteration.get(record.iteration_number, "unknown")
            rows.append(
                [
                    str(record.iteration_number),
                    record.phase.value,
                    self._truncate_cell(record.hypothesis.parameter_key, 28),
                    self._truncate_cell(record.hypothesis.proposed_value, 12),
                    f"{record.duration_seconds:.2f}s",
                    usage_display,
                    status,
                    self._truncate_cell(baseline_summary, 30),
                    self._truncate_cell(benchmark_summary, 30),
                ]
            )
        rendered = ["  Iteration history"]
        rendered.extend(
            self._render_pretty_table(
                [
                    "iter",
                    "phase",
                    "parameter",
                    "value",
                    "duration",
                    "tokens",
                    "status",
                    "baseline_rps",
                    "benchmarked_rps",
                ],
                rows,
            )
        )
        return rendered

    def _render_tune(
        self,
        tune: TuneState | None,
        baseline: BaselineResult | None,
    ) -> str:
        if tune is None:
            return ""

        accepted = sum(1 for item in tune.history if item.status is HypothesisStatus.ACCEPTED)
        input_tokens = sum(
            record.hypothesis.model_usage.input_tokens
            for record in tune.iteration_records
            if record.hypothesis.model_usage is not None
        )
        output_tokens = sum(
            record.hypothesis.model_usage.output_tokens
            for record in tune.iteration_records
            if record.hypothesis.model_usage is not None
        )
        total_tokens = sum(
            record.hypothesis.model_usage.total_tokens
            for record in tune.iteration_records
            if record.hypothesis.model_usage is not None
        )
        total_duration_seconds = sum(record.duration_seconds for record in tune.iteration_records)
        best_iteration_config = "none"
        best_score = "n/a"
        final_retained_config = "none"
        if tune.best_configuration is not None:
            best_score = f"{tune.best_configuration.score:.2%}"
            best_iteration_config = (
                ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(tune.best_iteration_config_values().items())
                )
                or "none"
            )
        retained_values = tune.final_retained_config_values()
        if retained_values:
            final_retained_config = ", ".join(
                f"{key}={value}" for key, value in sorted(retained_values.items())
            )
        active = ", ".join(sorted(tune.active_changes)) or "none"
        lines = [
            "Tune",
            f"  Current phase: {tune.current_phase.value}",
            f"  Iterations: {tune.total_iterations}",
            f"  Accepted hypotheses: {accepted}",
            f"  Active changes: {active}",
            f"  Total duration: {total_duration_seconds:.2f}s",
            f"  Model tokens: input={input_tokens} output={output_tokens} total={total_tokens}",
            f"  Best score: {best_score}",
            f"  Best iteration config: {best_iteration_config}",
            f"  Final retained config: {final_retained_config}",
            f"  Drift detected: {tune.drift_detected}",
        ]
        lines.extend(self._render_best_iteration_table(tune, baseline))
        lines.extend(self._render_iteration_history_table(tune, baseline))
        return "\n".join(lines)
