"""PrettyTable formatting helpers for tune logging."""
from __future__ import annotations

from typing import Protocol

try:
    from prettytable import PrettyTable
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal test envs
    PrettyTable = None  # type: ignore[assignment]


class _RenderableTable(Protocol):
    align: str
    max_width: int

    def add_row(self, row: list[object]) -> None:
        ...


class _FallbackTable:
    def __init__(self, field_names: list[str]) -> None:
        self.field_names = field_names
        self.rows: list[list[str]] = []
        self.align = "l"
        self.max_width = 35

    def add_row(self, row: list[object]) -> None:
        self.rows.append([self._clip(str(cell)) for cell in row])

    def _clip(self, value: str) -> str:
        if self.max_width > 3 and len(value) > self.max_width:
            return value[: self.max_width - 3] + "..."
        return value

    def __str__(self) -> str:
        headers = [self._clip(name) for name in self.field_names]
        widths = [len(h) for h in headers]
        for row in self.rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def _line() -> str:
            return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

        def _render_row(values: list[str]) -> str:
            cells = [f" {value.ljust(widths[idx])} " for idx, value in enumerate(values)]
            return "|" + "|".join(cells) + "|"

        parts = [_line(), _render_row(headers), _line()]
        for row in self.rows:
            parts.append(_render_row(row))
        parts.append(_line())
        return "\n".join(parts)


def _table(*field_names: str, max_width: int = 35) -> _RenderableTable:
    t: _RenderableTable
    if PrettyTable is None:
        t = _FallbackTable(list(field_names))
    else:
        t = PrettyTable(list(field_names))
    t.align = "l"
    t.max_width = max_width
    return t


def resolver_layer_table(
    layer_name: str,
    rows: list[tuple[str, str, str, str, str]],
) -> str:
    """Table of per-param resolver decisions for one layer.

    rows: [(param_key, current, target, source, action), ...]
    """
    t = _table("Parameter", "Current", "Target", "Source", "Action")
    for row in rows:
        t.add_row(list(row))
    return f"Resolver [{layer_name}]\n{t}"


def resolver_apply_table(layer_name: str, hyps: list[tuple[str, str]]) -> str:
    """Table of param=value pairs being applied for a resolver layer.

    hyps: [(parameter_key, proposed_value), ...]
    """
    t = _table("Parameter", "Value")
    for param, value in hyps:
        t.add_row([param, value])
    return f"Resolver applying [{layer_name}]\n{t}"


def resolver_summary_table(
    total_applied: int,
    layer_statuses: dict[str, str],
    layer_param_counts: dict[str, int],
) -> str:
    """Summary table across all resolver layers."""
    t = _table("Layer", "Status", "Params")
    for layer in sorted(layer_statuses):
        t.add_row([layer, layer_statuses[layer], layer_param_counts.get(layer, 0)])
    return (
        f"Unified resolver: {total_applied} params retained "
        f"across {sum(1 for v in layer_statuses.values() if v != 'ok')} active layers\n{t}"
    )


def apply_table(
    rows: list[tuple[str, str, str, str, str]],
    header: str = "Applied changes",
) -> str:
    """Table of applied hypothesis changes.

    rows: [(parameter, layer, previous, applied, mode), ...]
    """
    t = _table("Parameter", "Layer", "Previous", "Applied", "Mode")
    for row in rows:
        t.add_row(list(row))
    return f"{header}\n{t}"


def pre_apply_rejection_table(layer: str, parameter: str, reason: str) -> str:
    """Single-row table for a pre-apply rejection."""
    t = _table("Layer", "Parameter", "Reason", max_width=60)
    t.add_row([layer, parameter, reason])
    return f"Pre-apply rejection\n{t}"


def benchmark_summary_table(benchmark_result: object) -> str:
    """Prettytable for benchmark workload summaries."""
    from tune.domain.benchmark_models import TuneBenchmarkResult
    result: TuneBenchmarkResult = benchmark_result  # type: ignore[assignment]
    t = _table("Workload", "RPS", "Latency ms", "Variance", "Stable")
    for s in result.workload_summaries:
        t.add_row([
            s.workload_name,
            f"{s.median_requests_per_second:,.0f}",
            f"{s.median_latency_ms:.2f}",
            f"{s.relative_variance:.2%}",
            "✓" if s.stable else "✗",
        ])
    status = "stable" if result.stable else "UNSTABLE"
    return (
        f"Benchmark run summary: {status} | "
        f"variance_threshold={result.variance_threshold:.1%}\n{t}"
    )


def evaluation_table(evaluation_result: object) -> str:
    """Prettytable for per-workload evaluation with decision and signal."""
    from tune.domain.evaluation_models import EvaluationResult, EvaluationDecision
    result: EvaluationResult = evaluation_result  # type: ignore[assignment]
    decision = result.decision.value.upper()
    t = _table("Workload", "Baseline RPS", "Current RPS", "Change", "Signal", max_width=15)
    for w in result.workload_evaluations:
        change_pct = w.relative_change * 100
        sign = "+" if change_pct >= 0 else ""
        signal = "above noise" if w.above_noise_floor else "noise"
        t.add_row([
            w.workload_name,
            f"{w.baseline_requests_per_second:,.0f}",
            f"{w.current_requests_per_second:,.0f}",
            f"{sign}{change_pct:.1f}%",
            signal,
        ])
    return (
        f"Evaluate: decision={decision} | "
        f"guardrails={result.guardrails_held} | drift={result.drift_detected}\n{t}"
    )


def apply_failed_table(parameter: str, error: str) -> str:
    """Single-row table for an apply failure."""
    t = _table("Parameter", "Error", max_width=80)
    t.add_row([parameter, error])
    return f"Apply failed\n{t}"
