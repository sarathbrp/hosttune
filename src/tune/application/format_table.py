"""PrettyTable formatting helpers for tune logging."""
from __future__ import annotations

from prettytable import PrettyTable


def _table(*field_names: str, max_width: int = 35) -> PrettyTable:
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


def apply_failed_table(parameter: str, error: str) -> str:
    """Single-row table for an apply failure."""
    t = _table("Parameter", "Error", max_width=80)
    t.add_row([parameter, error])
    return f"Apply failed\n{t}"
