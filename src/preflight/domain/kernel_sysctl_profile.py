"""
Kernel sysctl profile for preflight (layer 1).

- **1a:** `KernelProbe` reads `PREFLIGHT_SYSCTL_KEYS` during discovery.
- **1b:** After onboard, `HostTuneInstance` merges `relevant_sysctls` from the service YAML
  (deduped, contract-only keys appended) and re-reads those values on the target.
"""

from __future__ import annotations

# Network and VM knobs commonly relevant to HTTP / reverse-proxy tuning.
PREFLIGHT_SYSCTL_KEYS: tuple[str, ...] = (
    "net.core.somaxconn",
    "net.ipv4.tcp_max_syn_backlog",
    "net.core.netdev_max_backlog",
    "net.core.rmem_max",
    "net.core.wmem_max",
    "net.ipv4.tcp_rmem",
    "net.ipv4.tcp_wmem",
    "net.ipv4.tcp_tw_reuse",
    "net.ipv4.tcp_fin_timeout",
    "vm.swappiness",
    "vm.dirty_ratio",
    "vm.vfs_cache_pressure",
)

PREFLIGHT_SYSCTL_KEY_SET: frozenset[str] = frozenset(PREFLIGHT_SYSCTL_KEYS)


def merged_sysctl_profile_key_order(contract_sysctl_names: tuple[str, ...]) -> tuple[str, ...]:
    """
    Preflight base keys first, then service-contract sysctls not already in the base list
    (YAML order for duplicates skipped).
    """
    seen: set[str] = set(PREFLIGHT_SYSCTL_KEYS)
    ordered = list(PREFLIGHT_SYSCTL_KEYS)
    for name in contract_sysctl_names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)


def contract_sysctl_names_only_extra(contract_sysctl_names: tuple[str, ...]) -> tuple[str, ...]:
    """Contract sysctl names that need an extra `sysctl -n` (not in the preflight base set)."""
    return tuple(n for n in contract_sysctl_names if n not in PREFLIGHT_SYSCTL_KEY_SET)


def sysctl_profile_read_command(keys: tuple[str, ...] = PREFLIGHT_SYSCTL_KEYS) -> str:
    """Shell loop: one `sysctl -n` per key; prints `name=value` or `name=` if unreadable."""
    keys_str = " ".join(keys)
    return (
        "for k in " + keys_str + "; do "
        'if v=$(sysctl -n "$k" 2>/dev/null); then printf \'%s=%s\\n\' "$k" "$v"; '
        "else printf '%s=\\n' \"$k\"; fi; done"
    )


def format_sysctl_profile_compact(
    profile: tuple[tuple[str, str], ...],
    *,
    max_chars: int = 1200,
) -> str:
    """Single-line summary for prompts; truncates if very long."""
    if not profile:
        return "not captured"
    parts = [f"{name}={value}" if value else f"{name}=<unreadable>" for name, value in profile]
    body = "; ".join(parts)
    if len(body) <= max_chars:
        return body
    return f"{body[:max_chars]}... [truncated, {len(body) - max_chars} chars omitted]"
