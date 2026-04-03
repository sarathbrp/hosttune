"""
Canonical current-value reads for the candidate catalog.

Live probes are preferred; preflight/snapshot are fallbacks for operator visibility
when live reads are unavailable — broad context stays in preflight/snapshot layers.
"""

from __future__ import annotations

import logging
import re
import shlex

from preflight.domain.models import CommandExecutor

_VALID_RING_NAMES = frozenset({"rx", "tx", "rx-mini", "rx-jumbo"})


def sysctl_value_from_profile(
    sysctl_profile: tuple[tuple[str, str], ...],
    sysctl_name: str,
) -> str | None:
    for key, value in sysctl_profile:
        if key == sysctl_name:
            stripped = value.strip()
            return stripped if stripped else None
    return None


def read_sysctl_catalog_current(
    sysctl_name: str,
    executor: CommandExecutor | None,
    sysctl_profile: tuple[tuple[str, str], ...],
) -> str | None:
    if executor is not None:
        result = executor.run(f"sysctl -n {shlex.quote(sysctl_name)}")
        if result.exit_code == 0:
            out = result.stdout.strip()
            if out:
                return out
            logging.getLogger(__name__).debug(
                "sysctl -n %s returned empty; falling back to preflight profile",
                sysctl_name,
            )
    return sysctl_value_from_profile(sysctl_profile, sysctl_name)


def try_read_network_ring_current(
    executor: CommandExecutor,
    interface_name: str,
    ring_name: str,
) -> str | None:
    """Match `ethtool -g` Current hardware settings (same awk as apply rollback)."""
    if ring_name not in _VALID_RING_NAMES:
        msg = f"Unsupported ring name: {ring_name!r}"
        raise ValueError(msg)
    command = (
        f"ethtool -g {shlex.quote(interface_name)} | "
        'awk \'BEGIN{section=""} '
        '/Current hardware settings:/{section="current"; next} '
        '/Pre-set maximums:/{section="max"; next} '
        f'section=="current" && $1=="{ring_name}:" {{print $2; exit}}\''
    )
    result = executor.run(command)
    if result.exit_code != 0:
        return None
    out = result.stdout.strip()
    return out if out else None


def read_network_ring_catalog_current(
    ring_name: str,
    interface_name: str,
    executor: CommandExecutor | None,
    preflight_current: int,
) -> str:
    if executor is not None and interface_name:
        live = try_read_network_ring_current(executor, interface_name, ring_name)
        if live is not None:
            return live
    return str(preflight_current)


def parse_directive_from_nginx_dump(
    directive_name: str, runtime_state_output: str | None
) -> str | None:
    """Parse `nginx -T`-style text for a top-level `name value;` directive."""
    if not runtime_state_output or not runtime_state_output.strip():
        return None
    pattern = re.compile(
        rf"(?m)^\s*{re.escape(directive_name)}\s+([^;]+);",
    )
    match = pattern.search(runtime_state_output)
    if not match:
        return None
    return match.group(1).strip()


def grep_directive_from_config_file(
    executor: CommandExecutor,
    config_path: str,
    directive_name: str,
) -> str | None:
    command = (
        f"grep -E '^\\s*{re.escape(directive_name)}\\s+' " f"{shlex.quote(config_path)} | tail -n 1"
    )
    result = executor.run(command)
    if result.exit_code != 0 or result.stdout == "":
        return None
    line = result.stdout.strip().rstrip(";")
    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        return None
    return parts[1].strip()


def read_service_directive_catalog_current(
    executor: CommandExecutor | None,
    config_path: str,
    directive_name: str,
    runtime_state_output: str | None,
) -> str | None:
    if executor is not None:
        from_file = grep_directive_from_config_file(executor, config_path, directive_name)
        if from_file is not None:
            return from_file
    return parse_directive_from_nginx_dump(directive_name, runtime_state_output)
