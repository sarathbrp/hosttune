from __future__ import annotations

from enum import StrEnum

_SERVICE_DIRECTIVE_PREFIX = "service.directive."
_SYSCTL_PREFIX = "sysctl."
_NETWORK_RING_PREFIX = "network.ring."
_RUNTIME_PRLIMIT_PREFIX = "runtime.prlimit."
_SYSTEMD_UNIT_PREFIX = "systemd.unit."

_RUNTIME_DIRECTIVES = frozenset({"worker_rlimit_nofile"})


class TuningLayer(StrEnum):
    """Logical stack layer for a tunable (Problem 4 — phase policy may use this later)."""

    KERNEL = "kernel"
    NETWORK = "network"
    SERVICE = "service"
    RUNTIME = "runtime"


def tuning_layer_for_parameter_key(parameter_key: str) -> TuningLayer:
    """
    Map a catalog parameter_key to its tuning layer.

    Rules:
    - sysctl.* → KERNEL
    - network.ring.* → NETWORK
    - runtime.prlimit.* → RUNTIME (process limits via prlimit)
    - systemd.unit.* → RUNTIME (systemd unit LimitNOFILE / LimitNPROC via set-property)
    - service.directive.worker_rlimit_nofile → RUNTIME (fd / process limits surface)
    - other service.directive.* → SERVICE
    """
    if parameter_key.startswith(_SYSCTL_PREFIX):
        return TuningLayer.KERNEL
    if parameter_key.startswith(_NETWORK_RING_PREFIX):
        return TuningLayer.NETWORK
    if parameter_key.startswith(_RUNTIME_PRLIMIT_PREFIX):
        return TuningLayer.RUNTIME
    if parameter_key.startswith(_SYSTEMD_UNIT_PREFIX):
        return TuningLayer.RUNTIME
    if parameter_key.startswith(_SERVICE_DIRECTIVE_PREFIX):
        directive = parameter_key.removeprefix(_SERVICE_DIRECTIVE_PREFIX)
        if directive in _RUNTIME_DIRECTIVES:
            return TuningLayer.RUNTIME
        return TuningLayer.SERVICE
    msg = f"Unknown parameter_key prefix for tuning layer: {parameter_key!r}"
    raise ValueError(msg)


def resolve_tuning_layer(parameter_key: str, yaml_override: str | None) -> TuningLayer:
    """Use YAML `tuning_layer` when set; otherwise derive from `parameter_key` prefix."""
    if yaml_override is not None:
        return TuningLayer(yaml_override)
    return tuning_layer_for_parameter_key(parameter_key)
