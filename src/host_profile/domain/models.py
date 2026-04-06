from __future__ import annotations

from dataclasses import dataclass

from onboard.domain.models import ApplyMode, PriorityTier


@dataclass(frozen=True)
class HostProfileIdentity:
    name: str
    platform: str  # e.g. "rhel"
    version: str  # e.g. "9"
    variant: str | None  # None = bare metal; "vm" = virtual machine


@dataclass(frozen=True)
class NetworkQueueConstraint:
    """Controls ethtool -L combined N expansion."""

    min_combined: int
    # 0 = resolve to logical_cores at catalog build time
    max_combined: int
    allow_irq_affinity: bool
    priority_tier: PriorityTier
    apply_mode: ApplyMode


@dataclass(frozen=True)
class CpuGovernorConstraint:
    """Controls cpupower frequency-set -g <governor>."""

    allowed_governors: tuple[str, ...]
    forbidden_governors: tuple[str, ...]
    preferred_governor: str
    priority_tier: PriorityTier
    apply_mode: ApplyMode


@dataclass(frozen=True)
class HostSysctlTunable:
    """Host-level sysctl beyond the service-specific relevant_sysctls list."""

    name: str
    priority_tier: PriorityTier
    rationale_hint: str


@dataclass(frozen=True)
class EnvironmentBlocker:
    """External constraint that makes tuning pointless until cleared."""

    name: str
    probe_command: str
    fix_command: str | None  # None = signal only, no autofix
    priority: str  # "critical" or "high"
    detail: str
    threshold_above: int | None = None  # trigger if probe output > N
    threshold_below: int | None = None  # trigger if probe output < N


@dataclass(frozen=True)
class HostTunableSurface:
    network_queues: NetworkQueueConstraint | None
    cpu_governor: CpuGovernorConstraint | None
    host_sysctls: tuple[HostSysctlTunable, ...]
    environment_blockers: tuple[EnvironmentBlocker, ...] = ()


@dataclass(frozen=True)
class HostProfile:
    identity: HostProfileIdentity
    tunable_surface: HostTunableSurface
