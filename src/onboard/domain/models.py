from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ConfigFormat(StrEnum):
    NGINX = "nginx"
    INI = "ini"
    YAML = "yaml"
    JSON = "json"


class ProbeType(StrEnum):
    HTTP = "http"
    TCP = "tcp"
    SYSTEMD_STATUS = "systemd_status"
    COMMAND = "command"


class ApplyMode(StrEnum):
    RELOAD = "reload"
    RESTART = "restart"
    REBOOT = "reboot"


class DirectiveValueType(StrEnum):
    INTEGER = "integer"
    ENUM = "enum"
    STRING = "string"


class PriorityTier(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FindingSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class ServiceIdentity:
    service_name: str
    systemd_unit_name: str
    rhel_versions: tuple[str, ...]
    service_versions: tuple[str, ...]
    config_paths: tuple[str, ...]
    config_format: ConfigFormat
    working_directory: str | None
    log_paths: tuple[str, ...]


@dataclass(frozen=True)
class ServiceHealthCheck:
    probe_type: ProbeType
    target: str
    expected_status_code: int | None
    expected_string: str | None
    expected_exit_code: int | None
    timeout_seconds: int
    retries: int
    warmup_seconds: int


@dataclass(frozen=True)
class ProcessState:
    pid_file: str | None
    worker_process_hint: str | None
    open_connections_command: str | None


@dataclass(frozen=True)
class ServiceSnapshotContract:
    files_to_snapshot: tuple[str, ...]
    runtime_state_command: str | None
    process_state: ProcessState
    restore_sequence: tuple[str, ...]
    snapshot_storage_location: str


@dataclass(frozen=True)
class ReloadContract:
    supported: bool
    command: str | None


@dataclass(frozen=True)
class RestartContract:
    supported: bool
    command: str | None
    expected_downtime_seconds: int


@dataclass(frozen=True)
class ServiceRestartContract:
    reload: ReloadContract
    restart: RestartContract
    change_categories: dict[str, ApplyMode]
    drain_policy: str | None
    dependency_chain: tuple[str, ...]
    post_restart_validation: str


@dataclass(frozen=True)
class DirectiveConstraint:
    value_type: DirectiveValueType
    apply_mode: ApplyMode
    priority_tier: PriorityTier
    min_value: int | None
    max_value: int | None
    allowed_values: tuple[str, ...]
    forbidden_values: tuple[str, ...]
    # Optional YAML override; values are TuningLayer strings (kernel|network|service|runtime).
    tuning_layer: str | None = None


@dataclass(frozen=True)
class SysctlTunable:
    """Kernel sysctl knob listed under tunable_surface (priority drives Wide Sweep ordering)."""

    name: str
    priority_tier: PriorityTier
    tuning_layer: str | None = None


@dataclass(frozen=True)
class ServiceTunableSurface:
    allowed_directives: dict[str, DirectiveConstraint]
    forbidden_directives: tuple[str, ...]
    interdependencies: tuple[str, ...]
    relevant_sysctls: tuple[SysctlTunable, ...]
    network_ring_priority_tier: PriorityTier
    # Process-level limits (prlimit); YAML keys map to runtime.prlimit.<name> in the catalog.
    runtime_limits: dict[str, DirectiveConstraint]
    # systemd unit resource limits (LimitNOFILE, LimitNPROC); YAML keys map to systemd.unit.<name>.
    systemd_unit_limits: dict[str, DirectiveConstraint]
    # Optional YAML override for network.ring.* catalog entries (rx/tx).
    network_ring_tuning_layer: str | None = None


@dataclass(frozen=True)
class ServiceBenchmarkHints:
    primary_metric: str
    guardrail_metrics: tuple[str, ...]
    expected_variance: float
    warmup_seconds: int
    interference_sources: tuple[str, ...]


@dataclass(frozen=True)
class ServiceDefinition:
    identity: ServiceIdentity
    health_check: ServiceHealthCheck
    snapshot: ServiceSnapshotContract
    restart: ServiceRestartContract
    tunable_surface: ServiceTunableSurface
    benchmark_hints: ServiceBenchmarkHints


@dataclass(frozen=True)
class CompatibilityFinding:
    severity: FindingSeverity
    message: str


@dataclass(frozen=True)
class CompatibilityReport:
    compatible: bool
    findings: tuple[CompatibilityFinding, ...]


@dataclass(frozen=True)
class OnboardResult:
    service_name: str
    service: ServiceDefinition
    compatibility: CompatibilityReport
