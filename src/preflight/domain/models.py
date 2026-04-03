from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

TargetMode = Literal["local", "ssh"]


@dataclass(frozen=True)
class LocalTargetConfig:
    mode: Literal["local"] = "local"


@dataclass(frozen=True)
class SshTargetConfig:
    host: str
    user: str
    private_key_path: Path
    port: int = 22
    connect_timeout_seconds: int = 5
    mode: Literal["ssh"] = "ssh"


TargetConfig = LocalTargetConfig | SshTargetConfig


@dataclass(frozen=True)
class EngagementPolicy:
    allow_reload: bool
    allow_restart: bool
    allow_reboot: bool
    rollback_required: bool
    max_iterations: int
    benchmark_stability_threshold: float


@dataclass(frozen=True)
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class BenchmarkResult:
    command: str
    exit_code: int
    primary_metric_name: str
    primary_metric_value: float
    raw_output: str


@dataclass(frozen=True)
class CapabilityFlag:
    name: str
    available: bool
    detail: str


@dataclass(frozen=True)
class CapabilityMap:
    flags: tuple[CapabilityFlag, ...] = ()


@dataclass(frozen=True)
class PlatformInfo:
    hostname: str
    operating_system: str
    kernel_version: str
    virtualization_type: str
    is_container: bool


@dataclass(frozen=True)
class CpuInfo:
    architecture: str
    logical_cores: int
    threads_per_core: int
    cores_per_socket: int
    sockets: int
    numa_nodes: int
    hyperthreading_enabled: bool


@dataclass(frozen=True)
class MemoryInfo:
    total_memory_kib: int
    swap_total_kib: int
    hugepages_total: int
    transparent_hugepages_mode: str


@dataclass(frozen=True)
class KernelInfo:
    sysctl_writable: bool
    selinux_mode: str
    tuned_profile: str


@dataclass(frozen=True)
class NetworkInfo:
    interface_name: str
    driver_name: str
    firmware_version: str
    rx_ring_current: int
    rx_ring_max: int
    tx_ring_current: int
    tx_ring_max: int
    combined_queues: int
    ring_buffer_tuning_supported: bool


@dataclass(frozen=True)
class StorageInfo:
    device_name: str
    device_type: str
    scheduler: str
    scheduler_meaningful: bool


@dataclass(frozen=True)
class DiscoverySnapshot:
    target: TargetConfig
    policy: EngagementPolicy
    platform: PlatformInfo
    cpu: CpuInfo
    memory: MemoryInfo
    kernel: KernelInfo
    network: NetworkInfo
    storage: StorageInfo
    capability_map: CapabilityMap
    benchmark_result: BenchmarkResult | None = None
    raw_probe_results: dict[str, CommandResult] | None = None


class CommandExecutor(Protocol):
    def run(self, command: str) -> CommandResult:
        """Run a command on the configured target."""


class BenchmarkRunner(Protocol):
    def run(self, executor: CommandExecutor) -> BenchmarkResult:
        """Run the configured benchmark and return normalized metrics."""


class DiscoveryProbe(Protocol):
    @property
    def name(self) -> str:
        """Return a stable probe name."""

    def collect(self, executor: CommandExecutor) -> object:
        """Collect and normalize one discovery concern."""
