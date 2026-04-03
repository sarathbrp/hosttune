from __future__ import annotations

from preflight.domain.models import CommandResult
from preflight.infrastructure.parsers.cpu_parser import CpuParser
from preflight.infrastructure.parsers.kernel_parser import KernelParser
from preflight.infrastructure.parsers.memory_parser import MemoryParser
from preflight.infrastructure.parsers.platform_parser import PlatformParser
from preflight.infrastructure.probes.cpu_probe import CpuProbe
from preflight.infrastructure.probes.kernel_probe import KernelProbe
from preflight.infrastructure.probes.memory_probe import MemoryProbe
from preflight.infrastructure.probes.platform_probe import PlatformProbe


class FakeExecutor:
    def __init__(self, responses: dict[str, CommandResult]) -> None:
        self._responses = responses

    def run(self, command: str) -> CommandResult:
        return self._responses[command]


def test_platform_probe_collects_platform_info() -> None:
    probe = PlatformProbe(parser=PlatformParser())
    executor = FakeExecutor(
        {
            "hostname": CommandResult("hostname", 0, "node-a", ""),
            ". /etc/os-release && printf '%s' \"$PRETTY_NAME\"": CommandResult("os", 0, "RHEL 9.4", ""),
            "uname -r": CommandResult("uname", 0, "5.14.0", ""),
            "systemd-detect-virt || true": CommandResult("virt", 0, "kvm", ""),
            "test -f /.dockerenv && printf 'container'": CommandResult("container", 1, "", ""),
        }
    )

    platform = probe.collect(executor)
    raw_results = probe.collect_raw(executor)

    assert platform.hostname == "node-a"
    assert platform.virtualization_type == "kvm"
    assert platform.is_container is False
    assert raw_results["hostname"].stdout == "node-a"


def test_cpu_probe_collects_cpu_info() -> None:
    probe = CpuProbe(parser=CpuParser())
    executor = FakeExecutor(
        {
            "lscpu": CommandResult(
                "lscpu",
                0,
                "\n".join(
                    [
                        "Architecture: x86_64",
                        "CPU(s): 8",
                        "Thread(s) per core: 2",
                        "Core(s) per socket: 4",
                        "Socket(s): 1",
                        "NUMA node(s): 1",
                    ]
                ),
                "",
            )
        }
    )

    cpu = probe.collect(executor)

    assert cpu.logical_cores == 8
    assert cpu.hyperthreading_enabled is True


def test_memory_probe_collects_memory_info() -> None:
    probe = MemoryProbe(parser=MemoryParser())
    executor = FakeExecutor(
        {
            "cat /proc/meminfo": CommandResult("meminfo", 0, "MemTotal: 1024 kB\nSwapTotal: 0 kB", ""),
            "cat /proc/sys/vm/nr_hugepages": CommandResult("hugepages", 0, "4", ""),
            "cat /sys/kernel/mm/transparent_hugepage/enabled": CommandResult(
                "thp", 0, "always [madvise] never", ""
            ),
        }
    )

    memory = probe.collect(executor)

    assert memory.total_memory_kib == 1024
    assert memory.hugepages_total == 4


def test_kernel_probe_collects_kernel_info() -> None:
    probe = KernelProbe(parser=KernelParser())
    executor = FakeExecutor(
        {
            "test -w /proc/sys/vm/swappiness && printf 'writable'": CommandResult(
                "sysctl", 0, "writable", ""
            ),
            "getenforce || printf 'unknown'": CommandResult("getenforce", 0, "Permissive", ""),
            "tuned-adm active | awk -F': ' 'NR==1 {print $2}' || true": CommandResult(
                "tuned-adm", 0, "throughput-performance", ""
            ),
        }
    )

    kernel = probe.collect(executor)

    assert kernel.sysctl_writable is True
    assert kernel.tuned_profile == "throughput-performance"
