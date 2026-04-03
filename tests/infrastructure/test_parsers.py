from preflight.domain.models import CommandResult
from preflight.infrastructure.parsers.cpu_parser import CpuParser
from preflight.infrastructure.parsers.kernel_parser import KernelParser
from preflight.infrastructure.parsers.memory_parser import MemoryParser
from preflight.infrastructure.parsers.platform_parser import PlatformParser


def test_platform_parser_normalizes_runtime_state() -> None:
    platform = PlatformParser().parse(
        hostname=CommandResult("hostname", 0, "node-a", ""),
        os_release=CommandResult("os", 0, "RHEL 9.4", ""),
        kernel=CommandResult("uname", 0, "5.14.0", ""),
        virtualization=CommandResult("virt", 0, "kvm", ""),
        container=CommandResult("container", 0, "container", ""),
    )

    assert platform.virtualization_type == "kvm"
    assert platform.is_container is True


def test_cpu_parser_extracts_topology() -> None:
    cpu = CpuParser().parse(
        CommandResult(
            "lscpu",
            0,
            "\n".join(
                [
                    "Architecture: x86_64",
                    "CPU(s): 16",
                    "Thread(s) per core: 2",
                    "Core(s) per socket: 8",
                    "Socket(s): 1",
                    "NUMA node(s): 2",
                ]
            ),
            "",
        )
    )

    assert cpu.logical_cores == 16
    assert cpu.hyperthreading_enabled is True


def test_memory_parser_extracts_memory_settings() -> None:
    memory = MemoryParser().parse(
        meminfo=CommandResult("meminfo", 0, "MemTotal: 1024 kB\nSwapTotal: 512 kB", ""),
        hugepages=CommandResult("hugepages", 0, "32", ""),
        thp_mode=CommandResult("thp", 0, "always [madvise] never", ""),
    )

    assert memory.total_memory_kib == 1024
    assert memory.swap_total_kib == 512
    assert memory.hugepages_total == 32


def test_kernel_parser_extracts_permissions() -> None:
    kernel = KernelParser().parse(
        sysctl_probe=CommandResult("sysctl", 0, "writable", ""),
        selinux_mode=CommandResult("getenforce", 0, "Permissive", ""),
        tuned_profile=CommandResult("tuned-adm", 0, "throughput-performance", ""),
    )

    assert kernel.sysctl_writable is True
    assert kernel.selinux_mode == "Permissive"
