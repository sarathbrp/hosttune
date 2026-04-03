from preflight.domain.models import CommandResult
from preflight.infrastructure.parsers.cgroup_parser import CgroupParser
from preflight.infrastructure.parsers.cpu_parser import CpuParser
from preflight.infrastructure.parsers.irq_parser import IrqParser
from preflight.infrastructure.parsers.kernel_parser import KernelParser
from preflight.infrastructure.parsers.memory_parser import MemoryParser
from preflight.infrastructure.parsers.platform_parser import PlatformParser
from preflight.infrastructure.parsers.storage_parser import StorageParser


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
        sysctl_profile_dump=CommandResult("sh", 0, "net.core.somaxconn=128\nvm.swappiness=60\n", ""),
    )

    assert kernel.sysctl_writable is True
    assert kernel.selinux_mode == "Permissive"
    assert ("net.core.somaxconn", "128") in kernel.sysctl_profile
    assert ("vm.swappiness", "60") in kernel.sysctl_profile


def test_storage_parser_converts_readahead_sectors_to_kb() -> None:
    storage = StorageParser().parse(
        device_name=CommandResult("dev", 0, "sda", ""),
        rotational=CommandResult("rot", 0, "0", ""),
        scheduler=CommandResult("sched", 0, "[mq-deadline] none", ""),
        readahead=CommandResult("blockdev", 0, "256", ""),
    )

    assert storage.readahead_kb == 128  # 256 sectors * 512 bytes / 1024 = 128 KiB


def test_storage_parser_handles_missing_readahead() -> None:
    storage = StorageParser().parse(
        device_name=CommandResult("dev", 0, "sda", ""),
        rotational=CommandResult("rot", 0, "0", ""),
        scheduler=CommandResult("sched", 0, "[mq-deadline] none", ""),
        readahead=CommandResult("blockdev", 1, "", ""),
    )

    assert storage.readahead_kb == -1


def test_irq_parser_detects_active_irqbalance() -> None:
    irq = IrqParser().parse(
        irqbalance_status=CommandResult("systemctl", 0, "active", ""),
        nic_irq_cpu_list=CommandResult("awk", 0, "0-3", ""),
    )

    assert irq.irqbalance_active is True
    assert irq.nic_irq_cpu_summary == "0-3"


def test_irq_parser_handles_inactive_irqbalance_and_unknown_cpus() -> None:
    irq = IrqParser().parse(
        irqbalance_status=CommandResult("systemctl", 1, "inactive", ""),
        nic_irq_cpu_list=CommandResult("awk", 0, "", ""),
    )

    assert irq.irqbalance_active is False
    assert irq.nic_irq_cpu_summary == "unknown"


def test_cgroup_parser_detects_v2() -> None:
    cgroup = CgroupParser().parse(
        cgroup_fs_type=CommandResult("stat", 0, "cgroup2fs", ""),
        controllers=CommandResult("cat", 0, "cpuset cpu io memory hugetlb pids rdma misc", ""),
    )

    assert cgroup.cgroup_version == "v2"
    assert cgroup.cpu_controller_available is True
    assert cgroup.memory_controller_available is True


def test_cgroup_parser_detects_v1() -> None:
    cgroup = CgroupParser().parse(
        cgroup_fs_type=CommandResult("stat", 0, "tmpfs", ""),
        controllers=CommandResult("cat", 1, "", ""),
    )

    assert cgroup.cgroup_version == "v1"
    assert cgroup.cpu_controller_available is False
    assert cgroup.memory_controller_available is False


def test_kernel_parser_sysctl_profile_orders_keys_and_fills_missing() -> None:
    from preflight.domain.kernel_sysctl_profile import PREFLIGHT_SYSCTL_KEYS

    profile = KernelParser.parse_sysctl_profile_stdout(
        "net.core.somaxconn=4096\n",
        keys=PREFLIGHT_SYSCTL_KEYS,
    )
    by_name = dict(profile)
    assert by_name["net.core.somaxconn"] == "4096"
    assert by_name["vm.swappiness"] == ""
