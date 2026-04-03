from onboard.domain.models import (
    CompatibilityReport,
    FindingSeverity,
    ServiceDefinition,
)
from onboard.infrastructure.service_compatibility_evaluator import ServiceCompatibilityEvaluator
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight.domain.models import (
    CapabilityMap,
    CgroupInfo,
    CpuInfo,
    DiscoverySnapshot,
    EngagementPolicy,
    IrqInfo,
    KernelInfo,
    LocalTargetConfig,
    MemoryInfo,
    NetworkInfo,
    PlatformInfo,
    StorageInfo,
)

from tests.onboard.test_service_definition_validator import build_valid_definition


class FakeExecutor:
    def __init__(self, command_results: dict[str, int]) -> None:
        self._command_results = command_results

    def run(self, command: str):  # type: ignore[no-untyped-def]
        exit_code = self._command_results.get(command, 0)
        from preflight.domain.models import CommandResult

        return CommandResult(command=command, exit_code=exit_code, stdout="", stderr="")


def build_preflight_snapshot(os_name: str = "Red Hat Enterprise Linux 9.4") -> DiscoverySnapshot:
    return DiscoverySnapshot(
        target=LocalTargetConfig(),
        policy=EngagementPolicy(
            allow_reload=False,
            allow_restart=False,
            allow_reboot=False,
            rollback_required=True,
            max_iterations=10,
            benchmark_stability_threshold=0.1,
        ),
        platform_summary="bare_metal_linux",
        platform=PlatformInfo(
            hostname="node-a",
            operating_system=os_name,
            kernel_version="5.14.0",
            virtualization_type="none",
            is_container=False,
        ),
        cpu=CpuInfo("x86_64", 16, 2, 8, 1, 2, True),
        memory=MemoryInfo(1024, 512, 0, "[always] madvise never"),
        kernel=KernelInfo(True, "Enforcing", "unknown"),
        network=NetworkInfo("eth0", "ixgbe", "1.2.3", 512, 4096, 512, 4096, 8, True),
        storage=StorageInfo("sda", "ssd", "[mq-deadline] none", True),
        irq=IrqInfo(irqbalance_active=False, nic_irq_cpu_summary="unknown"),
        cgroup=CgroupInfo(cgroup_version="unknown", cpu_controller_available=False, memory_controller_available=False),
        capability_map=CapabilityMap(flags=()),
    )


def build_service() -> ServiceDefinition:
    return ServiceDefinitionValidator().validate(build_valid_definition())


def test_evaluator_accepts_compatible_service() -> None:
    report = ServiceCompatibilityEvaluator().evaluate(
        preflight=build_preflight_snapshot(),
        service=build_service(),
        executor=FakeExecutor({}),
    )

    assert isinstance(report, CompatibilityReport)
    assert report.compatible is True


def test_evaluator_reports_missing_systemd_unit() -> None:
    report = ServiceCompatibilityEvaluator().evaluate(
        preflight=build_preflight_snapshot(),
        service=build_service(),
        executor=FakeExecutor({"systemctl status nginx.service >/dev/null 2>&1": 1}),
    )

    assert report.compatible is False
    assert any(finding.severity is FindingSeverity.ERROR for finding in report.findings)
