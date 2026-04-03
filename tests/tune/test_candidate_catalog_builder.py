from dataclasses import replace
from typing import cast

from baseline.domain.models import BaselineResult, BenchmarkConfig, WorkloadBenchmarkResult
from onboard.domain.models import CompatibilityReport, OnboardResult, PriorityTier
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight.domain.models import (
    CapabilityFlag,
    CapabilityMap,
    CommandResult,
    CpuInfo,
    DiscoverySnapshot,
    EngagementPolicy,
    KernelInfo,
    LocalTargetConfig,
    MemoryInfo,
    NetworkInfo,
    PlatformInfo,
    StorageInfo,
)
from snapshot.domain.models import SnapshotResult
from tune.application.candidate_catalog_builder import CandidateCatalogBuilder
from tune.domain.hypothesis_models import CandidateAvailability, CandidateSource
from tune.domain.tune_context import TuneContext
from tune.domain.tuning_layer import TuningLayer, tuning_layer_for_parameter_key

from tests.onboard.test_service_definition_validator import build_valid_definition


class FakeExecutor:
    def run(self, command: str) -> CommandResult:
        if command.startswith("grep -E"):
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="worker_processes 112;",
                stderr="",
            )
        if command.startswith("sysctl -n"):
            return CommandResult(command=command, exit_code=0, stdout="4096", stderr="")
        if command.startswith("cat "):
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="12345\n",
                stderr="",
            )
        if "awk " in command and "/proc/" in command and "limits" in command:
            return CommandResult(
                command=command,
                exit_code=0,
                stdout="8192 1048576\n",
                stderr="",
            )
        return CommandResult(command=command, exit_code=0, stdout="", stderr="")


def build_tune_context() -> TuneContext:
    preflight = DiscoverySnapshot(
        target=LocalTargetConfig(),
        policy=EngagementPolicy(
            allow_reload=True,
            allow_restart=False,
            allow_reboot=False,
            rollback_required=True,
            max_iterations=10,
            benchmark_stability_threshold=0.1,
        ),
        platform_summary="bare_metal_linux",
        platform=PlatformInfo("node-a", "RHEL 9.4", "5.14.0", "none", False),
        cpu=CpuInfo("x86_64", 112, 2, 28, 2, 2, True),
        memory=MemoryInfo(1024, 512, 0, "always madvise [never]"),
        kernel=KernelInfo(True, "Permissive", "unknown"),
        network=NetworkInfo("eth0", "ixgbe", "1.0.0", 512, 4096, 512, 4096, 8, True),
        storage=StorageInfo("sda", "ssd", "[none] mq-deadline", True),
        capability_map=CapabilityMap(
            flags=(
                CapabilityFlag("kernel_sysctl_tuning", True, "supported"),
                CapabilityFlag("network_ring_buffer_tuning", True, "supported"),
                CapabilityFlag("network_queue_tuning", True, "supported"),
                CapabilityFlag("runtime_prlimit_tuning", True, "supported"),
            )
        ),
    )
    onboard = OnboardResult(
        service_name="nginx",
        service=ServiceDefinitionValidator().validate(build_valid_definition()),
        compatibility=CompatibilityReport(compatible=True, findings=()),
    )
    snapshot = SnapshotResult(
        service_name="nginx",
        snapshot_directory="/var/tmp/hosttune",
        captured_paths=("/etc/nginx/nginx.conf",),
        runtime_state_output="nginx -T",
        process_state={"pid_file": "1234"},
        restore_sequence=("systemctl restart nginx",),
    )
    baseline = BaselineResult(
        service_name="nginx",
        benchmark_command="benchmark.sh hosttune",
        benchmark_target="10.1.90.178",
        workload_results=(
            WorkloadBenchmarkResult("homepage", "/tmp/homepage.json", 1000.0, 10000, 2.0),
        ),
        expected_variance=0.05,
        warmup_seconds=10,
        guardrail_metrics=("p95_latency",),
        comparison_output=None,
    )
    return TuneContext(
        preflight=preflight,
        onboard=onboard,
        snapshot=snapshot,
        baseline=baseline,
        benchmark_config=BenchmarkConfig(
            runner_target=LocalTargetConfig(),
            contestant_name="hosttune",
            script_path="/root/hackathon-tools/benchmark.sh",
            results_directory="/root/hackathon-results",
            workloads=("homepage", "small"),
            compare_script_path="/root/hackathon-tools/compare-results.sh",
        ),
        artifacts=None,
    )


def test_candidate_catalog_builder_includes_service_directives_and_sysctls() -> None:
    context = build_tune_context()

    candidates = CandidateCatalogBuilder().build(context, FakeExecutor())

    candidate_keys = {candidate.parameter_key for candidate in candidates}
    assert "service.directive.worker_processes" in candidate_keys
    assert "sysctl.net.core.somaxconn" in candidate_keys
    assert "sysctl.net.ipv4.ip_local_port_range" in candidate_keys
    assert "network.ring.rx" in candidate_keys
    assert "runtime.prlimit.nofile_soft" in candidate_keys
    prlimit_nofile = next(
        candidate
        for candidate in candidates
        if candidate.parameter_key == "runtime.prlimit.nofile_soft"
    )
    assert prlimit_nofile.tuning_layer is TuningLayer.RUNTIME
    assert prlimit_nofile.source is CandidateSource.RUNTIME_PRLIMIT
    assert prlimit_nofile.current_value == "8192"
    assert "service.directive.worker_rlimit_nofile" in candidate_keys
    assert any(
        candidate.source == CandidateSource.SERVICE_DIRECTIVE for candidate in candidates
    )
    assert any(candidate.source == CandidateSource.SERVICE_SYSCTL for candidate in candidates)
    assert candidates[0].priority_tier is PriorityTier.HIGH
    worker_processes = next(
        candidate
        for candidate in candidates
        if candidate.parameter_key == "service.directive.worker_processes"
    )
    assert worker_processes.current_value == "112"
    assert worker_processes.priority_tier is PriorityTier.HIGH
    worker_rlimit = next(
        candidate
        for candidate in candidates
        if candidate.parameter_key == "service.directive.worker_rlimit_nofile"
    )
    assert worker_rlimit.domain == "runtime"
    port_range = next(
        candidate
        for candidate in candidates
        if candidate.parameter_key == "sysctl.net.ipv4.ip_local_port_range"
    )
    assert port_range.priority_tier is PriorityTier.MEDIUM
    rx_ring = next(candidate for candidate in candidates if candidate.parameter_key == "network.ring.rx")
    assert rx_ring.priority_tier is PriorityTier.MEDIUM


def test_candidate_catalog_applies_yaml_tuning_layer_overrides() -> None:
    data = build_valid_definition()
    surface = cast(dict[str, object], data["tunable_surface"])
    data = {
        **data,
        "tunable_surface": {
            **surface,
            "relevant_sysctls": [
                {"name": "net.core.somaxconn", "priority_tier": "high", "tuning_layer": "service"},
            ],
            "network_ring_tuning_layer": "runtime",
        },
    }
    service = ServiceDefinitionValidator().validate(data)
    base = build_tune_context()
    context = replace(base, onboard=replace(base.onboard, service=service))
    candidates = CandidateCatalogBuilder().build(context, FakeExecutor())
    somaxconn = next(c for c in candidates if c.parameter_key == "sysctl.net.core.somaxconn")
    assert somaxconn.tuning_layer is TuningLayer.SERVICE
    rx_ring = next(c for c in candidates if c.parameter_key == "network.ring.rx")
    assert rx_ring.tuning_layer is TuningLayer.RUNTIME
    assert tuning_layer_for_parameter_key(somaxconn.parameter_key) is TuningLayer.KERNEL
    assert tuning_layer_for_parameter_key(rx_ring.parameter_key) is TuningLayer.NETWORK


def test_catalog_marks_sysctl_deferred_when_kernel_network_is_reboot() -> None:
    data = build_valid_definition()
    restart = cast(dict[str, object], data["restart"])
    categories = cast(dict[str, object], restart["change_categories"])
    data = {
        **data,
        "restart": {
            **restart,
            "change_categories": {**categories, "kernel_network": "reboot"},
        },
    }
    service = ServiceDefinitionValidator().validate(data)
    base = build_tune_context()
    context = replace(base, onboard=replace(base.onboard, service=service))
    candidates = CandidateCatalogBuilder().build(context, FakeExecutor())
    somaxconn = next(c for c in candidates if c.parameter_key == "sysctl.net.core.somaxconn")
    assert somaxconn.availability is CandidateAvailability.DEFERRED
    assert somaxconn.apply_mode.value == "reboot"
    assert all(
        c.availability is CandidateAvailability.ACTIVE
        for c in candidates
        if c.parameter_key.startswith("service.directive.")
    )
