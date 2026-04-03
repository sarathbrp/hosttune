from baseline.domain.models import BaselineResult, BenchmarkConfig, WorkloadBenchmarkResult
from onboard.domain.models import CompatibilityReport, OnboardResult
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight.domain.models import (
    CapabilityFlag,
    CapabilityMap,
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
from tune.domain.hypothesis_models import CandidateSource
from tune.domain.tune_context import TuneContext

from tests.onboard.test_service_definition_validator import build_valid_definition


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
                CapabilityFlag("network_queue_tuning", True, "supported"),
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

    candidates = CandidateCatalogBuilder().build(context)

    candidate_keys = {candidate.parameter_key for candidate in candidates}
    assert "service.directive.worker_processes" in candidate_keys
    assert "sysctl.net.core.somaxconn" in candidate_keys
    assert any(
        candidate.source == CandidateSource.SERVICE_DIRECTIVE for candidate in candidates
    )
    assert any(candidate.source == CandidateSource.SERVICE_SYSCTL for candidate in candidates)
