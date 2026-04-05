from preflight.domain.models import (
    CapabilityFlag,
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
from preflight.interfaces.console_reporter import ConsoleReporter
from snapshot.domain.models import SnapshotResult
from baseline.domain.models import BaselineResult, WorkloadBenchmarkResult
from onboard.domain.models import CompatibilityReport, OnboardResult
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from onboard.domain.models import ApplyMode
from tune.domain.benchmark_models import BenchmarkSample, BenchmarkWorkloadSummary, TuneBenchmarkResult
from tune.domain.iteration_record import TuneIterationRecord
from tune.domain.hypothesis_models import (
    CandidateSource,
    HypothesisStatus,
    TunePhase,
    TuningHypothesis,
)
from tune.domain.tune_state import TuneState
from tune.domain.apply_models import AppliedChange
from tune.domain.tuning_layer import TuningLayer

from tests.onboard.test_service_definition_validator import build_valid_definition


def test_console_reporter_serializes_snapshot() -> None:
    snapshot = DiscoverySnapshot(
        target=LocalTargetConfig(),
        policy=EngagementPolicy(
            allow_reload=True,
            allow_restart=False,
            allow_reboot=False,
            rollback_required=True,
            max_iterations=3,
            benchmark_stability_threshold=0.1,
        ),
        platform_summary="bare_metal_linux",
        platform=PlatformInfo(
            hostname="node-a",
            operating_system="RHEL",
            kernel_version="5.14.0",
            virtualization_type="none",
            is_container=False,
        ),
        cpu=CpuInfo(
            architecture="x86_64",
            logical_cores=16,
            threads_per_core=2,
            cores_per_socket=8,
            sockets=1,
            numa_nodes=2,
            hyperthreading_enabled=True,
        ),
        memory=MemoryInfo(
            total_memory_kib=1024,
            swap_total_kib=512,
            hugepages_total=8,
            transparent_hugepages_mode="always [madvise] never",
        ),
        kernel=KernelInfo(
            sysctl_writable=True,
            selinux_mode="Permissive",
            tuned_profile="throughput-performance",
        ),
        network=NetworkInfo(
            interface_name="eth0",
            driver_name="ixgbe",
            firmware_version="1.0.0",
            rx_ring_current=512,
            rx_ring_max=4096,
            tx_ring_current=512,
            tx_ring_max=4096,
            combined_queues=8,
            ring_buffer_tuning_supported=True,
        ),
        storage=StorageInfo(
            device_name="sda",
            device_type="ssd",
            scheduler="[mq-deadline] none",
            scheduler_meaningful=True,
        ),
        irq=IrqInfo(irqbalance_active=False, nic_irq_cpu_summary="unknown"),
        cgroup=CgroupInfo(cgroup_version="unknown", cpu_controller_available=False, memory_controller_available=False),
        capability_map=CapabilityMap(flags=(CapabilityFlag("irq_affinity", True, "supported"),)),
    )

    rendered = ConsoleReporter().render(snapshot)

    assert '"platform_summary": "bare_metal_linux"' in rendered
    assert '"hostname": "node-a"' in rendered
    assert '"irq_affinity"' in rendered


def test_console_reporter_renders_human_readable_runtime() -> None:
    reporter = ConsoleReporter()
    snapshot = DiscoverySnapshot(
        target=LocalTargetConfig(),
        policy=EngagementPolicy(
            allow_reload=True,
            allow_restart=False,
            allow_reboot=False,
            rollback_required=True,
            max_iterations=3,
            benchmark_stability_threshold=0.1,
        ),
        platform_summary="bare_metal_linux",
        platform=PlatformInfo(
            hostname="node-a",
            operating_system="RHEL",
            kernel_version="5.14.0",
            virtualization_type="none",
            is_container=False,
        ),
        cpu=CpuInfo("x86_64", 16, 2, 8, 1, 2, True),
        memory=MemoryInfo(1024, 512, 8, "always [madvise] never"),
        kernel=KernelInfo(True, "Permissive", "throughput-performance"),
        network=NetworkInfo("eth0", "ixgbe", "1.0.0", 512, 4096, 512, 4096, 8, True),
        storage=StorageInfo("sda", "ssd", "[mq-deadline] none", True),
        irq=IrqInfo(irqbalance_active=False, nic_irq_cpu_summary="unknown"),
        cgroup=CgroupInfo(cgroup_version="unknown", cpu_controller_available=False, memory_controller_available=False),
        capability_map=CapabilityMap(flags=()),
    )
    onboard = OnboardResult(
        service_name="nginx",
        service=ServiceDefinitionValidator().validate(build_valid_definition()),
        compatibility=CompatibilityReport(compatible=True, findings=()),
    )
    runtime_snapshot = SnapshotResult(
        service_name="nginx",
        snapshot_directory="/var/tmp/hosttune",
        captured_paths=("/etc/nginx/nginx.conf",),
        runtime_state_output="nginx -T",
        process_state={"pid_file": "1234"},
        restore_sequence=("systemctl restart nginx",),
    )
    baseline = BaselineResult(
        service_name="nginx",
        benchmark_command="TARGET_HOST=10.1.90.178 /root/hackathon-tools/benchmark.sh hosttune",
        benchmark_target="10.1.90.178",
        workload_results=(
            WorkloadBenchmarkResult(
                workload_name="homepage",
                result_path="/root/hackathon-results/hosttune_homepage.json",
                requests_per_second=1234.5,
                total_requests=9999,
                average_latency_ms=4.2,
            ),
        ),
        expected_variance=0.05,
        warmup_seconds=10,
        guardrail_metrics=("p95_latency",),
        comparison_output=(
            "=== Performance Comparison ===\n"
            "Workload   |  Baseline (rps) |   Current (rps) |          Change |     Status\n"
            "--------------------------------------------------------------------------------\n"
            "homepage   |          374706 |         1085909 |          189.8% | IMPROVED\n"
        ),
    )
    tune = TuneState.initialize(3)
    tune.current_phase = TunePhase.DOMAIN_FOCUS
    tune.total_iterations = 2
    tune.history = [
        type(
            "HistoryRecord",
            (),
            {"iteration_number": 1, "status": HypothesisStatus.ACCEPTED},
        )(),
        type(
            "HistoryRecord",
            (),
            {"iteration_number": 2, "status": HypothesisStatus.REJECTED},
        )(),
    ]
    tune.best_configuration = type(
        "BestConfig",
        (),
        {
            "score": 0.9203,
            "parameter_values": {"service.directive.access_log": "off"},
            "iteration_number": 1,
        },
    )()
    tune.active_changes = {
        "service.directive.access_log": AppliedChange(
            hypothesis=TuningHypothesis(
                phase=TunePhase.WIDE_SWEEP,
                parameter_key="service.directive.access_log",
                parameter_name="access_log",
                domain="service_config",
                tuning_layer=TuningLayer.SERVICE,
                proposed_value="off",
                source=CandidateSource.SERVICE_DIRECTIVE,
                apply_mode=ApplyMode.RELOAD,
                rationale="test",
            ),
            target_path="/etc/nginx/nginx.conf",
            previous_value="/var/log/nginx/access.log",
            applied_value="off",
            apply_mode=ApplyMode.RELOAD,
            apply_command="",
            rollback_command="",
        )
    }
    tune.iteration_records = [
        TuneIterationRecord(
            iteration_number=1,
            phase=TunePhase.WIDE_SWEEP,
            hypothesis=type(
                "Hypothesis",
                (),
                {
                    "model_usage": None,
                    "parameter_key": "service.directive.access_log",
                    "proposed_value": "off",
                },
            )(),
            applied_change=None,
            validation_result=None,
            benchmark_result=TuneBenchmarkResult(
                validation_result=None,
                benchmark_command="benchmark",
                run_count=1,
                stable=True,
                variance_threshold=0.05,
                workload_summaries=(
                    BenchmarkWorkloadSummary(
                        workload_name="homepage",
                        samples=(BenchmarkSample(1, 1085909.0, 9999, 4.2),),
                        median_requests_per_second=1085909.0,
                        median_total_requests=9999,
                        median_latency_ms=4.2,
                        relative_variance=0.0,
                        stable=True,
                    ),
                ),
            ),
            evaluation_result=None,
            attribution_verification=None,
            active_parameter_keys=(),
            started_at_utc="2026-04-04T00:00:00+00:00",
            completed_at_utc="2026-04-04T00:00:01+00:00",
            duration_seconds=1.0,
        )
    ]

    rendered = reporter.render_runtime(snapshot, onboard, runtime_snapshot, baseline, tune)

    assert "Preflight" in rendered
    assert "Kernel sysctl profile" in rendered
    assert "Onboard" in rendered
    assert "Snapshot" in rendered
    assert "Baseline" in rendered
    assert "Tune" in rendered
    assert "homepage" in rendered
    assert "Comparison" in rendered
    assert "| baseline_rps " in rendered
    assert "Best iteration: 1" in rendered
    assert "Best comparison" in rendered
    assert "| best_rps " in rendered
    assert "Best iteration config: service.directive.access_log=off" in rendered
    assert "Final retained config: service.directive.access_log=off" in rendered
    assert "Iteration history" in rendered
    assert "service.directive.access_log" in rendered
    assert "homepage=1234.50" in rendered
    assert "homepage=1085909.00" in rendered
    assert "+----" in rendered
