from __future__ import annotations

import argparse
import sys
from pathlib import Path

from baseline.application.baseline_runner import BaselineRunner
from baseline.domain.models import BenchmarkConfig
from host_profile.infrastructure.host_profile_loader import HostProfileLoader
from onboard.application.onboard_runner import OnboardRunner
from onboard.infrastructure.service_compatibility_evaluator import ServiceCompatibilityEvaluator
from onboard.infrastructure.service_definition_loader import ServiceDefinitionLoader
from onboard.infrastructure.service_definition_validator import ServiceDefinitionValidator
from preflight.application.discovery_runner import DiscoveryRunner
from preflight.application.hosttune_instance import HostTuneInstance
from preflight.domain.capability_builder import CapabilityMapBuilder
from preflight.domain.models import (
    BenchmarkResult,
    CommandExecutor,
    LocalTargetConfig,
    SshTargetConfig,
)
from preflight.infrastructure.config_loader import ConfigLoader
from preflight.infrastructure.executors.local_executor import LocalCommandExecutor
from preflight.infrastructure.executors.ssh_executor import SshCommandExecutor
from preflight.infrastructure.parsers.cgroup_parser import CgroupParser
from preflight.infrastructure.parsers.cpu_parser import CpuParser
from preflight.infrastructure.parsers.irq_parser import IrqParser
from preflight.infrastructure.parsers.kernel_parser import KernelParser
from preflight.infrastructure.parsers.memory_parser import MemoryParser
from preflight.infrastructure.parsers.network_parser import NetworkParser
from preflight.infrastructure.parsers.platform_parser import PlatformParser
from preflight.infrastructure.parsers.storage_parser import StorageParser
from preflight.infrastructure.probes.cgroup_probe import CgroupProbe
from preflight.infrastructure.probes.cpu_probe import CpuProbe
from preflight.infrastructure.probes.irq_probe import IrqProbe
from preflight.infrastructure.probes.kernel_probe import KernelProbe
from preflight.infrastructure.probes.memory_probe import MemoryProbe
from preflight.infrastructure.probes.network_probe import NetworkProbe
from preflight.infrastructure.probes.platform_probe import PlatformProbe
from preflight.infrastructure.probes.storage_probe import StorageProbe
from preflight.infrastructure.runtime_artifact_store import RuntimeArtifactStore
from preflight.interfaces.console_reporter import ConsoleReporter
from preflight.interfaces.execution_logger import (
    ColorExecutionLogger,
    ExecutionLogger,
    NullExecutionLogger,
)
from snapshot.application.snapshot_runner import SnapshotRunner
from tune.application.apply_coordinator import (
    ApplyCoordinator,
    CpuGovernorApplier,
    NetworkRingApplier,
    NginxDirectiveApplier,
    NicQueueApplier,
    PrlimitApplier,
    SysctlApplier,
    SystemdCgroupControlApplier,
    SystemdUnitLimitApplier,
)
from tune.application.attribution_verifier import AttributionVerifier
from tune.application.benchmark_executor import TuneBenchmarkExecutor
from tune.application.candidate_catalog_builder import CandidateCatalogBuilder
from tune.application.health_validator import HealthValidator
from tune.application.hypothesis_factory import build_langgraph_hypothesis_generator
from tune.application.hypothesis_generator import DeterministicHypothesisGenerator
from tune.application.phase_controller import PhaseController
from tune.application.pre_apply_validator import PreApplyValidator
from tune.application.result_evaluator import ResultEvaluator
from tune.application.rollback_coordinator import RollbackCoordinator
from tune.application.rule_based_triage import RuleBasedTriage, TriageRulesLoader
from tune.application.tune_engine import TuneEngine
from tune.application.tune_recorder import TuneRecorder

SERVICE_REGISTRY_PATH = Path("service-monitor")


class ShellBenchmarkRunner:
    def __init__(self, command: str) -> None:
        self._command = command

    def run(self, executor: CommandExecutor) -> BenchmarkResult:
        result = executor.run(self._command)
        primary_value = float(result.stdout.splitlines()[0]) if result.stdout else 0.0
        return BenchmarkResult(
            command=self._command,
            exit_code=result.exit_code,
            primary_metric_name="score",
            primary_metric_value=primary_value,
            raw_output=result.stdout,
        )


def build_executor(target: LocalTargetConfig | SshTargetConfig) -> CommandExecutor:
    if isinstance(target, LocalTargetConfig):
        return LocalCommandExecutor()
    return SshCommandExecutor(target)


def build_discovery_runner(
    _benchmark_command: str | None,
    logger: ExecutionLogger | None = None,
) -> DiscoveryRunner:
    return DiscoveryRunner(
        platform_probe=PlatformProbe(parser=PlatformParser()),
        cpu_probe=CpuProbe(parser=CpuParser()),
        memory_probe=MemoryProbe(parser=MemoryParser()),
        kernel_probe=KernelProbe(parser=KernelParser()),
        network_probe=NetworkProbe(parser=NetworkParser()),
        storage_probe=StorageProbe(parser=StorageParser()),
        irq_probe=IrqProbe(parser=IrqParser()),
        cgroup_probe=CgroupProbe(parser=CgroupParser()),
        capability_builder=CapabilityMapBuilder(),
        logger=logger or NullExecutionLogger(),
    )


def build_instance(
    verbose: bool = False,
    debug: bool = False,
    color: bool = True,
) -> HostTuneInstance:
    logger: ExecutionLogger
    if debug or verbose:
        logger = ColorExecutionLogger(debug=debug, color=color)
    else:
        logger = NullExecutionLogger()
    return HostTuneInstance(
        config_loader=ConfigLoader(),
        discovery_runner_factory=lambda benchmark_command: build_discovery_runner(
            benchmark_command,
            logger=logger,
        ),
        onboard_runner_factory=lambda: build_onboard_runner(logger),
        snapshot_runner_factory=lambda: build_snapshot_runner(logger),
        baseline_runner_factory=lambda benchmark_config: build_baseline_runner(
            benchmark_config,
            logger,
        ),
        executor_factory=build_executor,
        artifact_store=RuntimeArtifactStore(),
        logger=logger,
        host_profile_loader=HostProfileLoader(),
    )


def build_onboard_runner(logger: ExecutionLogger | None = None) -> OnboardRunner:
    return OnboardRunner(
        loader=ServiceDefinitionLoader(registry_path=SERVICE_REGISTRY_PATH),
        validator=ServiceDefinitionValidator(),
        evaluator=ServiceCompatibilityEvaluator(),
        logger=logger or NullExecutionLogger(),
    )


def build_snapshot_runner(logger: ExecutionLogger | None = None) -> SnapshotRunner:
    return SnapshotRunner(logger=logger or NullExecutionLogger())


def build_baseline_runner(
    benchmark_config: BenchmarkConfig,
    logger: ExecutionLogger | None = None,
) -> BaselineRunner:
    return BaselineRunner(
        benchmark_config=benchmark_config,
        logger=logger or NullExecutionLogger(),
    )


def build_tune_engine(
    logger: ExecutionLogger | None = None,
    *,
    prompt_compression: bool = False,
    kb_batch_apply: bool = False,
    skip_marginal_attribution: bool = False,
    marginal_attribution_multiplier: float = 2.0,
    use_unified_resolver: bool = False,
    dependency_graph_path: str = "tuning-dependency-graph.yaml",
    compiled_path: Path | None = None,
    mlflow_enabled: bool = False,
    mlflow_tracking_uri: str = "http://localhost:5000",
    mlflow_experiment_name: str = "hosttune",
    skip_attribution: bool = False,
    stopping_marginal_gain_threshold: float = 0.03,
    stopping_marginal_gain_iterations: int = 2,
    stopping_historical_best_pct: float = 0.85,
    stopping_telemetry_stop_enabled: bool = True,
) -> TuneEngine:
    execution_logger = logger or NullExecutionLogger()
    try:
        hypothesis_generator = build_langgraph_hypothesis_generator(
            logger=execution_logger,
            prompt_compression=prompt_compression,
            compiled_path=compiled_path,
        )
        mode_label = "compressed" if prompt_compression else "standard"
        execution_logger.stage_detail(
            "tune",
            f"Using LangGraph-backed hypothesis generation (prompt={mode_label}).",
        )
    except (ImportError, ModuleNotFoundError, ValueError) as error:
        execution_logger.stage_detail(
            "tune",
            f"LangGraph unavailable, falling back to deterministic hypotheses: {error}",
        )
        hypothesis_generator = DeterministicHypothesisGenerator()
    tune_benchmark_executor = TuneBenchmarkExecutor(logger=execution_logger)
    triage_path = Path("triage-rules.yaml")
    triage = (
        RuleBasedTriage(ruleset=TriageRulesLoader().load(triage_path))
        if triage_path.exists()
        else None
    )
    unified_resolver = None
    if use_unified_resolver:
        graph_path = Path(dependency_graph_path)
        if graph_path.exists():
            from tune.application.unified_resolver import UnifiedResolver

            unified_resolver = UnifiedResolver(
                graph_path=graph_path,
                triage=triage,
                logger=execution_logger,
            )
            execution_logger.stage_detail(
                "tune", "Unified resolver enabled."
            )
    return TuneEngine(
        candidate_catalog_builder=CandidateCatalogBuilder(),
        phase_controller=PhaseController(
            use_unified_resolver=use_unified_resolver,
            marginal_gain_threshold=stopping_marginal_gain_threshold,
            marginal_gain_iterations=stopping_marginal_gain_iterations,
            historical_best_pct=stopping_historical_best_pct,
            telemetry_stop_enabled=stopping_telemetry_stop_enabled,
        ),
        hypothesis_generator=hypothesis_generator,
        triage=triage,
        unified_resolver=unified_resolver,
        kb_batch_apply=kb_batch_apply,
        skip_marginal_attribution=skip_marginal_attribution,
        marginal_attribution_multiplier=marginal_attribution_multiplier,
        apply_coordinator=ApplyCoordinator(
            service_directive_applier=NginxDirectiveApplier(),
            sysctl_applier=SysctlApplier(),
            network_ring_applier=NetworkRingApplier(),
            runtime_limit_applier=PrlimitApplier(),
            systemd_unit_limit_applier=SystemdUnitLimitApplier(),
            cgroup_resource_control_applier=SystemdCgroupControlApplier(),
            nic_queue_applier=NicQueueApplier(),
            cpu_governor_applier=CpuGovernorApplier(),
        ),
        pre_apply_validator=PreApplyValidator(),
        health_validator=HealthValidator(),
        benchmark_executor=tune_benchmark_executor,
        attribution_verifier=AttributionVerifier(
            benchmark_executor=tune_benchmark_executor,
            health_validator=HealthValidator(),
        ),
        result_evaluator=ResultEvaluator(),
        rollback_coordinator=RollbackCoordinator(),
        recorder=TuneRecorder(),
        logger=execution_logger,
        compiled_path=compiled_path,
        mlflow_enabled=mlflow_enabled,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        skip_attribution=skip_attribution,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run step1 discovery and optional baseline benchmark."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print customer-safe stage and summary logs to stderr.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print internal debug logs including exact commands.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour output (default: auto-detect from terminal).",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=Path("config.yaml"),
        type=Path,
        help="Path to the YAML configuration file. Defaults to config.yaml.",
    )
    args = parser.parse_args()

    instance = build_instance(
        verbose=args.verbose or args.debug,
        debug=args.debug,
        color=not args.no_color,
    )
    try:
        loaded_config = instance.config_loader.load(args.config)
        preflight = instance.load_preflight(args.config)
        instance.load_host_profile(args.config)
        onboard = instance.load_onboard(args.config)
        snapshot = instance.load_snapshot(args.config)
        baseline = None
        tune = None
        if loaded_config.benchmark_config is not None:
            instance.clear_environment_blockers(args.config)
            baseline = instance.load_baseline(args.config)
            tune_logger = getattr(instance, "logger", None)
            tune = instance.run_tune(
                args.config,
                build_tune_engine(
                    tune_logger,
                    prompt_compression=loaded_config.prompt_compression,
                    kb_batch_apply=loaded_config.kb_batch_apply,
                    skip_marginal_attribution=loaded_config.skip_marginal_attribution,
                    marginal_attribution_multiplier=loaded_config.marginal_attribution_multiplier,
                    use_unified_resolver=loaded_config.use_unified_resolver,
                    dependency_graph_path=loaded_config.dependency_graph_path,
                    compiled_path=Path(loaded_config.dspy_compiled_path)
                    if loaded_config.dspy_compiled_path
                    else None,
                    mlflow_enabled=loaded_config.mlflow_enabled,
                    mlflow_tracking_uri=loaded_config.mlflow_tracking_uri,
                    mlflow_experiment_name=loaded_config.mlflow_experiment_name,
                    skip_attribution=loaded_config.skip_attribution,
                    stopping_marginal_gain_threshold=loaded_config.stopping_marginal_gain_threshold,
                    stopping_marginal_gain_iterations=loaded_config.stopping_marginal_gain_iterations,
                    stopping_historical_best_pct=loaded_config.stopping_historical_best_pct,
                    stopping_telemetry_stop_enabled=loaded_config.stopping_telemetry_stop_enabled,
                ),
            )
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(
        ConsoleReporter().render_runtime(
            preflight=preflight,
            onboard=onboard,
            snapshot=snapshot,
            baseline=baseline,
            tune=tune,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
