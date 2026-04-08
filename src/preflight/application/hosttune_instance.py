from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Protocol

from baseline.application.baseline_runner import BaselineRunner
from baseline.domain.models import BaselineResult, BenchmarkConfig
from onboard.application.onboard_runner import OnboardRunner
from onboard.domain.models import OnboardResult, SysctlTunable
from preflight.application.discovery_runner import DiscoveryRunner
from preflight.domain.kernel_sysctl_profile import (
    contract_sysctl_names_only_extra,
    merged_sysctl_profile_key_order,
    sysctl_profile_read_command,
)
from preflight.domain.models import (
    CommandExecutor,
    DiscoverySnapshot,
    LocalTargetConfig,
    SshTargetConfig,
)
from preflight.domain.runtime_artifacts import RuntimeArtifacts
from preflight.infrastructure.config_loader import ConfigLoader, LoadedConfig
from preflight.infrastructure.executors.logging_executor import LoggingCommandExecutor
from preflight.infrastructure.knowledge_base import KnowledgeBase, host_fingerprint_for_snapshot
from preflight.infrastructure.parsers.kernel_parser import KernelParser
from preflight.infrastructure.runtime_artifact_store import RuntimeArtifactStore
from preflight.interfaces.execution_logger import ExecutionLogger, NullExecutionLogger
from snapshot.application.snapshot_runner import SnapshotRunner
from snapshot.domain.models import SnapshotResult
from tune.domain.tune_context import TuneContext
from tune.domain.tune_state import TuneState

ExecutorFactory = Callable[[LocalTargetConfig | SshTargetConfig], CommandExecutor]
DiscoveryRunnerFactory = Callable[[str | None], DiscoveryRunner]
OnboardRunnerFactory = Callable[[], OnboardRunner]
SnapshotRunnerFactory = Callable[[], SnapshotRunner]
BaselineRunnerFactory = Callable[[BenchmarkConfig], BaselineRunner]


@dataclass
class HostTuneInstance:
    config_loader: ConfigLoader
    discovery_runner_factory: DiscoveryRunnerFactory
    onboard_runner_factory: OnboardRunnerFactory
    snapshot_runner_factory: SnapshotRunnerFactory
    baseline_runner_factory: BaselineRunnerFactory
    executor_factory: ExecutorFactory
    artifact_store: RuntimeArtifactStore
    logger: ExecutionLogger = NullExecutionLogger()
    host_profile_loader: object | None = None  # HostProfileLoader — avoids circular import
    preflight: DiscoverySnapshot | None = None
    onboard: OnboardResult | None = None
    snapshot: SnapshotResult | None = None
    baseline: BaselineResult | None = None
    tune: TuneState | None = None
    benchmark_config: BenchmarkConfig | None = None
    artifacts: RuntimeArtifacts | None = None
    host_profile: object | None = None  # HostProfile once loaded
    knowledge_base: KnowledgeBase | None = None

    def load_preflight(self, config_path: Path) -> DiscoverySnapshot:
        loaded_config = self.config_loader.load(config_path)
        self._ensure_artifacts()
        self.logger.stage_start("preflight")
        snapshot = self._run_preflight(loaded_config)
        self.logger.stage_end("preflight")
        self.preflight = snapshot
        self._persist_stage_result("preflight", snapshot)
        self._record_stage_event(
            component="snapshot_engine",
            event_type="preflight_completed",
            payload={
                "platform_summary": snapshot.platform_summary,
                "cpu_logical_cores": snapshot.cpu.logical_cores,
                "numa_nodes": snapshot.cpu.numa_nodes,
                "total_memory_kib": snapshot.memory.total_memory_kib,
                "nic_driver": snapshot.network.driver_name,
                "hostname": snapshot.platform.hostname,
            },
        )
        return snapshot

    def load_host_profile(self, config_path: Path) -> object | None:
        """Load host profile by name from config.yaml (optional stage)."""
        loaded_config = self.config_loader.load(config_path)
        name = loaded_config.host_profile_name
        if name is None or self.host_profile_loader is None:
            return None
        self.logger.stage_start("host_profile")
        profile = self.host_profile_loader.load(name)  # type: ignore[union-attr]
        self.logger.stage_end("host_profile")
        self.host_profile = profile
        self._persist_stage_result("host_profile", profile)
        return profile

    def load_onboard(self, config_path: Path) -> OnboardResult:
        if self.preflight is None:
            msg = "Preflight must be loaded before onboard."
            raise ValueError(msg)
        loaded_config = self.config_loader.load(config_path)
        self.logger.stage_start("onboard")
        executor = self._build_stage_executor(loaded_config.target, "onboard")
        runner = self.onboard_runner_factory()
        result = runner.run(
            service_name=loaded_config.service_name,
            preflight=self.preflight,
            executor=executor,
        )
        self._enrich_preflight_sysctl_profile_with_contract(result, executor)
        self.logger.stage_end("onboard")
        self.onboard = result
        self._persist_stage_result("onboard", result)
        self._record_stage_event(
            component="snapshot_engine",
            event_type="onboard_completed",
            payload={
                "service_name": result.service_name,
                "systemd_unit_name": result.service.identity.systemd_unit_name,
                "allowed_directives": sorted(result.service.tunable_surface.allowed_directives),
                "relevant_sysctls": [
                    entry.name for entry in result.service.tunable_surface.relevant_sysctls
                ],
            },
        )
        return result

    def load_snapshot(self, config_path: Path) -> SnapshotResult:
        if self.onboard is None:
            msg = "Onboard must be loaded before snapshot."
            raise ValueError(msg)
        loaded_config = self.config_loader.load(config_path)
        self.logger.stage_start("snapshot")
        executor = self._build_stage_executor(loaded_config.target, "snapshot")
        runner = self.snapshot_runner_factory()
        result = runner.run(self.onboard.service, executor)
        self.logger.stage_end("snapshot")
        self.snapshot = result
        self._persist_stage_result("snapshot", result)
        self._record_stage_event(
            component="snapshot_engine",
            event_type="snapshot_completed",
            payload={
                "snapshot_directory": result.snapshot_directory,
                "captured_paths": list(result.captured_paths),
                "restore_steps": len(result.restore_sequence),
                "process_state_keys": sorted(result.process_state.keys()),
            },
        )
        return result

    def load_baseline(self, config_path: Path) -> BaselineResult:
        if self.onboard is None:
            msg = "Onboard must be loaded before baseline."
            raise ValueError(msg)
        loaded_config = self.config_loader.load(config_path)
        if loaded_config.benchmark_config is None:
            msg = "benchmark must be configured before baseline."
            raise ValueError(msg)
        self.logger.stage_start("baseline")
        benchmark_executor = self._build_stage_executor(
            loaded_config.benchmark_config.runner_target,
            "baseline",
        )
        runner = self.baseline_runner_factory(loaded_config.benchmark_config)
        result = runner.run(self.onboard.service, benchmark_executor, loaded_config.target)
        self.logger.stage_end("baseline")
        self.baseline = result
        self.benchmark_config = loaded_config.benchmark_config
        self._persist_stage_result("baseline", result)
        self._record_stage_event(
            component="snapshot_engine",
            event_type="baseline_completed",
            payload={
                "benchmark_target": result.benchmark_target,
                "expected_variance": result.expected_variance,
                "warmup_seconds": result.warmup_seconds,
                "workloads": [
                    {
                        "workload_name": workload.workload_name,
                        "requests_per_second": workload.requests_per_second,
                        "total_requests": workload.total_requests,
                        "average_latency_ms": workload.average_latency_ms,
                    }
                    for workload in result.workload_results
                ],
            },
        )
        return result

    def build_tune_context(self) -> TuneContext:
        if self.preflight is None:
            msg = "Preflight must be loaded before building TuneContext."
            raise ValueError(msg)
        if self.onboard is None:
            msg = "Onboard must be loaded before building TuneContext."
            raise ValueError(msg)
        if self.snapshot is None:
            msg = "Snapshot must be loaded before building TuneContext."
            raise ValueError(msg)
        if self.baseline is None:
            msg = "Baseline must be loaded before building TuneContext."
            raise ValueError(msg)
        if self.benchmark_config is None:
            msg = "Benchmark config must be loaded before building TuneContext."
            raise ValueError(msg)

        if self.artifacts is not None and self.knowledge_base is not None:
            self.knowledge_base.record_run(
                run_id=self.artifacts.session_id,
                preflight=self.preflight,
                service_name=self.onboard.service_name,
                benchmark_target=self.baseline.benchmark_target,
            )
        return TuneContext(
            preflight=self.preflight,
            onboard=self.onboard,
            snapshot=self.snapshot,
            baseline=self.baseline,
            benchmark_config=self.benchmark_config,
            artifacts=self.artifacts,
            host_profile=self.host_profile,  # type: ignore[arg-type]
            knowledge_base=self.knowledge_base,
        )

    def clear_environment_blockers(self, config_path: Path) -> None:
        """Detect and fix environment blockers before baseline.

        Reads blocker definitions from the host profile and probes the
        target system. Fixes are applied only when the engagement policy
        allows environment cleanup.
        """
        if self.host_profile is None:
            return
        surface = self.host_profile.tunable_surface
        blockers = getattr(
            self.host_profile.tunable_surface, "environment_blockers", ()
        )
        hierarchy = getattr(surface, "performance_hierarchy", None)
        if not blockers and hierarchy is None:
            return
        loaded_config = self.config_loader.load(config_path)
        allow_fix = loaded_config.policy.allow_environment_cleanup
        executor = self._build_stage_executor(loaded_config.target, "env_diagnostic")
        if self.preflight is None:
            return
        interface_name = self.preflight.network.interface_name
        self.logger.stage_start("env_diagnostic")
        detected: list[str] = []
        fixed: list[str] = []
        self._run_environment_blocker_rules(
            blockers=blockers,
            executor=executor,
            interface_name=interface_name,
            allow_fix=allow_fix,
            detected=detected,
            fixed=fixed,
        )
        self._run_hierarchy_env_diagnostic(
            hierarchy=hierarchy,
            executor=executor,
            interface_name=interface_name,
            allow_fix=allow_fix,
            detected=detected,
            fixed=fixed,
        )
        if detected:
            self.logger.stage_detail(
                "env_diagnostic",
                f"Summary: {len(detected)} blocker(s) detected, "
                f"{len(fixed)} fixed.",
            )
        else:
            self.logger.stage_detail(
                "env_diagnostic", "No blockers detected."
            )
        self.logger.stage_end("env_diagnostic")

    def _run_environment_blocker_rules(
        self,
        *,
        blockers: object,
        executor: CommandExecutor,
        interface_name: str,
        allow_fix: bool,
        detected: list[str],
        fixed: list[str],
    ) -> None:
        for blocker in blockers:
            probe_cmd = blocker.probe_command.replace("{interface}", interface_name)
            result = executor.run(probe_cmd)
            output = result.stdout.strip()
            triggered = False
            if blocker.threshold_above is not None:
                try:
                    value = int(output.splitlines()[0].strip())
                    triggered = value > blocker.threshold_above
                except (ValueError, IndexError):
                    pass
            elif blocker.threshold_below is not None:
                try:
                    value = int(output.splitlines()[0].strip())
                    triggered = value < blocker.threshold_below
                except (ValueError, IndexError):
                    pass
            else:
                triggered = bool(output)
            if not triggered:
                continue
            detected.append(blocker.name)
            self.logger.stage_detail(
                "env_diagnostic",
                f"Blocker detected: {blocker.name} ({blocker.priority}) "
                f"— {blocker.detail}",
            )
            if blocker.fix_command is None:
                self.logger.stage_detail(
                    "env_diagnostic",
                    f"  {blocker.name}: no autofix (signal only).",
                )
                continue
            if not allow_fix:
                self.logger.stage_detail(
                    "env_diagnostic",
                    f"  {blocker.name}: fix available but "
                    "policy.allow_environment_cleanup=false.",
                )
                continue
            fix_cmd = blocker.fix_command.replace("{interface}", interface_name)
            self.logger.stage_detail("env_diagnostic", f"  Fixing: {fix_cmd}")
            fix_result = executor.run(fix_cmd)
            if fix_result.exit_code == 0:
                fixed.append(blocker.name)
                self.logger.stage_detail(
                    "env_diagnostic",
                    f"  {blocker.name}: fixed successfully.",
                )
            else:
                detail = fix_result.stderr.strip() or fix_result.stdout.strip()
                self.logger.stage_detail(
                    "env_diagnostic",
                    f"  {blocker.name}: fix failed "
                    f"(exit={fix_result.exit_code}): {detail}",
                )

    def _run_hierarchy_env_diagnostic(
        self,
        *,
        hierarchy: object | None,
        executor: CommandExecutor,
        interface_name: str,
        allow_fix: bool,
        detected: list[str],
        fixed: list[str],
    ) -> None:
        if hierarchy is None:
            return
        groups = tuple(getattr(hierarchy, "groups", ()) or ())
        for group in groups:
            group_id = str(getattr(group, "group_id", "group"))
            for parameter in tuple(getattr(group, "parameters", ()) or ()):
                name = str(getattr(parameter, "name", "")).strip()
                inspect_cmd = getattr(parameter, "inspect_cmd", None)
                target_perf = str(getattr(parameter, "target_perf", "")).strip()
                if not name or not isinstance(inspect_cmd, str) or inspect_cmd == "":
                    continue
                needs_fix = self._hierarchy_parameter_needs_fix(
                    parameter_name=name,
                    inspect_cmd=inspect_cmd,
                    target_perf=target_perf,
                    executor=executor,
                    interface_name=interface_name,
                )
                if not needs_fix:
                    continue
                detected_name = f"{group_id}.{name}"
                detected.append(detected_name)
                self.logger.stage_detail(
                    "env_diagnostic",
                    f"Hierarchy mismatch: {detected_name} (target={target_perf or 'unspecified'})",
                )
                fix_cmd = self._hierarchy_fix_command(name, interface_name)
                if fix_cmd is None:
                    self.logger.stage_detail(
                        "env_diagnostic",
                        f"  {detected_name}: signal only (no deterministic fix command).",
                    )
                    continue
                if not allow_fix:
                    self.logger.stage_detail(
                        "env_diagnostic",
                        f"  {detected_name}: fix available but "
                        "policy.allow_environment_cleanup=false.",
                    )
                    continue
                self.logger.stage_detail("env_diagnostic", f"  Fixing: {fix_cmd}")
                fix_result = executor.run(fix_cmd)
                if fix_result.exit_code == 0:
                    fixed.append(detected_name)
                    self.logger.stage_detail(
                        "env_diagnostic",
                        f"  {detected_name}: fixed successfully.",
                    )
                else:
                    detail = fix_result.stderr.strip() or fix_result.stdout.strip()
                    self.logger.stage_detail(
                        "env_diagnostic",
                        f"  {detected_name}: fix failed "
                        f"(exit={fix_result.exit_code}): {detail}",
                    )

    def _hierarchy_parameter_needs_fix(
        self,
        *,
        parameter_name: str,
        inspect_cmd: str,
        target_perf: str,
        executor: CommandExecutor,
        interface_name: str,
    ) -> bool:
        result = executor.run(inspect_cmd.replace("{interface}", interface_name))
        if result.exit_code != 0:
            return False
        output = result.stdout.strip()
        value = self._normalize_inspect_value(output)
        name = parameter_name.strip().lower()
        if name == "cpuquota":
            low = value.lower()
            return low not in {"", "0", "0%", "infinity"}
        if name == "memorymax":
            low = value.lower()
            return low not in {"", "infinity", "18446744073709551615"}
        if name in {"cpuweight", "ioweight"}:
            if value in {"", "[not set]"}:
                return False
            number = self._first_int(value)
            return number is not None and number != 100
        if name == "limitnofile":
            minimum = self._min_two_ints(output)
            return minimum is not None and minimum < 1_048_576
        if name == "limitnproc":
            minimum = self._min_two_ints(output)
            return minimum is not None and minimum < 65_535
        if name == "irqbalance":
            return value.strip().lower() != "active"
        if name == "nic_irq_affinity":
            return self._irq_affinity_pinned_to_cpu0(output)
        if name == "thp":
            return "[madvise]" not in output and value.strip().lower() != "madvise"
        if name == "cpu_governor":
            return value.strip().lower() != "performance"
        if name in {"tc_bandwidth_cap", "tc_latency"}:
            if target_perf.strip().lower() != "none":
                return False
            low = output.lower()
            return any(token in low for token in ("netem", "tbf", "htb"))
        return False

    def _hierarchy_fix_command(self, parameter_name: str, interface_name: str) -> str | None:
        name = parameter_name.strip().lower()
        if name == "cpuquota":
            return "systemctl set-property nginx.service CPUQuota="
        if name == "memorymax":
            return "systemctl set-property nginx.service MemoryMax="
        if name == "cpuweight":
            return "systemctl set-property nginx.service CPUWeight=100"
        if name == "ioweight":
            return "systemctl set-property nginx.service IOWeight=100"
        if name == "limitnofile":
            return self._systemd_limit_dropin_fix_command("LimitNOFILE", "1048576")
        if name == "limitnproc":
            return self._systemd_limit_dropin_fix_command("LimitNPROC", "65535")
        if name == "irqbalance":
            return "systemctl enable --now irqbalance"
        if name == "nic_irq_affinity":
            return "systemctl enable --now irqbalance && systemctl restart irqbalance"
        if name == "thp":
            return "echo madvise > /sys/kernel/mm/transparent_hugepage/enabled"
        if name == "cpu_governor":
            return "cpupower frequency-set -g performance"
        if name in {"tc_bandwidth_cap", "tc_latency"}:
            return f"tc qdisc del dev {interface_name} root 2>/dev/null || true"
        return None

    def _normalize_inspect_value(self, output: str) -> str:
        for line in output.splitlines():
            text = line.strip()
            if not text:
                continue
            if "=" in text:
                return text.split("=", 1)[1].strip()
            return text
        return ""

    def _first_int(self, text: str) -> int | None:
        match = re.search(r"(\d+)", text)
        if match is None:
            return None
        return int(match.group(1))

    def _min_two_ints(self, text: str) -> int | None:
        values = [int(value) for value in re.findall(r"\d+", text)]
        if len(values) < 2:
            return None
        return min(values[0], values[1])

    def _irq_affinity_pinned_to_cpu0(self, output: str) -> bool:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return False
        allowed = {"0", "0-0"}
        return all(line in allowed for line in lines)

    def _systemd_limit_dropin_fix_command(self, property_name: str, value: str) -> str:
        if property_name.startswith("Limit") and len(property_name) > len("Limit"):
            suffix = f"limit_{property_name[len('Limit'):].lower()}"
        else:
            suffix = re.sub(r"(?<!^)(?=[A-Z])", "_", property_name).lower()
        return (
            "mkdir -p /etc/systemd/system/nginx.service.d && "
            "for _f in /etc/systemd/system/nginx.service.d/*.conf; do "
            f"[ -f \"$_f\" ] && grep -q '^{property_name}=' \"$_f\" 2>/dev/null && rm -f \"$_f\"; "
            "done; true && "
            f"printf '[Service]\\n{property_name}={value}\\n' > "
            f"/etc/systemd/system/nginx.service.d/zz_hosttune_{suffix}.conf && "
            "systemctl daemon-reload && "
            "systemctl restart nginx.service"
        )

    def run_tune(
        self,
        config_path: Path,
        tune_engine: TuneEngineProtocol,
    ) -> TuneState:
        loaded_config = self.config_loader.load(config_path)
        context = self.build_tune_context()
        target_executor = self._build_stage_executor(loaded_config.target, "tune")
        benchmark_executor = self._build_stage_executor(
            loaded_config.benchmark_config.runner_target,
            "tune",
        )
        result = tune_engine.run(
            context=context,
            target_executor=target_executor,
            benchmark_executor=benchmark_executor,
        )
        self.tune = result
        self._persist_stage_result("tune", result)
        if (
            self.artifacts is not None
            and self.knowledge_base is not None
            and self.preflight is not None
        ):
            self.knowledge_base.record_event(
                run_id=self.artifacts.session_id,
                component="convergence_logic",
                event_type="run_completed",
                service_name=self.onboard.service_name if self.onboard is not None else None,
                host_fingerprint=host_fingerprint_for_snapshot(
                    self.preflight,
                    self.onboard.service_name if self.onboard is not None else None,
                ),
                payload={
                    "stop_reason": result.stop_reason or "unknown",
                    "current_phase": result.current_phase.value,
                    "total_iterations": result.total_iterations,
                    "active_changes": sorted(result.active_changes),
                },
            )
            self.knowledge_base.finalize_run(
                run_id=self.artifacts.session_id,
                stop_reason=result.stop_reason or "unknown",
                best_score=(
                    None if result.best_configuration is None else result.best_configuration.score
                ),
                best_iteration=(
                    None
                    if result.best_configuration is None
                    else result.best_configuration.iteration_number
                ),
                best_config=(
                    None
                    if result.best_configuration is None
                    else result.best_iteration_config_values()
                ),
                final_retained_config=result.final_retained_config_values(),
            )
            # Store degradation recipe for future pattern matching.
            self._store_degradation_recipe(result)
        return result

    def _store_degradation_recipe(self, result: TuneState) -> None:
        """Store the fix sequence as a degradation recipe for future lookups."""
        if (
            result.best_configuration is None
            or result.best_configuration.score <= 0
            or self.baseline is None
            or self.artifacts is None
            or self.knowledge_base is None
            or self.preflight is None
            or self.onboard is None
        ):
            return
        from tune.domain.hypothesis_models import HypothesisStatus

        fix_sequence = [
            {
                "parameter_key": rec.hypothesis.parameter_key,
                "value": rec.hypothesis.proposed_value,
                "apply_mode": rec.hypothesis.apply_mode.value,
            }
            for rec in result.history
            if rec.status is HypothesisStatus.ACCEPTED
        ]
        if not fix_sequence:
            return
        from preflight.infrastructure.knowledge_base import (
            compute_degradation_fingerprint,
        )

        names, vector = compute_degradation_fingerprint(
            self.baseline.workload_results
        )
        import json

        fingerprint = host_fingerprint_for_snapshot(
            self.preflight, self.onboard.service_name
        )
        self.knowledge_base.store_degradation_recipe(
            run_id=self.artifacts.session_id,
            host_fingerprint=fingerprint,
            service_name=self.onboard.service_name,
            fingerprint_json=json.dumps(
                {"workload_names": names, "rps_vector": vector}
            ),
            fix_sequence_json=json.dumps(fix_sequence),
            best_score=result.best_configuration.score,
            workload_count=len(names),
        )
        self.logger.stage_detail(
            "tune",
            f"Stored degradation recipe: {len(fix_sequence)} fixes, "
            f"score={result.best_configuration.score:.2%}",
        )

    def _run_preflight(self, loaded_config: LoadedConfig) -> DiscoverySnapshot:
        runner = self.discovery_runner_factory(None)
        executor = self._build_stage_executor(loaded_config.target, "preflight")
        return runner.run(
            executor=executor,
            target=loaded_config.target,
            policy=loaded_config.policy,
        )

    def _build_stage_executor(
        self,
        target: LocalTargetConfig | SshTargetConfig,
        stage_name: str,
    ) -> CommandExecutor:
        executor = self.executor_factory(target)
        return LoggingCommandExecutor(
            inner=executor,
            logger=self.logger,
            stage_name=stage_name,
        )

    def _ensure_artifacts(self) -> RuntimeArtifacts:
        if self.artifacts is None:
            self.artifacts = self.artifact_store.create_session()
            kb_path = self.artifact_store.knowledge_base_path()
            self.artifacts.stage_files["knowledge_base"] = kb_path
            if self.knowledge_base is None:
                self.knowledge_base = KnowledgeBase(kb_path)
        return self.artifacts

    def _persist_stage_result(self, stage_name: str, payload: object) -> None:
        artifacts = self._ensure_artifacts()
        file_path = self.artifact_store.write_stage_result(artifacts, stage_name, payload)
        self.logger.artifact_written(stage_name, str(file_path))

    def _record_stage_event(
        self,
        *,
        component: str,
        event_type: str,
        payload: object,
    ) -> None:
        if self.artifacts is None or self.knowledge_base is None or self.preflight is None:
            return
        self.knowledge_base.record_event(
            run_id=self.artifacts.session_id,
            component=component,
            event_type=event_type,
            service_name=self.onboard.service_name if self.onboard is not None else None,
            host_fingerprint=host_fingerprint_for_snapshot(
                self.preflight,
                self.onboard.service_name if self.onboard is not None else None,
            ),
            payload=payload,
        )

    def _enrich_preflight_sysctl_profile_with_contract(
        self,
        onboard: OnboardResult,
        executor: CommandExecutor,
    ) -> None:
        """Append service `relevant_sysctls` not in the fixed preflight list (preflight 1b)."""
        if self.preflight is None:
            return
        contract_names = _unique_relevant_sysctl_names(
            onboard.service.tunable_surface.relevant_sysctls
        )
        if not contract_names:
            return
        extra_keys = contract_sysctl_names_only_extra(contract_names)
        profile = self.preflight.kernel.sysctl_profile
        if not extra_keys and not profile:
            return
        merged_order = merged_sysctl_profile_key_order(contract_names)
        values = dict(profile)
        if extra_keys:
            dump = executor.run(sysctl_profile_read_command(extra_keys))
            if dump.exit_code != 0:
                import logging

                logging.getLogger(__name__).warning(
                    "sysctl profile enrichment failed (exit=%d): %s",
                    dump.exit_code,
                    dump.stderr.strip() or dump.stdout.strip(),
                )
                return
            extra_pairs = KernelParser.parse_sysctl_profile_stdout(
                dump.stdout,
                keys=extra_keys,
            )
            values.update(dict(extra_pairs))
        new_profile = tuple((key, values.get(key, "")) for key in merged_order)
        if new_profile == self.preflight.kernel.sysctl_profile:
            return
        new_kernel = replace(self.preflight.kernel, sysctl_profile=new_profile)
        self.preflight = replace(self.preflight, kernel=new_kernel)
        self._persist_stage_result("preflight", self.preflight)


def _unique_relevant_sysctl_names(entries: tuple[SysctlTunable, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in entries:
        if entry.name not in seen:
            seen.add(entry.name)
            ordered.append(entry.name)
    return tuple(ordered)


class TuneEngineProtocol(Protocol):
    def run(
        self,
        context: TuneContext,
        target_executor: CommandExecutor,
        benchmark_executor: CommandExecutor,
    ) -> TuneState:
        """Execute the tune stage for a prepared TuneContext."""
