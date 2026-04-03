from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from preflight.domain.models import CommandExecutor
from tune.domain.apply_models import AppliedChange
from tune.domain.hypothesis_models import TuningHypothesis
from tune.domain.tune_context import TuneContext


class ChangeApplier(Protocol):
    def apply(
        self,
        context: TuneContext,
        hypothesis: TuningHypothesis,
        executor: CommandExecutor,
    ) -> AppliedChange:
        """Apply one deterministic tuning change."""


@dataclass
class ApplyCoordinator:
    service_directive_applier: ChangeApplier
    sysctl_applier: ChangeApplier
    network_ring_applier: ChangeApplier
    runtime_limit_applier: ChangeApplier

    def apply(
        self,
        context: TuneContext,
        hypothesis: TuningHypothesis,
        executor: CommandExecutor,
    ) -> AppliedChange:
        if hypothesis.parameter_key.startswith("service.directive."):
            return self.service_directive_applier.apply(context, hypothesis, executor)
        if hypothesis.parameter_key.startswith("sysctl."):
            return self.sysctl_applier.apply(context, hypothesis, executor)
        if hypothesis.parameter_key.startswith("network.ring."):
            return self.network_ring_applier.apply(context, hypothesis, executor)
        if hypothesis.parameter_key.startswith("runtime.prlimit."):
            return self.runtime_limit_applier.apply(context, hypothesis, executor)
        msg = f"No applier available for parameter_key: {hypothesis.parameter_key}"
        raise ValueError(msg)


@dataclass
class NginxDirectiveApplier:
    def apply(
        self,
        context: TuneContext,
        hypothesis: TuningHypothesis,
        executor: CommandExecutor,
    ) -> AppliedChange:
        config_path = self._resolve_config_path(context)
        current_value = self._read_directive_value(
            executor=executor,
            config_path=config_path,
            directive_name=hypothesis.parameter_name,
        )
        apply_command = self._build_replace_command(
            config_path=config_path,
            directive_name=hypothesis.parameter_name,
            directive_value=hypothesis.proposed_value,
        )
        apply_result = executor.run(apply_command)
        if apply_result.exit_code != 0:
            msg = f"Failed to apply nginx directive: {apply_result.stderr or apply_result.stdout}"
            raise ValueError(msg)
        rollback_command = self._build_replace_command(
            config_path=config_path,
            directive_name=hypothesis.parameter_name,
            directive_value=current_value,
        )
        return AppliedChange(
            hypothesis=hypothesis,
            target_path=config_path,
            previous_value=current_value,
            applied_value=hypothesis.proposed_value,
            apply_mode=hypothesis.apply_mode,
            apply_command=apply_command,
            rollback_command=rollback_command,
        )

    def _resolve_config_path(self, context: TuneContext) -> str:
        for path in context.onboard.service.identity.config_paths:
            if PurePosixPath(path).suffix == ".conf":
                return path
        msg = "No concrete nginx config file path is available for directive apply."
        raise ValueError(msg)

    def _read_directive_value(
        self,
        executor: CommandExecutor,
        config_path: str,
        directive_name: str,
    ) -> str:
        grep_command = (
            f"grep -E '^\\s*{re.escape(directive_name)}\\s+' "
            f"{shlex.quote(config_path)} | tail -n 1"
        )
        result = executor.run(grep_command)
        if result.exit_code != 0 or result.stdout == "":
            msg = f"Directive {directive_name} not found in {config_path}"
            raise ValueError(msg)
        line = result.stdout.strip().rstrip(";")
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            msg = f"Unable to parse directive line for {directive_name}: {line}"
            raise ValueError(msg)
        return parts[1].strip()

    def _build_replace_command(
        self,
        config_path: str,
        directive_name: str,
        directive_value: str,
    ) -> str:
        python_script = (
            "import pathlib,re,sys; "
            "path=pathlib.Path(sys.argv[1]); "
            "name=sys.argv[2]; "
            "value=sys.argv[3]; "
            "text=path.read_text(); "
            "pattern=rf'(?m)^\\s*{re.escape(name)}\\s+[^;]+;'; "
            "replacement=f'{name} {value};'; "
            "updated,count=re.subn(pattern,replacement,text); "
            "count or sys.exit('directive not found'); "
            "path.write_text(updated)"
        )
        return " ".join(
            (
                "python3",
                "-c",
                shlex.quote(python_script),
                shlex.quote(config_path),
                shlex.quote(directive_name),
                shlex.quote(directive_value),
            )
        )


@dataclass
class SysctlApplier:
    def apply(
        self,
        context: TuneContext,
        hypothesis: TuningHypothesis,
        executor: CommandExecutor,
    ) -> AppliedChange:
        _ = context
        sysctl_name = hypothesis.parameter_name
        current_value = self._read_current_value(executor, sysctl_name)
        apply_command = self._build_sysctl_command(sysctl_name, hypothesis.proposed_value)
        apply_result = executor.run(apply_command)
        if apply_result.exit_code != 0:
            msg = f"Failed to apply sysctl change: {apply_result.stderr or apply_result.stdout}"
            raise ValueError(msg)
        rollback_command = self._build_sysctl_command(sysctl_name, current_value)
        return AppliedChange(
            hypothesis=hypothesis,
            target_path=sysctl_name,
            previous_value=current_value,
            applied_value=hypothesis.proposed_value,
            apply_mode=hypothesis.apply_mode,
            apply_command=apply_command,
            rollback_command=rollback_command,
        )

    def _read_current_value(self, executor: CommandExecutor, sysctl_name: str) -> str:
        command = f"sysctl -n {shlex.quote(sysctl_name)}"
        result = executor.run(command)
        if result.exit_code != 0:
            msg = f"Failed to read current sysctl value for {sysctl_name}"
            raise ValueError(msg)
        return result.stdout.strip()

    def _build_sysctl_command(self, sysctl_name: str, value: str) -> str:
        return f"sysctl -w {shlex.quote(sysctl_name)}={shlex.quote(value)}"


@dataclass
class NetworkRingApplier:
    def apply(
        self,
        context: TuneContext,
        hypothesis: TuningHypothesis,
        executor: CommandExecutor,
    ) -> AppliedChange:
        interface_name = context.preflight.network.interface_name
        ring_name = hypothesis.parameter_name
        current_value = self._read_current_value(executor, interface_name, ring_name)
        apply_command = self._build_ethtool_command(
            interface_name=interface_name,
            ring_name=ring_name,
            value=hypothesis.proposed_value,
        )
        apply_result = executor.run(apply_command)
        if apply_result.exit_code != 0:
            msg = (
                f"Failed to apply network ring change: {apply_result.stderr or apply_result.stdout}"
            )
            raise ValueError(msg)
        rollback_command = self._build_ethtool_command(
            interface_name=interface_name,
            ring_name=ring_name,
            value=current_value,
        )
        return AppliedChange(
            hypothesis=hypothesis,
            target_path=f"{interface_name}:{ring_name}",
            previous_value=current_value,
            applied_value=hypothesis.proposed_value,
            apply_mode=hypothesis.apply_mode,
            apply_command=apply_command,
            rollback_command=rollback_command,
        )

    def _read_current_value(
        self,
        executor: CommandExecutor,
        interface_name: str,
        ring_name: str,
    ) -> str:
        command = (
            f"ethtool -g {shlex.quote(interface_name)} | "
            'awk \'BEGIN{section=""} '
            '/Current hardware settings:/{section="current"; next} '
            '/Pre-set maximums:/{section="max"; next} '
            f'section=="current" && $1=="{ring_name}:" {{print $2; exit}}\''
        )
        result = executor.run(command)
        if result.exit_code != 0 or result.stdout.strip() == "":
            msg = f"Failed to read current {ring_name} ring value for {interface_name}"
            raise ValueError(msg)
        return result.stdout.strip()

    def _build_ethtool_command(
        self,
        interface_name: str,
        ring_name: str,
        value: str,
    ) -> str:
        return " ".join(
            (
                "ethtool",
                "-G",
                shlex.quote(interface_name),
                shlex.quote(ring_name),
                shlex.quote(value),
            )
        )


@dataclass
class PrlimitApplier:
    """Apply process NOFILE soft limit via prlimit(1) on the service PID from snapshot.pid_file."""

    @staticmethod
    def read_service_pid(executor: CommandExecutor, context: TuneContext) -> str:
        pid_path = context.onboard.service.snapshot.process_state.pid_file
        if not pid_path:
            msg = "Service snapshot contract has no pid_file; cannot use prlimit"
            raise ValueError(msg)
        result = executor.run(f"cat {shlex.quote(pid_path)}")
        if result.exit_code != 0 or not result.stdout.strip():
            msg = f"Failed to read PID from {pid_path}"
            raise ValueError(msg)
        return result.stdout.strip().split()[0]

    @staticmethod
    def read_nofile_soft_hard(executor: CommandExecutor, pid: str) -> tuple[int, int]:
        limits_path = f"/proc/{pid}/limits"
        result = executor.run(
            f"awk '/^Max open files/ {{print $4, $5}}' {shlex.quote(limits_path)}"
        )
        if result.exit_code != 0 or not result.stdout.strip():
            msg = f"Failed to read NOFILE limits for pid {pid}"
            raise ValueError(msg)
        parts = result.stdout.strip().split()
        if len(parts) < 2:
            msg = f"Unexpected limits line for pid {pid}: {result.stdout!r}"
            raise ValueError(msg)
        return int(parts[0]), int(parts[1])

    @staticmethod
    def current_nofile_soft(executor: CommandExecutor, context: TuneContext) -> str:
        pid = PrlimitApplier.read_service_pid(executor, context)
        soft, _hard = PrlimitApplier.read_nofile_soft_hard(executor, pid)
        return str(soft)

    def apply(
        self,
        context: TuneContext,
        hypothesis: TuningHypothesis,
        executor: CommandExecutor,
    ) -> AppliedChange:
        if hypothesis.parameter_name != "nofile_soft":
            msg = f"Unsupported runtime prlimit: {hypothesis.parameter_name!r}"
            raise ValueError(msg)
        pid = self.read_service_pid(executor, context)
        prev_soft, prev_hard = self.read_nofile_soft_hard(executor, pid)
        new_soft = int(hypothesis.proposed_value)
        new_hard = max(prev_hard, new_soft)
        apply_command = f"prlimit --pid {shlex.quote(pid)} --nofile={new_soft}:{new_hard}"
        apply_result = executor.run(apply_command)
        if apply_result.exit_code != 0:
            msg = f"Failed to apply prlimit NOFILE: {apply_result.stderr or apply_result.stdout}"
            raise ValueError(msg)
        rollback_command = f"prlimit --pid {shlex.quote(pid)} --nofile={prev_soft}:{prev_hard}"
        return AppliedChange(
            hypothesis=hypothesis,
            target_path=f"pid={pid}:nofile",
            previous_value=str(prev_soft),
            applied_value=str(new_soft),
            apply_mode=hypothesis.apply_mode,
            apply_command=apply_command,
            rollback_command=rollback_command,
        )
