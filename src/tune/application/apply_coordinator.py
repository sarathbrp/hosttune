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
        escaped_name = re.escape(directive_name)
        replacement = f"{directive_name} {directive_value};"
        escaped_replacement = replacement.replace("\\", "\\\\").replace("/", "\\/")
        perl_expr = f"s/^\\s*{escaped_name}\\s+[^;]+;/{escaped_replacement}/m"
        return f"perl -0pi -e {shlex.quote(perl_expr)} {shlex.quote(config_path)}"


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
