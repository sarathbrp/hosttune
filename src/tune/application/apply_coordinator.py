from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from onboard.domain.models import ApplyMode
from preflight.domain.models import CommandExecutor
from tune.application.candidate_value_reads import (
    directive_source_path_from_nginx_dump,
    grep_directive_from_config_file,
    try_read_network_ring_current,
)
from tune.domain.apply_models import AppliedChange
from tune.domain.hypothesis_models import TuningHypothesis
from tune.domain.tune_context import TuneContext

_SSH_NOISE_PATTERNS = ("Identity file", "not accessible", "Warning:")


def _cmd_error(result: object) -> str:
    """Return a clean error string from a command result, stripping SSH noise."""
    stderr = getattr(result, "stderr", "") or ""
    stdout = getattr(result, "stdout", "") or ""
    exit_code = getattr(result, "exit_code", "?")
    clean = "\n".join(
        line for line in stderr.splitlines()
        if not any(p in line for p in _SSH_NOISE_PATTERNS)
    ).strip()
    detail = clean or stdout.strip() or stderr.strip()
    return f"(exit={exit_code}) {detail}"


_SYSTEMD_LIMIT_PROPERTIES: dict[str, str] = {
    "limit_nofile": "LimitNOFILE",
    "limit_nproc": "LimitNPROC",
}

_SYSTEMD_CGROUP_PROPERTIES: dict[str, str] = {
    "cpu_quota_percent": "CPUQuota",
    "memory_max_mib": "MemoryMax",
}


def _service_reload_command(context: TuneContext, apply_mode: ApplyMode) -> str | None:
    """Return the reload or restart shell command from the service contract.

    Returns None if the service has no command configured for the mode.
    Raises ValueError if the engagement policy disallows the required operation —
    the change was applied to the config/sysctl but cannot be activated.
    """
    policy = context.preflight.policy
    restart = context.onboard.service.restart
    if apply_mode is ApplyMode.RESTART:
        if not policy.allow_restart:
            msg = (
                "Service restart is required to activate this change but "
                "allow_restart=false in engagement policy."
            )
            raise ValueError(msg)
        if restart.restart.supported and restart.restart.command:
            return restart.restart.command.strip()
    elif apply_mode is ApplyMode.RELOAD:
        if not policy.allow_reload:
            msg = (
                "Service reload is required to activate this change but "
                "allow_reload=false in engagement policy."
            )
            raise ValueError(msg)
        if restart.reload.supported and restart.reload.command:
            return restart.reload.command.strip()
    if apply_mode in (ApplyMode.RELOAD, ApplyMode.RESTART):
        import logging

        logging.getLogger(__name__).warning(
            "apply_mode=%s but no %s command configured; "
            "change written to disk but not activated in running service",
            apply_mode.value,
            apply_mode.value,
        )
    return None


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
    systemd_unit_limit_applier: ChangeApplier
    cgroup_resource_control_applier: ChangeApplier | None = None
    nic_queue_applier: ChangeApplier | None = None
    cpu_governor_applier: ChangeApplier | None = None

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
        if hypothesis.parameter_key.startswith("network.queue."):
            if self.nic_queue_applier is None:
                msg = "NIC queue candidate proposed but nic_queue_applier not configured."
                raise ValueError(msg)
            return self.nic_queue_applier.apply(context, hypothesis, executor)
        if hypothesis.parameter_key.startswith("platform.cpu_governor."):
            if self.cpu_governor_applier is None:
                msg = "CPU governor candidate proposed but cpu_governor_applier not configured."
                raise ValueError(msg)
            return self.cpu_governor_applier.apply(context, hypothesis, executor)
        if hypothesis.parameter_key.startswith("runtime.prlimit."):
            return self.runtime_limit_applier.apply(context, hypothesis, executor)
        if hypothesis.parameter_key.startswith("systemd.unit."):
            return self.systemd_unit_limit_applier.apply(context, hypothesis, executor)
        if hypothesis.parameter_key.startswith("systemd.cgroup."):
            if self.cgroup_resource_control_applier is None:
                msg = (
                    "Cgroup resource control candidate proposed but "
                    "cgroup_resource_control_applier not configured."
                )
                raise ValueError(msg)
            return self.cgroup_resource_control_applier.apply(context, hypothesis, executor)
        msg = f"No applier available for parameter_key: {hypothesis.parameter_key}"
        raise ValueError(msg)


@dataclass
class NginxDirectiveApplier:
    _MAIN_CONTEXT_DIRECTIVES = frozenset(
        {"worker_processes", "worker_rlimit_nofile", "worker_cpu_affinity"}
    )
    _EVENTS_CONTEXT_DIRECTIVES = frozenset({"worker_connections", "multi_accept"})
    _HTTP_CONTEXT_DIRECTIVES = frozenset(
        {
            "access_log",
            "sendfile",
            "keepalive_timeout",
            "keepalive_requests",
            "aio",
            "open_file_cache",
            "gzip",
            "tcp_nopush",
            "limit_rate",
        }
    )

    def apply(
        self,
        context: TuneContext,
        hypothesis: TuningHypothesis,
        executor: CommandExecutor,
    ) -> AppliedChange:
        config_path = self._resolve_config_path(context, hypothesis.parameter_name)
        current_value = self._read_directive_value(
            executor=executor,
            config_path=config_path,
            directive_name=hypothesis.parameter_name,
        )
        was_present = current_value is not None
        apply_command = (
            self._build_replace_command(
                config_path=config_path,
                directive_name=hypothesis.parameter_name,
                directive_value=hypothesis.proposed_value,
            )
            if was_present
            else self._build_insert_command(
                config_path=config_path,
                directive_name=hypothesis.parameter_name,
                directive_value=hypothesis.proposed_value,
            )
        )
        apply_result = executor.run(apply_command)
        if apply_result.exit_code != 0:
            operation = "insert" if not was_present else "replace"
            msg = (
                f"Failed to {operation} nginx directive {hypothesis.parameter_name!r} "
                f"in {config_path}: {_cmd_error(apply_result)}"
            )
            raise ValueError(msg)
        restore_command = (
            self._build_replace_command(
                config_path=config_path,
                directive_name=hypothesis.parameter_name,
                directive_value=current_value,
            )
            if was_present
            else self._build_delete_command(
                config_path=config_path,
                directive_name=hypothesis.parameter_name,
            )
        )
        # Reload/restart nginx so the edited config takes effect before benchmarking.
        service_cmd = _service_reload_command(context, hypothesis.apply_mode)
        if service_cmd:
            reload_result = executor.run(service_cmd)
            if reload_result.exit_code != 0:
                # Restore the config file before raising so we don't leave a broken state.
                restore_result = executor.run(restore_command)
                restore_note = ""
                if restore_result.exit_code != 0:
                    restore_note = (
                        f" WARNING: config rollback also failed: "
                        f"{restore_result.stderr or restore_result.stdout}"
                    )
                msg = (
                    f"nginx directive applied but service {hypothesis.apply_mode.value} failed: "
                    f"{reload_result.stderr or reload_result.stdout}{restore_note}"
                )
                raise ValueError(msg)
            full_apply_command = f"{apply_command} && {service_cmd}"
            rollback_command = f"{restore_command} && {service_cmd}"
        else:
            full_apply_command = apply_command
            rollback_command = restore_command
        return AppliedChange(
            hypothesis=hypothesis,
            target_path=config_path,
            previous_value=current_value if current_value is not None else "__absent__",
            applied_value=hypothesis.proposed_value,
            apply_mode=hypothesis.apply_mode,
            apply_command=full_apply_command,
            rollback_command=rollback_command,
        )

    def _resolve_config_path(self, context: TuneContext, directive_name: str) -> str:
        runtime_state_output = context.snapshot.runtime_state_output
        dump_path = directive_source_path_from_nginx_dump(directive_name, runtime_state_output)
        if dump_path is not None:
            return dump_path
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
    ) -> str | None:
        return grep_directive_from_config_file(executor, config_path, directive_name)

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

    def _directive_context(self, directive_name: str) -> str:
        if directive_name in self._MAIN_CONTEXT_DIRECTIVES:
            return "main"
        if directive_name in self._EVENTS_CONTEXT_DIRECTIVES:
            return "events"
        if directive_name in self._HTTP_CONTEXT_DIRECTIVES:
            return "http"
        msg = f"Directive {directive_name!r} has no known insertion context"
        raise ValueError(msg)

    def _build_insert_command(
        self,
        config_path: str,
        directive_name: str,
        directive_value: str,
    ) -> str:
        context_name = self._directive_context(directive_name)
        python_script = (
            "import pathlib,sys\n"
            "path=pathlib.Path(sys.argv[1])\n"
            "name=sys.argv[2]\n"
            "value=sys.argv[3]\n"
            "context=sys.argv[4]\n"
            "lines=path.read_text().splitlines()\n"
            "entry=f'{name} {value};'\n"
            "def block_bounds(block):\n"
            "    start=None; depth=0\n"
            "    for idx,line in enumerate(lines):\n"
            "        stripped=line.strip()\n"
            "        if start is None and stripped.startswith(f'{block} {{'):\n"
            "            start=idx; depth=stripped.count('{')-stripped.count('}')\n"
            "            continue\n"
            "        if start is not None:\n"
            "            depth += line.count('{')-line.count('}')\n"
            "            if depth==0:\n"
            "                return start, idx\n"
            "    return None\n"
            "if context=='main':\n"
            "    markers=('events {','http {')\n"
            "    insert_at=next("
            "(i for i,line in enumerate(lines) "
            "if line.strip().startswith(markers)), None)\n"
            "    if insert_at is None:\n"
            "        sys.exit('main-context markers (events/http) not found in config')\n"
            "    lines.insert(insert_at, entry)\n"
            "else:\n"
            "    bounds=block_bounds(context)\n"
            "    bounds or sys.exit('directive context not found')\n"
            "    start,end=bounds\n"
            "    indent='    '\n"
            "    for probe in lines[start+1:end]:\n"
            "        stripped=probe.strip()\n"
            "        if stripped:\n"
            "            indent=probe[:len(probe)-len(probe.lstrip())] or '    '\n"
            "            break\n"
            "    lines.insert(end, f'{indent}{entry}')\n"
            "path.write_text('\\n'.join(lines)+'\\n')\n"
        )
        return " ".join(
            (
                "python3",
                "-c",
                shlex.quote(python_script),
                shlex.quote(config_path),
                shlex.quote(directive_name),
                shlex.quote(directive_value),
                shlex.quote(context_name),
            )
        )

    def _build_delete_command(
        self,
        config_path: str,
        directive_name: str,
    ) -> str:
        python_script = (
            "import pathlib,re,sys; "
            "path=pathlib.Path(sys.argv[1]); "
            "name=sys.argv[2]; "
            "lines=path.read_text().splitlines(); "
            "pattern=re.compile(rf'^\\s*{re.escape(name)}\\s+[^;]+;\\s*$'); "
            "removed=False; "
            "out=[]; "
            "for line in lines:\n"
            "    if (not removed) and pattern.match(line):\n"
            "        removed=True\n"
            "        continue\n"
            "    out.append(line)\n"
            "removed or sys.exit('directive not found'); "
            "path.write_text('\\n'.join(out)+'\\n')"
        )
        return " ".join(
            (
                "python3",
                "-c",
                shlex.quote(python_script),
                shlex.quote(config_path),
                shlex.quote(directive_name),
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
            msg = f"Failed to apply sysctl change: {_cmd_error(apply_result)}"
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
                f"Failed to apply network ring change: {_cmd_error(apply_result)}"
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
        current = try_read_network_ring_current(executor, interface_name, ring_name)
        if current is None:
            msg = f"Failed to read current {ring_name} ring value for {interface_name}"
            raise ValueError(msg)
        return current

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
            msg = f"Failed to apply prlimit NOFILE: {_cmd_error(apply_result)}"
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


@dataclass
class SystemdUnitLimitApplier:
    """Apply systemd unit LimitNOFILE / LimitNPROC via systemctl set-property."""

    @staticmethod
    def property_name(limit_yaml_name: str) -> str:
        prop = _SYSTEMD_LIMIT_PROPERTIES.get(limit_yaml_name)
        if prop is None:
            msg = f"Unsupported systemd unit limit: {limit_yaml_name!r}"
            raise ValueError(msg)
        return prop

    @staticmethod
    def read_property_value(executor: CommandExecutor, unit: str, prop: str) -> str:
        cmd = f"systemctl show {shlex.quote(unit)} --property={shlex.quote(prop)} --value"
        result = executor.run(cmd)
        if result.exit_code != 0:
            msg = f"Failed to read {prop} for unit {unit!r}"
            raise ValueError(msg)
        return result.stdout.strip()

    @staticmethod
    def _post_set_commands(context: TuneContext, apply_mode: ApplyMode) -> list[str]:
        tail: list[str] = ["systemctl daemon-reload"]
        restart = context.onboard.service.restart
        if apply_mode is ApplyMode.RESTART:
            if restart.restart.supported and restart.restart.command:
                tail.append(restart.restart.command.strip())
            else:
                import logging

                logging.getLogger(__name__).warning(
                    "systemd unit limit requires restart but no restart command "
                    "is configured; limit will not take effect until manual restart"
                )
        elif apply_mode is ApplyMode.RELOAD and restart.reload.supported and restart.reload.command:
            tail.append(restart.reload.command.strip())
        return tail

    def apply(
        self,
        context: TuneContext,
        hypothesis: TuningHypothesis,
        executor: CommandExecutor,
    ) -> AppliedChange:
        limit_name = hypothesis.parameter_name
        prop = self.property_name(limit_name)
        unit = context.onboard.service.identity.systemd_unit_name
        previous_raw = self.read_property_value(executor, unit, prop)
        new_value = hypothesis.proposed_value.strip()
        set_cmd = (
            f"systemctl set-property {shlex.quote(unit)} " f"{shlex.quote(f'{prop}={new_value}')}"
        )
        apply_parts = [set_cmd, *self._post_set_commands(context, hypothesis.apply_mode)]
        apply_command = " && ".join(apply_parts)
        apply_result = executor.run(apply_command)
        if apply_result.exit_code != 0:
            msg = f"Failed to apply systemd unit limit: {_cmd_error(apply_result)}"
            raise ValueError(msg)
        rollback_set = (
            f"systemctl set-property {shlex.quote(unit)} "
            f"{shlex.quote(f'{prop}={previous_raw}')}"
        )
        rollback_parts = [
            rollback_set,
            *self._post_set_commands(context, hypothesis.apply_mode),
        ]
        rollback_command = " && ".join(rollback_parts)
        return AppliedChange(
            hypothesis=hypothesis,
            target_path=f"{unit}:{prop}",
            previous_value=previous_raw,
            applied_value=new_value,
            apply_mode=hypothesis.apply_mode,
            apply_command=apply_command,
            rollback_command=rollback_command,
        )


@dataclass
class SystemdCgroupControlApplier:
    """Apply systemd cgroup resource controls via systemctl set-property."""

    @staticmethod
    def property_name(control_name: str) -> str:
        prop = _SYSTEMD_CGROUP_PROPERTIES.get(control_name)
        if prop is None:
            msg = f"Unsupported cgroup resource control: {control_name!r}"
            raise ValueError(msg)
        return prop

    @staticmethod
    def read_property_value(executor: CommandExecutor, unit: str, prop: str) -> str:
        cmd = f"systemctl show {shlex.quote(unit)} --property={shlex.quote(prop)} --value"
        result = executor.run(cmd)
        if result.exit_code != 0:
            msg = f"Failed to read {prop} for unit {unit!r}"
            raise ValueError(msg)
        return result.stdout.strip()

    @staticmethod
    def normalize_property_value(prop: str, raw_value: str) -> str | None:
        value = raw_value.strip()
        if not value or value == "infinity":
            return None
        if prop == "CPUQuota":
            return value.removesuffix("%")
        if prop == "MemoryMax":
            if value.isdigit():
                return str(int(value) // (1024 * 1024))
            stripped = value.removesuffix("M").strip()
            return stripped if stripped.isdigit() else value
        return value

    @staticmethod
    def property_assignment(prop: str, proposed_value: str) -> str:
        value = proposed_value.strip()
        if prop == "CPUQuota":
            return f"{prop}={value}%"
        if prop == "MemoryMax":
            return f"{prop}={value}M"
        return f"{prop}={value}"

    def apply(
        self,
        context: TuneContext,
        hypothesis: TuningHypothesis,
        executor: CommandExecutor,
    ) -> AppliedChange:
        control_name = hypothesis.parameter_name
        prop = self.property_name(control_name)
        unit = context.onboard.service.identity.systemd_unit_name
        previous_raw = self.read_property_value(executor, unit, prop)
        previous_value = self.normalize_property_value(prop, previous_raw) or "infinity"
        new_value = hypothesis.proposed_value.strip()
        set_cmd = (
            f"systemctl set-property {shlex.quote(unit)} "
            f"{shlex.quote(self.property_assignment(prop, new_value))}"
        )
        apply_parts = [
            set_cmd,
            *SystemdUnitLimitApplier._post_set_commands(context, hypothesis.apply_mode),
        ]
        apply_command = " && ".join(apply_parts)
        apply_result = executor.run(apply_command)
        if apply_result.exit_code != 0:
            msg = (
                f"Failed to apply systemd cgroup control: {_cmd_error(apply_result)}"
            )
            raise ValueError(msg)
        rollback_set = (
            f"systemctl set-property {shlex.quote(unit)} "
            f"{shlex.quote(f'{prop}={previous_raw}')}"
        )
        rollback_parts = [
            rollback_set,
            *SystemdUnitLimitApplier._post_set_commands(context, hypothesis.apply_mode),
        ]
        rollback_command = " && ".join(rollback_parts)
        return AppliedChange(
            hypothesis=hypothesis,
            target_path=f"{unit}:{prop}",
            previous_value=previous_value,
            applied_value=new_value,
            apply_mode=hypothesis.apply_mode,
            apply_command=apply_command,
            rollback_command=rollback_command,
        )


@dataclass
class NicQueueApplier:
    """Expand or shrink NIC combined queue count via ethtool -L."""

    def apply(
        self,
        context: TuneContext,
        hypothesis: TuningHypothesis,
        executor: CommandExecutor,
    ) -> AppliedChange:
        iface = shlex.quote(context.preflight.network.interface_name)
        current = self._read_current_combined(executor, iface)
        new_value = hypothesis.proposed_value.strip()
        apply_command = f"ethtool -L {iface} combined {shlex.quote(new_value)}"
        apply_result = executor.run(apply_command)
        if apply_result.exit_code != 0:
            msg = (
                f"Failed to set NIC queue count to {new_value}: {_cmd_error(apply_result)}"
            )
            raise ValueError(msg)
        rollback_command = f"ethtool -L {iface} combined {shlex.quote(current)}"
        return AppliedChange(
            hypothesis=hypothesis,
            target_path=f"{context.preflight.network.interface_name}:combined_queues",
            previous_value=current,
            applied_value=new_value,
            apply_mode=hypothesis.apply_mode,
            apply_command=apply_command,
            rollback_command=rollback_command,
        )

    def _read_current_combined(self, executor: CommandExecutor, iface: str) -> str:
        cmd = (
            f"ethtool -l {iface} 2>/dev/null | "
            "awk '/Current hardware settings/{found=1} found && /Combined/{print $2; exit}'"
        )
        result = executor.run(cmd)
        value = result.stdout.strip()
        if not value.isdigit():
            msg = (
                f"Failed to read current NIC combined queue count for {iface}: "
                f"ethtool -l returned {result.stdout!r}"
            )
            raise ValueError(msg)
        return value


@dataclass
class CpuGovernorApplier:
    """Set CPU frequency scaling governor via cpupower frequency-set -g."""

    def apply(
        self,
        context: TuneContext,
        hypothesis: TuningHypothesis,
        executor: CommandExecutor,
    ) -> AppliedChange:
        current = self._read_current_governor(executor)
        new_governor = hypothesis.proposed_value.strip()
        apply_command = f"cpupower frequency-set -g {shlex.quote(new_governor)}"
        apply_result = executor.run(apply_command)
        if apply_result.exit_code != 0:
            msg = (
                f"Failed to set CPU governor to {new_governor!r}: {_cmd_error(apply_result)}"
            )
            raise ValueError(msg)
        rollback_command = f"cpupower frequency-set -g {shlex.quote(current)}"
        return AppliedChange(
            hypothesis=hypothesis,
            target_path="cpu:scaling_governor",
            previous_value=current,
            applied_value=new_governor,
            apply_mode=hypothesis.apply_mode,
            apply_command=apply_command,
            rollback_command=rollback_command,
        )

    def _read_current_governor(self, executor: CommandExecutor) -> str:
        result = executor.run(
            "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null"
        )
        value = result.stdout.strip()
        if result.exit_code != 0 or not value:
            msg = (
                "Failed to read current CPU governor: "
                f"exit={result.exit_code} stdout={result.stdout!r}"
            )
            raise ValueError(msg)
        return value
