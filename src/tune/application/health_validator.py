from __future__ import annotations

import shlex
from dataclasses import dataclass

from onboard.domain.models import ProbeType
from preflight.domain.models import CommandExecutor
from tune.domain.apply_models import AppliedChange
from tune.domain.tune_context import TuneContext
from tune.domain.validation_models import ValidationCheck, ValidationResult


def _normalize_sysctl_value(value: str) -> str:
    """Normalize kernel sysctl values for comparison.

    The kernel may return multi-field values (e.g. ip_local_port_range) with
    tabs or multiple spaces between fields. Normalize to single-space separated
    so '1024\t65535' and '1024 65535' compare as equal.
    """
    return " ".join(value.split())


def _systemd_unit_limit_observation_matches(observed: str, expected_applied: str) -> bool:
    obs = observed.strip()
    exp = expected_applied.strip()
    if obs == exp:
        return True
    try:
        return int(obs.split()[0]) == int(exp)
    except ValueError:
        import logging

        logging.getLogger(__name__).warning(
            "systemd unit limit comparison failed: observed=%r expected=%r "
            "(non-numeric value like 'infinity' may indicate the limit was not applied)",
            obs,
            exp,
        )
        return False


def _systemd_cgroup_observation_matches(prop: str, observed: str, expected_applied: str) -> bool:
    obs = observed.strip()
    exp = expected_applied.strip()
    if prop == "CPUQuota":
        return obs.removesuffix("%") == exp
    if prop == "MemoryMax":
        if obs == "infinity":
            return exp == "infinity"
        if obs.isdigit() and exp.isdigit():
            return int(obs) == int(exp) * 1024 * 1024
        return obs.removesuffix("M") == exp
    return obs == exp


@dataclass
class HealthValidator:
    def validate_baseline(
        self,
        context: TuneContext,
        executor: CommandExecutor,
    ) -> tuple[ValidationCheck, ...]:
        checks = self._build_service_health_checks(context, executor)
        return tuple(checks)

    def validate(
        self,
        context: TuneContext,
        applied_change: AppliedChange,
        executor: CommandExecutor,
    ) -> ValidationResult:
        checks = self._build_service_health_checks(context, executor)
        syntax_check = self._validate_config_syntax(context, applied_change, executor)
        if syntax_check is not None:
            checks.append(syntax_check)
        checks.append(self._validate_effective_value(applied_change, executor))
        healthy = all(check.passed for check in checks)
        return ValidationResult(
            applied_change=applied_change,
            healthy=healthy,
            checks=tuple(checks),
        )

    def _build_service_health_checks(
        self,
        context: TuneContext,
        executor: CommandExecutor,
    ) -> list[ValidationCheck]:
        return [
            self._validate_service_state(context, executor),
            self._validate_health_check(context, executor),
        ]

    def _validate_service_state(
        self,
        context: TuneContext,
        executor: CommandExecutor,
    ) -> ValidationCheck:
        unit_name = context.onboard.service.identity.systemd_unit_name
        command = f"systemctl is-active {shlex.quote(unit_name)}"
        result = executor.run(command)
        passed = result.exit_code == 0 and result.stdout.strip() == "active"
        detail = result.stdout or result.stderr or "inactive"
        return ValidationCheck(
            name="systemd_active",
            passed=passed,
            detail=detail,
        )

    def _validate_health_check(
        self,
        context: TuneContext,
        executor: CommandExecutor,
    ) -> ValidationCheck:
        health_check = context.onboard.service.health_check
        if health_check.probe_type is ProbeType.HTTP:
            return self._validate_http_probe(context, executor)
        if health_check.probe_type is ProbeType.SYSTEMD_STATUS:
            return self._validate_service_state(context, executor)
        if health_check.probe_type is ProbeType.COMMAND:
            return self._validate_command_probe(
                health_check.target, health_check.expected_exit_code, executor
            )
        if health_check.probe_type is ProbeType.TCP:
            return self._validate_tcp_probe(health_check.target, executor)
        msg = f"Unsupported health check probe type: {health_check.probe_type.value}"
        raise ValueError(msg)

    def _validate_http_probe(
        self,
        context: TuneContext,
        executor: CommandExecutor,
    ) -> ValidationCheck:
        health_check = context.onboard.service.health_check
        expected_status = health_check.expected_status_code or 200
        curl_command = (
            "body_file=$(mktemp /tmp/hosttune-health-body.XXXXXX) && "
            'status_code=$(curl -sS -o "$body_file" '
            f"-w '%{{http_code}}' --max-time {health_check.timeout_seconds} "
            f"{shlex.quote(health_check.target)}) && "
            "printf '%s\\n__HOSTTUNE_BODY__\\n' \"$status_code\" && "
            'cat "$body_file" && rm -f "$body_file"'
        )
        result = executor.run(curl_command)
        status_code, body = self._split_http_probe_output(result.stdout)
        passed = result.exit_code == 0 and status_code == str(expected_status)
        if health_check.expected_string is not None:
            passed = passed and health_check.expected_string in body
        body_match = health_check.expected_string is None or health_check.expected_string in body
        detail = f"status={status_code or 'unknown'} " f"body_match={body_match}"
        return ValidationCheck(
            name="health_probe",
            passed=passed,
            detail=detail,
        )

    def _validate_command_probe(
        self,
        command: str,
        expected_exit_code: int | None,
        executor: CommandExecutor,
    ) -> ValidationCheck:
        result = executor.run(command)
        expected_code = 0 if expected_exit_code is None else expected_exit_code
        return ValidationCheck(
            name="health_probe",
            passed=result.exit_code == expected_code,
            detail=result.stdout or result.stderr or f"exit_code={result.exit_code}",
        )

    def _validate_tcp_probe(
        self,
        target: str,
        executor: CommandExecutor,
    ) -> ValidationCheck:
        host, port = self._split_host_port(target)
        command = f"bash -lc 'echo > /dev/tcp/{host}/{port}'"
        result = executor.run(command)
        return ValidationCheck(
            name="health_probe",
            passed=result.exit_code == 0,
            detail=result.stderr or result.stdout or "connected",
        )

    def _validate_effective_value(
        self,
        applied_change: AppliedChange,
        executor: CommandExecutor,
    ) -> ValidationCheck:
        if applied_change.hypothesis.parameter_key.startswith("sysctl."):
            command = f"sysctl -n {shlex.quote(applied_change.target_path)}"
            result = executor.run(command)
            observed = result.stdout.strip()
            # Normalize whitespace: kernel may return tabs/multiple spaces between
            # fields (e.g. ip_local_port_range returns "1024\t65535").
            passed = result.exit_code == 0 and (
                _normalize_sysctl_value(observed)
                == _normalize_sysctl_value(applied_change.applied_value)
            )
            detail = f"observed={observed or 'unknown'}"
            return ValidationCheck(
                name="effective_value",
                passed=passed,
                detail=detail,
            )

        if applied_change.hypothesis.parameter_key.startswith("service.directive."):
            directive_name = applied_change.hypothesis.parameter_name
            command = (
                f"grep -E '^\\s*{directive_name}\\s+' "
                f"{shlex.quote(applied_change.target_path)} | tail -n 1"
            )
            result = executor.run(command)
            line = result.stdout.strip()
            passed = result.exit_code == 0 and applied_change.applied_value in line
            detail = line or result.stderr or "directive missing"
            return ValidationCheck(
                name="effective_value",
                passed=passed,
                detail=detail,
            )

        if applied_change.hypothesis.parameter_key.startswith("network.ring."):
            interface_name, ring_name = applied_change.target_path.split(":", maxsplit=1)
            command = (
                f"ethtool -g {shlex.quote(interface_name)} | "
                'awk \'BEGIN{section=""} '
                '/Current hardware settings:/{section="current"; next} '
                '/Pre-set maximums:/{section="max"; next} '
                f'section=="current" && tolower($1)=="{ring_name}:" {{print $2; exit}}\''
            )
            result = executor.run(command)
            observed = result.stdout.strip()
            passed = result.exit_code == 0 and observed == applied_change.applied_value
            detail = f"observed={observed or 'unknown'}"
            return ValidationCheck(
                name="effective_value",
                passed=passed,
                detail=detail,
            )

        if applied_change.hypothesis.parameter_key.startswith("runtime.prlimit."):
            pid = applied_change.target_path.split("=", maxsplit=1)[1].split(":")[0]
            command = f"awk '/^Max open files/ {{print $4}}' /proc/{shlex.quote(pid)}/limits"
            result = executor.run(command)
            observed = result.stdout.strip()
            passed = result.exit_code == 0 and observed == applied_change.applied_value
            detail = f"observed={observed or 'unknown'}"
            return ValidationCheck(
                name="effective_value",
                passed=passed,
                detail=detail,
            )

        if applied_change.hypothesis.parameter_key.startswith("systemd.unit."):
            unit, prop = applied_change.target_path.rsplit(":", maxsplit=1)
            command = f"systemctl show {shlex.quote(unit)} --property={shlex.quote(prop)} --value"
            result = executor.run(command)
            observed = result.stdout.strip()
            passed = result.exit_code == 0 and _systemd_unit_limit_observation_matches(
                observed,
                applied_change.applied_value,
            )
            detail = f"observed={observed or 'unknown'}"
            return ValidationCheck(
                name="effective_value",
                passed=passed,
                detail=detail,
            )

        if applied_change.hypothesis.parameter_key.startswith("systemd.cgroup."):
            unit, prop = applied_change.target_path.rsplit(":", maxsplit=1)
            command = f"systemctl show {shlex.quote(unit)} --property={shlex.quote(prop)} --value"
            result = executor.run(command)
            observed = result.stdout.strip()
            passed = result.exit_code == 0 and _systemd_cgroup_observation_matches(
                prop,
                observed,
                applied_change.applied_value,
            )
            detail = f"observed={observed or 'unknown'}"
            return ValidationCheck(
                name="effective_value",
                passed=passed,
                detail=detail,
            )

        if applied_change.hypothesis.parameter_key.startswith("network.queue."):
            iface = applied_change.target_path.split(":", maxsplit=1)[0]
            command = (
                f"ethtool -l {shlex.quote(iface)} 2>/dev/null | "
                "awk '/Current hardware settings/{found=1} "
                "found && /Combined/{print $2; exit}'"
            )
            result = executor.run(command)
            observed = result.stdout.strip()
            passed = result.exit_code == 0 and observed == applied_change.applied_value
            detail = f"observed={observed or 'unknown'}"
            return ValidationCheck(
                name="effective_value",
                passed=passed,
                detail=detail,
            )

        if applied_change.hypothesis.parameter_key.startswith("platform.cpu_governor."):
            result = executor.run(
                "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor " "2>/dev/null"
            )
            observed = result.stdout.strip()
            passed = result.exit_code == 0 and observed == applied_change.applied_value
            detail = f"observed={observed or 'unknown'}"
            return ValidationCheck(
                name="effective_value",
                passed=passed,
                detail=detail,
            )

        return ValidationCheck(
            name="effective_value",
            passed=False,
            detail="No effective value validator for applied change type.",
        )

    def _validate_config_syntax(
        self,
        context: TuneContext,
        applied_change: AppliedChange,
        executor: CommandExecutor,
    ) -> ValidationCheck | None:
        if not applied_change.hypothesis.parameter_key.startswith("service.directive."):
            return None
        if context.onboard.service.identity.config_format != "nginx":
            return None

        command = "nginx -t"
        result = executor.run(command)
        passed = result.exit_code == 0
        detail = result.stderr.strip() or result.stdout.strip() or "nginx -t returned no output"
        return ValidationCheck(
            name="config_syntax",
            passed=passed,
            detail=detail,
        )

    def _split_host_port(self, target: str) -> tuple[str, str]:
        if ":" not in target:
            msg = f"TCP target must be host:port, got {target!r}"
            raise ValueError(msg)
        host, port = target.rsplit(":", maxsplit=1)
        return host, port

    def _split_http_probe_output(self, output: str) -> tuple[str, str]:
        marker = "\n__HOSTTUNE_BODY__\n"
        if marker not in output:
            return output.strip(), ""
        status_code, body = output.split(marker, maxsplit=1)
        return status_code.strip(), body
