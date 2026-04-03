from __future__ import annotations

import shlex
from dataclasses import dataclass

from onboard.domain.models import ProbeType
from preflight.domain.models import CommandExecutor
from tune.domain.apply_models import AppliedChange
from tune.domain.tune_context import TuneContext
from tune.domain.validation_models import ValidationCheck, ValidationResult


@dataclass
class HealthValidator:
    def validate(
        self,
        context: TuneContext,
        applied_change: AppliedChange,
        executor: CommandExecutor,
    ) -> ValidationResult:
        checks = [
            self._validate_service_state(context, executor),
            self._validate_health_check(context, executor),
            self._validate_effective_value(applied_change, executor),
        ]
        healthy = all(check.passed for check in checks)
        return ValidationResult(
            applied_change=applied_change,
            healthy=healthy,
            checks=tuple(checks),
        )

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
            passed = result.exit_code == 0 and observed == applied_change.applied_value
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

        return ValidationCheck(
            name="effective_value",
            passed=False,
            detail="No effective value validator for applied change type.",
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
