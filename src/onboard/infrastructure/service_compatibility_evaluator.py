from __future__ import annotations

from onboard.domain.models import (
    CompatibilityFinding,
    CompatibilityReport,
    FindingSeverity,
    ServiceDefinition,
)
from preflight.domain.models import CommandExecutor, DiscoverySnapshot


class ServiceCompatibilityEvaluator:
    def evaluate(
        self,
        preflight: DiscoverySnapshot,
        service: ServiceDefinition,
        executor: CommandExecutor,
    ) -> CompatibilityReport:
        findings: list[CompatibilityFinding] = []
        self._check_rhel_version(preflight, service, findings)
        self._check_config_paths(service, executor, findings)
        self._check_working_directory(service, executor, findings)
        self._check_log_paths(service, executor, findings)
        self._check_systemd_unit(service, executor, findings)
        self._check_health_target(service, findings)
        compatible = all(finding.severity is not FindingSeverity.ERROR for finding in findings)
        return CompatibilityReport(compatible=compatible, findings=tuple(findings))

    def _check_rhel_version(
        self,
        preflight: DiscoverySnapshot,
        service: ServiceDefinition,
        findings: list[CompatibilityFinding],
    ) -> None:
        operating_system = preflight.platform.operating_system
        if "Red Hat" not in operating_system and "RHEL" not in operating_system:
            findings.append(
                CompatibilityFinding(
                    severity=FindingSeverity.WARNING,
                    message=f"Preflight OS did not identify as RHEL: {operating_system}",
                )
            )
            return
        if not any(version in operating_system for version in service.identity.rhel_versions):
            findings.append(
                CompatibilityFinding(
                    severity=FindingSeverity.ERROR,
                    message=(
                        f"Service supports RHEL versions {service.identity.rhel_versions}, "
                        f"but preflight reported {operating_system}"
                    ),
                )
            )

    def _check_config_paths(
        self,
        service: ServiceDefinition,
        executor: CommandExecutor,
        findings: list[CompatibilityFinding],
    ) -> None:
        for path in service.identity.config_paths:
            result = executor.run(f"test -e {path!s}")
            if result.exit_code != 0:
                findings.append(
                    CompatibilityFinding(
                        severity=FindingSeverity.ERROR,
                        message=f"Configured path does not exist: {path}",
                    )
                )

    def _check_working_directory(
        self,
        service: ServiceDefinition,
        executor: CommandExecutor,
        findings: list[CompatibilityFinding],
    ) -> None:
        directory = service.identity.working_directory
        if directory is None:
            return
        result = executor.run(f"test -d {directory!s}")
        if result.exit_code != 0:
            findings.append(
                CompatibilityFinding(
                    severity=FindingSeverity.ERROR,
                    message=f"Working directory does not exist: {directory}",
                )
            )

    def _check_log_paths(
        self,
        service: ServiceDefinition,
        executor: CommandExecutor,
        findings: list[CompatibilityFinding],
    ) -> None:
        for path in service.identity.log_paths:
            result = executor.run(f"test -e {path!s} || test -d $(dirname {path!s})")
            if result.exit_code != 0:
                findings.append(
                    CompatibilityFinding(
                        severity=FindingSeverity.WARNING,
                        message=f"Log path or parent directory missing: {path}",
                    )
                )

    def _check_systemd_unit(
        self,
        service: ServiceDefinition,
        executor: CommandExecutor,
        findings: list[CompatibilityFinding],
    ) -> None:
        command = f"systemctl status {service.identity.systemd_unit_name} >/dev/null 2>&1"
        result = executor.run(command)
        if result.exit_code != 0:
            findings.append(
                CompatibilityFinding(
                    severity=FindingSeverity.ERROR,
                    message=f"Systemd unit not available: {service.identity.systemd_unit_name}",
                )
            )

    def _check_health_target(
        self,
        service: ServiceDefinition,
        findings: list[CompatibilityFinding],
    ) -> None:
        target = service.health_check.target.strip()
        if target == "":
            findings.append(
                CompatibilityFinding(
                    severity=FindingSeverity.ERROR,
                    message="Health check target must not be empty.",
                )
            )
