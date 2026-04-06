from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from host_profile.domain.models import (
    CpuGovernorConstraint,
    EnvironmentBlocker,
    HostProfile,
    HostProfileIdentity,
    HostSysctlTunable,
    HostTunableSurface,
    NetworkQueueConstraint,
)
from onboard.domain.models import ApplyMode, PriorityTier


@dataclass(frozen=True)
class HostProfileValidator:
    def validate(self, data: dict[str, Any]) -> HostProfile:
        identity = self._parse_identity(cast(dict[str, Any], data.get("identity", {})))
        surface = self._parse_tunable_surface(cast(dict[str, Any], data.get("tunable_surface", {})))
        return HostProfile(identity=identity, tunable_surface=surface)

    def _parse_identity(self, data: dict[str, Any]) -> HostProfileIdentity:
        return HostProfileIdentity(
            name=self._require_str(data, "name"),
            platform=self._require_str(data, "platform"),
            version=self._require_str(data, "version"),
            variant=self._optional_str(data.get("variant")),
        )

    def _parse_tunable_surface(self, data: dict[str, Any]) -> HostTunableSurface:
        nq_raw = data.get("network_queues")
        cg_raw = data.get("cpu_governor")
        sysctls_raw = data.get("host_sysctls", [])
        blockers_raw = data.get("environment_blockers", [])
        return HostTunableSurface(
            network_queues=(
                self._parse_network_queues(cast(dict[str, Any], nq_raw)) if nq_raw else None
            ),
            cpu_governor=(
                self._parse_cpu_governor(cast(dict[str, Any], cg_raw)) if cg_raw else None
            ),
            host_sysctls=self._parse_host_sysctls(sysctls_raw),
            environment_blockers=self._parse_environment_blockers(blockers_raw),
        )

    def _parse_network_queues(self, data: dict[str, Any]) -> NetworkQueueConstraint:
        return NetworkQueueConstraint(
            min_combined=self._require_int(data, "min_combined"),
            max_combined=self._require_int(data, "max_combined"),
            allow_irq_affinity=self._require_bool(data, "allow_irq_affinity"),
            priority_tier=PriorityTier(self._require_str(data, "priority_tier")),
            apply_mode=ApplyMode(data.get("apply_mode", "reload")),
        )

    def _parse_cpu_governor(self, data: dict[str, Any]) -> CpuGovernorConstraint:
        allowed = data.get("allowed_governors", [])
        forbidden = data.get("forbidden_governors", [])
        if not isinstance(allowed, list):
            msg = "cpu_governor.allowed_governors must be a list"
            raise ValueError(msg)
        return CpuGovernorConstraint(
            allowed_governors=tuple(str(g) for g in allowed),
            forbidden_governors=tuple(str(g) for g in forbidden),
            preferred_governor=self._require_str(data, "preferred_governor"),
            priority_tier=PriorityTier(self._require_str(data, "priority_tier")),
            apply_mode=ApplyMode(data.get("apply_mode", "reload")),
        )

    def _parse_host_sysctls(self, items: object) -> tuple[HostSysctlTunable, ...]:
        if not isinstance(items, list):
            return ()
        result: list[HostSysctlTunable] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            result.append(
                HostSysctlTunable(
                    name=self._require_str(item, "name"),
                    priority_tier=PriorityTier(self._require_str(item, "priority_tier")),
                    rationale_hint=item.get("rationale_hint", ""),
                )
            )
        return tuple(result)

    def _parse_environment_blockers(
        self, items: object
    ) -> tuple[EnvironmentBlocker, ...]:
        if not isinstance(items, list):
            return ()
        result: list[EnvironmentBlocker] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            raw_fix = item.get("fix_command")
            fix_command = str(raw_fix) if raw_fix is not None else None
            raw_above = item.get("threshold_above")
            if raw_above is not None and not isinstance(raw_above, int):
                msg = f"threshold_above must be an integer, got {raw_above!r}"
                raise ValueError(msg)
            threshold_above = int(raw_above) if raw_above is not None else None
            raw_below = item.get("threshold_below")
            if raw_below is not None and not isinstance(raw_below, int):
                msg = f"threshold_below must be an integer, got {raw_below!r}"
                raise ValueError(msg)
            threshold_below = int(raw_below) if raw_below is not None else None
            result.append(
                EnvironmentBlocker(
                    name=self._require_str(item, "name"),
                    probe_command=self._require_str(item, "probe_command"),
                    fix_command=fix_command,
                    priority=item.get("priority", "high"),
                    detail=item.get("detail", ""),
                    threshold_above=threshold_above,
                    threshold_below=threshold_below,
                )
            )
        return tuple(result)

    def _require_str(self, data: dict[str, Any], key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or value == "":
            msg = f"Expected non-empty string for {key!r}"
            raise ValueError(msg)
        return value

    def _optional_str(self, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            msg = "Expected string or null"
            raise ValueError(msg)
        return value or None

    def _require_int(self, data: dict[str, Any], key: str) -> int:
        value = data.get(key)
        if not isinstance(value, int):
            msg = f"Expected integer for {key!r}"
            raise ValueError(msg)
        return value

    def _require_bool(self, data: dict[str, Any], key: str) -> bool:
        value = data.get(key)
        if not isinstance(value, bool):
            msg = f"Expected boolean for {key!r}"
            raise ValueError(msg)
        return value
