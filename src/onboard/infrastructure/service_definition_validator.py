from __future__ import annotations

from typing import Any, cast

from onboard.domain.models import (
    ApplyMode,
    ConfigFormat,
    DirectiveConstraint,
    DirectiveValueType,
    ParameterGroup,
    ParameterGroupMember,
    PerformanceHierarchy,
    PerformanceHierarchyGroup,
    PerformanceHierarchyParameter,
    PriorityTier,
    ProbeType,
    ProcessState,
    ReloadContract,
    RestartContract,
    ServiceBenchmarkHints,
    ServiceDefinition,
    ServiceHealthCheck,
    ServiceIdentity,
    ServiceRestartContract,
    ServiceSnapshotContract,
    ServiceTunableSurface,
    SysctlTunable,
)

_VALID_TUNING_LAYER_STRINGS = frozenset({"kernel", "network", "service", "runtime"})


class ServiceDefinitionValidator:
    REQUIRED_BLOCKS = (
        "identity",
        "health_check",
        "snapshot",
        "restart",
        "tunable_surface",
        "benchmark_hints",
    )

    def validate(self, data: dict[str, Any]) -> ServiceDefinition:
        missing = [block for block in self.REQUIRED_BLOCKS if block not in data]
        if missing:
            msg = f"Missing required service definition blocks: {', '.join(missing)}"
            raise ValueError(msg)

        return ServiceDefinition(
            identity=self._parse_identity(cast(dict[str, Any], data["identity"])),
            health_check=self._parse_health_check(cast(dict[str, Any], data["health_check"])),
            snapshot=self._parse_snapshot(cast(dict[str, Any], data["snapshot"])),
            restart=self._parse_restart(cast(dict[str, Any], data["restart"])),
            tunable_surface=self._parse_tunable_surface(
                cast(dict[str, Any], data["tunable_surface"])
            ),
            benchmark_hints=self._parse_benchmark_hints(
                cast(dict[str, Any], data["benchmark_hints"])
            ),
        )

    def _parse_identity(self, data: dict[str, Any]) -> ServiceIdentity:
        return ServiceIdentity(
            service_name=self._require_str(data, "service_name"),
            systemd_unit_name=self._require_str(data, "systemd_unit_name"),
            rhel_versions=self._require_str_tuple(data, "rhel_versions"),
            service_versions=self._str_tuple(data.get("service_versions", [])),
            config_paths=self._require_str_tuple(data, "config_paths"),
            config_format=ConfigFormat(self._require_str(data, "config_format")),
            working_directory=self._optional_str(data.get("working_directory")),
            log_paths=self._str_tuple(data.get("log_paths", [])),
        )

    def _parse_health_check(self, data: dict[str, Any]) -> ServiceHealthCheck:
        return ServiceHealthCheck(
            probe_type=ProbeType(self._require_str(data, "probe_type")),
            target=self._require_str(data, "target"),
            expected_status_code=self._optional_int(data.get("expected_status_code")),
            expected_string=self._optional_str(data.get("expected_string")),
            expected_exit_code=self._optional_int(data.get("expected_exit_code")),
            timeout_seconds=self._require_int(data, "timeout_seconds"),
            retries=self._require_int(data, "retries"),
            warmup_seconds=self._require_int(data, "warmup_seconds"),
        )

    def _parse_snapshot(self, data: dict[str, Any]) -> ServiceSnapshotContract:
        process_state_data = cast(dict[str, Any], data.get("process_state", {}))
        return ServiceSnapshotContract(
            files_to_snapshot=self._require_str_tuple(data, "files_to_snapshot"),
            runtime_state_command=self._optional_str(data.get("runtime_state_command")),
            process_state=ProcessState(
                pid_file=self._optional_str(process_state_data.get("pid_file")),
                worker_process_hint=self._optional_str(
                    process_state_data.get("worker_process_hint")
                ),
                open_connections_command=self._optional_str(
                    process_state_data.get("open_connections_command")
                ),
            ),
            restore_sequence=self._require_str_tuple(data, "restore_sequence"),
            snapshot_storage_location=self._require_str(data, "snapshot_storage_location"),
        )

    def _parse_restart(self, data: dict[str, Any]) -> ServiceRestartContract:
        reload_data = cast(dict[str, Any], data.get("reload", {}))
        restart_data = cast(dict[str, Any], data.get("restart", {}))
        change_categories_data = cast(dict[str, Any], data.get("change_categories", {}))
        return ServiceRestartContract(
            reload=ReloadContract(
                supported=self._require_bool(reload_data, "supported"),
                command=self._optional_str(reload_data.get("command")),
            ),
            restart=RestartContract(
                supported=self._require_bool(restart_data, "supported"),
                command=self._optional_str(restart_data.get("command")),
                expected_downtime_seconds=self._require_int(
                    restart_data, "expected_downtime_seconds"
                ),
            ),
            change_categories={
                category: ApplyMode(self._require_str(change_categories_data, category))
                for category in change_categories_data
            },
            drain_policy=self._optional_str(data.get("drain_policy")),
            dependency_chain=self._str_tuple(data.get("dependency_chain", [])),
            post_restart_validation=self._require_str(data, "post_restart_validation"),
        )

    def _parse_tunable_surface(self, data: dict[str, Any]) -> ServiceTunableSurface:
        directives = cast(dict[str, Any], data.get("allowed_directives", {}))
        ring_tier_raw = data.get("network_ring_priority_tier", "high")
        if not isinstance(ring_tier_raw, str):
            msg = "Expected string for 'network_ring_priority_tier'"
            raise ValueError(msg)
        ring_layer_raw = data.get("network_ring_tuning_layer")
        runtime_limits_raw = cast(dict[str, Any], data.get("runtime_limits", {}))
        systemd_limits_raw = cast(dict[str, Any], data.get("systemd_unit_limits", {}))
        cgroup_controls_raw = cast(dict[str, Any], data.get("cgroup_resource_controls", {}))
        parameter_groups_raw = data.get("parameter_groups", [])
        performance_hierarchy_raw = data.get("performance_hierarchy")
        return ServiceTunableSurface(
            allowed_directives={
                name: self._parse_directive_constraint(cast(dict[str, Any], value))
                for name, value in directives.items()
            },
            forbidden_directives=self._str_tuple(data.get("forbidden_directives", [])),
            interdependencies=self._str_tuple(data.get("interdependencies", [])),
            relevant_sysctls=self._parse_relevant_sysctls(data.get("relevant_sysctls", [])),
            network_ring_priority_tier=PriorityTier(ring_tier_raw),
            runtime_limits={
                name: self._parse_directive_constraint(cast(dict[str, Any], value))
                for name, value in runtime_limits_raw.items()
            },
            systemd_unit_limits={
                name: self._parse_directive_constraint(cast(dict[str, Any], value))
                for name, value in systemd_limits_raw.items()
            },
            network_ring_tuning_layer=self._optional_tuning_layer_string(ring_layer_raw),
            cgroup_resource_controls={
                name: self._parse_directive_constraint(cast(dict[str, Any], value))
                for name, value in cgroup_controls_raw.items()
            },
            parameter_groups=self._parse_parameter_groups(parameter_groups_raw),
            performance_hierarchy=self._parse_performance_hierarchy(performance_hierarchy_raw),
        )

    def _parse_parameter_groups(self, value: object) -> tuple[ParameterGroup, ...]:
        if value is None or value == []:
            return ()
        if not isinstance(value, list):
            msg = "Expected list for 'parameter_groups'"
            raise ValueError(msg)
        groups: list[ParameterGroup] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                msg = f"parameter_groups[{index}] must be an object"
                raise ValueError(msg)
            name = self._require_str(item, "name")
            members_raw = item.get("members")
            if not isinstance(members_raw, list) or not members_raw:
                msg = f"parameter_groups[{index}].members must be a non-empty list"
                raise ValueError(msg)
            members: list[ParameterGroupMember] = []
            for m_index, member in enumerate(members_raw):
                if not isinstance(member, dict):
                    msg = f"parameter_groups[{index}].members[{m_index}] must be an object"
                    raise ValueError(msg)
                members.append(
                    ParameterGroupMember(
                        parameter_key=self._require_str(member, "parameter_key"),
                        target_value=self._require_str(member, "target_value"),
                    )
                )
            groups.append(ParameterGroup(name=name, members=tuple(members)))
        return tuple(groups)

    def _parse_performance_hierarchy(
        self, value: object
    ) -> PerformanceHierarchy | None:
        if value is None or value == {}:
            return None
        if not isinstance(value, dict):
            msg = "Expected mapping for 'performance_hierarchy'"
            raise ValueError(msg)
        raw = cast(dict[str, Any], value)
        groups_raw = raw.get("groups")
        if groups_raw is None:
            return None
        groups = self._parse_performance_hierarchy_groups(groups_raw)
        return PerformanceHierarchy(
            version=self._optional_str(raw.get("version")),
            description=self._optional_str(raw.get("description")),
            groups=groups,
        )

    def _parse_performance_hierarchy_groups(
        self, value: object
    ) -> tuple[PerformanceHierarchyGroup, ...]:
        if isinstance(value, dict):
            parsed: list[PerformanceHierarchyGroup] = []
            for group_id, item in cast(dict[str, Any], value).items():
                if not isinstance(group_id, str) or group_id == "":
                    msg = "performance_hierarchy.groups keys must be non-empty strings"
                    raise ValueError(msg)
                if not isinstance(item, dict):
                    msg = f"performance_hierarchy.groups[{group_id!r}] must be an object"
                    raise ValueError(msg)
                item_dict = cast(dict[str, Any], item)
                parsed.append(
                    PerformanceHierarchyGroup(
                        group_id=group_id,
                        description=self._require_str(item_dict, "description"),
                        parameters=self._parse_performance_hierarchy_parameters(
                            item_dict.get("parameters"),
                        ),
                        depends_on_groups=self._str_tuple(item_dict.get("depends_on_groups", [])),
                    )
                )
            return tuple(parsed)
        if isinstance(value, list):
            parsed = []
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    msg = f"performance_hierarchy.groups[{index}] must be an object"
                    raise ValueError(msg)
                item_dict = cast(dict[str, Any], item)
                group_id = self._require_str(item_dict, "group_id")
                parsed.append(
                    PerformanceHierarchyGroup(
                        group_id=group_id,
                        description=self._require_str(item_dict, "description"),
                        parameters=self._parse_performance_hierarchy_parameters(
                            item_dict.get("parameters"),
                        ),
                        depends_on_groups=self._str_tuple(item_dict.get("depends_on_groups", [])),
                    )
                )
            return tuple(parsed)
        msg = "performance_hierarchy.groups must be a mapping or list"
        raise ValueError(msg)

    def _parse_performance_hierarchy_parameters(
        self, value: object
    ) -> tuple[PerformanceHierarchyParameter, ...]:
        if isinstance(value, dict):
            parsed: list[PerformanceHierarchyParameter] = []
            for parameter_name, item in cast(dict[str, Any], value).items():
                if not isinstance(parameter_name, str) or parameter_name == "":
                    msg = (
                        "performance_hierarchy group parameter names must be non-empty strings"
                    )
                    raise ValueError(msg)
                if not isinstance(item, dict):
                    msg = f"performance_hierarchy parameter {parameter_name!r} must be an object"
                    raise ValueError(msg)
                item_dict = cast(dict[str, Any], item)
                parsed.append(
                    PerformanceHierarchyParameter(
                        name=parameter_name,
                        target_perf=self._require_str(item_dict, "target_perf"),
                        inspect_cmd=self._optional_str(item_dict.get("inspect_cmd")),
                        detail=self._optional_str(item_dict.get("detail")),
                        candidate_key=self._optional_str(item_dict.get("candidate_key")),
                        target_type=self._optional_str(item_dict.get("target_type")),
                        target_value=self._optional_target_value(item_dict.get("target_value")),
                        enforceable=self._optional_bool(item_dict.get("enforceable"), False),
                        depends_on_groups=self._str_tuple(item_dict.get("depends_on_groups", [])),
                    )
                )
            return tuple(parsed)
        if isinstance(value, list):
            parsed = []
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    msg = f"performance_hierarchy parameter[{index}] must be an object"
                    raise ValueError(msg)
                item_dict = cast(dict[str, Any], item)
                parsed.append(
                    PerformanceHierarchyParameter(
                        name=self._require_str(item_dict, "name"),
                        target_perf=self._require_str(item_dict, "target_perf"),
                        inspect_cmd=self._optional_str(item_dict.get("inspect_cmd")),
                        detail=self._optional_str(item_dict.get("detail")),
                        candidate_key=self._optional_str(item_dict.get("candidate_key")),
                        target_type=self._optional_str(item_dict.get("target_type")),
                        target_value=self._optional_target_value(item_dict.get("target_value")),
                        enforceable=self._optional_bool(item_dict.get("enforceable"), False),
                        depends_on_groups=self._str_tuple(item_dict.get("depends_on_groups", [])),
                    )
                )
            return tuple(parsed)
        msg = "performance_hierarchy group parameters must be a mapping or list"
        raise ValueError(msg)

    def _optional_target_value(self, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, (str, int, float)):
            return str(value)
        msg = "Expected target_value to be string/number/null"
        raise ValueError(msg)

    def _parse_relevant_sysctls(self, value: object) -> tuple[SysctlTunable, ...]:
        if value is None or value == []:
            return ()
        if not isinstance(value, list):
            msg = "Expected list for 'relevant_sysctls'"
            raise ValueError(msg)
        entries: list[SysctlTunable] = []
        for index, item in enumerate(value):
            if isinstance(item, str):
                if item == "":
                    msg = f"Empty sysctl name at relevant_sysctls[{index}]"
                    raise ValueError(msg)
                entries.append(SysctlTunable(name=item, priority_tier=PriorityTier.HIGH))
                continue
            if isinstance(item, dict):
                item_dict = cast(dict[str, Any], item)
                name = self._require_str(item_dict, "name")
                tier = PriorityTier(self._require_str(item_dict, "priority_tier"))
                entries.append(
                    SysctlTunable(
                        name=name,
                        priority_tier=tier,
                        tuning_layer=self._optional_tuning_layer_string(
                            item_dict.get("tuning_layer")
                        ),
                    )
                )
                continue
            msg = (
                f"Each relevant_sysctls entry must be a string or object with "
                f"name and priority_tier; invalid at index {index}"
            )
            raise ValueError(msg)
        return tuple(entries)

    def _parse_directive_constraint(self, data: dict[str, Any]) -> DirectiveConstraint:
        return DirectiveConstraint(
            value_type=DirectiveValueType(self._require_str(data, "value_type")),
            apply_mode=ApplyMode(self._require_str(data, "apply_mode")),
            priority_tier=PriorityTier(self._require_str(data, "priority_tier")),
            min_value=self._optional_int(data.get("min_value")),
            max_value=self._optional_int(data.get("max_value")),
            allowed_values=self._str_tuple(data.get("allowed_values", [])),
            forbidden_values=self._str_tuple(data.get("forbidden_values", [])),
            tuning_layer=self._optional_tuning_layer_string(data.get("tuning_layer")),
        )

    def _optional_tuning_layer_string(self, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or value == "":
            msg = "Expected non-empty string or null for 'tuning_layer'"
            raise ValueError(msg)
        if value not in _VALID_TUNING_LAYER_STRINGS:
            allowed = ", ".join(sorted(_VALID_TUNING_LAYER_STRINGS))
            msg = f"Invalid tuning_layer {value!r}; expected one of: {allowed}"
            raise ValueError(msg)
        return value

    def _parse_benchmark_hints(self, data: dict[str, Any]) -> ServiceBenchmarkHints:
        return ServiceBenchmarkHints(
            primary_metric=self._require_str(data, "primary_metric"),
            guardrail_metrics=self._str_tuple(data.get("guardrail_metrics", [])),
            expected_variance=self._require_float(data, "expected_variance"),
            warmup_seconds=self._require_int(data, "warmup_seconds"),
            interference_sources=self._str_tuple(data.get("interference_sources", [])),
        )

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
            msg = "Expected string or null."
            raise ValueError(msg)
        return value

    def _require_int(self, data: dict[str, Any], key: str) -> int:
        value = data.get(key)
        if not isinstance(value, int):
            msg = f"Expected integer for {key!r}"
            raise ValueError(msg)
        return value

    def _optional_int(self, value: object) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int):
            msg = "Expected integer or null."
            raise ValueError(msg)
        return value

    def _optional_bool(self, value: object, default: bool) -> bool:
        if value is None:
            return default
        if not isinstance(value, bool):
            msg = "Expected boolean or null."
            raise ValueError(msg)
        return value

    def _require_float(self, data: dict[str, Any], key: str) -> float:
        value = data.get(key)
        if not isinstance(value, int | float):
            msg = f"Expected numeric value for {key!r}"
            raise ValueError(msg)
        return float(value)

    def _require_bool(self, data: dict[str, Any], key: str) -> bool:
        value = data.get(key)
        if not isinstance(value, bool):
            msg = f"Expected boolean for {key!r}"
            raise ValueError(msg)
        return value

    def _require_str_tuple(self, data: dict[str, Any], key: str) -> tuple[str, ...]:
        value = data.get(key)
        return self._str_tuple(value, key)

    def _str_tuple(self, value: object, key: str = "list") -> tuple[str, ...]:
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            msg = f"Expected list[str] for {key!r}"
            raise ValueError(msg)
        return tuple(value)
