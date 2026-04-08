import sqlite3

from preflight.infrastructure.knowledge_base import KnowledgeBase

from tests.tune.test_candidate_catalog_builder import build_tune_context


def test_knowledge_base_records_and_queries_similar_runs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    context = build_tune_context()
    path = tmp_path / "artifacts" / "knowledge_base.sqlite"
    knowledge_base = KnowledgeBase(path)

    knowledge_base.record_run(
        run_id="run-old",
        preflight=context.preflight,
        service_name=context.onboard.service_name,
        benchmark_target=context.baseline.benchmark_target,
    )
    knowledge_base.record_event(
        run_id="run-old",
        component="best_config_tracker",
        event_type="best_config_updated",
        service_name=context.onboard.service_name,
        payload={"score": 0.25, "parameter_values": {"service.directive.access_log": "off"}},
    )
    knowledge_base.finalize_run(
        run_id="run-old",
        stop_reason="converged",
        best_score=0.25,
        best_iteration=2,
        best_config={"service.directive.access_log": "off"},
        final_retained_config={
            "service.directive.access_log": "off",
            "sysctl.net.core.somaxconn": "8192",
        },
    )

    knowledge_base.record_run(
        run_id="run-new",
        preflight=context.preflight,
        service_name=context.onboard.service_name,
        benchmark_target=context.baseline.benchmark_target,
    )

    similar = knowledge_base.find_similar_runs(
        service_name=context.onboard.service_name,
        cpu_logical_cores=context.preflight.cpu.logical_cores,
        numa_nodes=context.preflight.cpu.numa_nodes,
        platform_summary=context.preflight.platform_summary,
        nic_driver=context.preflight.network.driver_name,
        exclude_run_id="run-new",
    )
    summary = knowledge_base.summarize_similar_runs(
        service_name=context.onboard.service_name,
        cpu_logical_cores=context.preflight.cpu.logical_cores,
        numa_nodes=context.preflight.cpu.numa_nodes,
        platform_summary=context.preflight.platform_summary,
        nic_driver=context.preflight.network.driver_name,
        exclude_run_id="run-new",
    )
    run_summary = knowledge_base.get_run_summary("run-old")
    best_config = knowledge_base.get_best_config("run-old")

    assert len(similar) == 1
    assert similar[0]["run_id"] == "run-old"
    assert "service.directive.access_log=off" in summary
    assert run_summary is not None
    assert run_summary["stop_reason"] == "converged"
    assert run_summary["final_retained_config"]["sysctl.net.core.somaxconn"] == "8192"
    assert best_config is not None
    assert best_config["best_iteration"] == 2


def test_knowledge_base_appends_events(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "artifacts" / "knowledge_base.sqlite"
    knowledge_base = KnowledgeBase(path)
    knowledge_base.record_event(
        run_id="run-1",
        component="snapshot_engine",
        event_type="preflight_completed",
        payload={"platform_summary": "bare_metal_linux"},
    )
    knowledge_base.record_event(
        run_id="run-1",
        component="snapshot_engine",
        event_type="baseline_completed",
        payload={"benchmark_target": "127.0.0.1"},
    )

    events = knowledge_base.get_run_events("run-1")

    assert [event["event_type"] for event in events] == [
        "preflight_completed",
        "baseline_completed",
    ]
    with sqlite3.connect(path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert count == 2


def test_knowledge_base_recovers_confidence_from_legacy_events(tmp_path) -> None:  # type: ignore[no-untyped-def]
    context = build_tune_context()
    path = tmp_path / "artifacts" / "knowledge_base.sqlite"
    knowledge_base = KnowledgeBase(path)

    knowledge_base.record_run(
        run_id="run-old",
        preflight=context.preflight,
        service_name=context.onboard.service_name,
        benchmark_target=context.baseline.benchmark_target,
    )
    knowledge_base.record_event(
        run_id="run-old",
        iteration_number=1,
        component="tuning_executor",
        event_type="change_applied",
        service_name=context.onboard.service_name,
        payload={
            "parameter_key": "service.directive.access_log",
            "applied_value": "off",
        },
    )
    knowledge_base.record_event(
        run_id="run-old",
        iteration_number=1,
        component="benchmark_runner",
        event_type="evaluation_completed",
        service_name=context.onboard.service_name,
        payload={
            "decision": "accept",
            "summary": "ok",
            "guardrails_held": True,
            "drift_detected": False,
            "workloads": [],
        },
    )
    knowledge_base.record_event(
        run_id="run-old",
        iteration_number=2,
        component="tuning_executor",
        event_type="change_applied",
        service_name=context.onboard.service_name,
        payload={
            "parameter_key": "sysctl.net.core.netdev_max_backlog",
            "applied_value": "100000",
        },
    )
    knowledge_base.record_event(
        run_id="run-old",
        iteration_number=2,
        component="benchmark_runner",
        event_type="evaluation_completed",
        service_name=context.onboard.service_name,
        payload={
            "decision": "reject",
            "summary": "regressed",
            "guardrails_held": True,
            "drift_detected": False,
            "workloads": [],
        },
    )
    knowledge_base.finalize_run(
        run_id="run-old",
        stop_reason="completed",
        best_score=0.1,
        best_iteration=1,
        best_config={"service.directive.access_log": "off"},
        final_retained_config={"service.directive.access_log": "off"},
    )

    scores = knowledge_base.get_parameter_confidence_scores(
        service_name=context.onboard.service_name,
        cpu_logical_cores=context.preflight.cpu.logical_cores,
        numa_nodes=context.preflight.cpu.numa_nodes,
        platform_summary=context.preflight.platform_summary,
        nic_driver=context.preflight.network.driver_name,
    )
    blocked = knowledge_base.get_prior_blocked_pairs(
        service_name=context.onboard.service_name,
        cpu_logical_cores=context.preflight.cpu.logical_cores,
        numa_nodes=context.preflight.cpu.numa_nodes,
        platform_summary=context.preflight.platform_summary,
        nic_driver=context.preflight.network.driver_name,
    )

    assert scores["service.directive.access_log"] == (1, 1, 1.0)
    assert scores["sysctl.net.core.netdev_max_backlog"] == (1, 0, 0.0)
    assert ("sysctl.net.core.netdev_max_backlog", "100000") in blocked


def test_knowledge_base_reads_applied_parameter_values_payload(tmp_path) -> None:  # type: ignore[no-untyped-def]
    context = build_tune_context()
    path = tmp_path / "artifacts" / "knowledge_base.sqlite"
    knowledge_base = KnowledgeBase(path)

    knowledge_base.record_run(
        run_id="run-old",
        preflight=context.preflight,
        service_name=context.onboard.service_name,
        benchmark_target=context.baseline.benchmark_target,
    )
    knowledge_base.record_event(
        run_id="run-old",
        iteration_number=1,
        component="benchmark_runner",
        event_type="evaluation_completed",
        service_name=context.onboard.service_name,
        payload={
            "decision": "accept",
            "applied_parameter_values": [
                {"parameter_key": "service.directive.sendfile", "proposed_value": "on"},
                {"parameter_key": "service.directive.tcp_nopush", "proposed_value": "on"},
            ],
            "summary": "ok",
            "guardrails_held": True,
            "drift_detected": False,
            "workloads": [],
        },
    )
    knowledge_base.record_event(
        run_id="run-old",
        iteration_number=2,
        component="benchmark_runner",
        event_type="evaluation_completed",
        service_name=context.onboard.service_name,
        payload={
            "decision": "inconclusive",
            "applied_parameter_values": [
                {"parameter_key": "service.directive.keepalive_timeout", "proposed_value": "15"},
            ],
            "summary": "noise",
            "guardrails_held": True,
            "drift_detected": False,
            "workloads": [],
        },
    )
    knowledge_base.finalize_run(
        run_id="run-old",
        stop_reason="completed",
        best_score=0.2,
        best_iteration=1,
        best_config={"service.directive.sendfile": "on"},
        final_retained_config={"service.directive.sendfile": "on"},
    )

    scores = knowledge_base.get_parameter_confidence_scores(
        service_name=context.onboard.service_name,
        cpu_logical_cores=context.preflight.cpu.logical_cores,
        numa_nodes=context.preflight.cpu.numa_nodes,
        platform_summary=context.preflight.platform_summary,
        nic_driver=context.preflight.network.driver_name,
    )
    blocked = knowledge_base.get_prior_blocked_pairs(
        service_name=context.onboard.service_name,
        cpu_logical_cores=context.preflight.cpu.logical_cores,
        numa_nodes=context.preflight.cpu.numa_nodes,
        platform_summary=context.preflight.platform_summary,
        nic_driver=context.preflight.network.driver_name,
    )

    assert scores["service.directive.sendfile"] == (1, 1, 1.0)
    assert scores["service.directive.tcp_nopush"] == (1, 1, 1.0)
    assert scores["service.directive.keepalive_timeout"] == (1, 0, 0.0)
    assert ("service.directive.keepalive_timeout", "15") in blocked


def test_find_similar_runs_prefers_high_homepage_rps(tmp_path) -> None:  # type: ignore[no-untyped-def]
    context = build_tune_context()
    path = tmp_path / "artifacts" / "knowledge_base.sqlite"
    knowledge_base = KnowledgeBase(path)

    knowledge_base.record_run(
        run_id="run-a",
        preflight=context.preflight,
        service_name=context.onboard.service_name,
        benchmark_target=context.baseline.benchmark_target,
    )
    knowledge_base.record_event(
        run_id="run-a",
        component="best_config_tracker",
        event_type="best_config_updated",
        service_name=context.onboard.service_name,
        payload={
            "score": 10.0,
            "parameter_values": {"service.directive.access_log": "off"},
            "workloads": [
                {"workload_name": "homepage", "current_requests_per_second": 100.0},
            ],
        },
    )
    knowledge_base.finalize_run(
        run_id="run-a",
        stop_reason="done",
        best_score=10.0,
        best_iteration=1,
        best_config={"service.directive.access_log": "off"},
        final_retained_config={"service.directive.access_log": "off"},
    )

    knowledge_base.record_run(
        run_id="run-b",
        preflight=context.preflight,
        service_name=context.onboard.service_name,
        benchmark_target=context.baseline.benchmark_target,
    )
    knowledge_base.record_event(
        run_id="run-b",
        component="best_config_tracker",
        event_type="best_config_updated",
        service_name=context.onboard.service_name,
        payload={
            "score": 1.0,
            "parameter_values": {"service.directive.sendfile": "on"},
            "workloads": [
                {"workload_name": "homepage", "current_requests_per_second": 200.0},
            ],
        },
    )
    knowledge_base.finalize_run(
        run_id="run-b",
        stop_reason="done",
        best_score=1.0,
        best_iteration=1,
        best_config={"service.directive.sendfile": "on"},
        final_retained_config={"service.directive.sendfile": "on"},
    )

    knowledge_base.record_run(
        run_id="run-current",
        preflight=context.preflight,
        service_name=context.onboard.service_name,
        benchmark_target=context.baseline.benchmark_target,
    )

    similar = knowledge_base.find_similar_runs(
        service_name=context.onboard.service_name,
        cpu_logical_cores=context.preflight.cpu.logical_cores,
        numa_nodes=context.preflight.cpu.numa_nodes,
        platform_summary=context.preflight.platform_summary,
        nic_driver=context.preflight.network.driver_name,
        exclude_run_id="run-current",
        limit=2,
    )

    assert [item["run_id"] for item in similar] == ["run-b", "run-a"]
