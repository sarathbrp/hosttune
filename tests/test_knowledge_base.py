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
