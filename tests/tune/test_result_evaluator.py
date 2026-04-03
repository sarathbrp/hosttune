from tune.application.result_evaluator import ResultEvaluator
from tune.domain.benchmark_models import BenchmarkWorkloadSummary, TuneBenchmarkResult
from tune.domain.evaluation_models import EvaluationDecision

from tests.tune.test_benchmark_executor import build_validation_result
from tests.tune.test_candidate_catalog_builder import build_tune_context


def build_benchmark_result(
    *,
    homepage_rps: float,
    small_rps: float,
    stable: bool,
) -> TuneBenchmarkResult:
    return TuneBenchmarkResult(
        validation_result=build_validation_result(),
        benchmark_command="benchmark.sh hosttune",
        run_count=3,
        stable=stable,
        variance_threshold=0.05,
        workload_summaries=(
            BenchmarkWorkloadSummary(
                workload_name="homepage",
                samples=(),
                median_requests_per_second=homepage_rps,
                median_total_requests=10000,
                median_latency_ms=2.0,
                relative_variance=0.01,
                stable=stable,
            ),
            BenchmarkWorkloadSummary(
                workload_name="small",
                samples=(),
                median_requests_per_second=small_rps,
                median_total_requests=9000,
                median_latency_ms=1.5,
                relative_variance=0.01,
                stable=stable,
            ),
        ),
    )


def test_result_evaluator_accepts_real_improvement() -> None:
    context = build_tune_context()
    benchmark_result = build_benchmark_result(
        homepage_rps=1200.0,
        small_rps=1000.0,
        stable=True,
    )

    evaluation = ResultEvaluator().evaluate(context, benchmark_result)

    assert evaluation.decision is EvaluationDecision.ACCEPT
    assert evaluation.guardrails_held is True
    assert evaluation.drift_detected is False


def test_result_evaluator_rejects_clear_regression() -> None:
    context = build_tune_context()
    benchmark_result = build_benchmark_result(
        homepage_rps=800.0,
        small_rps=700.0,
        stable=True,
    )

    evaluation = ResultEvaluator().evaluate(context, benchmark_result)

    assert evaluation.decision is EvaluationDecision.REJECT


def test_result_evaluator_marks_unstable_results_inconclusive() -> None:
    context = build_tune_context()
    benchmark_result = build_benchmark_result(
        homepage_rps=1300.0,
        small_rps=1100.0,
        stable=False,
    )

    evaluation = ResultEvaluator().evaluate(context, benchmark_result)

    assert evaluation.decision is EvaluationDecision.INCONCLUSIVE
    assert evaluation.drift_detected is True
