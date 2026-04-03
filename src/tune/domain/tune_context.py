from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from baseline.domain.models import BaselineResult, BenchmarkConfig
from onboard.domain.models import OnboardResult
from preflight.domain.models import DiscoverySnapshot
from preflight.domain.runtime_artifacts import RuntimeArtifacts
from snapshot.domain.models import SnapshotResult

if TYPE_CHECKING:
    from host_profile.domain.models import HostProfile


@dataclass(frozen=True)
class TuneContext:
    preflight: DiscoverySnapshot
    onboard: OnboardResult
    snapshot: SnapshotResult
    baseline: BaselineResult
    benchmark_config: BenchmarkConfig
    artifacts: RuntimeArtifacts | None
    host_profile: HostProfile | None = None  # optional; loaded from host-profiles/ by name

    @property
    def effective_variance_threshold(self) -> float:
        """Strictest of: service noise floor (YAML) and operator threshold (config.yaml policy).

        benchmark_hints.expected_variance = natural measurement jitter for this service/workload.
        policy.benchmark_stability_threshold = operator minimum signal to accept.
        Take the max so whichever is the harder bar wins.
        """
        threshold = max(
            self.baseline.expected_variance,
            self.preflight.policy.benchmark_stability_threshold,
        )
        if threshold > 0.50 or threshold <= 0.0:
            import logging

            logging.getLogger(__name__).warning(
                "effective_variance_threshold=%.4f is outside reasonable range "
                "(0.01-0.50); check expected_variance and "
                "benchmark_stability_threshold config",
                threshold,
            )
        return threshold
