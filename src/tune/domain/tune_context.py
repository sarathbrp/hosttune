from __future__ import annotations

from dataclasses import dataclass

from baseline.domain.models import BaselineResult
from onboard.domain.models import OnboardResult
from preflight.domain.models import DiscoverySnapshot
from preflight.domain.runtime_artifacts import RuntimeArtifacts
from snapshot.domain.models import SnapshotResult


@dataclass(frozen=True)
class TuneContext:
    preflight: DiscoverySnapshot
    onboard: OnboardResult
    snapshot: SnapshotResult
    baseline: BaselineResult
    artifacts: RuntimeArtifacts | None
