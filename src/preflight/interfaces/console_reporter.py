from __future__ import annotations

import json
from dataclasses import asdict

from baseline.domain.models import BaselineResult
from onboard.domain.models import OnboardResult
from preflight.domain.models import DiscoverySnapshot
from snapshot.domain.models import SnapshotResult


class ConsoleReporter:
    def render(self, snapshot: DiscoverySnapshot) -> str:
        return json.dumps(asdict(snapshot), indent=2, default=str)

    def render_runtime(
        self,
        preflight: DiscoverySnapshot,
        onboard: OnboardResult | None,
        snapshot: SnapshotResult | None,
        baseline: BaselineResult | None,
    ) -> str:
        payload: dict[str, object] = {"preflight": asdict(preflight)}
        if onboard is not None:
            payload["onboard"] = asdict(onboard)
        if snapshot is not None:
            payload["snapshot"] = asdict(snapshot)
        if baseline is not None:
            payload["baseline"] = asdict(baseline)
        return json.dumps(payload, indent=2, default=str)
