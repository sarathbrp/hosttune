from __future__ import annotations

import json
from dataclasses import asdict

from preflight.domain.models import DiscoverySnapshot


class ConsoleReporter:
    def render(self, snapshot: DiscoverySnapshot) -> str:
        return json.dumps(asdict(snapshot), indent=2, default=str)
