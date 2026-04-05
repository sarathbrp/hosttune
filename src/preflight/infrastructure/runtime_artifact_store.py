from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from preflight.domain.runtime_artifacts import RuntimeArtifacts


class RuntimeArtifactStore:
    SESSION_ID_LENGTH = 12

    def __init__(self, base_directory: Path = Path("artifacts")) -> None:
        self._base_directory = base_directory

    def create_session(self) -> RuntimeArtifacts:
        session_id = uuid4().hex[: self.SESSION_ID_LENGTH]
        session_directory = self._base_directory / session_id
        session_directory.mkdir(parents=True, exist_ok=True)
        return RuntimeArtifacts(
            session_id=session_id,
            session_directory=session_directory,
        )

    def knowledge_base_path(self) -> Path:
        return self._base_directory / "knowledge_base.sqlite"

    def write_stage_result(
        self,
        artifacts: RuntimeArtifacts,
        stage_name: str,
        payload: object,
    ) -> Path:
        file_path = artifacts.session_directory / f"{stage_name}_{artifacts.session_id}.jsonl"
        record = {
            "session_id": artifacts.session_id,
            "stage": stage_name,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "payload": self._serialize_payload(payload),
        }
        with file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str))
            handle.write("\n")
        artifacts.stage_files[stage_name] = file_path
        return file_path

    def _serialize_payload(self, payload: object) -> object:
        if is_dataclass(payload):
            return asdict(payload)
        return payload
