from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from tune.domain.iteration_record import TuneIterationRecord
from tune.domain.scoreboard_models import TuneScoreboard
from tune.domain.tune_context import TuneContext


class TuneRecorder:
    def record(
        self,
        context: TuneContext,
        record: TuneIterationRecord,
    ) -> Path:
        artifacts = context.artifacts
        if artifacts is None:
            msg = "TuneContext must include artifacts before recording iterations."
            raise ValueError(msg)

        hypotheses_directory = artifacts.session_directory / "hypotheses"
        hypotheses_directory.mkdir(parents=True, exist_ok=True)
        file_path = hypotheses_directory / f"tune_iterations_{artifacts.session_id}.jsonl"
        payload = {
            "session_id": artifacts.session_id,
            "iteration_number": record.iteration_number,
            "phase": record.phase.value,
            "record": asdict(record),
        }
        with file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str))
            handle.write("\n")
        artifacts.stage_files["tune_iterations"] = file_path
        return file_path

    def record_scoreboard(
        self,
        context: TuneContext,
        scoreboard: TuneScoreboard,
    ) -> Path:
        artifacts = context.artifacts
        if artifacts is None:
            msg = "TuneContext must include artifacts before recording scoreboard."
            raise ValueError(msg)

        hypotheses_directory = artifacts.session_directory / "hypotheses"
        hypotheses_directory.mkdir(parents=True, exist_ok=True)
        file_path = hypotheses_directory / f"tune_scoreboard_{artifacts.session_id}.json"
        payload = {
            "session_id": artifacts.session_id,
            "scoreboard": asdict(scoreboard),
        }
        file_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        artifacts.stage_files["tune_scoreboard"] = file_path
        return file_path
