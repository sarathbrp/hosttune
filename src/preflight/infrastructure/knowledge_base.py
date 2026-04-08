from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from preflight.domain.models import DiscoverySnapshot


def _serialize_payload(payload: object) -> object:
    if is_dataclass(payload):
        return asdict(payload)
    if isinstance(payload, dict):
        return {key: _serialize_payload(value) for key, value in payload.items()}
    if isinstance(payload, tuple | list):
        return [_serialize_payload(item) for item in payload]
    return payload


def _cpu_core_band(logical_cores: int) -> str:
    if logical_cores < 16:
        return "1-15"
    if logical_cores < 32:
        return "16-31"
    if logical_cores < 64:
        return "32-63"
    if logical_cores < 128:
        return "64-127"
    return "128+"


def _memory_band(total_memory_kib: int) -> str:
    total_gib = total_memory_kib / (1024 * 1024)
    if total_gib < 64:
        return "<64GiB"
    if total_gib < 256:
        return "64-255GiB"
    if total_gib < 512:
        return "256-511GiB"
    return "512GiB+"


def host_fingerprint_for_snapshot(
    preflight: DiscoverySnapshot,
    service_name: str | None = None,
) -> str:
    raw = "|".join(
        (
            service_name or "",
            preflight.platform_summary,
            _cpu_core_band(preflight.cpu.logical_cores),
            str(preflight.cpu.numa_nodes),
            _memory_band(preflight.memory.total_memory_kib),
            preflight.network.driver_name,
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_degradation_fingerprint(
    workload_results: tuple[Any, ...] | list[Any],
) -> tuple[list[str], list[float]]:
    """Compute normalized RPS vector from baseline workload results.

    Returns (workload_names, rps_vector) sorted by name.
    The vector is L2-normalized for cosine similarity comparison.
    """
    sorted_results = sorted(workload_results, key=lambda w: w.workload_name)
    names = [w.workload_name for w in sorted_results]
    rps_values = [w.requests_per_second for w in sorted_results]
    norm = sum(v * v for v in rps_values) ** 0.5
    if norm == 0:
        return names, [0.0] * len(rps_values)
    return names, [v / norm for v in rps_values]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class KnowledgeBase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def record_run(
        self,
        *,
        run_id: str,
        preflight: DiscoverySnapshot,
        service_name: str,
        benchmark_target: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        fingerprint = host_fingerprint_for_snapshot(preflight, service_name)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, created_at_utc, updated_at_utc, target_host, service_name,
                    platform_summary, cpu_logical_cores, cpu_core_band, numa_nodes,
                    total_memory_kib, memory_band, nic_driver, host_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    updated_at_utc=excluded.updated_at_utc,
                    target_host=excluded.target_host,
                    service_name=excluded.service_name,
                    platform_summary=excluded.platform_summary,
                    cpu_logical_cores=excluded.cpu_logical_cores,
                    cpu_core_band=excluded.cpu_core_band,
                    numa_nodes=excluded.numa_nodes,
                    total_memory_kib=excluded.total_memory_kib,
                    memory_band=excluded.memory_band,
                    nic_driver=excluded.nic_driver,
                    host_fingerprint=excluded.host_fingerprint
                """,
                (
                    run_id,
                    now,
                    now,
                    benchmark_target,
                    service_name,
                    preflight.platform_summary,
                    preflight.cpu.logical_cores,
                    _cpu_core_band(preflight.cpu.logical_cores),
                    preflight.cpu.numa_nodes,
                    preflight.memory.total_memory_kib,
                    _memory_band(preflight.memory.total_memory_kib),
                    preflight.network.driver_name,
                    fingerprint,
                ),
            )
            connection.commit()

    def finalize_run(
        self,
        *,
        run_id: str,
        stop_reason: str,
        best_score: float | None,
        best_iteration: int | None,
        best_config: dict[str, str] | None,
        final_retained_config: dict[str, str] | None,
    ) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                UPDATE runs
                SET completed_at_utc=?,
                    updated_at_utc=?,
                    stop_reason=?,
                    best_score=?,
                    best_iteration=?,
                    best_config_json=?,
                    final_retained_config_json=?
                WHERE run_id=?
                """,
                (
                    datetime.now(UTC).isoformat(),
                    datetime.now(UTC).isoformat(),
                    stop_reason,
                    best_score,
                    best_iteration,
                    json.dumps(best_config or {}, sort_keys=True),
                    json.dumps(final_retained_config or {}, sort_keys=True),
                    run_id,
                ),
            )
            connection.commit()

    def record_event(
        self,
        *,
        run_id: str,
        component: str,
        event_type: str,
        payload: object,
        iteration_number: int | None = None,
        phase: str | None = None,
        service_name: str | None = None,
        host_fingerprint: str | None = None,
    ) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO events (
                    run_id, iteration_number, phase, component, event_type, created_at_utc,
                    service_name, host_fingerprint, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    iteration_number,
                    phase,
                    component,
                    event_type,
                    datetime.now(UTC).isoformat(),
                    service_name,
                    host_fingerprint,
                    json.dumps(_serialize_payload(payload), default=str, sort_keys=True),
                ),
            )
            connection.commit()

    def get_run_events(self, run_id: str) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT run_id, iteration_number, phase, component, event_type, created_at_utc,
                       service_name, host_fingerprint, payload_json
                FROM events
                WHERE run_id=?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        return [
            {
                **dict(row),
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def get_run_summary(self, run_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        payload = dict(row)
        best_config_json = payload.pop("best_config_json", None)
        final_retained_config_json = payload.pop("final_retained_config_json", None)
        payload["best_config"] = json.loads(best_config_json or "{}")
        payload["final_retained_config"] = json.loads(final_retained_config_json or "{}")
        return payload

    def get_best_config(self, run_id: str) -> dict[str, Any] | None:
        summary = self.get_run_summary(run_id)
        if summary is None or summary.get("best_iteration") is None:
            return None
        return {
            "best_score": summary["best_score"],
            "best_iteration": summary["best_iteration"],
            "best_config": summary["best_config"],
        }

    def find_similar_runs(
        self,
        *,
        service_name: str,
        cpu_logical_cores: int,
        numa_nodes: int,
        platform_summary: str,
        nic_driver: str | None,
        exclude_run_id: str | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        band = _cpu_core_band(cpu_logical_cores)
        params: list[object] = [service_name, band, numa_nodes, platform_summary]
        query = """
            SELECT run_id, target_host, service_name, platform_summary, cpu_logical_cores,
                   cpu_core_band, numa_nodes, total_memory_kib, nic_driver, stop_reason,
                   best_score, best_iteration, best_config_json, completed_at_utc
            FROM runs
            WHERE service_name=?
              AND cpu_core_band=?
              AND numa_nodes=?
              AND platform_summary=?
              AND completed_at_utc IS NOT NULL
              AND best_score IS NOT NULL
              AND COALESCE(best_config_json, '') NOT IN ('', '{}')
        """
        if nic_driver:
            query += " AND nic_driver=?"
            params.append(nic_driver)
        if exclude_run_id is not None:
            query += " AND run_id<>?"
            params.append(exclude_run_id)
        query += " ORDER BY completed_at_utc DESC"
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, tuple(params)).fetchall()
        if not rows:
            return []
        homepage_rps_by_run = self._best_homepage_rps_by_run([row["run_id"] for row in rows])
        ranked_rows = sorted(
            rows,
            key=lambda row: (
                homepage_rps_by_run.get(row["run_id"], 0.0),
                float(row["best_score"] or 0.0),
                str(row["completed_at_utc"] or ""),
            ),
            reverse=True,
        )
        results: list[dict[str, Any]] = []
        for row in ranked_rows[:limit]:
            item = dict(row)
            item["best_config"] = json.loads(item.pop("best_config_json") or "{}")
            item["best_homepage_rps"] = homepage_rps_by_run.get(item["run_id"], 0.0)
            results.append(item)
        return results

    def summarize_similar_runs(
        self,
        *,
        service_name: str,
        cpu_logical_cores: int,
        numa_nodes: int,
        platform_summary: str,
        nic_driver: str | None,
        exclude_run_id: str | None = None,
        limit: int = 3,
    ) -> str:
        runs = self.find_similar_runs(
            service_name=service_name,
            cpu_logical_cores=cpu_logical_cores,
            numa_nodes=numa_nodes,
            platform_summary=platform_summary,
            nic_driver=nic_driver,
            exclude_run_id=exclude_run_id,
            limit=limit,
        )
        if not runs:
            return ""
        lines = []
        for run in runs:
            config = (
                ", ".join(f"{key}={value}" for key, value in sorted(run["best_config"].items()))
                or "none"
            )
            score = "n/a" if run["best_score"] is None else f"{run['best_score']:.2%}"
            lines.append(
                f"- run={run['run_id']}; score={score}; "
                f"best={config}; stop={run['stop_reason'] or 'unknown'}"
            )
        return "\n".join(lines)

    def get_prior_blocked_pairs(
        self,
        *,
        service_name: str,
        cpu_logical_cores: int,
        numa_nodes: int,
        platform_summary: str,
        nic_driver: str | None,
        exclude_run_id: str | None = None,
        limit_runs: int = 3,
    ) -> list[tuple[str, str]]:
        """Parameter/value pairs that failed in prior similar runs.

        Returns (parameter_key, proposed_value) tuples from evaluation_completed
        events where the decision was reject or inconclusive.
        """
        runs = self.find_similar_runs(
            service_name=service_name,
            cpu_logical_cores=cpu_logical_cores,
            numa_nodes=numa_nodes,
            platform_summary=platform_summary,
            nic_driver=nic_driver,
            exclude_run_id=exclude_run_id,
            limit=limit_runs,
        )
        if not runs:
            return []
        run_ids = [run["run_id"] for run in runs]
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for decision, key, value in self._collect_parameter_decisions(run_ids):
            if decision not in ("reject", "inconclusive"):
                continue
            if key and value and (key, value) not in seen:
                seen.add((key, value))
                pairs.append((key, value))
        return pairs

    def get_prior_best_config(
        self,
        *,
        service_name: str,
        cpu_logical_cores: int,
        numa_nodes: int,
        platform_summary: str,
        nic_driver: str | None,
        exclude_run_id: str | None = None,
    ) -> dict[str, str] | None:
        """Best config from the highest-scoring prior similar run.

        Returns None if no prior run exists or none had an accepted config.
        """
        runs = self.find_similar_runs(
            service_name=service_name,
            cpu_logical_cores=cpu_logical_cores,
            numa_nodes=numa_nodes,
            platform_summary=platform_summary,
            nic_driver=nic_driver,
            exclude_run_id=exclude_run_id,
            limit=1,
        )
        if not runs:
            return None
        best_run = runs[0]
        config = best_run.get("best_config")
        if not config or best_run.get("best_score") is None:
            return None
        return config

    def get_best_workload_rps(
        self,
        *,
        service_name: str,
        exclude_run_id: str | None = None,
    ) -> dict[str, float]:
        """All-time best RPS per workload for this service from KB events.

        Queries all best_config_updated events and returns {workload_name: max_rps}.
        Always top-1 — as sessions improve, this automatically reflects the new best.
        """
        query = """
            SELECT payload_json FROM events
            WHERE event_type = 'best_config_updated'
              AND service_name = ?
        """
        params: list[object] = [service_name]
        if exclude_run_id:
            query += " AND run_id != ?"
            params.append(exclude_run_id)
        best: dict[str, float] = {}
        with sqlite3.connect(self.path) as connection:
            for row in connection.execute(query, tuple(params)).fetchall():
                try:
                    payload = json.loads(row[0])
                    for w in payload.get("workloads", []):
                        name = w.get("workload_name")
                        rps = w.get("current_requests_per_second")
                        if name and rps and float(rps) > best.get(name, 0.0):
                            best[name] = float(rps)
                except Exception:
                    continue
        return best

    def get_parameter_confidence_scores(
        self,
        *,
        service_name: str,
        cpu_logical_cores: int,
        numa_nodes: int,
        platform_summary: str,
        nic_driver: str | None,
        exclude_run_id: str | None = None,
        limit_runs: int = 10,
    ) -> dict[str, tuple[int, int, float]]:
        """Per-parameter confidence from prior similar runs.

        Returns {parameter_key: (tests, accepted, confidence_ratio)}.
        Aggregates evaluation_completed events across similar runs.
        """
        runs = self.find_similar_runs(
            service_name=service_name,
            cpu_logical_cores=cpu_logical_cores,
            numa_nodes=numa_nodes,
            platform_summary=platform_summary,
            nic_driver=nic_driver,
            exclude_run_id=exclude_run_id,
            limit=limit_runs,
        )
        if not runs:
            return {}
        run_ids = [run["run_id"] for run in runs]
        counts: dict[str, list[int]] = {}  # key -> [tests, accepted]
        for decision, key, _value in self._collect_parameter_decisions(run_ids):
            if not key:
                continue
            entry = counts.setdefault(key, [0, 0])
            entry[0] += 1
            if decision == "accept":
                entry[1] += 1
        return {
            key: (tests, accepted, accepted / tests if tests > 0 else 0.0)
            for key, (tests, accepted) in counts.items()
        }

    def _best_homepage_rps_by_run(self, run_ids: list[str]) -> dict[str, float]:
        if not run_ids:
            return {}
        placeholders = ", ".join("?" for _ in run_ids)
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT run_id, payload_json FROM events"  # noqa: S608
                f" WHERE run_id IN ({placeholders})"
                " AND event_type = 'best_config_updated'"
                " ORDER BY id ASC",
                tuple(run_ids),
            ).fetchall()
        best: dict[str, float] = {}
        for row in rows:
            run_id = str(row["run_id"])
            payload = json.loads(row["payload_json"])
            workloads = payload.get("workloads", [])
            if not isinstance(workloads, list):
                continue
            for workload in workloads:
                if not isinstance(workload, dict):
                    continue
                if workload.get("workload_name") != "homepage":
                    continue
                rps = workload.get("current_requests_per_second")
                if rps is None:
                    continue
                try:
                    value = float(rps)
                except (TypeError, ValueError):
                    continue
                if value > best.get(run_id, 0.0):
                    best[run_id] = value
        return best

    def _collect_parameter_decisions(
        self,
        run_ids: list[str],
    ) -> list[tuple[str, str, str]]:
        """Return [(decision, parameter_key, proposed_value)] from run events.

        Supports:
        - New payloads carrying parameter_key/proposed_value.
        - Batch payloads carrying applied_parameter_values.
        - Legacy payloads with neither key/value by joining same-iteration
          change_applied events.
        """
        if not run_ids:
            return []
        placeholders = ", ".join("?" for _ in run_ids)
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT run_id, iteration_number, component, event_type, payload_json FROM events"  # noqa: S608
                f" WHERE run_id IN ({placeholders})"
                " AND ("
                "      (component = 'benchmark_runner' AND event_type = 'evaluation_completed')"
                "   OR (component = 'tuning_executor' AND event_type = 'change_applied')"
                " )"
                " ORDER BY id ASC",
                tuple(run_ids),
            ).fetchall()
        applied_by_iteration: dict[tuple[str, int], dict[str, str]] = {}
        decisions: list[tuple[str, str, str]] = []
        for row in rows:
            run_id = str(row["run_id"])
            iteration_raw = row["iteration_number"]
            payload = json.loads(row["payload_json"])
            event_type = str(row["event_type"])
            if event_type == "change_applied":
                if iteration_raw is None:
                    continue
                parameter_key = str(payload.get("parameter_key", "")).strip()
                applied_value = payload.get("applied_value")
                if parameter_key == "" or applied_value is None:
                    continue
                iter_key = (run_id, int(iteration_raw))
                bucket = applied_by_iteration.setdefault(iter_key, {})
                bucket[parameter_key] = str(applied_value)
                continue
            if event_type != "evaluation_completed":
                continue
            decision = str(payload.get("decision", "")).strip()
            if decision == "":
                continue
            parameter_values: dict[str, str] = {}
            # Preferred: explicit batch mapping.
            applied_parameter_values = payload.get("applied_parameter_values")
            if isinstance(applied_parameter_values, list):
                for item in applied_parameter_values:
                    if not isinstance(item, dict):
                        continue
                    key = str(item.get("parameter_key", "")).strip()
                    value = item.get("proposed_value", item.get("applied_value"))
                    if key == "" or value is None:
                        continue
                    parameter_values[key] = str(value)
            # Explicit single-parameter fields.
            key_single = str(payload.get("parameter_key", "")).strip()
            value_single = payload.get("proposed_value")
            if key_single and value_single is not None:
                parameter_values[key_single] = str(value_single)
            # Legacy fallback: infer from change_applied within same iteration.
            if not parameter_values and iteration_raw is not None:
                iter_key = (run_id, int(iteration_raw))
                parameter_values.update(applied_by_iteration.get(iter_key, {}))
            for parameter_key, proposed_value in parameter_values.items():
                decisions.append((decision, parameter_key, proposed_value))
        return decisions

    def store_degradation_recipe(
        self,
        *,
        run_id: str,
        host_fingerprint: str,
        service_name: str,
        fingerprint_json: str,
        fix_sequence_json: str,
        best_score: float,
        blockers_fixed_json: str | None = None,
        workload_count: int,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO degradation_recipes (
                    run_id, host_fingerprint, service_name,
                    degradation_fingerprint_json, fix_sequence_json,
                    best_score, blockers_fixed_json, workload_count,
                    created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    degradation_fingerprint_json=excluded.degradation_fingerprint_json,
                    fix_sequence_json=excluded.fix_sequence_json,
                    best_score=excluded.best_score,
                    blockers_fixed_json=excluded.blockers_fixed_json,
                    created_at_utc=excluded.created_at_utc
                """,
                (
                    run_id,
                    host_fingerprint,
                    service_name,
                    fingerprint_json,
                    fix_sequence_json,
                    best_score,
                    blockers_fixed_json,
                    workload_count,
                    now,
                ),
            )
            connection.commit()

    def lookup_degradation_recipe(
        self,
        *,
        service_name: str,
        host_fingerprint: str,
        current_fingerprint: list[float],
        current_workload_names: list[str],
        similarity_threshold: float = 0.90,
        exclude_run_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Find best-matching recipe by cosine similarity.

        Returns None if no recipe exceeds the threshold.
        """
        params: list[object] = [service_name, host_fingerprint]
        query = (
            "SELECT run_id, degradation_fingerprint_json, "
            "fix_sequence_json, best_score, blockers_fixed_json "
            "FROM degradation_recipes "
            "WHERE service_name=? AND host_fingerprint=?"
        )
        if exclude_run_id is not None:
            query += " AND run_id<>?"
            params.append(exclude_run_id)
        query += " ORDER BY best_score DESC LIMIT 20"
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, tuple(params)).fetchall()
        best_match: dict[str, Any] | None = None
        best_similarity = 0.0
        for row in rows:
            stored = json.loads(row["degradation_fingerprint_json"])
            stored_names = stored.get("workload_names", [])
            stored_vector = stored.get("rps_vector", [])
            if sorted(stored_names) != sorted(current_workload_names):
                continue
            if len(stored_vector) != len(current_fingerprint):
                continue
            similarity = _cosine_similarity(
                current_fingerprint, stored_vector
            )
            if similarity >= similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = {
                    "run_id": row["run_id"],
                    "fix_sequence": json.loads(row["fix_sequence_json"]),
                    "best_score": row["best_score"],
                    "similarity": similarity,
                    "blockers_fixed": json.loads(
                        row["blockers_fixed_json"] or "[]"
                    ),
                }
        return best_match

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    completed_at_utc TEXT,
                    target_host TEXT,
                    service_name TEXT,
                    platform_summary TEXT,
                    cpu_logical_cores INTEGER,
                    cpu_core_band TEXT,
                    numa_nodes INTEGER,
                    total_memory_kib INTEGER,
                    memory_band TEXT,
                    nic_driver TEXT,
                    host_fingerprint TEXT,
                    stop_reason TEXT,
                    best_score REAL,
                    best_iteration INTEGER,
                    best_config_json TEXT,
                    final_retained_config_json TEXT
                )
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)").fetchall()}
            if "final_retained_config_json" not in columns:
                connection.execute("ALTER TABLE runs ADD COLUMN final_retained_config_json TEXT")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    iteration_number INTEGER,
                    phase TEXT,
                    component TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    service_name TEXT,
                    host_fingerprint TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id, id)")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runs_similarity
                ON runs(service_name, cpu_core_band, numa_nodes, platform_summary, nic_driver)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS degradation_recipes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    host_fingerprint TEXT NOT NULL,
                    service_name TEXT NOT NULL,
                    degradation_fingerprint_json TEXT NOT NULL,
                    fix_sequence_json TEXT NOT NULL,
                    best_score REAL NOT NULL,
                    blockers_fixed_json TEXT,
                    workload_count INTEGER NOT NULL,
                    created_at_utc TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_recipes_lookup
                ON degradation_recipes(service_name, host_fingerprint)
                """
            )
            connection.commit()
