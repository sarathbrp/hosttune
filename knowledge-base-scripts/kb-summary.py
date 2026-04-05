#!/usr/bin/env python3
"""Run overview — all runs with scores, best configs, and stop reasons.

Usage:
    python3 kb-summary.py [--db PATH]
"""

import argparse
import json
import sqlite3


def main():
    parser = argparse.ArgumentParser(description="KB run summary")
    parser.add_argument(
        "--db", default="artifacts/knowledge_base.sqlite", help="Path to knowledge_base.sqlite"
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    cur.execute("""
        SELECT run_id, created_at_utc, service_name, platform_summary,
               cpu_core_band, numa_nodes, nic_driver, best_score,
               best_iteration, stop_reason, best_config_json
        FROM runs ORDER BY created_at_utc DESC
    """)
    rows = cur.fetchall()

    print(f"Total runs: {len(rows)}")
    print()
    print(
        f"{'Run ID':<16} {'Score':>7} {'Iter':>5} {'Cores':<8} {'NUMA':>4} {'Stop':<16} {'Config'}"
    )
    print("-" * 110)

    for r in rows:
        run_id, created, svc, plat, cores, numa, nic, score, best_iter, stop, cfg_json = r
        cfg = json.loads(cfg_json) if cfg_json else {}
        cfg_short = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in cfg.items())
        if len(cfg_short) > 55:
            cfg_short = cfg_short[:52] + "..."
        print(
            f"{run_id:<16} {(score or 0) * 100:>6.1f}% {best_iter or '-':>5} "
            f"{cores or '?':<8} {numa or '?':>4} {stop or '?':<16} {cfg_short}"
        )

    # Summary stats
    print()
    scores = [r[7] for r in rows if r[7] is not None]
    if scores:
        print(
            f"Best score: {max(scores) * 100:.1f}%  |  Avg: {sum(scores) / len(scores) * 100:.1f}%  |  Runs: {len(rows)}"
        )

    # Event counts
    cur.execute("SELECT count(*) FROM events")
    total_events = cur.fetchone()[0]
    cur.execute("SELECT count(DISTINCT run_id) FROM events")
    runs_with_events = cur.fetchone()[0]
    print(f"Total events: {total_events} across {runs_with_events} runs")

    conn.close()


if __name__ == "__main__":
    main()
