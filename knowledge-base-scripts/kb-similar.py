#!/usr/bin/env python3
"""Find similar runs by hardware profile.

Matches: same service_name, similar CPU core band, same NUMA nodes,
same platform_summary, optionally same NIC driver.

Usage:
    python3 kb-similar.py [--db PATH] [--service nginx] [--nic-driver DRIVER]
"""

import argparse
import json
import sqlite3


def main():
    parser = argparse.ArgumentParser(description="Find similar KB runs")
    parser.add_argument(
        "--db", default="artifacts/knowledge_base.sqlite", help="Path to knowledge_base.sqlite"
    )
    parser.add_argument("--service", default="nginx", help="Service name filter")
    parser.add_argument("--nic-driver", default=None, help="Optional NIC driver filter")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    # Get reference hardware from latest run
    cur.execute(
        "SELECT cpu_core_band, numa_nodes, platform_summary, nic_driver "
        "FROM runs ORDER BY created_at_utc DESC LIMIT 1"
    )
    ref = cur.fetchone()
    if not ref:
        print("No runs found")
        return

    ref_cores, ref_numa, ref_plat, ref_nic = ref
    print(f"Reference hardware: {ref_plat} | cores={ref_cores} | NUMA={ref_numa} | NIC={ref_nic}")
    print()

    # Build query
    query = """
        SELECT run_id, created_at_utc, cpu_core_band, numa_nodes,
               nic_driver, best_score, best_iteration, stop_reason,
               best_config_json
        FROM runs
        WHERE service_name = ?
          AND platform_summary = ?
          AND cpu_core_band = ?
          AND numa_nodes = ?
    """
    params = [args.service, ref_plat, ref_cores, ref_numa]

    if args.nic_driver:
        query += " AND nic_driver = ?"
        params.append(args.nic_driver)

    query += " ORDER BY best_score DESC"
    cur.execute(query, params)
    rows = cur.fetchall()

    print(
        f"Similar runs: {len(rows)} ({args.service} / {ref_plat} / {ref_cores} / NUMA={ref_numa})"
    )
    print()
    print(
        f"{'#':<3} {'Run ID':<16} {'Score':>7} {'Iter':>5} {'NIC':<12} {'Stop':<16} {'Best Config'}"
    )
    print("-" * 110)

    for i, r in enumerate(rows, 1):
        run_id, created, cores, numa, nic, score, best_iter, stop, cfg_json = r
        cfg = json.loads(cfg_json) if cfg_json else {}
        cfg_short = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in cfg.items())
        if len(cfg_short) > 45:
            cfg_short = cfg_short[:42] + "..."
        print(
            f"{i:<3} {run_id:<16} {(score or 0) * 100:>6.1f}% {best_iter or '-':>5} "
            f"{(nic or '?')[:11]:<12} {stop or '?':<16} {cfg_short}"
        )

    # Cross-run parameter frequency
    if rows:
        print()
        print("=== Most effective parameters (across all similar runs) ===")
        param_wins = {}
        for r in rows:
            cfg = json.loads(r[8]) if r[8] else {}
            score = r[5] or 0
            for k, v in cfg.items():
                key = f"{k}={v}"
                if key not in param_wins:
                    param_wins[key] = {"count": 0, "total_score": 0}
                param_wins[key]["count"] += 1
                param_wins[key]["total_score"] += score

        sorted_params = sorted(param_wins.items(), key=lambda x: x[1]["total_score"], reverse=True)
        print(f"{'Parameter=Value':<55} {'Runs':>5} {'Avg Score':>10}")
        print("-" * 75)
        for param, stats in sorted_params[:10]:
            avg = stats["total_score"] / stats["count"] * 100
            print(f"{param:<55} {stats['count']:>5} {avg:>9.1f}%")

    conn.close()


if __name__ == "__main__":
    main()
