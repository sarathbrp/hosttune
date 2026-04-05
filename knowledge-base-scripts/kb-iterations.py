#!/usr/bin/env python3
"""Per-iteration detail for a specific run (or latest).

Usage:
    python3 kb-iterations.py [--db PATH] [--run RUN_ID]
"""

import argparse
import json
import sqlite3


def main():
    parser = argparse.ArgumentParser(description="KB iteration details")
    parser.add_argument("--db", default="artifacts/knowledge_base.sqlite", help="Path to knowledge_base.sqlite")
    parser.add_argument("--run", default=None, help="Run ID (default: latest)")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    # Resolve run ID
    if args.run:
        run_id = args.run
    else:
        cur.execute("SELECT run_id FROM runs ORDER BY created_at_utc DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            print("No runs found")
            return
        run_id = row[0]

    # Run info
    cur.execute(
        "SELECT service_name, platform_summary, cpu_core_band, best_score, "
        "best_iteration, stop_reason, best_config_json FROM runs WHERE run_id = ?",
        (run_id,),
    )
    r = cur.fetchone()
    if not r:
        print(f"Run {run_id} not found")
        return

    svc, plat, cores, score, best_iter, stop, cfg_json = r
    print(f"Run: {run_id}")
    print(f"Service: {svc} | Platform: {plat} | Cores: {cores}")
    print(f"Best score: {(score or 0) * 100:.1f}% at iter {best_iter} | Stop: {stop}")
    if cfg_json:
        print(f"Best config: {', '.join(f'{k.split(chr(46))[-1]}={v}' for k, v in json.loads(cfg_json).items())}")
    print()

    # All events per iteration
    cur.execute(
        """SELECT iteration_number, event_type, payload_json
           FROM events WHERE run_id = ?
           AND event_type IN (
               'llm_proposal_selected', 'llm_skipped_autofix',
               'evaluation_completed', 'apply_failed',
               'rollback_completed', 'best_config_updated',
               'change_applied'
           )
           ORDER BY iteration_number, id""",
        (run_id,),
    )

    print(f"{'Iter':<5} {'Event':<25} {'Detail'}")
    print("-" * 90)

    for row in cur.fetchall():
        it, et, pj = row
        p = json.loads(pj) if pj else {}

        if et == "llm_proposal_selected":
            detail = f"{p.get('parameter_key', '?')}={p.get('proposed_value', '?')}"
        elif et == "llm_skipped_autofix":
            detail = f"AUTOFIX {p.get('parameter_key', '?')}={p.get('proposed_value', p.get('value', '?'))}"
        elif et == "evaluation_completed":
            dec = p.get("decision", "?")
            avg = p.get("average_relative_change")
            avg_str = f" avg={avg * 100:+.2f}%" if avg is not None else ""
            detail = f"{dec}{avg_str}"
        elif et == "apply_failed":
            detail = f"FAILED: {str(p.get('error', '?'))[:60]}"
        elif et == "rollback_completed":
            detail = f"ROLLBACK {p.get('parameter_key', p.get('reason', '?'))}"
        elif et == "best_config_updated":
            detail = f"NEW BEST score={p.get('score', '?')}"
        elif et == "change_applied":
            detail = f"{p.get('parameter', p.get('parameter_key', '?'))}: {p.get('previous', '?')} -> {p.get('applied', p.get('value', '?'))}"
        else:
            detail = str(p)[:60]

        print(f"{it or '-':<5} {et:<25} {detail}")

    # Token usage
    cur.execute(
        "SELECT payload_json FROM events WHERE run_id = ? AND event_type = 'run_completed'",
        (run_id,),
    )
    rc = cur.fetchone()
    if rc and rc[0]:
        rp = json.loads(rc[0])
        tokens = rp.get("token_usage", {})
        if tokens:
            print(f"\nTokens: input={tokens.get('input', 0):,} output={tokens.get('output', 0):,} total={tokens.get('total', 0):,}")

    conn.close()


if __name__ == "__main__":
    main()
