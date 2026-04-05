#!/usr/bin/env python3
"""Parameter scoreboard — which params help vs hurt across all runs.

Usage:
    python3 kb-scoreboard.py [--db PATH] [--run RUN_ID]
"""

import argparse
import json
import sqlite3


def main():
    parser = argparse.ArgumentParser(description="KB parameter scoreboard")
    parser.add_argument("--db", default="artifacts/knowledge_base.sqlite", help="Path to knowledge_base.sqlite")
    parser.add_argument("--run", default=None, help="Run ID (default: all runs)")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    # Collect all evaluation events
    if args.run:
        cur.execute(
            """SELECT e.iteration_number, e.run_id, e.payload_json,
                      p.payload_json as proposal_json
               FROM events e
               LEFT JOIN events p ON p.run_id = e.run_id
                   AND p.iteration_number = e.iteration_number
                   AND p.event_type IN ('llm_proposal_selected', 'llm_skipped_autofix')
               WHERE e.event_type = 'evaluation_completed'
                 AND e.run_id = ?
               ORDER BY e.run_id, e.iteration_number""",
            (args.run,),
        )
    else:
        cur.execute(
            """SELECT e.iteration_number, e.run_id, e.payload_json,
                      p.payload_json as proposal_json
               FROM events e
               LEFT JOIN events p ON p.run_id = e.run_id
                   AND p.iteration_number = e.iteration_number
                   AND p.event_type IN ('llm_proposal_selected', 'llm_skipped_autofix')
               WHERE e.event_type = 'evaluation_completed'
               ORDER BY e.run_id, e.iteration_number"""
        )

    # Aggregate by parameter
    param_stats: dict[str, dict] = {}
    for row in cur.fetchall():
        it, run_id, eval_json, prop_json = row
        ev = json.loads(eval_json) if eval_json else {}
        pr = json.loads(prop_json) if prop_json else {}

        param = pr.get("parameter_key", "unknown")
        value = pr.get("proposed_value", "?")
        decision = ev.get("decision", "?")
        avg_change = ev.get("average_relative_change")

        key = param
        if key not in param_stats:
            param_stats[key] = {
                "accepted": 0,
                "promising": 0,
                "inconclusive": 0,
                "reject": 0,
                "total": 0,
                "total_change": 0,
                "values_tried": set(),
                "runs": set(),
            }

        stats = param_stats[key]
        stats["total"] += 1
        stats["values_tried"].add(str(value))
        stats["runs"].add(run_id)

        if decision == "accept":
            stats["accepted"] += 1
        elif decision == "promising":
            stats["promising"] += 1
        elif decision == "inconclusive":
            stats["inconclusive"] += 1
        elif decision == "reject":
            stats["reject"] += 1

        if avg_change is not None:
            stats["total_change"] += avg_change

    # Display
    scope = f"run {args.run}" if args.run else "all runs"
    print(f"Parameter scoreboard ({scope})")
    print()
    print(
        f"{'Parameter':<45} {'Tests':>5} {'Acc':>4} {'Pro':>4} {'Inc':>4} "
        f"{'Rej':>4} {'Runs':>4} {'Values Tried'}"
    )
    print("-" * 110)

    # Sort by accepted count desc, then total
    sorted_params = sorted(
        param_stats.items(),
        key=lambda x: (x[1]["accepted"], -x[1]["reject"]),
        reverse=True,
    )

    for param, s in sorted_params:
        vals = ", ".join(sorted(s["values_tried"])[:3])
        if len(s["values_tried"]) > 3:
            vals += f" +{len(s['values_tried']) - 3}"
        print(
            f"{param:<45} {s['total']:>5} {s['accepted']:>4} {s['promising']:>4} "
            f"{s['inconclusive']:>4} {s['reject']:>4} {len(s['runs']):>4} {vals}"
        )

    # Summary
    print()
    total_tests = sum(s["total"] for s in param_stats.values())
    total_accept = sum(s["accepted"] for s in param_stats.values())
    total_reject = sum(s["reject"] for s in param_stats.values())
    print(
        f"Total: {total_tests} tests | {total_accept} accepted | "
        f"{total_reject} rejected | {len(param_stats)} unique params"
    )

    conn.close()


if __name__ == "__main__":
    main()
