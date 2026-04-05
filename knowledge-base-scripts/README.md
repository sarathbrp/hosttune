# Knowledge Base Scripts

Query and analyze the hosttune `knowledge_base.sqlite` database.

## Setup

Copy scripts to System 2 or run locally with the sqlite file:

```bash
# On System 2
cd /opt/hosttune
python3 /path/to/kb-summary.py

# Or locally (copy the DB first)
scp root@system2:/opt/hosttune/artifacts/knowledge_base.sqlite .
python3 kb-summary.py --db knowledge_base.sqlite
```

No dependencies — uses Python 3 stdlib only (`sqlite3`, `json`).

## Scripts

### kb-summary.py — Run Overview

Shows all runs ranked by score with best configs.

```bash
python3 kb-summary.py --db artifacts/knowledge_base.sqlite
```

```
Total runs: 4
Run ID          Score  Iter Cores    NUMA Stop            Config
──────────────────────────────────────────────────────────────────
a3eee909a3ec    98.1%     1 64-127      2 no_candidates   access_log=off
8425479b22e5    87.1%     2 64-127      2 no_candidates   access_log=off, somaxconn=8192
d9c11da2d934    16.3%     3 64-127      2 no_candidates   worker_connections=8192, worker_rlimit...
b4f6f1ab1da3     0.0%     - 64-127      2 no_candidates   (none)
```

### kb-iterations.py — Iteration Detail

Shows proposals, evaluations, rollbacks, and apply failures per iteration.

```bash
# Latest run
python3 kb-iterations.py --db artifacts/knowledge_base.sqlite

# Specific run
python3 kb-iterations.py --db artifacts/knowledge_base.sqlite --run a3eee909a3ec
```

```
Run: a3eee909a3ec
Best score: 98.1% at iter 1 | Stop: no_candidates

Iter  Event                     Detail
──────────────────────────────────────────────────────────────
1     llm_skipped_autofix       AUTOFIX access_log=off
1     evaluation_completed      accept
2     llm_proposal_selected     somaxconn=8192
2     evaluation_completed      inconclusive
...
```

### kb-similar.py — Find Similar Runs

Matches runs with same hardware profile (service, CPU band, NUMA, platform).
Shows which parameters are most effective across similar hardware.

```bash
# Default: match latest run's hardware
python3 kb-similar.py --db artifacts/knowledge_base.sqlite

# Filter by service
python3 kb-similar.py --db artifacts/knowledge_base.sqlite --service nginx

# Also match NIC driver
python3 kb-similar.py --db artifacts/knowledge_base.sqlite --nic-driver bnxt_en
```

```
Reference hardware: bare_metal_linux | cores=64-127 | NUMA=2 | NIC=bnxt_en

Similar runs: 4 (nginx / bare_metal_linux / 64-127 / NUMA=2)

=== Most effective parameters (across all similar runs) ===
Parameter=Value                                        Runs  Avg Score
───────────────────────────────────────────────────────────────────
access_log=off                                            3      72.4%
somaxconn=8192                                            1      87.1%
worker_connections=8192                                   1      16.3%
```

### kb-scoreboard.py — Parameter Scoreboard

Win/loss record for every parameter tested across all runs (or a specific run).

```bash
# All runs
python3 kb-scoreboard.py --db artifacts/knowledge_base.sqlite

# Specific run
python3 kb-scoreboard.py --db artifacts/knowledge_base.sqlite --run d9c11da2d934
```

```
Parameter                                     Tests  Acc  Pro  Inc  Rej Runs Values Tried
──────────────────────────────────────────────────────────────────────────────────────
service.directive.access_log                      3    2    0    1    0    3 off
sysctl.net.core.somaxconn                         5    2    0    3    0    3 8192, 16384, 32768
sysctl.net.core.netdev_max_backlog                5    1    0    4    0    3 5000, 10000, 20000
service.directive.worker_connections              3    1    0    1    1    3 8192, 32768, 65535
...
Total: 35 tests | 8 accepted | 4 rejected | 12 unique params
```

## Database Schema

```
runs    — One row per hosttune session (run_id, hardware, best_score, best_config)
events  — All lifecycle events (proposals, evaluations, rollbacks, benchmarks)
```

Key event types:
- `llm_proposal_selected` — LLM proposed a parameter change
- `llm_skipped_autofix` — Applied from prior knowledge without LLM
- `change_applied` — Parameter was written to the system
- `evaluation_completed` — Benchmark result evaluated (accept/reject/inconclusive)
- `rollback_completed` — Bad change reverted
- `best_config_updated` — New best configuration found
