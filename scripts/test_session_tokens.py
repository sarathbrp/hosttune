#!/usr/bin/env python3
"""
Test: one-shot vs multi-turn session token usage.

Replays the actual hypothesis prompts from run fa34a6d3d7f2 in two modes:
  Mode A (baseline): each iteration = independent system+user message (current behavior)
  Mode B (session):  static context sent once as system message, each iteration
                     sends only the delta as a new user message in the same conversation

Compares input token counts to measure savings.

Usage:
  python3 test_session_tokens.py
  # Reads .env from /opt/hosttune/.env and artifacts from the last run.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path("/opt/hosttune/artifacts/fa34a6d3d7f2/hypotheses")
ENV_PATH = Path("/opt/hosttune/.env")

ITER_FILES = sorted(ARTIFACTS_DIR.glob("iter*_hybrid_hypothesizer.json"))

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
def load_env(path: Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


load_env(ENV_PATH)

BASE_URL = os.environ["GPT_OSS_BASE_URL"]
API_KEY = os.environ["GPT_OSS_API_KEY"]
MODEL = os.environ["GPT_OSS_MODEL"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_client():
    from openai import OpenAI
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def extract_prompt(artifact: dict) -> str:
    return artifact["prompt"]


SYSTEM_MSG = (
    "You are the single hybrid hypothesizer for HostTune. "
    "A deterministic triage layer has already run. "
    "Return exactly one JSON object with keys: parameter_key, proposed_value, "
    "tuning_layer, apply_mode, rationale, expected_benchmark_impact, rollback_plan."
)


# ---------------------------------------------------------------------------
# Split prompt into static vs dynamic sections
# ---------------------------------------------------------------------------
def split_prompt(prompt: str) -> tuple[str, str]:
    """
    Split the hypothesis prompt into static (unchanging) and dynamic (per-iteration) parts.

    Static sections (same every iteration):
      - Context policy
      - Host facts
      - Host profile
      - Available tunable surfaces (capabilities)
      - Service contract
      - Snapshot digest + raw config
      - Selected service YAML reference snippet
      - Baseline workload results
      - Output rules

    Dynamic sections (change per iteration):
      - Phase, iteration number
      - Rule-based triage result + triggered rules
      - Prior similar run memory (could be static but small)
      - Last benchmark runtime telemetry
      - Current tune state (active changes, best config)
      - History lines
      - Selectable candidates
      - Blocked prior parameter/value pairs
    """
    lines = prompt.split("\n")
    static_lines = []
    dynamic_lines = []

    # Markers for static sections
    static_section_starts = {
        "Context policy:",
        "Host facts:",
        "Host profile:",
        "Available tunable surfaces:",
        "Service contract:",
        "Snapshot digest:",
        "Limit baselines:",
        "Output rules:",
        "Selected service YAML reference snippet:",
        "Baseline workload results:",
    }

    # Markers for dynamic sections
    dynamic_section_starts = {
        "Current phase:",
        "Phase objective:",
        "Iteration:",
        "Rule-based triage result:",
        "Triggered rules:",
        "Last benchmark runtime telemetry:",
        "Current tune state:",
        "Selectable candidates:",
        "Deferred candidates",
        "Blocked prior parameter/value pairs:",
        "Prior similar run memory:",
    }

    # Also treat the raw config blocks as static
    static_raw_markers = {
        "Selected runtime config snippet:",
        "# configuration file",
        "include /usr/share/nginx",
    }

    current_target = dynamic_lines  # default to dynamic
    in_raw_config = False

    for line in lines:
        stripped = line.strip()

        # Check if this line starts a new section
        matched_static = any(stripped.startswith(m) for m in static_section_starts)
        matched_dynamic = any(stripped.startswith(m) for m in dynamic_section_starts)
        matched_raw_static = any(stripped.startswith(m) for m in static_raw_markers)

        if matched_static or matched_raw_static:
            current_target = static_lines
            in_raw_config = matched_raw_static
        elif matched_dynamic:
            current_target = dynamic_lines
            in_raw_config = False
        elif stripped.startswith("You are the single hybrid"):
            # System preamble — goes to dynamic (has phase/iteration)
            current_target = dynamic_lines

        current_target.append(line)

    return "\n".join(static_lines), "\n".join(dynamic_lines)


# ---------------------------------------------------------------------------
# Mode A: One-shot (current behavior)
# ---------------------------------------------------------------------------
def run_oneshot(client, prompts: list[str]) -> list[dict]:
    results = []
    for i, prompt in enumerate(prompts):
        iter_num = i + 2  # iter002 is first
        print(f"  [one-shot] iter {iter_num:03d} ... ", end="", flush=True)
        t0 = time.time()
        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
        )
        elapsed = time.time() - t0
        usage = completion.usage
        result = {
            "iteration": iter_num,
            "mode": "one-shot",
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "prompt_chars": len(prompt),
            "elapsed_s": round(elapsed, 2),
        }
        results.append(result)
        print(
            f"in={usage.prompt_tokens} out={usage.completion_tokens} "
            f"total={usage.total_tokens} ({elapsed:.1f}s)"
        )
    return results


# ---------------------------------------------------------------------------
# Mode B: Multi-turn session
# ---------------------------------------------------------------------------
def run_session(client, prompts: list[str]) -> list[dict]:
    results = []

    # Extract static context from first prompt
    static_context, _ = split_prompt(prompts[0])

    # Build system message with static context baked in
    session_system = (
        SYSTEM_MSG + "\n\n"
        "=== STATIC HOST & SERVICE CONTEXT (does not change between iterations) ===\n"
        + static_context
    )

    # Conversation history for multi-turn
    messages = [{"role": "system", "content": session_system}]

    for i, prompt in enumerate(prompts):
        iter_num = i + 2
        _, dynamic = split_prompt(prompt)

        print(f"  [session]  iter {iter_num:03d} ... ", end="", flush=True)

        # Add user message with only dynamic content
        user_msg = (
            f"=== ITERATION {iter_num} DYNAMIC CONTEXT ===\n"
            + dynamic
        )
        messages.append({"role": "user", "content": user_msg})

        t0 = time.time()
        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            messages=messages,
        )
        elapsed = time.time() - t0
        usage = completion.usage

        # Add assistant response to conversation
        assistant_content = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_content})

        result = {
            "iteration": iter_num,
            "mode": "session",
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "prompt_chars": len(user_msg),
            "system_chars": len(session_system) if i == 0 else 0,
            "conversation_messages": len(messages),
            "elapsed_s": round(elapsed, 2),
        }
        results.append(result)
        print(
            f"in={usage.prompt_tokens} out={usage.completion_tokens} "
            f"total={usage.total_tokens} msgs={len(messages)} ({elapsed:.1f}s)"
        )
    return results


# ---------------------------------------------------------------------------
# Mode C: Compressed one-shot (static stripped, no session)
# ---------------------------------------------------------------------------
def run_compressed_oneshot(client, prompts: list[str]) -> list[dict]:
    """One-shot but with static context in system msg and only dynamic in user msg."""
    results = []
    static_context, _ = split_prompt(prompts[0])
    compressed_system = (
        SYSTEM_MSG + "\n\n"
        "=== STATIC HOST & SERVICE CONTEXT ===\n"
        + static_context
    )

    for i, prompt in enumerate(prompts):
        iter_num = i + 2
        _, dynamic = split_prompt(prompt)
        user_msg = f"=== ITERATION {iter_num} ===\n" + dynamic

        print(f"  [compressed] iter {iter_num:03d} ... ", end="", flush=True)
        t0 = time.time()
        completion = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": compressed_system},
                {"role": "user", "content": user_msg},
            ],
        )
        elapsed = time.time() - t0
        usage = completion.usage
        result = {
            "iteration": iter_num,
            "mode": "compressed-oneshot",
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "system_chars": len(compressed_system),
            "user_chars": len(user_msg),
            "elapsed_s": round(elapsed, 2),
        }
        results.append(result)
        print(
            f"in={usage.prompt_tokens} out={usage.completion_tokens} "
            f"total={usage.total_tokens} ({elapsed:.1f}s)"
        )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not ITER_FILES:
        print(f"No iteration files found in {ARTIFACTS_DIR}")
        sys.exit(1)

    print(f"Found {len(ITER_FILES)} iteration artifacts")
    print(f"Model: {MODEL}")
    print(f"Endpoint: {BASE_URL}")
    print()

    # Load all prompts
    prompts = []
    for path in ITER_FILES:
        data = json.loads(path.read_text())
        prompts.append(extract_prompt(data))

    # Show split analysis first (no LLM calls)
    print("=" * 70)
    print("PROMPT SPLIT ANALYSIS (static vs dynamic)")
    print("=" * 70)
    for i, prompt in enumerate(prompts):
        static, dynamic = split_prompt(prompt)
        iter_num = i + 2
        print(
            f"  iter {iter_num:03d}: total={len(prompt):,} chars | "
            f"static={len(static):,} ({100*len(static)/len(prompt):.0f}%) | "
            f"dynamic={len(dynamic):,} ({100*len(dynamic)/len(prompt):.0f}%)"
        )
    print()

    client = build_client()

    # --- Mode A: one-shot (baseline) ---
    print("=" * 70)
    print("MODE A: ONE-SHOT (current behavior)")
    print("=" * 70)
    oneshot_results = run_oneshot(client, prompts)
    print()

    # --- Mode C: compressed one-shot ---
    print("=" * 70)
    print("MODE C: COMPRESSED ONE-SHOT (static in system, dynamic in user)")
    print("=" * 70)
    compressed_results = run_compressed_oneshot(client, prompts)
    print()

    # --- Mode B: multi-turn session ---
    print("=" * 70)
    print("MODE B: MULTI-TURN SESSION (static once, deltas per turn)")
    print("=" * 70)
    session_results = run_session(client, prompts)
    print()

    # --- Summary ---
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_oneshot_in = sum(r["input_tokens"] for r in oneshot_results)
    total_oneshot_out = sum(r["output_tokens"] for r in oneshot_results)
    total_compressed_in = sum(r["input_tokens"] for r in compressed_results)
    total_compressed_out = sum(r["output_tokens"] for r in compressed_results)
    total_session_in = sum(r["input_tokens"] for r in session_results)
    total_session_out = sum(r["output_tokens"] for r in session_results)

    print(f"  One-shot:        input={total_oneshot_in:,} output={total_oneshot_out:,} total={total_oneshot_in+total_oneshot_out:,}")
    print(f"  Compressed:      input={total_compressed_in:,} output={total_compressed_out:,} total={total_compressed_in+total_compressed_out:,}")
    print(f"  Session:         input={total_session_in:,} output={total_session_out:,} total={total_session_in+total_session_out:,}")
    print()

    if total_oneshot_in > 0:
        saving_c = (1 - total_compressed_in / total_oneshot_in) * 100
        saving_b = (1 - total_session_in / total_oneshot_in) * 100
        print(f"  Compressed vs one-shot: {saving_c:+.1f}% input tokens")
        print(f"  Session vs one-shot:    {saving_b:+.1f}% input tokens")
    print()

    # Per-iteration comparison
    print(f"  {'Iter':<6} {'One-shot':>10} {'Compressed':>12} {'Session':>10} {'C savings':>10} {'S savings':>10}")
    print(f"  {'----':<6} {'--------':>10} {'----------':>12} {'-------':>10} {'---------':>10} {'---------':>10}")
    for a, c, b in zip(oneshot_results, compressed_results, session_results):
        c_save = (1 - c["input_tokens"] / a["input_tokens"]) * 100 if a["input_tokens"] else 0
        b_save = (1 - b["input_tokens"] / a["input_tokens"]) * 100 if a["input_tokens"] else 0
        print(
            f"  {a['iteration']:<6} {a['input_tokens']:>10,} {c['input_tokens']:>12,} "
            f"{b['input_tokens']:>10,} {c_save:>+9.1f}% {b_save:>+9.1f}%"
        )

    # Save raw results
    out_path = ARTIFACTS_DIR / "token_test_results.json"
    out_path.write_text(json.dumps({
        "oneshot": oneshot_results,
        "compressed": compressed_results,
        "session": session_results,
    }, indent=2))
    print(f"\n  Raw results saved to: {out_path}")


if __name__ == "__main__":
    main()
