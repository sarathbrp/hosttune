"""Offline DSPy prompt optimizer for the hypothesis predictor.

Reads past session artifacts (prompt + accept/reject outcome) and runs
BootstrapFewShot to bake the best few-shot demos into the prompt.

Usage:
    python scripts/compile_hypothesis_prompt.py \\
        --sessions-dir ~/.perf-agent/sessions \\
        --output compiled_hypothesis.json

The compiled file is loaded at startup by LangGraphHypothesisClient when
compiled_path is set. If the file is absent, the unoptimized predictor runs.

Requirements: 30+ accepted iterations for BootstrapFewShot to be effective.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_session_examples(sessions_dir: Path) -> list[dict[str, object]]:
    """Join prompt artifacts with evaluation outcomes per (session, iteration)."""
    examples: list[dict[str, object]] = []

    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue

        hypotheses_dir = session_dir / "hypotheses"
        if not hypotheses_dir.exists():
            continue

        # Build outcome map: iteration_number -> accepted (bool)
        outcomes: dict[int, bool] = {}
        for iteration_file in hypotheses_dir.glob("tune_iterations_*.jsonl"):
            for line in iteration_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                eval_result = record.get("record", {}).get("evaluation_result")
                if eval_result is None:
                    continue
                iteration = record.get("iteration_number", -1)
                decision = eval_result.get("decision", "")
                outcomes[iteration] = decision == "ACCEPT"

        if not outcomes:
            continue

        # Load prompt artifacts for the LLM path iterations
        for artifact_file in hypotheses_dir.glob("iter*_hybrid_hypothesizer.json"):
            try:
                data = json.loads(artifact_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            iteration = data.get("iteration")
            if iteration not in outcomes:
                continue
            prompt = data.get("prompt", "")
            response = data.get("response", "")
            if not prompt or not response:
                continue
            examples.append(
                {
                    "context": prompt,
                    "response": response,
                    "accepted": outcomes[iteration],
                    "session": session_dir.name,
                    "iteration": iteration,
                }
            )

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile DSPy hypothesis prompt optimizer")
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        default=Path.home() / ".perf-agent" / "sessions",
        help="Directory containing session subdirectories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("compiled_hypothesis.json"),
        help="Output path for compiled program",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=4,
        help="Max bootstrapped demos to include in optimized prompt",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="Path to .env file with GPT_OSS_* settings",
    )
    args = parser.parse_args()

    # Import project modules (must be on PYTHONPATH)
    try:
        import dspy
        from dspy.teleprompt import BootstrapFewShot

        from tune.application.dspy_hypothesis_module import HypothesisPredictor, configure_dspy
        from tune.infrastructure.model_config import ModelEndpointConfigLoader
    except ImportError as exc:
        print(f"Import error: {exc}", file=sys.stderr)
        print(
            "Run from project root with: "
            "PYTHONPATH=src python scripts/compile_hypothesis_prompt.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # Configure LM
    config = ModelEndpointConfigLoader().load(args.env)
    configure_dspy(config)

    # Load training examples
    print(f"Scanning sessions in: {args.sessions_dir}")
    raw_examples = _load_session_examples(args.sessions_dir)
    print(f"Found {len(raw_examples)} prompt/outcome pairs")

    accepted = [e for e in raw_examples if e["accepted"]]
    rejected = [e for e in raw_examples if not e["accepted"]]
    print(f"  Accepted: {len(accepted)}  Rejected: {len(rejected)}")

    if len(accepted) < 10:
        print(
            f"Warning: only {len(accepted)} accepted examples. "
            "BootstrapFewShot works best with 30+. Continuing anyway.",
            file=sys.stderr,
        )

    if not accepted:
        print("No accepted examples found — cannot compile. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Build DSPy trainset from accepted examples only.
    # The metric checks if the run was accepted by the benchmark.
    trainset = [
        dspy.Example(context=e["context"]).with_inputs("context")
        for e in accepted
    ]

    def metric(example: dspy.Example, prediction: dspy.Prediction, trace: object = None) -> bool:  # noqa: ARG001
        # All trainset examples are accepted runs — any valid prediction qualifies.
        return hasattr(prediction, "hypothesis") and prediction.hypothesis is not None

    # Compile
    print(f"Running BootstrapFewShot with max_demos={args.max_demos}...")
    optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=args.max_demos)
    student = HypothesisPredictor()
    compiled = optimizer.compile(student, trainset=trainset)

    # Save
    compiled.save(str(args.output))
    print(f"Compiled program saved to: {args.output}")
    print(
        f"Load it with: LangGraphHypothesisClient(..., compiled_path=Path('{args.output}'))"
    )


if __name__ == "__main__":
    main()
