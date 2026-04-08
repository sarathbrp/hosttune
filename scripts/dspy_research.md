# DSPy Research Report — Integration into perf-agent

**Date:** 2026-04-08  
**Scope:** DSPy 2.x for structured LLM output in a LangGraph + mypy-strict + OpenAI-compatible agent

---

## 1. What Is DSPy

DSPy (Declarative Self-improving Python) is a framework for *programming* LMs rather than prompting them.
The key shift: instead of crafting brittle system+user prompt strings, you declare *Signatures* that
describe inputs and outputs, then let DSPy (and optionally its optimizers) figure out the best prompt
to satisfy them. DSPy sits on top of LiteLLM, which handles the actual HTTP calls.

**Core abstractions:**

| Abstraction | What it is |
|---|---|
| `dspy.Signature` | Declares input/output fields with types. The "schema" for one LM call. |
| `dspy.Predict` | Bare module: calls LM once for the signature, returns a `Prediction`. |
| `dspy.ChainOfThought` | Adds a hidden `reasoning` field before the declared outputs; better for multi-step. |
| `dspy.Module` | Base class (like `nn.Module`). Compose Predict/ChainOfThought inside `forward()`. |
| Teleprompter / Optimizer | Compiles a Module: rewrites its prompts and/or injects few-shot demos based on a metric. |

---

## 2. Signatures and Typed Output

### Inline form (sufficient for most cases)

```python
"context: str, candidates: list[str] -> parameter_key: str, rationale: str"
```

### Class-based form (preferred for perf-agent)

```python
import dspy
import pydantic

class TuningHypothesisOutput(pydantic.BaseModel):
    parameter_key: str
    proposed_value: str
    tuning_layer: str
    apply_mode: str
    rationale: str
    expected_benchmark_impact: str
    rollback_plan: str

class ProposeHypothesis(dspy.Signature):
    """Given system context and candidate parameters, propose the single best tuning hypothesis."""

    system_context: str = dspy.InputField(desc="snapshot of current system state, benchmarks, history")
    candidates_json: str = dspy.InputField(desc="JSON array of candidate parameters with constraints")

    hypothesis: TuningHypothesisOutput = dspy.OutputField(
        desc="One tuning hypothesis with all required fields"
    )
```

DSPy natively handles Pydantic `BaseModel` as an output type. It serializes the JSON schema into
the prompt and parses/validates the LM response. This replaces the current hand-rolled
`json.loads` + `_require_string` validation in `hypothesis_generator.py`.

**Note:** `dspy.InputField(prefix=..., format=..., parser=...)` kwargs are *deprecated in DSPy 2.x*;
use only `desc=` for field hints.

---

## 3. DSPy vs. Current Raw JSON Prompting

| Dimension | Current (`hypothesis_generator.py`) | DSPy |
|---|---|---|
| Schema enforcement | Manual `_require_string`, per-field `isinstance` checks | Pydantic model, DSPy validates automatically |
| Retry on parse fail | Raises `ValueError`; caller retries iteration | DSPy has built-in retry + assertion support |
| Prompt construction | `format_hybrid_hypothesis_prompt()` string | Signature docstring + field `desc` |
| Field contract checks | `_validate_contract_fields` (tuning_layer, apply_mode) | Can stay as post-prediction validators or use `dspy.Assert` |
| Optimization | None (static prompt) | Can run BootstrapFewShot/MIPRO over iteration history |

---

## 4. OpenAI-Compatible Endpoint Configuration

DSPy uses LiteLLM under the hood. Connecting to any OpenAI-compatible endpoint:

```python
import dspy

lm = dspy.LM(
    "openai/your-model-name",       # "openai/" prefix = OpenAI-compatible
    api_base="https://your-endpoint/v1",
    api_key="YOUR_KEY",
)
dspy.configure(lm=lm)
```

This is a *global* configure. Per-call override is also possible:

```python
with dspy.context(lm=other_lm):
    result = predictor(...)
```

For the perf-agent, this maps directly onto the existing `LanggraphHypothesisClient`'s
`openai.OpenAI(base_url=..., api_key=...)` pattern. The DSPy LM call replaces
`client.chat.completions.create(...)`.

---

## 5. Concrete Migration: Raw Prompt → DSPy Signature → TypedPredictor

### Step 1 — Define the Pydantic output model (already exists as `TuningHypothesis`)

```python
# src/tune/domain/hypothesis_models.py  (already has TuningHypothesis dataclass)
# Add a parallel Pydantic model for DSPy:
import pydantic

class HypothesisProposal(pydantic.BaseModel):
    parameter_key: str
    proposed_value: str
    tuning_layer: str
    apply_mode: str
    rationale: str
    expected_benchmark_impact: str
    rollback_plan: str
```

### Step 2 — Define the Signature

```python
# src/tune/application/dspy_hypothesis_module.py
import dspy
from tune.domain.hypothesis_models import HypothesisProposal

class ProposeHypothesis(dspy.Signature):
    """Propose the single best Linux/system tuning parameter to improve benchmark throughput."""

    system_context: str = dspy.InputField(
        desc="Current system snapshot: CPU, memory, network, benchmark history, prior hypotheses"
    )
    candidate_params: str = dspy.InputField(
        desc="JSON array of allowed candidate parameters with constraints"
    )
    hypothesis: HypothesisProposal = dspy.OutputField(
        desc="One hypothesis. parameter_key and apply_mode must match the candidate exactly."
    )
```

### Step 3 — Build the Module

```python
class HypothesisPredictor(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        # ChainOfThought adds hidden reasoning field; good for multi-factor decisions
        self.predict = dspy.ChainOfThought(ProposeHypothesis)

    def forward(self, system_context: str, candidate_params: str) -> dspy.Prediction:
        return self.predict(
            system_context=system_context,
            candidate_params=candidate_params,
        )
```

### Step 4 — Wire into LangGraph node

```python
# LangGraph node — DSPy module is just a callable
def hypothesis_node(state: TuneState) -> dict[str, object]:
    predictor = HypothesisPredictor()  # or load compiled version
    result = predictor(
        system_context=build_context_string(state),
        candidate_params=json.dumps(state.candidates),
    )
    proposal: HypothesisProposal = result.hypothesis   # typed Pydantic object
    # post-validate contract fields (tuning_layer, apply_mode) — same logic as today
    _validate_contract_fields(proposal, state.candidates)
    return {"current_hypothesis": proposal}
```

The DSPy module is a plain Python callable — it drops straight into a LangGraph `StateGraph` node
with no special integration layer needed. The `langgraph_hypothesis_client.py` can be rewritten
as a thin wrapper or replaced entirely.

---

## 6. Optimizers (Teleprompters) — What Compiling Means

"Compiling" a DSPy program means running an optimizer that:
1. Iterates over a `trainset` of `dspy.Example` objects.
2. Executes the program on each example, collecting execution *traces* (the full chain of LM calls and outputs).
3. Scores each trace with a user-supplied `metric` function.
4. Selects the best traces as *few-shot demos* and/or rewrites the instruction in each Signature.
5. Returns a new `student` module whose Signatures now contain baked-in demos/instructions.

The compiled program can be saved/loaded as JSON: `program.save("compiled.json")` / `program.load("compiled.json")`.

### BootstrapFewShot (simplest, start here)

```python
from dspy.teleprompt import BootstrapFewShot

def hypothesis_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """Return True if the prediction passed benchmark acceptance."""
    return example.accepted  # from iteration history

trainset = [
    dspy.Example(
        system_context=ctx_str,
        candidate_params=candidates_str,
        accepted=True,          # label: was this hypothesis accepted?
    ).with_inputs("system_context", "candidate_params")
    for ctx_str, candidates_str in accepted_iterations
]

optimizer = BootstrapFewShot(metric=hypothesis_metric, max_bootstrapped_demos=4)
compiled_predictor = optimizer.compile(HypothesisPredictor(), trainset=trainset)
compiled_predictor.save("compiled_hypothesis.json")
```

**Training data needed:** 30+ accepted iteration records (context, proposal, accepted=True/False).
The perf-agent's `tune_recorder.py` and knowledge base already collect exactly this.

### MIPROv2 (instruction + few-shot optimization, more powerful)

Requires `pip install dspy[optuna]`. Uses Bayesian search (Optuna) to jointly optimize instruction
wording and demo selection. Needs ~100-300 labeled examples for reliable results.

```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(metric=hypothesis_metric, auto="light")  # light/medium/heavy
compiled = optimizer.compile(HypothesisPredictor(), trainset=trainset, valset=valset)
```

**Recommended split:** 20% train / 80% validation (DSPy's recommendation — prompt optimizers overfit
small trainsets).

---

## 7. LangGraph + DSPy Integration Pattern

DSPy modules are stateless callables. The integration is mechanical:

```python
# Pattern A: Module instantiated once, reused across nodes
predictor = HypothesisPredictor()
predictor.load("compiled_hypothesis.json")  # load compiled demos

def hypothesis_node(state: TuneState) -> dict[str, object]:
    result = predictor(system_context=..., candidate_params=...)
    return {"hypothesis": result.hypothesis}

graph = StateGraph(TuneState)
graph.add_node("hypothesis", hypothesis_node)
```

```python
# Pattern B: Different LM per node (e.g. cheaper model for triage, smarter for hypothesis)
def hypothesis_node(state: TuneState) -> dict[str, object]:
    with dspy.context(lm=strong_lm):
        result = predictor(...)
    return {"hypothesis": result.hypothesis}
```

No DSPy-specific LangGraph adapter exists or is needed. DSPy modules work as plain functions.

---

## 8. mypy Strict Typing — Gotchas

DSPy's internal classes (`Prediction`, `Module`, `Signature`) are **not fully typed** for mypy strict mode. Key issues:

1. **`dspy.Prediction` is a dynamic namespace object** — `result.hypothesis` returns `Any` at the mypy level. Fix with an explicit cast:
   ```python
   from typing import cast
   proposal = cast(HypothesisProposal, result.hypothesis)
   ```

2. **`dspy.Module.forward()` return type** — declare it explicitly:
   ```python
   def forward(self, system_context: str, candidate_params: str) -> dspy.Prediction:
   ```

3. **`dspy.InputField` / `dspy.OutputField`** return `pydantic.fields.FieldInfo` at runtime, which mypy sees as the annotated type in the Signature class body. This is fine for strict mode *within* the Signature class, but accessing `ProposeHypothesis.hypothesis` outside the class gives `FieldInfo`, not `HypothesisProposal`. Don't access Signature fields directly — access them on the `Prediction` return value with a cast.

4. **`dspy.configure(lm=lm)`** has `lm: Any` — no issue but no type safety on the LM.

5. **Pydantic v2 compatibility** — DSPy 2.x requires pydantic v2. If the project is on pydantic v1, an upgrade is needed. Check with `pip show pydantic`.

**Recommended mypy strategy:** keep DSPy boundary types wrapped in a thin typed adapter layer:

```python
# src/tune/application/dspy_hypothesis_module.py
def propose_hypothesis(context_str: str, candidates_str: str) -> HypothesisProposal:
    result = _predictor(system_context=context_str, candidate_params=candidates_str)
    proposal = cast(HypothesisProposal, result.hypothesis)
    # mypy sees HypothesisProposal from here outward
    return proposal
```

The rest of the codebase (hypothesis_generator.py, tune_engine.py) stays fully typed.

---

## 9. Recommended Migration Path

### Phase 0 (no behavior change) — install and configure

```bash
pip install dspy dspy[optuna]
```

```python
# src/tune/infrastructure/langgraph_hypothesis_client.py — add alongside existing client
lm = dspy.LM("openai/your-model", api_base=config.base_url, api_key=config.api_key)
dspy.configure(lm=lm)
```

### Phase 1 — replace raw JSON parsing with DSPy TypedPredictor

- Create `src/tune/application/dspy_hypothesis_module.py` with `ProposeHypothesis` signature and `HypothesisPredictor` module.
- Expose a `propose_hypothesis(context_str, candidates_str) -> HypothesisProposal` function.
- Keep the post-prediction contract validators (`_validate_contract_fields`, `_validate_against_history`, etc.) — these are business logic, not prompt engineering.
- Replace the `json.loads` + manual field extraction in `LlmHypothesisGenerator.generate()` with a call to `propose_hypothesis(...)`.

### Phase 2 — collect training data from iteration records

- `tune_recorder.py` already logs accepted/rejected outcomes per iteration. Emit `dspy.Example` objects from accepted runs.
- Store them in a `trainset.jsonl` alongside the compiled program.

### Phase 3 — compile with BootstrapFewShot

- After 30+ accepted iterations, run BootstrapFewShot offline (not in the hot path).
- Load `compiled_hypothesis.json` at startup; fall back to uncompiled if absent.

### Phase 4 (optional) — MIPROv2

- After 100+ examples, run MIPROv2 to jointly optimize instructions + demos.
- Evaluate improvement using the `result_evaluator.py` benchmark metric.

---

## 10. What DSPy Does NOT Help With

- **Deterministic triage layer** — keep `rule_based_triage.py` exactly as-is. DSPy is for the LLM path only.
- **KNOWLEDGE_DRIVEN phase** — the KB-scored candidate selection has no LLM call; keep it deterministic.
- **Apply/rollback coordination** — pure infrastructure, no LLM involvement.
- **Benchmark execution** — out of scope for DSPy.
- **LangGraph graph topology** — DSPy has no opinion on graph structure; it's just a callable node.
