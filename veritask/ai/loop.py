"""'AI proposes, physics disposes': loops where a model drafts an artifact and the platform verifies it.

1. review_predicate(): a model reads a task's success code + instruction and lists semantic gaps; each claimed gap
   is cross-checked against the static linter and turned into a candidate counterexample to *run* (a claim only
   counts once the counterexample actually fools the predicate in simulation).
2. draft_contract(): a model drafts a TaskContract (outcome terms + counterexamples) from an instruction or a
   human-video description; the draft is validated by schema, then by running the expert (must pass) and the
   proposed counterexamples (must fail) before it is accepted.
3. propose_counterexamples(): red-team prompting for behaviours that might satisfy the predicate without doing
   the task; the runner executes them and keeps only the ones that expose a real gap.

All functions take any object with .chat(text, images) (see ai/models.py) so local and remote models are
interchangeable; every model call is logged by the adapter.
"""
from __future__ import annotations

import json, re

REVIEW_PROMPT = """You audit success checks of robot manipulation benchmarks.
Task instruction: {instruction}
Below is the Python evaluate() function. A success check is WRONG if a robot could satisfy it without actually
accomplishing the instruction (e.g. a required condition is computed but not used, a momentary event is latched
forever, the object may still be held, a physical effect such as pouring/folding/pressing is never measured).
Return JSON only: {{"gaps": [{{"condition": "...", "why": "...", "counterexample": "one concrete robot behaviour that would pass the check but not do the task"}}]}}. Return {{"gaps": []}} if the check is sound.

```python
{code}
```"""


def _json_block(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def review_predicate(model, instruction: str, evaluate_src: str) -> dict:
    out = model.chat(REVIEW_PROMPT.format(instruction=instruction, code=evaluate_src))
    parsed = _json_block(out)
    return dict(raw=out, gaps=(parsed or {}).get("gaps", []), parse_ok=parsed is not None)


CONTRACT_PROMPT = """Write a task contract (JSON) for a robot manipulation task.
Instruction: {instruction}
Available outcome metrics (name: meaning): {metrics}
Return JSON with keys: task_id, instruction, outcome (list of {{metric, op, threshold, units, args, real_measurement}}),
require_released (bool), settle_seconds (float), counterexamples (list of {{name, description, expected:false}}).
Use only the listed metrics. Choose thresholds with a safety margin and explain nothing outside the JSON."""


def draft_contract(model, instruction: str, metric_docs: dict[str, str]) -> dict:
    out = model.chat(CONTRACT_PROMPT.format(instruction=instruction, metrics=json.dumps(metric_docs, ensure_ascii=False)))
    return dict(raw=out, contract=_json_block(out))
