"""Evidence ledger: every executed episode that bears on a contract's credibility, one JSON object per line.

A contract is only as credible as the evidence behind it. The ledger records, per episode, what was run (behaviour,
its role), what the contract said, what the benchmark's own predicate said, and (when available) an independent
ground truth, the binding margin and the physics engine. Task credibility cards (audit/card.py) are computed from it.

Roles
  counterexample   designed behaviour that must be rejected (contract counterexample with expected=false)
  control          designed positive control that must be accepted (expected=true)
  natural          episode from a policy / expert / data engine, no designed expectation
  near_miss        episode generated near the decision boundary (truth from privileged physics, if known)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

ROLES = ("counterexample", "control", "natural", "near_miss")


@dataclass
class EvidenceRow:
    contract: str                      # task_id of the contract
    behaviour: str
    role: str                          # one of ROLES
    verdict: bool | None               # contract verdict (None = unknown)
    bench: bool | None = None          # the benchmark's own predicate on the same episode (any-step unless noted)
    truth: bool | None = None          # independent ground truth, if any (e.g. human label, real outcome)
    margin: float | None = None        # min normalised margin of the contract terms
    binding_term: str | None = None
    engine: str = ""                   # physics engine / simulator, e.g. "sapien-physx", "newton-mpm", "mujoco4"
    seed: int | None = None
    experiment: str = ""               # e.g. "E4"
    source: str = ""                   # result file the row was derived from
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")


def save(path: str, rows: Iterable[EvidenceRow]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n"); n += 1
    return n


def load(path: str) -> list[EvidenceRow]:
    with open(path, encoding="utf-8") as f:
        return [EvidenceRow(**json.loads(l)) for l in f if l.strip()]
