"""Judge harness: evaluate a judge (VLM, reward model, rule) against episodes with physical ground truth.

A judge is any callable judge(sample) -> dict(verdict: bool|None, score: float|None, raw: str). Samples carry the
instruction, image paths (before/after or frames), and ground truth from the task contract evaluated on privileged
state. Results are converted into credibility.Episode records so the same metrics apply to judges and predicates.
"""
from __future__ import annotations

import json, time
from dataclasses import dataclass, field, asdict
from typing import Callable

from ..audit.credibility import Episode, confusion, by_margin, auroc


@dataclass
class JudgeSample:
    sample_id: str
    instruction: str
    images: list[str]
    truth: bool
    margin: float | None = None          # signed margin of the ground-truth metric (near-miss if small)
    behaviour: str = ""
    meta: dict = field(default_factory=dict)


YES_NO_PROMPT = (
    "You are judging whether a robot completed a manipulation task. Task: {instruction}\n"
    "The first image shows the scene before the robot acted, the last image shows the final state.\n"
    "Answer with exactly one word, YES if the task was completed in the final state, otherwise NO."
)


def parse_yes_no(text: str) -> bool | None:
    t = text.strip().upper()
    if t.startswith("YES"):
        return True
    if t.startswith("NO"):
        return False
    if "YES" in t and "NO" not in t:
        return True
    if "NO" in t and "YES" not in t:
        return False
    return None


def run_judge(judge: Callable[[JudgeSample], dict], samples: list[JudgeSample], log_path: str | None = None) -> dict:
    eps, rows = [], []
    for s in samples:
        t0 = time.time(); out = judge(s); dt = time.time() - t0
        eps.append(Episode(behaviour=s.behaviour, verdict=out.get("verdict"), truth=s.truth, score=out.get("score"), margin=s.margin))
        rows.append(dict(sample_id=s.sample_id, behaviour=s.behaviour, truth=s.truth, margin=s.margin, seconds=round(dt, 3), **out))
    pos = [e.score for e in eps if e.truth and e.score is not None]; neg = [e.score for e in eps if (not e.truth) and e.score is not None]
    res = dict(confusion=confusion(eps), by_margin=by_margin(eps), auroc=auroc(pos, neg), n=len(samples))
    if log_path:
        json.dump(dict(summary=res, rows=rows), open(log_path, "w"), indent=1, default=str)
    return res
