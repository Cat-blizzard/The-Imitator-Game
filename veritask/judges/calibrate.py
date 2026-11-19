"""Judge calibration and fine-tuning interfaces.

Implemented now:
  - PlattCalibrator: fit a logistic map from a judge's raw score (e.g. P(YES)) to P(success) on labelled
    sim episodes; report the threshold that maximises balanced accuracy on a held-out split.
  - export_sft_jsonl(): write near-miss episodes with physical ground truth as an instruction-tuning dataset
    (image paths + instruction + target answer + margin), so any fine-tuning stack can consume it.
Interface only (not implemented): FineTuneJob - hand the exported dataset to a trainer (LoRA on a local VLM,
or a provider fine-tuning API) and register the resulting model as a judge backend.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
import numpy as np


class PlattCalibrator:
    def __init__(self):
        self.a, self.b = 1.0, 0.0

    def fit(self, scores, labels, iters: int = 2000, lr: float = 0.5):
        s = np.asarray(scores, float); y = np.asarray(labels, float)
        a, b = 1.0, 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(a * s + b)))
            ga, gb = np.mean((p - y) * s), np.mean(p - y)
            a -= lr * ga * 10; b -= lr * gb
        self.a, self.b = float(a), float(b)
        return self

    def predict_proba(self, scores):
        return 1 / (1 + np.exp(-(self.a * np.asarray(scores, float) + self.b)))

    @staticmethod
    def best_threshold(probs, labels):
        p = np.asarray(probs); y = np.asarray(labels).astype(bool); best, bt = 0.0, 0.5
        for t in np.unique(p):
            pred = p >= t
            tpr = (pred & y).sum() / max(y.sum(), 1); tnr = (~pred & ~y).sum() / max((~y).sum(), 1)
            if (tpr + tnr) / 2 > best:
                best, bt = (tpr + tnr) / 2, float(t)
        return bt, best


def export_sft_jsonl(rows: list[dict], instruction: str, image_keys: tuple[str, ...], out_path: str) -> int:
    """rows: episodes with ground-truth 'truth' and image paths under image_keys (e.g. ('before', 'after_top'))."""
    n = 0
    with open(out_path, "w") as f:
        for r in rows:
            imgs = [(r[k]["top"] if isinstance(r[k], dict) else r[k]) for k in image_keys]  # 'before' is stored per view
            f.write(json.dumps(dict(images=imgs, instruction=instruction, answer="YES" if r["truth"] else "NO",
                                    margin=r.get("min_norm_margin"), meta={k: r[k] for k in ("world", "mode") if k in r}), ensure_ascii=False) + "\n")
            n += 1
    return n


@dataclass
class FineTuneJob:
    """Interface placeholder: train a judge on an exported dataset and return a backend spec for ai.backends.make()."""
    dataset_path: str
    base_model: str
    method: str = "lora"          # or a provider fine-tuning API

    def run(self) -> dict:
        raise NotImplementedError("fine-tuning not implemented yet; dataset export and calibration are available")
