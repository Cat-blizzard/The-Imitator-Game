"""Agent-in-the-loop backend: the platform writes model requests to a queue file and an interactive agent
(currently the Claude Code session doing this research, standing in for a frontier-model API) answers them.

Protocol (plain JSONL, so any agent or a human can serve it):
  <dir>/requests.jsonl   {"id", "prompt", "images": [paths], "meta": {...}}   written by enqueue()
  <dir>/answers.jsonl    {"id", "answer"}                                     written by the answering agent
Requests are blind: meta is for bookkeeping only and must not contain labels (keep labels in a separate key file).
"""
from __future__ import annotations

import json, os


class PendingAnswer(Exception):
    pass


class AgentQueue:
    def __init__(self, qdir: str):
        self.qdir = qdir; os.makedirs(qdir, exist_ok=True)
        self.req, self.ans = f"{qdir}/requests.jsonl", f"{qdir}/answers.jsonl"

    def enqueue(self, items: list[dict]) -> None:
        with open(self.req, "a") as f:
            for it in items:
                f.write(json.dumps(dict(id=it["id"], prompt=it["prompt"], images=list(it.get("images", [])), meta=it.get("meta", {})), ensure_ascii=False) + "\n")

    def answers(self) -> dict[str, str]:
        if not os.path.exists(self.ans):
            return {}
        out = {}
        for line in open(self.ans):
            if line.strip():
                d = json.loads(line); out[d["id"]] = d["answer"]
        return out

    def answer(self, rid: str, text: str) -> None:
        with open(self.ans, "a") as f:
            f.write(json.dumps(dict(id=rid, answer=text), ensure_ascii=False) + "\n")

    def chat_by_id(self, rid: str) -> str:
        a = self.answers()
        if rid not in a:
            raise PendingAnswer(rid)
        return a[rid]
