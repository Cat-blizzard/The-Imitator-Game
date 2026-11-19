"""Durable job ledger with explicit submission uncertainty and bounded retries."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .common import file_sha256, json_sha256, read_json


TRANSITIONS = {
    "prepared": {"request_prepared"},
    "request_prepared": {"submitting"},
    "submitting": {"submitted", "submission_unknown", "generation_failed"},
    "submission_unknown": {"submitted"},
    "submitted": {"generating", "generated", "generation_failed"},
    "generating": {"generating", "generated", "generation_failed"},
    "generation_failed": {"request_prepared"},
    "generated": {"reviewing"},
    "reviewing": {"accepted_for_pairing", "needs_review", "rejected", "review_failed"},
    "review_failed": {"reviewing"},
    "needs_review": {"reviewing", "request_prepared"},
    "rejected": {"request_prepared"},
    "accepted_for_pairing": set(),
}


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
          job_id TEXT PRIMARY KEY, job_path TEXT NOT NULL, job_sha256 TEXT NOT NULL,
          bundle_sha256 TEXT NOT NULL,
          group_id TEXT NOT NULL, goal_id TEXT NOT NULL, split TEXT NOT NULL,
          status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
          task_id TEXT, request_path TEXT, request_sha256 TEXT,
          output_path TEXT, review_path TEXT, decision_path TEXT, decision_sha256 TEXT,
          updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
          old_status TEXT, new_status TEXT NOT NULL, at TEXT NOT NULL,
          detail TEXT NOT NULL, FOREIGN KEY(job_id) REFERENCES jobs(job_id));
        CREATE TABLE IF NOT EXISTS anchors (
          group_id TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL,
          provenance_path TEXT, provenance_sha256 TEXT, created_at TEXT NOT NULL);
        """)

    def close(self):
        self.db.close()

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def get(self, job_id):
        row = self.db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown job ID")
        return dict(row)

    def rows(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM jobs ORDER BY job_id")]

    def register(self, catalog):
        with self.transaction():
            for entry in catalog["jobs"]:
                path = Path(entry["job_path"]).resolve(strict=True)
                declared = entry.get("artifact_sha256", {})
                required = {"job.json", "contract.json", "evidence.json", "prompt.txt", "blind_review_prompt.txt", "frames/frames.json"}
                if not required.issubset(declared):
                    raise ValueError("Catalog lacks required artifact hashes")
                for relative, expected_digest in declared.items():
                    artifact = (path.parent / relative).resolve(strict=True)
                    if not artifact.is_relative_to(path.parent) or file_sha256(artifact) != expected_digest:
                        raise ValueError("Declared catalog artifact hash mismatch")
                job = read_json(path)
                if job.get("schema") != "r2h_job_v1" or job.get("training_admitted") is not False:
                    raise ValueError("Invalid input job or training already admitted")
                job_id = entry["job_id"]
                if job["job_id"] != job_id:
                    raise ValueError("Catalog/job identity mismatch")
                for name in ("group_id", "goal_id", "split"):
                    if job.get(name, job.get("source", {}).get(name)) != entry[name]:
                        raise ValueError("Catalog/job identity mismatch: " + name)
                digest = file_sha256(path)
                bundle_digest = self._bundle_sha(job)
                old = self.db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
                if old:
                    if (old["job_sha256"] != digest or old["job_path"] != str(path)
                            or old["bundle_sha256"] != bundle_digest):
                        raise ValueError("Immutable job was changed")
                    continue
                self.db.execute("""INSERT INTO jobs
                  (job_id,job_path,job_sha256,bundle_sha256,group_id,goal_id,split,status,updated_at)
                  VALUES(?,?,?,?,?,?,?,?,?)""", (job_id, str(path), digest, bundle_digest, entry["group_id"],
                    entry["goal_id"], entry["split"], "prepared", now()))
                self._event(job_id, None, "prepared", {"source": "verified_local_catalog"})

    def load_job(self, job_id):
        row = self.get(job_id)
        if file_sha256(row["job_path"]) != row["job_sha256"]:
            raise ValueError("Job artifact changed after registration")
        job = read_json(row["job_path"])
        if self._bundle_sha(job) != row["bundle_sha256"]:
            raise ValueError("Task contract, evidence or prompts changed after registration")
        for key in ("contract", "evidence"):
            job[key] = read_json(job[key + "_path"])
        return job

    @staticmethod
    def _bundle_sha(job):
        keys = ("contract_path", "evidence_path", "prompt_path", "blind_review_prompt_path")
        return json_sha256({key: file_sha256(job[key]) for key in keys})

    def _event(self, job_id, old, new, detail):
        self.db.execute("INSERT INTO events(job_id,old_status,new_status,at,detail) VALUES(?,?,?,?,?)",
                        (job_id, old, new, now(), json.dumps(detail, sort_keys=True)))

    def transition(self, job_id, new_status, *, expected=None, detail=None, **fields):
        allowed = {"task_id", "request_path", "request_sha256", "output_path", "review_path", "decision_path", "decision_sha256"}
        if set(fields) - allowed:
            raise ValueError("Unknown ledger fields")
        with self.transaction():
            row = self.get(job_id)
            if expected is not None and row["status"] != expected:
                raise ValueError("Job was changed by another worker")
            if new_status not in TRANSITIONS.get(row["status"], set()):
                raise ValueError("Invalid job transition: " + row["status"] + " -> " + new_status)
            values = {"status": new_status, "updated_at": now(), **fields}
            self.db.execute("UPDATE jobs SET " + ",".join(k + "=?" for k in values) + " WHERE job_id=?",
                            [*values.values(), job_id])
            self._event(job_id, row["status"], new_status, detail or {})

    def claim_submission(self, job_id, max_attempts=2):
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        with self.transaction():
            row = self.get(job_id)
            if row["status"] != "request_prepared":
                raise ValueError("Submission not ready; unresolved requests must not be resubmitted")
            if row["attempts"] >= max_attempts:
                raise ValueError("Generation attempt budget exhausted")
            if file_sha256(row["request_path"]) != row["request_sha256"]:
                raise ValueError("Prepared request changed")
            self.db.execute("UPDATE jobs SET status='submitting',attempts=attempts+1,updated_at=? WHERE job_id=?",
                            (now(), job_id))
            self._event(job_id, row["status"], "submitting", {"attempt": row["attempts"] + 1})
        return self.get(job_id)

    def bind_anchor(self, group_id, path, provenance_path=None):
        if not self.db.execute("SELECT 1 FROM jobs WHERE group_id=?", (group_id,)).fetchone():
            raise ValueError("Unknown initial-state group")
        path = Path(path).resolve(strict=True)
        digest = file_sha256(path)
        provenance_digest = file_sha256(provenance_path) if provenance_path else None
        with self.transaction():
            old = self.db.execute("SELECT * FROM anchors WHERE group_id=?", (group_id,)).fetchone()
            if old and (old["sha256"] != digest or old["path"] != str(path)
                        or old["provenance_sha256"] != provenance_digest):
                raise ValueError("Shared anchor is immutable; create a new run for a different anchor")
            self.db.execute("INSERT OR IGNORE INTO anchors VALUES(?,?,?,?,?,?)",
                            (group_id, str(path), digest, str(Path(provenance_path).resolve()) if provenance_path else None,
                             provenance_digest, now()))
        return {"group_id": group_id, "path": str(path), "sha256": digest}

    def anchor(self, group_id):
        row = self.db.execute("SELECT * FROM anchors WHERE group_id=?", (group_id,)).fetchone()
        if row is None:
            raise ValueError("Missing shared human initial anchor for this group")
        result = dict(row)
        if file_sha256(result["path"]) != result["sha256"]:
            raise ValueError("Shared human anchor changed")
        if result["provenance_path"] and file_sha256(result["provenance_path"]) != result["provenance_sha256"]:
            raise ValueError("Shared human anchor provenance changed")
        return result

    def status(self):
        return {row["status"]: row["n"] for row in self.db.execute(
            "SELECT status,COUNT(*) n FROM jobs GROUP BY status ORDER BY status")}
