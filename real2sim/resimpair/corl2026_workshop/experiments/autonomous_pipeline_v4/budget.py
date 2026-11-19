"""Bounded, cumulative, single-submission calls. Semantic failures never retry."""
from __future__ import annotations

import fcntl
from pathlib import Path
import time

from .core import REGISTRY, bind, checked, digest, identifier, read, require, write
from .timing import span

KINDS = {"image", "video", "review"}


def counts(ledger):
    answer = {k: 0 for k in KINDS}
    for marker in Path(ledger).glob("stages/*/call_started.json"):
        kind = read(marker)["kind"]
        require(kind in KINDS, "Unrecognized counted operation")
        answer[kind] += 1
    return answer


class Budget:
    def __init__(self, path):
        self.path = Path(path)
        self.spec = read(path)
        s = self.spec
        require(s["schema"] == "resimpair_cumulative_budget_v4", "Explicit v4 budget required")
        identifier(s["campaign_id"])
        self.ledger = Path(s["ledger_path"])
        require(self.ledger.is_absolute() and self.ledger == self.ledger.resolve(), "Canonical ledger required")
        require(set(s["new_call_limits"]) == KINDS and s["max_concurrent"] == 1,
                "Fixed limits and serial execution required")
        for k in KINDS:
            require(type(s["new_call_limits"][k]) is int and s["new_call_limits"][k] >= 0,
                    "Nonnegative call limit required")
            require(s["lineage_cumulative_limits"][k] == s["ancestor_started_calls"][k] + s["new_call_limits"][k],
                    "Lineage may not reset historical usage")
        for bound in s["ancestor_evidence"]:
            checked(bound)
        require(1 <= s["timeout_seconds"] <= 600, "Bounded provider timeout required")
        REGISTRY.mkdir(parents=True, exist_ok=True)
        with (REGISTRY / ".registration.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(REGISTRY / (s["campaign_id"] + ".json"), s)
        self.ledger.mkdir(parents=True, exist_ok=True)
        write(self.ledger / "budget.json", s)

    def usage(self):
        current = counts(self.ledger)
        return {"new_started": current,
                "lineage_started": {k: current[k] + self.spec["ancestor_started_calls"][k] for k in KINDS},
                "new_remaining": {k: self.spec["new_call_limits"][k] - current[k] for k in KINDS},
                "currency_cost": None}

    def directory(self, kind, request):
        require(kind in KINDS, "Unknown operation")
        return self.ledger / "stages" / (kind + "_" + digest(bind(request))[:32])

    def call(self, kind, request, operation):
        """operation receives its durable stage directory; never gets an implicit retry."""
        stage = self.directory(kind, request)
        with (self.ledger / ".execution.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            metadata = {"kind": kind, "request": bind(request), "budget": bind(self.path)}
            write(stage / "request_metadata.json", metadata)
            if (stage / "result.json").exists():
                return read(stage / "result.json"), stage
            if (stage / "call_started.json").exists():
                raise RuntimeError("previous_call_has_no_result_no_resubmission")
            payload = read(request)
            code = payload.get("implementation", {})
            scope = digest({name: code[name]["sha256"] for name in ("models.py", "source.py", "review.py") if name in code}) if "models.py" in code else None
            for marker in self.ledger.glob("stages/*/call_started.json"):
                prior = marker.parent
                failure = read(prior / "failure.json") if (prior / "failure.json").exists() else {}
                require(not (scope and failure.get("shared_service_scope") == scope and
                             failure.get("http_status") in {400, 401, 403, 413, 429}),
                        "shared_provider_error_requires_a_new_protocol_revision")
                if not (prior / "result.json").exists():
                    from .billing import validate_retired_billed_review
                    require(failure.get("provider_outcome") == "definite_failure" or validate_retired_billed_review(prior),
                            "unresolved_provider_submission_no_new_call")
                elif read(marker)["kind"] == "video":
                    require((prior / "video_terminal.json").exists(), "video_in_flight_resume_existing_id")
            require(counts(self.ledger)[kind] < self.spec["new_call_limits"][kind], "campaign_call_limit_exhausted")
            write(stage / "call_started.json", {**metadata, "started_unix": time.time(),
                  "single_post": True, "automatic_retry": False})
            try:
                with span(stage, "api_" + kind):
                    result = operation(stage)
            except Exception as exc:
                # Raw provider errors may contain signed URLs; save stable codes only.
                http = getattr(exc, "http_status", None)
                complete = (stage / "provider_response.json").exists()
                definite = (complete and kind != "video") or (isinstance(http, int) and 400 <= http < 500)
                write(stage / "failure.json", {"exception_type": type(exc).__name__,
                      "code": getattr(exc, "code", "operation_failed"), "http_status": http,
                      "provider_outcome": "definite_failure" if definite else "unknown",
                      "provider_response_complete": complete, "count_preserved": True,
                      "shared_service_scope": scope,
                      "finished_unix": time.time(), "automatic_retry": False})
                raise
            write(stage / "result.json", result)
            return result, stage
