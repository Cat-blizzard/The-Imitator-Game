"""Local timing evidence; never a provider retry, quality rule, or payload log."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import os
import time
import uuid
import warnings

from .core import write

_parent = ContextVar("resimpair_timing_parent", default=None)


def _save(path, value):
    # Observability must not replace a task exception or change task behavior
    # when its own disk is full. A missing completion remains an unknown span.
    try:
        write(path, value)
    except OSError:
        warnings.warn("Timing evidence could not be persisted", RuntimeWarning)


@contextmanager
def span(directory, stage, *, cache_hit=False):
    identifier = uuid.uuid4().hex
    output = Path(directory) / "timings" / identifier
    record = {"schema": "resimpair_stage_timing_v1", "span_id": identifier,
              "parent_span_id": _parent.get(), "stage": stage, "pid": os.getpid(),
              "started_unix": time.time(), "cache_hit": bool(cache_hit)}
    _save(output / "started.json", record)
    token = _parent.set(identifier)
    started = time.perf_counter()
    status, failure = "completed", None
    result = {}
    try:
        yield result
    except BaseException as exc:
        status, failure = "failed", type(exc).__name__
        raise
    finally:
        elapsed = time.perf_counter() - started
        _parent.reset(token)
        # Only fixed scalar fields leave the observed operation, never a raw
        # provider error, request, URL, credential, or arbitrary returned object.
        allowed = {k: v for k, v in result.items() if k in {"cache_hit", "semantic_rejected", "frame_count", "bytes"}
                   and type(v) in {bool, int}}
        _save(output / "finished.json", {**record, **allowed, "status": status,
              "exception_type": failure, "finished_unix": time.time(),
              "elapsed_seconds": elapsed, "clock": "perf_counter", "inclusive_of_child_spans": True})


def traced(stage, directory, cache_file=None):
    """Time a public entry point while preserving its signature and exceptions."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            path = Path(directory(*args, **kwargs))
            cached = cache_file is not None and (path / cache_file).exists()
            with span(path, stage, cache_hit=cached) as timing:
                result = function(*args, **kwargs)
                if isinstance(result, dict):
                    timing["semantic_rejected"] = result.get("status") in {
                        "compiler_rejected", "initial_image_rejected", "generated_video_rejected",
                        "source_extraction_stopped", "runtime_stopped", "video_provider_failed"}
                    if isinstance(result.get("gate"), dict) and result["gate"].get("passed_in_sampled_visual_scope") is False:
                        timing["semantic_rejected"] = True
                    if result.get("status") == "existing_terminal_run_preserved":
                        timing["cache_hit"] = True
                return result
        return wrapped
    return decorate


class Accumulator:
    """Small in-memory operation counters; exclusive time avoids double counting.

    A report is persisted once, not once per simulation step. Only synchronous
    host elapsed time is claimed; the renderer already synchronizes readback.
    """
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.phase = "setup"
        self.rows = {}
        self.stack = []

    @contextmanager
    def section(self, operation):
        key = (self.phase, operation)
        entry = [self.clock(), 0.]
        self.stack.append(entry)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            elapsed = max(0., self.clock() - entry[0])
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            row = self.rows.setdefault(key, {"calls": 0, "failed_calls": 0,
                                             "inclusive_seconds": 0., "exclusive_seconds": 0.})
            row["calls"] += 1
            row["failed_calls"] += int(failed)
            row["inclusive_seconds"] += elapsed
            row["exclusive_seconds"] += max(0., elapsed - entry[1])

    def snapshot(self):
        return {"schema": "resimpair_operation_timing_v1", "clock": "perf_counter",
                "rows": [{"phase": p, "operation": op, **v} for (p, op), v in sorted(self.rows.items())],
                "scope": "host wall time; physics includes control/contact bookkeeping; render includes sensor readback; encode_enqueue is not complete background codec time",
                "exclusive_times_may_be_summed": True, "inclusive_times_must_not_be_summed": True}

    def persist(self, path):
        _save(Path(path), self.snapshot())
