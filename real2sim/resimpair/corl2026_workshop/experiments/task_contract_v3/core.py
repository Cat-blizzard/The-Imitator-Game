"""Small evidence helpers. Immutable artifacts and mutable state are distinct."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

VERSION = "task_contract_candidate_v3"
from resimpair_runtime import ROOT as WORKSPACE, RUNS
DATA = RUNS / "legacy_five_task"
RUN = DATA / "human_generation_v3"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha(path)}


def checked(bound):
    require(set(bound) == {"path", "sha256"}, "Malformed artifact binding")
    from resimpair_runtime import evidence_path
    mapped = evidence_path(bound['path'])
    if str(mapped) != bound['path']:
        require(sha(mapped) == bound['sha256'], 'Relocated evidence bytes changed')
        return mapped
    require(bind(bound["path"]) == bound, "Artifact bytes or canonical path changed: " + bound["path"])
    return Path(bound["path"])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == value, "Immutable artifact differs; use a new revision: " + str(path))
        return
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def replace_state(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def validate_graph(events, precedence, persistent, nframes):
    """Validate a DAG, without truncating atomic events to a prompt-size limit."""
    ids = [e["id"] for e in events]
    require(len(ids) == len(set(ids)) and ids, "Duplicate or empty event IDs")
    by_id = {e["id"]: e for e in events}
    for event in events:
        require(type(event["frame"]) is int and 0 <= event["frame"] < nframes,
                "Event outside decoded simulation timeline")
        require(event["evidence_kind"] == "measured", "Controller labels cannot supply event evidence")
        require(event["measurement"] and set(event["hands"]) <= {"left", "right"}, "Unbound event")
    remaining = set(ids)
    edges = set(tuple(edge) for edge in precedence)
    for left, right in edges:
        require(left in by_id and right in by_id and left != right, "Invalid precedence edge")
        require(by_id[left]["frame"] <= by_id[right]["frame"], "Precedence contradicts measured time")
    while remaining:
        roots = {x for x in remaining if not any(b == x and a in remaining for a, b in edges)}
        require(roots, "Cyclic event graph")
        remaining -= roots
    for interval in persistent:
        require(interval["start_event"] in by_id and interval["end_event"] in by_id,
                "Unknown persistent interval endpoint")
        require(interval["hand"] in {"left", "right"} and interval["holds_in_every_measured_frame"] is True,
                "Continuous support needs every-frame evidence")


def compile_source(observation, mapping, receipt_binding):
    """Compile observations plus explicit augmentation, never invent a source pass."""
    require(observation["schema"] == VERSION + "/source_observation", "Source observation required")
    checked(observation["video"])
    checked(observation["review_artifact"])
    for key in ["provenance", "source_spec"]:
        if key in observation:
            checked(observation[key])
    start, end = observation["window_frames"]
    require(type(start) is int and type(end) is int and 0 <= start < end, "Invalid source window")
    require(end <= observation["decoded_source_frames"], "Source window exceeds decoded source video")
    require(observation["source_split"] in {"train", "validation", "test"}, "Source split required")
    require(observation["source_session"] and observation["observations"], "Source session/observations missing")
    for item in observation["observations"]:
        require(item["text"] and item["frames"] and all(start <= f < end for f in item["frames"]),
                "Source observation cites frames outside its declared window")
    mode = mapping["correspondence"]
    require(mode in {"reviewed_atomic_equivalence", "count_layout_augmentation",
                     "subskill_composition", "mechanical_analogue"}, "Unknown source mapping mode")
    require(mapping["preserved"] and mapping["changes"] and mapping["limitations"], "Explicit mapping required")
    exact = mode == "reviewed_atomic_equivalence" and observation["complete_atomic_window_reviewed"] is True
    if mode != "reviewed_atomic_equivalence":
        require(mapping["changes"] != ["none"], "Augmentation cannot claim an unchanged source")
    return {"schema": VERSION + "/source_contract", "observation": observation, "mapping": mapping,
            "target_execution": receipt_binding, "source_split": observation["source_split"],
            "source_session": observation["source_session"], "same_source_derivatives_must_share_split": True,
            "exact_source_task_equivalence": exact, "development_synthesis_supported": True,
            "automatic_visual_inference": False, "training_admitted": False,
            "status": "compiled_reviewed_atomic_source" if exact else "compiled_explicit_development_augmentation"}
