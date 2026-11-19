"""Semantic provenance counterexamples, including fully consistent rehashes."""
import copy

import pytest

from data_generation_methods.human_to_robot_pipeline.source_lineage_audit import (
    resolve_selected_source, summarize,
)


def fixture():
    a = {"id": "a", "video": {"path": "/a.mp4", "sha256": "a" * 64},
         "session": "session_a", "split": "train"}
    b = {"id": "b", "video": {"path": "/b.mp4", "sha256": "b" * 64},
         "session": "session_b", "split": "train"}
    proposal = {"human_window_id": "a", "human_video": a["video"],
                "human_source_window": a, "split": "train"}
    return ({"proposal": copy.deepcopy(proposal), "contract": {"human_windows": [a, b]}},
            {"randomization": {"structured_proposal": copy.deepcopy(proposal)}, "split": "train"})


def test_shared_contract_selects_only_actual_source_and_legacy_resolution():
    metadata, receipt = fixture()
    assert resolve_selected_source(metadata, receipt)["id"] == "a"
    for p in [metadata["proposal"], receipt["randomization"]["structured_proposal"]]:
        p.pop("human_source_window")
    assert resolve_selected_source(metadata, receipt)["id"] == "a"


@pytest.mark.parametrize("mutation", ["selected_id", "video", "session", "split", "receipt", "duplicate"])
def test_mismatched_source_is_rejected_even_when_receipt_matches_input(mutation):
    metadata, receipt = fixture()
    p = metadata["proposal"]
    if mutation == "selected_id":
        p["human_window_id"] = "b"
    elif mutation == "video":
        p["human_video"]["sha256"] = "b" * 64
    elif mutation == "session":
        p["human_source_window"]["session"] = "other"
    elif mutation == "split":
        p["split"] = "test"
        receipt["split"] = "test"
    elif mutation == "duplicate":
        metadata["contract"]["human_windows"].append(copy.deepcopy(metadata["contract"]["human_windows"][0]))
    if mutation != "receipt":
        receipt["randomization"]["structured_proposal"] = copy.deepcopy(p)
    else:
        receipt["randomization"]["structured_proposal"]["human_window_id"] = "b"
    with pytest.raises(ValueError):
        resolve_selected_source(metadata, receipt)


def row(name="a"):
    return {"attempt": name, "receipt": {"sha256": name}, "execution_id": None,
            "selected_human_window": {"video": {"sha256": name}, "session": name},
            "human_spec": {"sha256": "same_registry"}, "dataset_split": "train",
            "exact_dynamic_sha256": name, "dynamic_array_signatures": {"states": [2, 32]}}


@pytest.mark.parametrize("shared", ["video", "session", "motion"])
def test_shared_lineage_never_crosses_dataset_splits(shared):
    a, b = row(), row("b")
    b["dataset_split"] = "test"
    if shared == "video":
        b["selected_human_window"]["video"] = a["selected_human_window"]["video"]
    elif shared == "session":
        b["selected_human_window"]["session"] = a["selected_human_window"]["session"]
    else:
        b["exact_dynamic_sha256"] = a["exact_dynamic_sha256"]
    with pytest.raises(ValueError, match="crosses dataset splits"):
        summarize([a, b])


@pytest.mark.parametrize("shared", ["attempt", "receipt", "execution_id"])
def test_duplicates_cannot_inflate_execution_counts(shared):
    a, b = row(), row("b")
    a["execution_id"] = "execution_a"
    b[shared] = a[shared]
    with pytest.raises(ValueError):
        summarize([a, b])


def test_different_exact_hashes_do_not_claim_new_behavior():
    result = summarize([row(), row("b")])
    assert result["qualified_attempt_records"] == 2
    assert result["legacy_records_without_execution_uuid"] == 2
    assert result["exact_dynamic_byte_groups"] == 2
    assert result["new_behavior_coverage"].startswith("unknown")
