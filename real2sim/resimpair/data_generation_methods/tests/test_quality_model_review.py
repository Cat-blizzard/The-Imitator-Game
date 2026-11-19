import copy
import pytest

from data_generation_methods.r2h_pipeline.quality_model_review import (
    CONSTRAINTS, DIMENSIONS, build_prompt, select_uniform_boundary_frames, validate_result,
)


def valid_result():
    evidence = [{"frame_index": 0, "timestamp_seconds": 0., "observation": "Visible cabinet"}]
    return {"review": {
        "dimensions": {k: {"score": 3, "evidence": copy.deepcopy(evidence),
                            "explanation": "Observed scope", "unobserved_or_ambiguous": ["Unsampled frames"]}
                       for k in DIMENSIONS},
        "constraints": {k: {"status": "supported_in_observed_frames",
                             "evidence": copy.deepcopy(evidence), "explanation": "Observed"}
                        for k in CONSTRAINTS},
        "visible_hand_inventory": [{"frame_index": 0, "timestamp_seconds": 0.,
                                    "minimum_distinct_visible_hands": 1,
                                    "exact_count_known": False, "description": "Partial hand"}],
        "limitations": ["Sparse frames cannot establish physical contact."],
    }}


FRAMES = [{"frame_index": 0, "timestamp_seconds": 0.},
          {"frame_index": 192, "timestamp_seconds": 8.}]


def test_same_sampler_has_24_frames_for_both_cabinet_clips():
    selected = select_uniform_boundary_frames(193)
    assert len(selected) == 24
    assert selected[:4] == [0, 1, 2, 3]
    assert selected[-4:] == [189, 190, 191, 192]


def test_review_accepts_valid_and_preserves_unknown_not_high_score():
    r = valid_result()
    r["review"]["dimensions"]["interaction_visibility"]["score"] = "unknown"
    r["review"]["dimensions"]["interaction_visibility"]["evidence"] = []
    assert validate_result(r, FRAMES)["dimensions"]["interaction_visibility"]["score"] == "unknown"


@pytest.mark.parametrize("change", ["missing", "wrong_frame", "wrong_pts", "boolean"])
def test_model_cannot_invent_visual_evidence(change):
    r = valid_result()
    d = r["review"]["dimensions"]["task_readability"]
    if change == "missing":
        d["evidence"] = []
    elif change == "wrong_frame":
        d["evidence"][0]["frame_index"] = 99
    elif change == "wrong_pts":
        d["evidence"][0]["timestamp_seconds"] = .4
    else:
        d["score"] = True
    with pytest.raises(Exception):
        validate_result(r, FRAMES)


def test_prompt_contract_rejects_previous_verdict_fields():
    task = {"initial": "Closed", "necessary_steps": ["Open"], "goal": "Open", "invariants": []}
    assert "No baseline" in build_prompt(task)
    task["old_grade"] = "bad"
    with pytest.raises(ValueError):
        build_prompt(task)
