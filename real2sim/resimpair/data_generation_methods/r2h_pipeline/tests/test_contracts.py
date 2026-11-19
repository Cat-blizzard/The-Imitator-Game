import copy
import pytest

from data_generation_methods.r2h_pipeline.contracts import (
    BLIND_GOAL_ALIASES, GOAL_PREFIX, INITIAL_HASH_KEYS, PHASES, PROGRAMS,
    PROGRAM_PHASES, blind_review_prompt, build_contract, generation_prompt, validate_contract,
)


def fixture_row_receipt(program=PROGRAMS[0]):
    row = {"program": program, "goal_key": GOAL_PREFIX + program,
           "group": "layout:0", "split": "train"}
    phases, events = [], []
    for i, name in enumerate(PROGRAM_PHASES[program]):
        index, destination, _ = PHASES[name]
        phases.append({"name": name, "object_index": index, "position": [0., 0., .1],
                       "rotation": [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                       "destination_support": None, "required_clear_objects": []})
        events.append({"phase": name, "object_index": index, "passed": True,
                       "first_frame": i * 3 + 1, "end_frame": i * 3 + 4})
    receipt = {**row, "length": 24, "fps": 20, "phase_targets": phases, "events": events,
               **{k: "a" * 64 for k in INITIAL_HASH_KEYS}}
    return row, receipt


@pytest.mark.parametrize("program,count", zip(PROGRAMS, (3, 5, 6)))
def test_complete_order_identity_and_no_human_gt_claim(program, count):
    row, receipt = fixture_row_receipt(program)
    contract = build_contract(row, receipt)
    assert len(contract["required_events"]) == count
    assert len(contract["event_order"]) == count - 1
    assert contract["training_admitted"] is False
    assert contract["human_wrist_3d_valid"] is False
    assert contract["review_spec"]["blind_goal_aliases"] == BLIND_GOAL_ALIASES
    assert "truncate" in generation_prompt(contract)


def test_missing_step_and_changed_object_are_rejected():
    row, receipt = fixture_row_receipt()
    bad = copy.deepcopy(receipt)
    bad["phase_targets"].pop()
    with pytest.raises(ValueError, match="phase"):
        build_contract(row, bad)
    bad = copy.deepcopy(receipt)
    bad["events"][0]["object_index"] = 0
    with pytest.raises(ValueError, match="identity"):
        build_contract(row, bad)


def test_overlap_and_split_mismatch_rejected():
    row, receipt = fixture_row_receipt()
    receipt["events"][1]["first_frame"] = 1
    with pytest.raises(ValueError, match="time range"):
        build_contract(row, receipt)
    row, receipt = fixture_row_receipt()
    receipt["split"] = "test"
    with pytest.raises(ValueError, match="split"):
        build_contract(row, receipt)


def test_replacing_private_review_expectation_or_admitting_training_rejected():
    row, receipt = fixture_row_receipt()
    contract = build_contract(row, receipt)
    contract["review_spec"]["required_events"] = []
    with pytest.raises(ValueError, match="expectations"):
        validate_contract(contract)
    contract = build_contract(row, receipt)
    contract["training_admitted"] = True
    with pytest.raises(ValueError, match="admit"):
        validate_contract(contract)


def test_blind_prompt_is_symmetric_and_has_no_private_answer():
    prompt = blind_review_prompt()
    assert all(choice in prompt for choice in BLIND_GOAL_ALIASES)
    assert all(goal not in prompt for goal in BLIND_GOAL_ALIASES.values())
    assert "/mnt/" not in prompt and "layout:0" not in prompt
    assert all(name not in prompt for name in PROGRAMS)


def test_non_target_invariant_is_structured_and_checked():
    row, receipt = fixture_row_receipt()
    contract = build_contract(row, receipt)
    assert contract["review_spec"]["forbidden_events"] == [
        {"role": "block0", "verb": "pick_and_place", "destination_role": "*"}]
    contract["review_spec"]["forbidden_events"] = []
    with pytest.raises(ValueError, match="non-target invariant"):
        validate_contract(contract)
