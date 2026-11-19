import pytest

from data_generation_methods.r2h_pipeline.actor_plan_guard import decide_counts


def inputs():
    plan = {"maximum_visible_hands": 2}
    rows = [{"frame_index": i, "timestamp_seconds": i/24,
             "detected_hand_count": 3 if i else 2} for i in range(30)]
    inventory = [
        {"frame_index": i, "timestamp_seconds": i/24,
         "minimum_distinct_visible_hands": 3, "exact_count_known": True} for i in (0,23)]
    return plan, inventory, rows, {"human_structure": 4}


def test_all_four_scores_do_not_cancel_observed_plan_violation():
    result = decide_counts(*inputs())
    assert result["status"] == "blocked_actor_plan_violation_observed"
    assert result["cross_source_same_frame_excess"][0]["frame_index"] == 23
    assert result["review_requests"][0]["trigger"] == "count_disagreement"
    assert result["ordinal_scores_retained"]["human_structure"] == 4
    assert not result["training_admitted"]


def test_single_hand_no_flag_still_never_certifies_whole_clip():
    plan, inventory, rows, scores = inputs()
    plan["maximum_visible_hands"] = 1
    for x in inventory:
        x["minimum_distinct_visible_hands"] = 1
    for x in rows:
        x["detected_hand_count"] = 1
    result = decide_counts(plan,inventory,rows,scores)
    assert not result["review_requests"]
    assert result["person_count_status"].startswith("unknown")
    assert not result["full_clip_actor_absence_verified"]


def test_local_detector_only_excess_requires_review_not_true_anomaly():
    plan, inventory, rows, scores = inputs()
    for x in inventory:
        x["minimum_distinct_visible_hands"] = 2
    result = decide_counts(plan,inventory,rows,scores)
    assert result["status"] == "needs_local_actor_review"


def test_time_mismatch_blocks_join():
    plan, inventory, rows, scores = inputs()
    inventory[0]["timestamp_seconds"] = .1
    with pytest.raises(ValueError, match="timestamp"):
        decide_counts(plan,inventory,rows,scores)


def test_review_budget_bounded_and_no_person_inference():
    result = decide_counts(*inputs(), max_requests=1)
    assert len(result["review_requests"]) == 1
    assert result["hand_count_not_converted_to_person_count"]
