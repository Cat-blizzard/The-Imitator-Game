import pytest

from data_generation_methods.r2h_pipeline.anchor_body_review import decision


def observation(n, exact=True):
    return dict(minimum_visible_hand_count=n, exact_count_known=exact,
                hands=[dict(location=f"location {i}", description="visible fingers")
                       for i in range(n)], limitations=[])


def test_extra_visible_hand_is_plan_mismatch_not_anatomy_proof():
    result = decision(observation(3), maximum_visible_hands=2)
    assert result["status"] == "rendering_plan_mismatch"
    assert result["anatomy_verified"] is False
    assert result["training_admitted"] is False


def test_uncertainty_is_not_pass():
    assert decision(observation(2, False), maximum_visible_hands=2)["status"] == "needs_review"


def test_inconsistent_inventory_is_not_pass():
    item = observation(2)
    item["hands"].pop()
    assert decision(item, maximum_visible_hands=2)["status"] == "needs_review"


def test_within_budget_is_not_task_or_training_approval():
    result = decision(observation(2), maximum_visible_hands=2)
    assert result["status"] == "inventory_within_budget"
    assert result["task_semantics_verified"] is False
    assert result["training_admitted"] is False


def test_explicit_budget_is_required():
    with pytest.raises(ValueError):
        decision(observation(2), maximum_visible_hands=0)
