"""Focused visible-hand inventory for an edited anchor, not a pose estimator.

The expected actor budget comes from the chosen rendering plan. Exceeding it
is a plan mismatch, not proof that the image is anatomically impossible.
"""
from __future__ import annotations

import json

from .providers import parse_review_json


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "minimum_visible_hand_count": {"type": "integer", "minimum": 0},
        "exact_count_known": {"type": "boolean"},
        "hands": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "location": {"type": "string", "minLength": 1},
                "description": {"type": "string", "minLength": 1},
            }, "required": ["location", "description"]}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["minimum_visible_hand_count", "exact_count_known", "hands", "limitations"],
}


def prompt():
    return (
        "Inspect this single static edited image. Inventory every distinct visible HUMAN "
        "HAND, including a partially visible hand underneath another hand. Distinguish "
        "hands from forearms, sleeves, shadows and reflections. List each hand once with "
        "its image-relative location and what visible fingers/palm establish that it is "
        "a separate hand. Give a conservative minimum count, and whether the total visible "
        "count is unambiguous. Do not infer hidden limbs or a desired answer. This is an "
        "inventory, not an anatomical correctness or task-success judgment. Return ONLY "
        "JSON matching this complete schema:\n" + json.dumps(SCHEMA)
    )


def decision(review, *, maximum_visible_hands):
    if type(maximum_visible_hands) is not int or maximum_visible_hands < 1:
        raise ValueError("A positive explicit rendering-plan hand budget is required")
    review = parse_review_json(json.dumps(review), SCHEMA)["review"]
    count = review["minimum_visible_hand_count"]
    if count != len(review["hands"]):
        state, reason = "needs_review", "inventory_count_disagrees_with_list"
    elif count > maximum_visible_hands:
        state, reason = "rendering_plan_mismatch", "too_many_visible_hands"
    elif not review["exact_count_known"]:
        state, reason = "needs_review", "visible_hand_count_ambiguous"
    else:
        state, reason = "inventory_within_budget", "visible_count_within_declared_limit"
    return {
        "status": state, "reason": reason,
        "maximum_visible_hands": maximum_visible_hands,
        "minimum_visible_hand_count": count,
        "scope": "visible_hand_inventory_only",
        "task_semantics_verified": False, "anatomy_verified": False,
        "quality_release_approved": False, "training_admitted": False,
    }
