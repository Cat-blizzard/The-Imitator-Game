"""Candidate RGB evidence vocabulary for mechanisms absent from the old five skills.

These are observable phase/role requirements, not estimated physical labels.
The extension is opt-in; no paid extraction or source equivalence is implied.
"""
from .core import require

VERSION="visible_mechanisms_v1"
REQUIREMENTS={
    "elastic_press": {"phases":["depress","withdraw","rebound"],"roles":["moving_control","fixed_support"]},
    "nonprehensile_rocker": {"phases":["touch_side","tilt","withdraw"],"roles":["moving_control","pivot_support"]},
    "constrained_slide": {"phases":["touch_control","slide_along_guide","withdraw"],"roles":["moving_control","guide"]},
    "tangential_rotation": {"phases":["tangential_touch","rotate","withdraw"],"roles":["rotary_control","pivot_support"]},
    "lever_equilibrium": {"phases":["touch_side","tilt","withdraw"],"roles":["beam","pivot_support"]},
    "rolling_capture": {"phases":["trigger","visible_rotation","arrive"],"roles":["rolling_body","destination"]},
    "tool_mediated_pull": {"phases":["acquire_tool","engage_tool","pull_cargo"],"roles":["tool_handle","tool_contact_feature","cargo"]},
    "tool_mediated_sweep": {"phases":["acquire_tool","contact_loose_bodies","sweep"],"roles":["tool_handle","tool_contact_feature","loose_bodies","destination"]},
    "contact_path_trace": {"phases":["acquire_tool","touch_surface","trace_path"],"roles":["tool_handle","tool_tip","contact_surface"]},
    "confined_tool_agitation": {"phases":["acquire_tool","enter_container","agitate_contents"],"roles":["tool_handle","tool_contact_feature","container","contents"]},
}


def evidence_schema(obj, text):
    return {"type":"array","items":obj({
        "skill":{"type":"string","enum":list(REQUIREMENTS)},
        "roles":{"type":"array","items":obj({"role":text,"entity_id":text,"visible_feature":text,
             "event_ids":{"type":"array","items":text,"minItems":1}})},
        "phases":{"type":"array","items":obj({"phase":text,
             "visibility":{"type":"string","enum":["observed","occluded","unknown"]},
             "event_ids":{"type":"array","items":text},"limitation":text})},
        "unobservable_properties":{"type":"array","items":text},
    })}


def validate_evidence(observations, claimed):
    events={e["id"]:e for e in observations["events"]}
    entities={e["id"] for e in observations["entities"]}
    rows=observations.get("mechanism_evidence",[])
    require(len({x["skill"] for x in rows})==len(rows),"duplicate_mechanism_evidence")
    by_skill={x["skill"]:x for x in rows}
    result={}
    for skill in set(claimed)&REQUIREMENTS.keys():
        require(skill in by_skill,"mechanism_evidence_missing:"+skill)
        row=by_skill[skill];phases={x["phase"]:x for x in row["phases"]}
        require(len(phases)==len(row["phases"]),"duplicate_visible_phase")
        for phase in REQUIREMENTS[skill]["phases"]:
            require(phase in phases and phases[phase]["visibility"]=="observed","mechanism_phase_not_observed:"+phase)
            refs=phases[phase]["event_ids"]
            require(bool(refs) and all(e in events and events[e]["observation"]=="observed" for e in refs),"mechanism_phase_without_observed_event")
        roles={x["role"] for x in row["roles"]}
        require(set(REQUIREMENTS[skill]["roles"])<=roles,"mechanism_functional_role_missing:"+skill)
        for role in row["roles"]:
            require(role["entity_id"] in entities and bool(role["visible_feature"].strip()),"mechanism_role_unknown_entity_or_feature")
            require(bool(role["event_ids"]) and all(e in events and events[e]["observation"]=="observed" for e in role["event_ids"]),"mechanism_role_without_observed_event")
        result[skill]={"status":"model_observed_phases_with_frame_binding","evidence_version":VERSION,
                       "independent_visual_truth":False,"rgb_physics_ground_truth":False}
    return result
