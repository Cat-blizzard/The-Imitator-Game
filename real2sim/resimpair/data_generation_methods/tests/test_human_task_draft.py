import copy
import pytest
from data_generation_methods.r2h_pipeline.human_task_draft import (
    compile_draft, extraction_prompt, validate_observations)


def sample():
    packet={"window_frames":[0,11],"video":{"frames":11,"path":"secret_task_name.mp4"},
            "frames":[{"frame_index_0based":i,"pts_seconds":i/10} for i in [0,2,5,8,10]]}
    def ev(i):return [{"frame_index":i,"pts_seconds":i/10,"visible_detail":"visible outline"}]
    r={"entities":[{"id":"hand","kind":"hand","visual_description":"a visible hand","identity_evidence":ev(0)},
                   {"id":"box","kind":"container","visual_description":"visible box","identity_evidence":ev(0)}],
       "initial_relations":[{"subject":"box","predicate":"closed","object":"",
                             "observation":"observed","description":"closed panel","evidence":ev(0)}],
       "final_relations":[{"subject":"box","predicate":"open","object":"",
                           "observation":"observed","description":"open panel","evidence":ev(10)}],
       "events":[{"id":"reach","verb":"reach","entity_ids":["hand","box"],"observation":"observed",
                  "temporal_scope":"visible_milestone","description":"hand at box","evidence":ev(2)},
                 {"id":"open","verb":"open","entity_ids":["box"],"observation":"observed",
                  "temporal_scope":"visible_milestone","description":"open panel","evidence":ev(8)}],
       "observed_order":[{"before_event":"reach","after_event":"open","reason":"visible earlier"}],
       "window_assessment":{"scope":"complete_atomic_candidate","begins_before_relevant_action":"yes",
                             "ends_after_visible_release_or_settle":"unknown","boundary_evidence":ev(0)+ev(10),
                             "possible_missing_prefix_or_suffix":"release unclear"},
       "hand_roles":[{"entity_id":"hand","role":"active_manipulator","description":"moves","evidence":ev(2)}],
       "hidden_state_observations":[],"dense_review_requests":[{"first_supplied_frame":2,
                                                             "last_supplied_frame":5,"reason":"need contact"}],
       "uncertainties":["No contact force"]}
    return packet,r


def test_draft_does_not_upgrade_observed_order_or_semantic_approval():
    p,r=sample()
    d=compile_draft(r,p)
    assert d["necessary_precedence"]["status"]=="unknown"
    assert not d["training_admitted"] and not d["manual_semantic_reviewed"]
    assert d["human_sim_equivalence"]=="unverified"


def test_blind_prompt_does_not_include_filename_or_previous_task_text():
    p,_=sample()
    assert "secret_task_name" not in extraction_prompt(p)


@pytest.mark.parametrize("change",["frame","pts","no_evidence"])
def test_unsupported_citations_are_rejected(change):
    p,r=sample()
    if change=="frame":r["events"][0]["evidence"][0]["frame_index"]=3
    if change=="pts":r["events"][0]["evidence"][0]["pts_seconds"]=.3
    if change=="no_evidence":r["events"][0]["evidence"]=[]
    with pytest.raises(ValueError):validate_observations(r,p)


def test_initial_relation_cannot_be_asserted_from_late_reveal():
    p,r=sample()
    r["initial_relations"][0]["evidence"]=copy.deepcopy(r["events"][1]["evidence"])
    with pytest.raises(ValueError):validate_observations(r,p)


def test_order_cannot_reverse_actual_evidence():
    p,r=sample()
    r["observed_order"]=[{"before_event":"open","after_event":"reach","reason":"guess"}]
    with pytest.raises(ValueError):validate_observations(r,p)


def test_static_support_hand_is_preserved():
    p,r=sample()
    r["hand_roles"][0]["role"]="support_or_stabilize"
    assert compile_draft(r,p)["observations"]["hand_roles"][0]["role"]=="support_or_stabilize"


def test_partial_window_remains_partial():
    p,r=sample()
    r["window_assessment"]["scope"]="partial_action"
    assert compile_draft(r,p)["observations"]["window_assessment"]["scope"]=="partial_action"


def test_hand_role_cannot_reference_nonhand_entity():
    p,r=sample()
    r["hand_roles"][0]["entity_id"]="box"
    with pytest.raises(ValueError):validate_observations(r,p)
