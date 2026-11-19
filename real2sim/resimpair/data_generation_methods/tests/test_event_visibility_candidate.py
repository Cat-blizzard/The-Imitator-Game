import numpy as np
import pytest
import hashlib

from data_generation_methods.human_to_robot_pipeline.audit_multiview_v2 import measured_grasp_mask
from data_generation_methods.r2h_pipeline.visibility_policy_candidate import plan_review, RISK_FIELDS


def test_object_identity_not_arm_index_and_any_robot():
    grasp = np.zeros((4, 2, 3), bool)
    grasp[0, 0, 1] = True
    grasp[1, 1, 1] = True
    grasp[2, 0, 0] = True
    mask = measured_grasp_mask(grasp, frame_count=4, object_count=3, object_index=1, robot_count=2)
    np.testing.assert_array_equal(mask, [True, True, False, False])


def test_ambiguous_legacy_axes_do_not_silently_select_robot():
    g = np.ones((4, 2), bool)
    with pytest.raises(ValueError, match="Ambiguous"):
        measured_grasp_mask(g, frame_count=4, object_count=2, object_index=0)
    assert measured_grasp_mask(g, frame_count=4, object_count=2, object_index=0,
                               axes=["frame", "object"]).all()


def test_conflicting_explicit_native_axes_are_rejected():
    with pytest.raises(ValueError, match="axes conflict"):
        measured_grasp_mask(np.ones((4, 2, 3)), frame_count=4, object_count=3,
                            object_index=1, axes=["frame", "object", "robot"])


@pytest.mark.parametrize("change", ["nan", "rank", "frames", "objects", "index", "robots"])
def test_malformed_grasp_evidence_rejected(change):
    g = np.zeros((4, 2, 3))
    kwargs = dict(frame_count=4, object_count=3, object_index=1, robot_count=2)
    if change == "nan": g[0, 0, 0] = np.nan
    if change == "rank": g = np.zeros(4)
    if change == "frames": kwargs["frame_count"] = 3
    if change == "objects": kwargs["object_count"] = 2
    if change == "index": kwargs["object_index"] = 3
    if change == "robots": kwargs["robot_count"] = 1
    with pytest.raises(ValueError): measured_grasp_mask(g, **kwargs)


def clear_scene():
    return dict(adapter_qualified=True, layout_reviewed=True, **{key: False for key in RISK_FIELDS})


def test_clear_scene_skips_extra_vlm_without_approving_sample():
    r = plan_review(clear_scene())
    assert r["visibility_vlm_check"] == 0
    assert r["preferred_reference_camera"] == "opposite"
    assert r["routing_status"] == "ready"
    assert not r["quality_release_approved"] and not r["paid_call_authorized"]
    assert "actor_structure" in r["mandatory_checks_unchanged"]


@pytest.mark.parametrize("risk", RISK_FIELDS)
def test_named_risks_route_but_do_not_execute(risk):
    s = clear_scene(); s[risk] = True
    r = plan_review(s)
    assert r["visibility_vlm_check"] == 1 and risk in r["reason"]
    assert r["calls_executed"] == 0


def test_missing_risk_metadata_is_not_claimed_clear():
    r = plan_review({})
    assert r["needs_local_risk_review"] and r["routing_status"] == "needs_local_review"


def test_recorded_event_gap_escalates_named_event():
    r = plan_review(clear_scene(), {"grasp": {"cameras": {"opposite": {"frames": 10, "frames_ge16": 7}}}})
    assert r["visibility_vlm_check"] == 1 and r["event_ids"] == ["grasp"]


def test_inconsistent_counts_and_string_flags_rejected():
    with pytest.raises(ValueError): plan_review({"adapter_qualified": "true"})
    with pytest.raises(ValueError):
        plan_review(clear_scene(), {"grasp": {"cameras": {"opposite": {"frames": 10, "frames_ge16": 11}}}})


def test_front_facing_cabinet_override_checks_selected_camera(tmp_path):
    p = tmp_path / "view_review.json"
    p.write_text('Native camera sees the cabinet opening; opposite sees the back.')
    s = clear_scene()
    s["reference_view"] = dict(camera_id="native", reason="Cabinet opening faces native camera",
        evidence=dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    r = plan_review(s, {"open": {"cameras": {
        "native": dict(frames=10, frames_ge16=10), "opposite": dict(frames=10, frames_ge16=0)}}})
    assert r["preferred_reference_camera"] == "native" and not r["visibility_vlm_check"]
    p.write_text("changed")
    with pytest.raises(ValueError, match="evidence"): plan_review(s)


def test_unbound_camera_override_is_rejected():
    s = clear_scene(); s["reference_view"] = dict(camera_id="native", reason="looks better")
    with pytest.raises(ValueError, match="bound evidence"): plan_review(s)
