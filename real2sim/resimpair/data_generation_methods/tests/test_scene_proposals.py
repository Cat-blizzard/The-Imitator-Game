import numpy as np
import pytest
from data_generation_methods.r2h_pipeline.scene_proposals import (
    apply_correction, hidden_state_candidates, project_world_bounds,
    register_images, transfer_box, validate_box, verify_proposal_bindings)
from data_generation_methods.r2h_pipeline.common import json_sha256, file_sha256


def test_hidden_state_uses_sustained_visible_pixels_not_category_words():
    pixels = np.array([[0, 10, 0], [0, 10, 6], [5, 10, 0],
                       [9, 10, 0], [12, 10, 0]])
    rows = hidden_state_candidates(pixels, np.ones_like(pixels, dtype=bool))
    assert [r["hidden_then_revealed"] for r in rows] == [True, False, False]
    assert rows[0]["first_sustained_reveal_frame"] == 2


def test_offscreen_is_not_hidden_occlusion():
    counts = np.array([[0], [10], [10], [10]])
    frame = np.array([[False], [True], [True], [True]])
    assert not hidden_state_candidates(counts, frame)[0]["hidden_then_revealed"]


def test_projection_uses_positive_camera_z_and_never_infers_visibility():
    e = np.c_[np.eye(3), np.zeros(3)]
    k = np.array([[100,0,100], [0,100,100], [0,0,1]])
    result = project_world_bounds([[-.1,-.1,1],[.1,.1,1.2]], e, k, 200, 200)
    np.testing.assert_allclose(result["box_xyxy_normalized"], [.45,.45,.55,.55])
    assert not result["visible_surface_verified"]
    assert project_world_bounds([[-.1,-.1,-1],[.1,.1,1]], e, k, 200, 200)["status"] == "unknown"


@pytest.mark.parametrize("box", [[0,0,0,.5], [-.1,0,.5,.5], [0,0,np.nan,1], [0,1,1,0]])
def test_invalid_boxes_rejected(box):
    with pytest.raises(ValueError):
        validate_box(box)


def test_no_texture_registration_remains_unknown():
    image = np.zeros((100,100,3), dtype=np.uint8)
    assert register_images(image, image)["status"] == "unknown"


def test_geometry_transfer_is_only_candidate_with_explicit_vlm_request():
    reg = {"status": "candidate", "homography": [[2,0,0],[0,2,0],[0,0,1]],
           "source_inlier_pixels": [[50,50]]}
    r = transfer_box([.4,.4,.6,.6], reg, (100,100,3), (200,200,3), padding=0)
    assert r["box_xyxy_normalized"] == pytest.approx([.4,.4,.6,.6])
    assert r["requires_vlm_correction"]
    assert not r["semantic_identity_verified"]


def make_patch():
    p = {"target": {"frame": {"sha256": "target"},
                   "roi_candidates": [{"id": "h", "role": "interaction_target",
                                       "status": "candidate",
                                       "box_xyxy_normalized": [.1,.2,.3,.4]}]}}
    patch = {"proposal_sha256": json_sha256(p), "target_frame_sha256": "target",
             "decisions": [{"roi_id":"h","status":"correct",
                            "box_xyxy_normalized":[.2,.3,.4,.5],"reason":"visible handle outline"}]}
    return p, patch


def test_correction_preserves_raw_proposal_and_cannot_approve_contact():
    p, patch = make_patch()
    original = json_sha256(p)
    r = apply_correction(p, patch)
    assert json_sha256(p) == original
    assert not r["physical_contact_approved"] and not r["training_admitted"]
    assert r["roi_candidates"][0]["box_xyxy_normalized"] == [.2,.3,.4,.5]


@pytest.mark.parametrize("field", ["proposal_sha256", "target_frame_sha256"])
def test_stale_correction_bindings_rejected(field):
    p, patch = make_patch()
    patch[field] = "stale"
    with pytest.raises(ValueError):
        apply_correction(p, patch)


def test_patch_cannot_add_unrelated_roi_or_duplicate_decision():
    p, patch = make_patch()
    patch["decisions"] *= 2
    with pytest.raises(ValueError):
        apply_correction(p, patch)


def test_unknown_roi_cannot_be_kept():
    p, patch = make_patch()
    p["target"]["roi_candidates"][0]["status"] = "unknown"
    patch["proposal_sha256"] = json_sha256(p)
    patch["decisions"][0]["status"] = "keep"
    with pytest.raises(ValueError):
        apply_correction(p, patch)


def test_changed_source_artifact_is_rejected_before_review(tmp_path):
    path = tmp_path/"receipt.json"
    path.write_text('{"objects": 1}')
    proposal = {"bindings": {"receipt": {"path": str(path), "sha256": file_sha256(path)}},
                "source_keyframes": [], "target": None}
    assert verify_proposal_bindings(proposal)
    path.write_text('{"objects": 2}')
    with pytest.raises(ValueError):
        verify_proposal_bindings(proposal)
