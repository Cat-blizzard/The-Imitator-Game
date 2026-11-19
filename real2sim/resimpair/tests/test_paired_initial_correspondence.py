import pytest
from resimpair_tasks.paired_initial_correspondence_v2 import CHECKS,validate_observation


def evidence(label):
    return {"image_label":label,"bbox_xyxy_normalized":[.1,.1,.3,.3],"observation":"Visible paired region"}


def answer(value="unknown"):
    return {"checks":[{"check_id":k,"value":value,"evidence":[] if value=="unknown" else [evidence("B"),evidence("C")],
                       "reason":"Insufficient evidence" if value=="unknown" else "Paired comparison"} for k in CHECKS],
            "object_correspondences":[],"limitations":["Synthetic schema fixture, not an image-quality result"]}


def test_unknown_never_passes():
    assert validate_observation(answer())["status"]=="unknown"


def test_visible_contradiction_fails_candidate_scope():
    a=answer("yes");a["checks"][1]["value"]="no"
    result=validate_observation(a)
    assert result["status"]=="fail"
    assert result["default_pipeline_promoted"] is False


def test_single_candidate_image_cannot_establish_pairing():
    a=answer("yes");a["checks"][0]["evidence"]=[evidence("C")]
    with pytest.raises((RuntimeError,ValueError)):validate_observation(a)


def test_reversed_box_rejected():
    a=answer("yes");a["checks"][0]["evidence"][0]["bbox_xyxy_normalized"]=[.3,.3,.1,.1]
    with pytest.raises((RuntimeError,ValueError)):validate_observation(a)
