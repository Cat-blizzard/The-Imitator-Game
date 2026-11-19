"""Fail-closed handoff and independent expansion admission; no API clients."""
from copy import deepcopy
import json

import pytest

from resimpair_tasks import human_pilot_v1 as pilot


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return pilot.bind(path)


@pytest.fixture
def qualified(tmp_path):
    request = put(tmp_path / "requested.json", {"tasks": ["bound-before-execution"]})
    original = {"planning_failure": None, "native_execution": True, "case": {"source_request": request},
                "measurements": {"measured": True}, "event_protocol": {}, "task_intent": {}, "scene": {}, "cameras": {}}
    original_ref = put(tmp_path / "execution.json", original)
    bridge = put(tmp_path / "bridge.json", {**original, "original_execution_bundle": original_ref})
    native = put(tmp_path / "native.json", {"passed": True, "bundle": original_ref})
    container = put(tmp_path / "container.json", {"passed": True, "retained_audit": native})
    requested = put(tmp_path / "requested_audit.json", {"passed": True, "native_audit": container, "source_request": request})
    path = tmp_path / "export/contract.json"
    contract = put(path, {"execution": bridge, "source_mapping": {"formal_pair_qualified": False, "source_status": "missing"}})
    provenance = {"contract": contract, "bridge": bridge, "container_acceptance": container,
                  "requested_goal_acceptance": requested}
    put(path.parent / "general_pickplace_provenance.json", provenance)
    return path, provenance


def test_qualified_receipts_are_bound_to_original_execution(qualified):
    path, _ = qualified
    result = pilot.qualify_contract(path)
    assert result["original_execution"]["path"].endswith("execution.json")
    assert result["requested_goal_audit"]


def test_pass_flag_does_not_bypass_wrong_execution(qualified):
    path, provenance = qualified
    root = path.parent.parent
    wrong = put(root / "other_execution.json", {"other": True})
    native = put(root / "native.json", {"passed": True, "bundle": wrong})
    container = put(root / "container.json", {"passed": True, "retained_audit": native})
    provenance["container_acceptance"] = container
    put(path.parent / "general_pickplace_provenance.json", provenance)
    with pytest.raises(ValueError, match="native_audit_execution_mismatch"):
        pilot.qualify_contract(path)


@pytest.mark.parametrize("missing", [True, False])
def test_requested_goal_missing_or_failed_cannot_reach_generation(qualified, missing):
    path, provenance = qualified
    if missing:
        provenance["requested_goal_acceptance"] = None
    else:
        p = pilot.resolve(provenance["requested_goal_acceptance"])
        data = pilot.read(p); data["passed"] = False
        provenance["requested_goal_acceptance"] = put(p, data)
    put(path.parent / "general_pickplace_provenance.json", provenance)
    with pytest.raises(ValueError, match="requested_goal"):
        pilot.qualify_contract(path)


def test_bound_native_bytes_changed_are_rejected(qualified):
    path, provenance = qualified
    pilot.resolve(provenance["bridge"]).write_text('{}')
    with pytest.raises(ValueError, match="hash changed"):
        pilot.qualify_contract(path)


def test_registration_cannot_exceed_campaign_ceiling(tmp_path):
    with pytest.raises(ValueError, match="finite_pilot_call_limits"):
        pilot.register(tmp_path / "new", {"image": 3, "video": 7, "review": 18}, 60)
    assert not (tmp_path / "new").exists()


@pytest.fixture
def independent(tmp_path):
    frame = tmp_path / "candidate.jpg"; frame.write_bytes(b'bounded-test-frame')
    video = tmp_path / "video.mp4"; video.write_bytes(b'bounded-test-video')
    packet = put(tmp_path / "packet.json", {"video": pilot.bind(video), "decode_complete": True,
                                           "frames": [{"image": pilot.bind(frame)}]})
    result = {"phase": "qualification", "status": "terminal", "rows": [
        {"key": key, "status": "sampled_visual_pass", "video": pilot.bind(video)} for key in ("task_a", "task_b")]}
    spot = {"schema": "resimpair_human_independent_spotcheck_v1", "status": "pass",
            "major_conflicts_found": False, "observer": "independent_test_observer",
            "independent_of_generation": True, "records": [
                {"key": row["key"], "video": row["video"], "frame_packet": packet, "checks": {
                    key: {"status": "pass", "candidate_frames": [pilot.bind(frame)], "observation": "test-only observation"}
                    for key in pilot.SPOT_CHECKS}} for row in result["rows"]]}
    return result, spot


def test_generator_pass_alone_is_insufficient_for_expansion(independent):
    result, spot = independent
    assert pilot.expansion_qualified(result, spot)
    with pytest.raises(ValueError, match="independent_spotcheck_required"):
        pilot.expansion_qualified(result, {})


@pytest.mark.parametrize("dimension", pilot.SPOT_CHECKS)
def test_unknown_independent_check_blocks_expansion(independent, dimension):
    result, spot = independent
    spot = deepcopy(spot)
    spot["records"][0]["checks"][dimension]["status"] = "unknown"
    with pytest.raises(ValueError, match="missing_evidence_or_unknown"):
        pilot.expansion_qualified(result, spot)


def test_wrong_video_spotcheck_does_not_qualify(independent):
    result, spot = independent
    spot["records"][0]["video"] = {"path": "some_other_video", "sha256": "wrong"}
    with pytest.raises(ValueError, match="video_mismatch"):
        pilot.expansion_qualified(result, spot)


def test_rejected_or_pending_qualification_cannot_expand(independent):
    result, spot = independent
    result["rows"][1]["status"] = "generated_video_review_unknown"
    with pytest.raises(ValueError, match="both_complete_passing"):
        pilot.expansion_qualified(result, spot)


def test_unrelated_frame_cannot_support_candidate_claim(independent, tmp_path):
    result, spot = independent
    other = tmp_path / "wrong.jpg"; other.write_bytes(b'unrelated-frame')
    spot["records"][0]["checks"]["one_operator"]["candidate_frames"] = [pilot.bind(other)]
    with pytest.raises(ValueError, match="frame_not_from_candidate"):
        pilot.expansion_qualified(result, spot)


def test_task_name_and_timing_variants_do_not_create_new_task():
    info = {"required_events": [{"id": "grasp_a", "action": "grasp", "entities": ["box"],
                                "human_hand": "left", "effector": "arm1", "suggested_seconds": 1.0}],
            "required_event_precedence": [], "final_relations": [{"id": "goal_a", "relation": "supported_by",
                "entities": ["box", "tray"], "parameters": {}}], "terminal_hands": {}, "persistent_constraints": []}
    renamed = deepcopy(info)
    renamed["required_events"][0].update(id="grasp_new_name", suggested_seconds=3.0)
    renamed["final_relations"][0]["id"] = "goal_new_name"
    assert pilot.semantic_task_signature(info) == pilot.semantic_task_signature(renamed)
    renamed["final_relations"][0]["entities"][1] = "other_tray"
    assert pilot.semantic_task_signature(info) != pilot.semantic_task_signature(renamed)
