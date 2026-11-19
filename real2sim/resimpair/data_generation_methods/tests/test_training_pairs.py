"""Small temporary artifacts exercise handoff guards; no training or API calls."""
from copy import deepcopy
import json

import numpy as np
import pytest

from data_generation_methods.r2h_pipeline.common import file_sha256
from data_generation_methods.r2h_pipeline.training_pairs import (
    COLLECTION_SCHEMA, SCHEMA, inspect_pair_inputs, lookup_pair,
    read_pair_inputs, reference_cache_key, validate_collection, validate_pair,
)


def bind(path):
    return {"path": str(path), "sha256": file_sha256(path)}


def write_source(pair, path=None):
    binding = pair["query"]["source_manifest"]
    from pathlib import Path
    target = Path(path or binding["path"])
    old = json.loads(Path(binding["path"]).read_text()) if Path(binding["path"]).exists() else {}
    old["episodes"] = [deepcopy(pair["query"]["episode_row"])]
    target.write_text(json.dumps(old))
    pair["query"]["source_manifest"] = bind(target)


@pytest.fixture
def pair(tmp_path):
    n = 5
    video = tmp_path / "robot.mp4"
    video.write_bytes(b"not-decoded-until-inspection")
    human = tmp_path / "human.mp4"
    human.write_bytes(b"independent-reference-fixture")
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"fixture": true}')
    poses = np.broadcast_to(np.eye(4), (n, 2, 4, 4)).copy()
    poses[:, :, 0, 3] = np.arange(n)[:, None] * .01
    target = poses.copy()
    target[:, :, 1, 3] = .02
    arrays = tmp_path / "arrays.npz"
    np.savez(arrays, states=np.zeros((n, 18)), poses=poses, target_poses=target,
             native_commands=np.zeros((n, 16)), openness=np.ones((n, 2)),
             eef_mask=np.ones((n, 14), dtype=bool), timestamps_seconds=np.arange(n)/20,
             pose_valid=np.ones((n, 2), dtype=bool))
    row = {
        "id": "robot-1", "domain": "robot", "source": "simulation_expansion",
        "split": "train", "goal_key": "fixture:move", "group": "robot-layout-1", "scene": "fixture",
        "action_space": "panda_eef14_v3", "target_start_offset": 0,
        "label_kind": "FK_of_executed_native_PD_joint_targets", "length": n, "fps": 20.,
        "start_seconds": 0., "video": {"kind": "file", "path": str(video)},
        "arrays": str(arrays), "collection_receipt": str(receipt),
        "source_video_sha256": file_sha256(video), "receipt_sha256": file_sha256(receipt),
        "BC_action_labels_verified": True,
    }
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(json.dumps({
        "episodes": [row],
        "action_spaces": {
            "panda_eef14_v3": {"representation": "eef_relative_v3", "action_dim": 14, "state_dim": 18,
                              "action_mean": [0.] * 14, "action_std": [1.] * 14,
                              "state_mean": [0.] * 18, "state_std": [1.] * 18},
            "human20": {"action_dim": 20, "state_dim": 20},
        },
    }))
    return {
        "schema": SCHEMA, "pair_id": "fixture-pair", "pair_scope": "task_only", "split": "train",
        "quality_release_approved": False, "training_admitted": False,
        "reference": {
            "episode_id": "synthetic-1", "source": "synthetic_human", "usage": "reference_only",
            "split": "train", "goal_key": "fixture:move", "group": "robot-layout-2",
            "video": bind(human), "fps": 24., "length": n, "start_seconds": 0.,
            "action_space": None, "arrays": None, "human_wrist_3d_gt": False,
            "allowed_query_episode_ids": ["robot-1"], "lineage_root_ids": ["real:session-1", "robot:2"],
            "quality_release_approved": False, "training_admitted": False,
        },
        "query": {"episode_row": row, "source_manifest": bind(manifest),
                  "lineage_root_ids": ["real:session-1", "robot:1"],
                  "artifacts": {"video": bind(video), "arrays": bind(arrays), "receipt": bind(receipt)}},
    }


def test_inspection_is_readonly_and_cannot_promote(pair):
    before = deepcopy(pair)
    result = validate_pair(pair)
    assert result["passed"] and result["cannot_train"] and not result["can_train"]
    assert not result["source_state_changed"] and not result["media_decoded"]
    assert result["reference_clock"]["fps"] == 24 and result["query_clock"]["fps"] == 20
    assert pair == before


def test_training_requires_both_approval_flags_at_both_levels(pair):
    for owner in (pair, pair["reference"]):
        owner["quality_release_approved"] = owner["training_admitted"] = True
    assert validate_pair(pair, mode="training")["can_train"]
    for name in (None, "reference"):
        for flag in ("quality_release_approved", "training_admitted"):
            bad = deepcopy(pair)
            owner = bad if name is None else bad[name]
            owner[flag] = False
            with pytest.raises(ValueError, match="Training requires"):
                validate_pair(bad, mode="training")
    with pytest.raises(ValueError, match="verified artifact"):
        validate_pair(pair, mode="training", verify_files=False)
    assert validate_pair(pair, mode="inspection")["cannot_train"]


@pytest.mark.parametrize("field,value", [
    ("source", "egodex"), ("usage", "actions"), ("action_space", "panda_eef14_v3"),
    ("arrays", "/tmp/borrowed_robot.npz"), ("human_wrist_3d_gt", True),
])
def test_human_action_or_source_laundering_is_rejected(pair, field, value):
    pair["reference"][field] = value
    with pytest.raises(ValueError):
        validate_pair(pair)


def test_explicit_reference_ineligibility_survives_other_approval_flags(pair):
    for owner in (pair, pair["reference"]):
        owner["quality_release_approved"] = owner["training_admitted"] = True
    pair["reference"]["reference_pool_eligible"] = False
    with pytest.raises(ValueError, match="explicitly forbids"):
        validate_pair(pair, mode="training")


@pytest.mark.parametrize("field,value", [
    ("quality_release_approved", 1), ("training_admitted", None),
    ("length", True), ("fps", float("nan")), ("start_seconds", .01),
])
def test_strict_boolean_and_clock_types(pair, field, value):
    pair["reference"][field] = value
    with pytest.raises(ValueError):
        validate_pair(pair)


def test_no_goal_based_expansion_or_hidden_alignment(pair):
    pair["reference"]["allowed_query_episode_ids"] = ["different_robot_same_goal"]
    with pytest.raises(ValueError, match="explicit allowed"):
        validate_pair(pair)
    with pytest.raises(ValueError, match="missing or duplicated"):
        lookup_pair([pair], "synthetic-1", "different_robot_same_goal")
    pair["reference"]["allowed_query_episode_ids"] = ["robot-1"]
    assert lookup_pair([pair], "synthetic-1", "robot-1") is pair
    pair["frame_alignment"] = True
    with pytest.raises(ValueError, match="cannot claim"):
        validate_pair(pair)


def test_original_robot_row_and_offset_cannot_drift(pair):
    pair["query"]["episode_row"]["fps"] = 24.
    with pytest.raises(ValueError, match="differs from its immutable"):
        validate_pair(pair)
    pair["query"]["episode_row"]["fps"] = 20.
    del pair["query"]["episode_row"]["target_start_offset"]
    write_source(pair)
    with pytest.raises(ValueError, match="target_start_offset=0"):
        validate_pair(pair)


def test_artifact_binding_must_match_row_and_content(pair):
    pair["query"]["artifacts"]["arrays"] = deepcopy(pair["reference"]["video"])
    with pytest.raises(ValueError, match="does not match"):
        validate_pair(pair)


def test_file_hash_mismatch_cannot_be_approved(pair):
    pair["reference"]["video"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_pair(pair)
    assert validate_pair(pair, verify_files=False)["cannot_train"]


def test_exact_goal_and_split_match_required(pair):
    pair["reference"]["goal_key"] += "_other_asset"
    with pytest.raises(ValueError, match="Exact goal mismatch"):
        validate_pair(pair)
    pair["reference"]["goal_key"] = "fixture:move"
    pair["reference"]["split"] = "validation"
    with pytest.raises(ValueError, match="split mismatch"):
        validate_pair(pair)


def test_explicit_cross_asset_relation_is_inspection_only_until_approved(pair):
    pair["reference"]["goal_key"] = "fixture:move_original_asset"
    pair["goal_relation"] = {
        "type": "explicit_task_only_equivalence",
        "reference_goal_key": "fixture:move_original_asset", "query_goal_key": "fixture:move",
        "training_approved": False, "evidence": [pair["query"]["source_manifest"]],
    }
    assert validate_pair(pair)["cannot_train"]
    for owner in (pair, pair["reference"]):
        owner["quality_release_approved"] = owner["training_admitted"] = True
    with pytest.raises(ValueError, match="approved goal equivalence"):
        validate_pair(pair, mode="training")
    pair["goal_relation"]["training_approved"] = True
    assert validate_pair(pair, mode="training")["can_train"]


def test_cross_demo_requirement_rejects_source_robot_group(pair):
    pair["cross_demo_require_distinct_group"] = True
    pair["reference"]["group"] = pair["query"]["episode_row"]["group"]
    with pytest.raises(ValueError, match="Cross-demo"):
        validate_pair(pair)


def test_collection_detects_duplicate_edges_and_conflicting_reference(pair, tmp_path):
    with pytest.raises(ValueError, match="Duplicate pair ID"):
        validate_collection([pair, deepcopy(pair)])
    second = deepcopy(pair)
    second["pair_id"] = "pair-2"
    with pytest.raises(ValueError, match="Duplicate explicit"):
        validate_collection([pair, second])
    second["query"]["episode_row"]["id"] = "robot-3"
    second["query"]["episode_row"]["group"] = "robot-layout-3"
    second["reference"]["allowed_query_episode_ids"] = ["robot-3"]
    write_source(second, tmp_path / "second_manifest.json")
    with pytest.raises(ValueError, match="Conflicting artifact"):
        validate_collection([pair, second])


def test_shared_lineage_cannot_cross_splits(pair, tmp_path):
    second = deepcopy(pair)
    second["pair_id"] = "validation-pair"
    second["split"] = second["reference"]["split"] = second["query"]["episode_row"]["split"] = "validation"
    second["reference"]["episode_id"] = "different_generated_retry"
    second["reference"]["group"] = "new-group-name"
    second["query"]["episode_row"]["id"] = "robot-validation"
    second["query"]["episode_row"]["group"] = "robot-layout-validation"
    second["reference"]["allowed_query_episode_ids"] = ["robot-validation"]
    write_source(second, tmp_path / "validation_manifest.json")
    with pytest.raises(ValueError, match="Cross-split shared lineage"):
        validate_collection([pair, second])
    result = validate_collection({"schema": COLLECTION_SCHEMA, "pairs": [pair]})
    assert result["explicit_pairs"] == 1 and result["cannot_train"]
    assert not result["reference_pool_expanded_by_goal"]


def settings():
    return {"encoder_id": "fixture", "encoder_revision": "sha256:encoder_weights",
            "frames": 32, "image_size": 224, "frame_selection": "rounded_full_clip_linspace_v1",
            "preprocessing": {"resize": "av_reformat", "color": "RGB", "range": "[0,1]"}}


def test_cache_includes_content_window_and_all_preprocessing(pair):
    ref, encoder = pair["reference"], settings()
    original = reference_cache_key(ref, encoder)
    changed = deepcopy(ref)
    changed["video"]["sha256"] = "b" * 64
    assert reference_cache_key(changed, encoder, verify_files=False) != original
    for key, value in (("start_seconds", 1 / 24), ("length", 4), ("fps", 30.)):
        changed = deepcopy(ref)
        changed[key] = value
        assert reference_cache_key(changed, encoder) != original
    for key, value in (("encoder_revision", "new-weights"), ("frames", 16), ("image_size", 256),
                       ("preprocessing", {"resize": "center_crop"}), ("precision", "float16")):
        changed_encoder = {**encoder, key: value}
        assert reference_cache_key(ref, changed_encoder) != original


def make_video(path, fps, n=5):
    av = pytest.importorskip("av")
    with av.open(str(path), mode="w") as movie:
        stream = movie.add_stream("libx264", rate=fps)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(n):
            rgb = np.full((32, 32, 3), 20 + i * 30, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                movie.mux(packet)
        for packet in stream.encode():
            movie.mux(packet)


def install_media(pair):
    from pathlib import Path
    video = Path(pair["query"]["artifacts"]["video"]["path"])
    human = Path(pair["reference"]["video"]["path"])
    make_video(video, 20)
    make_video(human, 24)
    pair["query"]["artifacts"]["video"] = bind(video)
    pair["query"]["episode_row"]["source_video_sha256"] = file_sha256(video)
    pair["reference"]["video"] = bind(human)
    write_source(pair)


def test_cpu_inspection_preserves_independent_clocks_and_last_horizon_mask(pair):
    install_media(pair)
    before = deepcopy(pair)
    result = inspect_pair_inputs(pair, frames=4, image_size=32, horizon=16, query_index=4)
    assert result["reference_video_shape"] == [4, 3, 32, 32]
    assert result["obs_shape"] == [3, 32, 32]
    assert result["actions_shape"] == [16, 20] and result["state_shape"] == [20]
    assert result["valid_horizon_steps"] == 1 and result["padded_horizon_steps"] == 15
    assert result["valid_action_elements"] == 14 and result["action_dt"] == .05
    assert result["reference_decode"]["source_frame_indices"] == [0, 1, 3, 4]
    assert not result["production_reader_exercised"]
    assert result["cannot_train"] and pair == before


def test_cpu_inspection_rejects_wrong_media_clock(pair):
    install_media(pair)
    pair["reference"]["fps"] = 20.
    with pytest.raises(ValueError, match="Decoded FPS differs"):
        inspect_pair_inputs(pair)


def test_fixed_reader_returns_native_labels_and_zero_padding(pair):
    install_media(pair)
    sample = read_pair_inputs(pair, frames=4, image_size=32, horizon=3, query_index=4)
    np.testing.assert_allclose(sample["actions"][0, :14],
                               [0., .02, 0., 0., 0., 0., 1.] * 2, atol=1e-7)
    assert not sample["actions"][1:].any()
    assert not sample["action_mask"][1:].any()
    assert not sample["action_mask"][:, 14:].any()
    assert sample["state_mask"][:18].all() and not sample["state_mask"][18:].any()
    assert sample["report"]["cannot_train"]


def test_cpu_inspection_rejects_out_of_bounds_reference_window(pair):
    install_media(pair)
    pair["reference"]["length"] = 6
    with pytest.raises(ValueError, match="window exceeds"):
        inspect_pair_inputs(pair)
