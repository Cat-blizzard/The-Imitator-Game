"""Explicit synthetic-reference handoff; never changes production readers or data.

``validate_pair`` checks one declared edge, not a goal-derived reference pool.
``inspection`` can read candidate pixels/robot labels but always returns
``cannot_train=True``. ``training`` requires explicit reference AND pair approval.
All functions are read-only and never promote a candidate's state.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

from .common import file_sha256, json_sha256


SCHEMA = "r2h_training_pair_v1"
COLLECTION_SCHEMA = "r2h_training_pair_collection_v1"


def _require(value, message):
    if not value:
        raise ValueError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _strings(value, label):
    _require(isinstance(value, list) and value and all(_text(v) for v in value)
             and len(set(value)) == len(value), "Invalid " + label)
    return value


def _binding(binding, label, verify_files):
    _require(isinstance(binding, dict) and _text(binding.get("path"))
             and isinstance(binding.get("sha256"), str) and len(binding["sha256"]) == 64
             and all(c in "0123456789abcdef" for c in binding["sha256"]),
             "Invalid artifact binding: " + label)
    path = Path(binding["path"])
    _require(path.is_absolute(), "Artifact paths must be absolute: " + label)
    if verify_files:
        _require(path.is_file() and file_sha256(path) == binding["sha256"],
                 "Artifact missing or SHA256 mismatch: " + label)
    return path


def _clock(row, label):
    _require(type(row.get("length")) is int and row["length"] >= 2,
             "Invalid frame count: " + label)
    _require(_number(row.get("fps")) and row["fps"] > 0
             and _number(row.get("start_seconds")) and row["start_seconds"] >= 0,
             "Invalid media clock: " + label)
    start = row["start_seconds"] * row["fps"]
    _require(abs(start - round(start)) < 1e-6, "Window is not on the frame grid: " + label)


def _source_query(pair, verify_files):
    query = pair.get("query")
    _require(isinstance(query, dict), "Missing query")
    row = query.get("episode_row")
    _require(isinstance(row, dict) and _text(row.get("id")), "Missing original query episode row")
    source = _binding(query.get("source_manifest"), "query.source_manifest", verify_files)
    manifest = json.loads(source.read_text())
    matches = [r for r in manifest.get("episodes", []) if r.get("id") == row["id"]]
    _require(len(matches) == 1 and json_sha256(matches[0]) == json_sha256(row),
             "Query episode row differs from its immutable source manifest")
    _require(row.get("domain") == "robot" and row.get("source") == "simulation_expansion"
             and row.get("action_space") == "panda_eef14_v3",
             "Handoff currently requires native simulation_expansion Panda EEF14 queries")
    _require(type(row.get("target_start_offset")) is int and row["target_start_offset"] == 0,
             "Explicit native target_start_offset=0 is required")
    _require(_text(row.get("label_kind")), "Missing native action label definition")
    _clock(row, "query")
    for key in ("goal_key", "group", "scene"):
        _require(_text(row.get(key)), "Missing query " + key)
    spec = manifest.get("action_spaces", {}).get(row["action_space"])
    _require(isinstance(spec, dict) and spec.get("representation") == "eef_relative_v3"
             and spec.get("action_dim") == 14 and spec.get("state_dim") == 18,
             "Source manifest must define unchanged EEF14/state18 actions")
    artifacts = query.get("artifacts")
    _require(isinstance(artifacts, dict), "Missing query artifact bindings")
    video = row.get("video", {})
    _require(isinstance(video, dict) and video.get("kind") == "file",
             "Query video must be an explicit file locator")
    for key, row_path in (("video", video.get("path")), ("arrays", row.get("arrays")),
                          ("receipt", row.get("collection_receipt"))):
        path = _binding(artifacts.get(key), "query." + key, verify_files)
        _require(_text(row_path) and path.resolve() == Path(row_path).resolve(),
                 "Query artifact does not match original row: " + key)
    for row_key, artifact_key in (("source_video_sha256", "video"), ("receipt_sha256", "receipt")):
        if row.get(row_key):
            _require(row[row_key] == artifacts[artifact_key]["sha256"],
                     "Original row artifact hash differs: " + row_key)
    _strings(query.get("lineage_root_ids"), "query lineage_root_ids")
    return row, spec, manifest


def validate_pair(pair, *, mode="inspection", verify_files=True):
    """Validate a single explicit edge; raise ValueError for an invalid handoff.

    Required reference fields: episode_id, source, usage, goal_key, group, split,
    video{path,sha256}, fps/length/start_seconds, allowed_query_episode_ids,
    lineage_root_ids, action_space=None, arrays=None, human_wrist_3d_gt=False,
    quality_release_approved and training_admitted (explicit booleans).
    The pair also carries explicit quality_release_approved/training_admitted.
    """
    _require(mode in ("inspection", "training"), "Unknown validation mode")
    _require(type(verify_files) is bool, "verify_files must be Boolean")
    _require(mode != "training" or verify_files, "Training requires verified artifact files")
    _require(isinstance(pair, dict) and pair.get("schema") == SCHEMA, "Unknown pair schema")
    _require(_text(pair.get("pair_id")), "Missing pair_id")
    _require(pair.get("pair_scope") == "task_only", "Only task_only pairing is supported")
    _require(pair.get("split") in ("train", "validation", "test"), "Invalid explicit split")
    ref = pair.get("reference")
    _require(isinstance(ref, dict) and _text(ref.get("episode_id")), "Missing reference identity")
    _require(ref.get("source") == "synthetic_human" and ref.get("usage") == "reference_only",
             "Reference must be synthetic_human/reference_only")
    _require("action_space" in ref and ref["action_space"] is None
             and "arrays" in ref and ref["arrays"] is None
             and ref.get("human_wrist_3d_gt") is False,
             "Synthetic references must carry no actions, arrays or wrist 3D ground truth")
    for key in ("human_action_retargeting", "time_or_motor_alignment", "frame_alignment"):
        _require(pair.get(key, False) is False, "Task-only pair cannot claim " + key)
    for key in ("goal_key", "group"):
        _require(_text(ref.get(key)), "Missing reference " + key)
    _clock(ref, "reference")
    _binding(ref.get("video"), "reference.video", verify_files)
    _strings(ref.get("lineage_root_ids"), "reference lineage_root_ids")
    _strings(ref.get("allowed_query_episode_ids"), "allowed_query_episode_ids")
    for owner in (pair, ref):
        for key in ("quality_release_approved", "training_admitted"):
            _require(type(owner.get(key)) is bool, "Missing explicit Boolean: " + key)
            if mode == "training":
                _require(owner[key], "Training requires " + key)
    for key in ("task_evidence", "quality_release"):
        if key in ref:
            _binding(ref[key], "reference." + key, verify_files)
    for evidence in ref.get("provenance_artifacts", []):
        _binding(evidence, "reference.provenance_artifact", verify_files)
    if "source_robot" in ref:
        origin = ref["source_robot"]
        path = _binding(origin.get("source_manifest"), "reference.source_robot_manifest", verify_files)
        matches = [r for r in json.loads(path.read_text()).get("episodes", [])
                   if r.get("id") == origin.get("episode_id")]
        _require(len(matches) == 1 and matches[0].get("group") == ref["group"]
                 and matches[0].get("split") == ref.get("split"),
                 "Reference group/split differs from its bound source robot")
        _require(matches[0].get("goal_key") == ref["goal_key"],
                 "Reference goal was relabeled from its source robot")
    row, spec, _ = _source_query(pair, verify_files)
    _require(ref.get("split") == pair["split"] == row.get("split"), "Pair split mismatch")
    _require(row["id"] in ref["allowed_query_episode_ids"],
             "Query is absent from explicit allowed_query_episode_ids")
    if ref["goal_key"] != row["goal_key"]:
        relation = pair.get("goal_relation", {})
        _require(relation.get("type") == "explicit_task_only_equivalence"
                 and relation.get("reference_goal_key") == ref["goal_key"]
                 and relation.get("query_goal_key") == row["goal_key"],
                 "Exact goal mismatch; no automatic alias expansion")
        _require(type(relation.get("training_approved")) is bool,
                 "Explicit goal relation needs a training_approved Boolean")
        _require(isinstance(relation.get("evidence"), list) and relation["evidence"],
                 "Explicit goal relation requires bound evidence")
        for evidence in relation["evidence"]:
            _binding(evidence, "goal_relation.evidence", verify_files)
        _require(mode != "training" or relation["training_approved"],
                 "Training requires independently approved goal equivalence")
    require_cross_demo = pair.get("cross_demo_require_distinct_group", False)
    _require(type(require_cross_demo) is bool, "cross_demo_require_distinct_group must be Boolean")
    if require_cross_demo or row.get("cross_demo_references_require_distinct_group") is True:
        _require(ref["group"] != row["group"], "Cross-demo reference shares query robot group")
    if mode == "training":
        _require(ref.get("reference_pool_eligible") is not False,
                 "Reference explicitly forbids training-pool eligibility")
        _require(row.get("BC_action_labels_verified") is True, "Native query BC labels are not verified")
        _require(row.get("training_admitted") is not False, "Query explicitly forbids training")
    shared = sorted(set(ref["lineage_root_ids"]) & set(pair["query"]["lineage_root_ids"]))
    return {
        "schema": "r2h_pair_handoff_validation_v1", "passed": True, "mode": mode,
        "pair_id": pair["pair_id"], "pair_sha256": json_sha256(pair),
        "reference_episode_id": ref["episode_id"], "query_episode_id": row["id"],
        "pair_scope": "task_only", "explicit_edge_only": True,
        "can_train": mode == "training", "cannot_train": mode != "training",
        "artifact_files_verified": verify_files, "media_decoded": False,
        "reference_clock": {k: ref[k] for k in ("fps", "length", "start_seconds")},
        "query_clock": {k: row[k] for k in ("fps", "length", "start_seconds")},
        "action_definition": {"action_space": row["action_space"],
                              "target_start_offset": row["target_start_offset"],
                              "label_kind": row["label_kind"], "spec_sha256": json_sha256(spec)},
        "shared_lineage_root_ids": shared,
        "independent_robot_group": ref["group"] != row["group"],
        "training_started": False, "source_state_changed": False,
    }


def validate_collection(pairs, *, mode="inspection", verify_files=True):
    """Check only declared edges, identities and transitive lineage split isolation.

    Accept a list of pairs or {schema: COLLECTION_SCHEMA, pairs: [...]}. A shared
    lineage root, reference content hash, reference ID, robot ID or robot group
    cannot occur in two splits. All records for one reference ID must be identical.
    """
    if isinstance(pairs, dict):
        _require(pairs.get("schema") == COLLECTION_SCHEMA, "Unknown collection schema")
        pairs = pairs.get("pairs")
    _require(isinstance(pairs, list) and pairs, "Require a nonempty explicit pair collection")
    reports, ids, edges, identities, split_keys = [], set(), set(), {}, defaultdict(set)
    for pair in pairs:
        report = validate_pair(pair, mode=mode, verify_files=verify_files)
        ref, query, split = pair["reference"], pair["query"], pair["split"]
        row = query["episode_row"]
        _require(pair["pair_id"] not in ids, "Duplicate pair ID")
        ids.add(pair["pair_id"])
        edge = ref["episode_id"], row["id"]
        _require(edge not in edges, "Duplicate explicit reference/query edge")
        edges.add(edge)
        for key, value in ((("reference", ref["episode_id"]), ref),
                           (("query", row["id"]), query)):
            digest = json_sha256(value)
            _require(key not in identities or identities[key] == digest,
                     "Conflicting artifact/metadata for repeated episode identity")
            identities[key] = digest
        roots = set(ref["lineage_root_ids"]) | set(query["lineage_root_ids"])
        keys = [("lineage", r) for r in roots]
        keys += [("reference_content", ref["video"]["sha256"]),
                 ("reference_id", ref["episode_id"]), ("query_id", row["id"]),
                 ("robot_group", row["group"]), ("robot_group", ref["group"]),
                 ("query_arrays_content", query["artifacts"]["arrays"]["sha256"])]
        for key in keys:
            split_keys[key].add(split)
            _require(len(split_keys[key]) == 1, "Cross-split shared lineage or media: " + repr(key))
        reports.append(report)
    return {"schema": "r2h_pair_collection_validation_v1", "passed": True, "mode": mode,
            "explicit_pairs": len(pairs), "unique_references": len({x[0] for x in edges}),
            "unique_robot_queries": len({x[1] for x in edges}),
            "lineage_keys_checked": len(split_keys), "reference_pool_expanded_by_goal": False,
            "can_train": mode == "training", "cannot_train": mode != "training",
            "source_state_changed": False, "training_started": False, "pairs": reports}


def lookup_pair(pairs, reference_episode_id, query_episode_id):
    """Return the sole declared edge; do not derive alternatives from goal names."""
    if isinstance(pairs, dict):
        _require(pairs.get("schema") == COLLECTION_SCHEMA, "Unknown collection schema")
        pairs = pairs.get("pairs")
    _require(isinstance(pairs, list), "Expected explicit pair list")
    found = [p for p in pairs if p.get("reference", {}).get("episode_id") == reference_episode_id
             and p.get("query", {}).get("episode_row", {}).get("id") == query_episode_id]
    _require(len(found) == 1, "Requested edge is missing or duplicated")
    _require(query_episode_id in found[0]["reference"].get("allowed_query_episode_ids", []),
             "Requested edge is not explicitly allowed")
    return found[0]


def reference_cache_key(reference, encoder_settings, *, verify_files=True):
    """Content-addressed identity of the ACTUAL reference media and input recipe.

    encoder_settings requires encoder_id, encoder_revision, frames, image_size,
    preprocessing (a nonempty object), and frame_selection. All settings enter
    the digest, including any caller-added precision/tokenizer configuration.
    """
    _require(isinstance(reference, dict), "Missing reference")
    _binding(reference.get("video"), "reference.video", verify_files)
    _clock(reference, "reference")
    _require(isinstance(encoder_settings, dict), "Missing encoder settings")
    for key in ("encoder_id", "encoder_revision", "frame_selection"):
        _require(_text(encoder_settings.get(key)), "Missing encoder setting: " + key)
    for key in ("frames", "image_size"):
        _require(type(encoder_settings.get(key)) is int and encoder_settings[key] > 0,
                 "Invalid encoder setting: " + key)
    _require(isinstance(encoder_settings.get("preprocessing"), dict)
             and encoder_settings["preprocessing"], "Missing preprocessing settings")
    return json_sha256({"schema": "r2h_reference_cache_identity_v1",
                        "media_sha256": reference["video"]["sha256"],
                        "clip": {key: reference[key] for key in ("start_seconds", "fps", "length")},
                        "encoder_settings": encoder_settings})


def _decode_window(binding, clock, wanted, image_size):
    import av
    import numpy as np

    offset = round(clock["start_seconds"] * clock["fps"])
    global_indices = offset + np.asarray(wanted, dtype=int)
    picks, times = {}, []
    with av.open(binding["path"]) as movie:
        stream = movie.streams.video[0]
        stream.codec_context.thread_count = 1
        _require(stream.average_rate is not None
                 and abs(float(stream.average_rate) - clock["fps"]) < 1e-6,
                 "Decoded FPS differs from declared independent media clock")
        tolerance = max(1e-8, float(stream.time_base) * .51)
        for index, frame in enumerate(movie.decode(video=0)):
            _require(frame.pts is not None, "Missing decoded presentation timestamp")
            times.append(float(frame.pts * frame.time_base))
            if index in global_indices:
                picks[index] = frame.reformat(width=image_size, height=image_size).to_ndarray(format="rgb24")
    _require(len(times) >= offset + clock["length"], "Declared window exceeds decoded media")
    error = float(np.max(np.abs(np.asarray(times) - np.arange(len(times)) / clock["fps"])))
    _require(error <= tolerance + 1e-8, "Non-CFR or shifted media requires explicit PTS mapping")
    _require(len(times) > 1 and np.all(np.diff(times) > 0), "Non-monotonic media timestamps")
    rgb = np.stack([picks[int(i)] for i in global_indices]).transpose(0, 3, 1, 2)
    return rgb, {"decoded_frames": len(times), "max_clock_error_seconds": error,
                 "source_frame_indices": global_indices.tolist(), "full_decode_passed": True}


def read_pair_inputs(pair, *, frames=32, image_size=224, horizon=16, query_index=0):
    """Read candidate RGB and native labels on CPU, without a training loader.

    This is an isolated shape/clock/action preflight, not a quality approval or a
    successful invocation of the old production reader. Returns CPU NumPy input
    arrays and an inspection report. Human references never acquire action labels.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    for name, value in (("frames", frames), ("image_size", image_size), ("horizon", horizon)):
        _require(type(value) is int and value > 0, "Invalid inspection setting: " + name)
    validated = validate_pair(pair, mode="inspection")
    ref, query = pair["reference"], pair["query"]
    row, spec, manifest = _source_query(pair, True)
    _require(type(query_index) is int and 0 <= query_index < row["length"], "Invalid query frame")
    indices = np.rint(np.linspace(0, ref["length"] - 1, frames)).astype(int)
    rgb, reference_clock = _decode_window(ref["video"], ref, indices, image_size)
    obs, query_clock = _decode_window(query["artifacts"]["video"], row, [query_index], image_size)
    with np.load(query["artifacts"]["arrays"]["path"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    n = row["length"]
    for key, shape in (("states", (n, 18)), ("poses", (n, 2, 4, 4)),
                       ("target_poses", (n, 2, 4, 4)), ("native_commands", (n, 16)),
                       ("openness", (n, 2)), ("eef_mask", (n, 14)),
                       ("timestamps_seconds", (n,))):
        _require(key in arrays and arrays[key].shape == shape, "Native array shape mismatch: " + key)
        _require(np.isfinite(arrays[key]).all(), "Nonfinite native array: " + key)
    _require(arrays["eef_mask"].dtype == np.bool_, "Native EEF mask must be Boolean")
    _require(np.allclose(arrays["timestamps_seconds"], np.arange(n) / row["fps"], rtol=0, atol=1e-8),
             "Native array timestamps differ from robot clock")
    t, end = query_index, min(n, query_index + horizon)
    current, targets = arrays["poses"][t], arrays["target_poses"][t:end]
    rt = current[..., :3, :3].swapaxes(-1, -2)
    dp = (rt @ (targets[..., :3, 3] - current[..., :3, 3])[..., None])[..., 0]
    rotation = rt @ targets[..., :3, :3]
    dr = Rotation.from_matrix(rotation.reshape(-1, 3, 3)).as_rotvec().reshape(end-t, 2, 3)
    raw_actions = np.concatenate((dp, dr, arrays["openness"][t:end, ..., None]), axis=-1).reshape(-1, 14)
    mask = arrays["eef_mask"][t:end].copy()
    if "pose_valid" in arrays:
        _require(arrays["pose_valid"].shape == (n, 2) and arrays["pose_valid"].dtype == np.bool_,
                 "Invalid native pose_valid mask")
        mask &= np.repeat(arrays["pose_valid"][t], 7)[None]
    _require(mask.any(), "Inspection query has no valid action targets")
    for kind, dim in (("action", 14), ("state", 18)):
        mean, std = np.asarray(spec[kind + "_mean"]), np.asarray(spec[kind + "_std"])
        _require(mean.shape == std.shape == (dim,) and np.isfinite(mean).all()
                 and np.isfinite(std).all() and (std > 0).all(), "Invalid unchanged source normalizers")
    normalized = (raw_actions - np.asarray(spec["action_mean"])) / np.asarray(spec["action_std"])
    state = (arrays["states"][t] - np.asarray(spec["state_mean"])) / np.asarray(spec["state_std"])
    _require(np.isfinite(normalized[mask]).all() and np.isfinite(state).all(), "Nonfinite normalized labels")
    max_action = max(s["action_dim"] for s in manifest["action_spaces"].values())
    max_state = max(s["state_dim"] for s in manifest["action_spaces"].values())
    actions = np.zeros((horizon, max_action), dtype=np.float32)
    action_mask = np.zeros_like(actions, dtype=bool)
    padded_state = np.zeros(max_state, dtype=np.float32)
    state_mask = np.zeros(max_state, dtype=bool)
    actions[:end-t, :14], action_mask[:end-t, :14] = normalized, mask
    padded_state[:18], state_mask[:18] = state, True
    # Catch changed artifacts during decoding without writing any state or media.
    validate_pair(pair, mode="inspection")
    report = {**validated, "schema": "r2h_pair_cpu_inspection_v1", "media_decoded": True,
            "inspection_adapter": "independent_CPU_no_production_reader_patch",
            "production_reader_exercised": False, "reference_video_shape": list(rgb.shape),
            "obs_shape": list(obs[0].shape), "actions_shape": list(actions.shape),
            "state_shape": list(padded_state.shape), "native_action_dim": 14, "native_state_dim": 18,
            "valid_action_elements": int(mask.sum()), "valid_horizon_steps": end-t,
            "padded_horizon_steps": horizon-(end-t), "action_dt": 1 / row["fps"],
            "query_index": t, "reference_decode": reference_clock, "query_decode": query_clock,
            "frame_alignment": False, "human_wrist_3d_gt": False,
            "can_train": False, "cannot_train": True, "training_started": False}
    return {"reference_video": rgb.astype(np.float32) / 255,
            "obs": obs[0].astype(np.float32) / 255, "actions": actions, "action_mask": action_mask,
            "state": padded_state, "state_mask": state_mask, "report": report}


def inspect_pair_inputs(pair, *, frames=32, image_size=224, horizon=16, query_index=0):
    """Return only the JSON report from the isolated fixed-sample CPU reader."""
    return read_pair_inputs(pair, frames=frames, image_size=image_size,
                            horizon=horizon, query_index=query_index)["report"]
