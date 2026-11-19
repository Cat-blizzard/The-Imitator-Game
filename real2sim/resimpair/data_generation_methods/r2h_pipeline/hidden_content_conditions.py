"""CPU-only hidden cabinet content preparation; no provider submission route.

The simulator's object identity, visibility and geometry are separate from RGB
recognizability and generated-image identity. Old evidence and thresholds remain
unchanged. This narrow candidate accepts the historical right-door-open episode.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np
import trimesh

from .common import file_sha256, write_json
from .scene_proposals import (binding, checked_path, frame_at, hidden_state_candidates,
                              project_world_bounds, verify_proposal_bindings)
from .sim_state_conditions import verify_camera_geometry, world_vertices

SCHEMA = "r2h_hidden_cabinet_content_conditions_v2"
REVIEW_SCHEMA = "r2h_hidden_content_rgb_observations_v2"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def bound_path(value):
    require(isinstance(value, dict) and len(value.get("sha256", "")) == 64,
            "Complete SHA256 evidence binding required")
    return checked_path(value).resolve(strict=True)


def visibility_events(counts, timestamps, *, frame_indices=None):
    """Instance-pixel observations, never a classifier for object recognition."""
    counts, timestamps = np.asarray(counts), np.asarray(timestamps)
    indices = np.arange(len(counts)) if frame_indices is None else np.asarray(frame_indices)
    require(counts.ndim == timestamps.ndim == indices.ndim == 1 and len(counts) >= 3
            and len(counts) == len(timestamps) == len(indices), "Visibility timeline shape invalid")
    require(np.isfinite(counts).all() and (counts >= 0).all()
            and np.isfinite(timestamps).all() and (np.diff(timestamps) > 0).all()
            and np.issubdtype(indices.dtype, np.integer) and (np.diff(indices) > 0).all(),
            "Visibility timeline values invalid")
    hits = np.flatnonzero(counts > 0)
    starts = [i for i in range(len(counts)-2) if (counts[i:i+3] >= 16).all()
              and (np.diff(indices[i:i+3]) == 1).all()]
    i = int(hits[0]) if len(hits) else None
    return {
        "first_any_instance_pixels_frame": int(indices[i]) if i is not None else None,
        "first_any_instance_pixels_count": int(counts[i]) if i is not None else None,
        "first_observed_transition_interval_frames": [int(indices[i-1]), int(indices[i])]
        if i is not None and i > 0 else None,
        "first_any_is_exact_recorded_frame": bool(i is not None and indices[0] == 0
                                                  and (np.diff(indices[:i+1]) == 1).all()),
        "sustained_candidate_frame": int(indices[starts[0]]) if starts else None,
        "sustained_rule": {"minimum_instance_pixels": 16, "consecutive_recorded_frames": 3,
                           "purpose": "engineering reference selection; not semantic recognition"},
        "first_rgb_recognizable_frame": "unknown",
        "first_rgb_recognizable_reason": "Instance segmentation does not prove RGB recognizability",
    }


def vertical_surface_hits(mesh, xy):
    """Exact vertical triangle intersections, avoiding ray-library/GPU dependencies."""
    tri = np.asarray(mesh.triangles)
    a = tri[:, 0, :2]
    b, c, d = tri[:, 1, :2]-a, tri[:, 2, :2]-a, np.asarray(xy)-a
    den = b[:, 0]*c[:, 1]-b[:, 1]*c[:, 0]
    valid = abs(den) > 1e-12
    u, v = np.zeros(len(tri)), np.zeros(len(tri))
    u[valid] = (d[valid, 0]*c[valid, 1]-d[valid, 1]*c[valid, 0])/den[valid]
    v[valid] = (b[valid, 0]*d[valid, 1]-b[valid, 1]*d[valid, 0])/den[valid]
    valid &= (u >= -1e-8) & (v >= -1e-8) & (u+v <= 1+1e-8)
    z = tri[:, 0, 2]+u*(tri[:, 1, 2]-tri[:, 0, 2])+v*(tri[:, 2, 2]-tri[:, 0, 2])
    return np.unique(np.round(z[valid], 10))


def _load(proposal_path, verification_path, object_index):
    proposal = json.loads(Path(proposal_path).read_text())
    verify_proposal_bindings(proposal)
    b = proposal["bindings"]
    receipt_path = bound_path(b["receipt"])
    receipt = json.loads(receipt_path.read_text())
    verification = json.loads(Path(verification_path).read_text())
    require(verification["receipt"] == binding(receipt_path), "Physical review belongs to another receipt")
    raw = verification["raw_verifier"]
    require(raw.get("passed") is True and raw["receipt_sha256"] == file_sha256(receipt_path)
            and all(v is True for v in raw["physical_checks"].values()),
            "Historical cabinet physical review is missing or failed")
    require(receipt.get("task") == "cabinet_open_right_only"
            and receipt.get("full_insert_remove_close_task_completed") is False,
            "Only the separately reviewed right-door-open task is supported")
    bindings = {"proposal": binding(proposal_path), "verification": binding(verification_path),
                "receipt": binding(receipt_path)}
    keys = {"arrays": (receipt["arrays"], receipt["sha256"]["arrays.npz"]),
            "video": (receipt["reference_video"]["path"], receipt["reference_video_sha256"]),
            "visibility": (receipt["qa_visibility"], receipt["qa_visibility_sha256"]),
            "articulation": (receipt["articulation_data"], receipt["articulation_data_sha256"]),
            "cabinet_adapter": (receipt["cabinet_adapter"], receipt["cabinet_adapter_sha256"])}
    for key, (path, digest) in keys.items():
        bindings[key] = {"path": path, "sha256": digest}
        bound_path(bindings[key])
        if key in b:
            require(bound_path(b[key]) == Path(path).resolve() and b[key]["sha256"] == digest,
                    "Proposal source binding disagrees with receipt: " + key)
    arrays = dict(np.load(bindings["arrays"]["path"], allow_pickle=False))
    qa = dict(np.load(bindings["visibility"]["path"], allow_pickle=False))
    art = dict(np.load(bindings["articulation"]["path"], allow_pickle=False))
    n = len(arrays["timestamps_seconds"])
    require(n == receipt["length"] and n >= 3, "Receipt/array length mismatch")
    require(arrays["object_states"].shape == (n, len(receipt["asset_specs"]), 13),
            "Measured object state shape mismatch")
    require(type(object_index) is int and 0 <= object_index < len(receipt["asset_specs"]),
            "Unknown object index")
    for field in ["object_pixels_512", "object_pixels_224", "projected_bounds_224",
                  "projected_entirely_in_frame", "visible_bounds_224"]:
        require(len(qa[field]) == n, "Sparse QA cannot be treated as complete frame-aligned QA")
    require(art["cabinet_qpos"].shape == (n, 2), "Articulation alignment mismatch")
    require(np.isfinite(arrays["object_states"]).all()
            and np.isfinite(art["cabinet_qpos"]).all(), "Nonfinite measured state")
    np.testing.assert_allclose(arrays["timestamps_seconds"], np.arange(n)/receipt["fps"], atol=1e-8, rtol=0)
    video = bindings["video"]
    known = [x for x in raw["full_videos"] if x["sha256"] == video["sha256"]]
    require(len(known) == 1 and known[0]["decoded_frames"] == n and known[0]["all_PTS_verified"],
            "Video is outside original cabinet verification")
    with av.open(video["path"]) as container:
        stream = container.streams.video[0]
        decoded = 0
        for i, frame in enumerate(container.decode(stream)):
            require(i < n and frame.pts is not None and frame.width == frame.height == 512,
                    "Native video shape or frame count mismatch")
            require(abs(float(frame.pts*stream.time_base)-arrays["timestamps_seconds"][i]) < 1e-6,
                    "Native video PTS differs from measured state")
            decoded += 1
    require(decoded == n, "Native video does not contain every state frame")
    camera = {"camera_from_world": qa["camera_from_world"].tolist(),
              "intrinsic_matrix": qa["intrinsic_512"].tolist(), "width": 512, "height": 512,
              "near": .01, "far": 100.}
    verify_camera_geometry(camera, receipt["randomization"]["camera"])
    np.testing.assert_allclose(qa["intrinsic_224"][:2], qa["intrinsic_512"][:2]*224/512,
                               atol=1e-4, rtol=0)
    asset = receipt["asset_specs"][object_index]
    require(asset["name"] == "block" and asset["uid"] == "robotwin_073_rubikscube_0",
            "This candidate has only established the source small-cube asset mapping")
    for kind in ["collision", "visual"]:
        path = asset[kind]
        bindings["object_"+kind] = {"path": path, "sha256": asset["files"][path]}
        bound_path(bindings["object_"+kind])
    mesh = trimesh.load(asset["collision"], force="mesh")
    np.testing.assert_allclose(mesh.bounds, asset["bounds_rotated_m"], atol=1e-7, rtol=0,
                               err_msg="Receipt asset dimensions disagree with bound mesh")
    adapter = json.loads(Path(bindings["cabinet_adapter"]["path"]).read_text())
    bindings["cabinet_body_collision"] = {"path": adapter["body_collision"],
                                           "sha256": adapter["body_collision_sha256"]}
    body = trimesh.load(bound_path(bindings["cabinet_body_collision"]), force="mesh")
    np.testing.assert_allclose(body.bounds, adapter["body_bounds_m"], atol=1e-7, rtol=0)
    counts = qa["object_pixels_512"][:, object_index]
    require(counts[0] == 0 and qa["projected_entirely_in_frame"][0, object_index]
            and counts.max() > 0, "Selected object is not initially hidden then revealed")
    states = arrays["object_states"][:, object_index]
    displacement = np.linalg.norm(states[:, :3]-states[0, :3], axis=1)
    require(displacement.max() < .01 and not arrays["grasp"][:, :, object_index].any(),
            "Object moved or was grasped; stationary hidden-content condition is unsupported")
    require(abs(art["cabinet_qpos"][0]).max() < .02 and art["cabinet_qpos"][-1, 1] > 1.
            and abs(art["cabinet_qpos"][:, 0]).max() < .02,
            "Measured initial/terminal door relation disagrees with open-right task")
    return receipt, arrays, qa, art, camera, asset, mesh, adapter, body, bindings


def _review(path, *, video, receipt, asset, object_index, slot_id, frames, qa):
    if path is None:
        return {"status": "unknown", "reason": "No direct RGB observation record",
                "first_observed_confirmed_reference": None, "observations": []}
    review = json.loads(Path(path).read_text())
    reviewer = review.get("reviewer")
    require(review.get("schema") == REVIEW_SCHEMA and isinstance(reviewer, dict)
            and set(reviewer) == {"id", "kind"} and isinstance(reviewer["id"], str)
            and reviewer["id"].strip() and reviewer["kind"] in {"human", "assistant_visual", "vlm"}
            and review.get("method") == "direct_source_rgb_inspection", "Typed v2 RGB review metadata required")
    require(isinstance(review.get("observations"), list) and len(review["observations"]) > 0,
            "At least one concrete RGB observation required")
    require(review.get("generated_identity_pass") is False,
            "Source RGB review cannot pass generated identity")
    require(review["video"] == video and review["receipt"] == receipt
            and review["object_index"] == object_index and review["asset_uid"] == asset["uid"]
            and review["physical_slot_id"] == slot_id, "RGB review object/execution/slot binding mismatch")
    observations, confirmed, seen = [], [], set()
    for entry in review["observations"]:
        index = entry["frame"]["frame_index_0based"]
        require(type(index) is int and str(index) in frames, "Review frame must be a prepared source frame")
        require(index not in seen, "Duplicate or conflicting RGB observation frame")
        seen.add(index)
        require(isinstance(entry.get("observation"), str) and len(entry["observation"].strip()) >= 8,
                "Concrete source RGB observation text is required")
        actual = frames[str(index)]
        bound_path(entry["frame"])
        for key in ["sha256", "decoded_RGB_sha256", "frame_index_0based", "width", "height"]:
            require(entry["frame"][key] == actual[key], "RGB observation image/frame binding mismatch")
        require(entry["frame"].get("video") == video
                and entry["frame"].get("timestamp_seconds") == actual["timestamp_seconds"],
                "RGB review video/time mismatch")
        require(entry.get("object_extent_visible") in ["yes", "no", "unknown"]
                and entry.get("small_cuboid_shape_supported") in ["yes", "no", "unknown"],
                "RGB review must preserve tri-state evidence")
        require(entry.get("generated_identity_pass") is False, "Source RGB review cannot pass generated identity")
        if entry["object_extent_visible"] == entry["small_cuboid_shape_supported"] == "yes":
            require(qa["object_pixels_512"][index, object_index] >= 16,
                    "Claimed clear RGB reference has no instance visibility support")
            if reviewer["kind"] in {"human", "assistant_visual"}:
                confirmed.append(index)
        observations.append(entry)
    return {"status": "unknown_vlm_only" if reviewer["kind"] == "vlm" else "reviewed_source_rgb_only",
            "reviewer": reviewer, "binding": binding(path),
            "reviewer_provenance_is_declared_not_cryptographically_authenticated": True,
            "first_observed_confirmed_reference": min(confirmed) if confirmed else None,
            "first_rgb_recognizable_frame": "unknown",
            "claim_scope": "Earliest confirmed among listed observations, not exact recognizability onset",
            "observations": observations}


def _request_spec(packet, frames):
    reveal = packet["rgb_observations"]["first_observed_confirmed_reference"]
    terminal = packet["source_frame_count"]-1
    blockers = ["approved_single_operator_body_plan_missing", "generated_condition_images_not_reviewed",
                "cabinet_paid_execution_adapter_not_connected", "no_API_budget_authorized"]
    if reveal is None:
        blockers.append("source_revealed_RGB_reference_unconfirmed")
    dimensions = packet["object"]["collision_local_extent_m"]
    common = (
        "Preserve the same cabinet, physical middle-shelf slot, stationary small cuboid object and "
        "relative geometry. Do not replace the cuboid with a flat grid, large board, or shelf-sized panel. "
        "Color alone is not identity. "
        f"Source collision extents in meters are {dimensions}; these are simulator dimensions, "
        "not measured human RGB scale. "
        "An approved single-operator body/hand ownership plan and approved generated initial anchor "
        "must be supplied before execution; both are currently missing. This is preparation only. "
    )
    stages = [{"stage_id": "closed_initial", "target_state": "doors_closed_object_occluded",
        "generation_allowed": False, "provider_called": False,
        "reference_roles": [], "arguments": {"image_path": frames["0"], "reference_images": [],
        "model": "gemini-nano-banana-2.1", "prompt": (
            "Prepare the CLOSED initial human anchor corresponding to this closed simulator image. "
            "Keep both doors closed and the already-present contained object fully occluded. "
            "Do not paint the hidden object on the door or open the cabinet to show it. "
            "This initial RGB cannot encode the hidden object's identity or visible size. "+common)},
        "source_reference_scope": "closed state only; no open-scene reference injected",
        "unresolved_inputs": ["approved_single_operator_body_plan"]}]
    if reveal is not None:
        for stage_id, target, reference, state in [
            ("revealed_keyframe", reveal, terminal, "right_door_partial_open_content_visible"),
            ("open_terminal", terminal, reveal, "right_door_open_hand_withdrawn_content_visible")]:
            stages.append({"stage_id": stage_id, "target_state": state,
                "generation_allowed": False, "provider_called": False,
                "reference_roles": ["same_source_object_shape_scale_and_physical_slot_only"],
                "arguments": {"image_path": frames[str(target)],
                    "reference_images": [frames[str(reference)]], "model": "gemini-nano-banana-2.1",
                    "prompt": ("Prepare the human "+stage_id+" anchor using the edit target's exact "
                        "door state, contact/release state and object placement. The supporting simulator "
                        "image supplies only the same revealed small cuboid's shape, relative size and "
                        "shelf slot; do not copy its different hand or door state. "+common)},
                "source_reference_scope": "source content identity/scale/slot; target image controls state",
                "unresolved_inputs": ["approved_single_operator_body_plan", "approved_generated_initial_anchor"]})
    return {"schema": "r2h_hidden_content_image_request_spec_v1", "generation_allowed": False,
            "provider_called": False, "blocking_reasons": blockers,
            "api": "GeminiClient.generate_image_with_references", "stages": stages,
            "source_condition_schema": SCHEMA,
            "downstream_video_path": {
                "status": "not_connected_no_generated_human_anchors",
                "first_frame": "requires approved generated closed_initial human anchor",
                "last_frame": "requires approved generated open_terminal human anchor",
                "hidden_identity_information_path": "visible content in reviewed human terminal anchor to last_frame condition",
                "first_frame_only_carries_hidden_RGB_identity": False,
                "intermediate_revealed_anchor_use": "unknown; provider reference support or segment strategy not evaluated",
                "video_generation_allowed": False},
            "planned_edit_difference": packet["planned_state_difference"],
            "not_a_video_generation_request": True}


def payload_preflight(request_spec):
    """Run the real provider's image/Parts/size construction, stop before transport.

    No credentials are read, no response is fabricated, no success image is made.
    """
    from .providers import GeminiClient
    class Prepared(Exception):
        pass
    class LocalOnly(GeminiClient):
        def _call(self, method, url, payload=None):
            parts = payload["contents"][0]["parts"]
            self.preflight = {"status": "provider_payload_constructed_without_transport",
                "transport_calls": 0, "provider_response": None, "generation_success": False,
                "image_parts": [{"mime_type": p["inlineData"]["mimeType"],
                    "sha256": hashlib.sha256(base64.b64decode(p["inlineData"]["data"])).hexdigest(),
                    "bytes": len(base64.b64decode(p["inlineData"]["data"]))}
                    for p in parts if "inlineData" in p],
                "text_part_count": sum("text" in p for p in parts),
                "serialized_payload_bytes": len(json.dumps(payload).encode())}
            raise Prepared()
    require(request_spec["generation_allowed"] is False and not request_spec["provider_called"],
            "Only prepare-only specs are accepted")
    if "stages" in request_spec:
        return {"status": "all_stage_payloads_constructed_without_transport", "transport_calls": 0,
                "generation_success": False,
                "stages": [{"stage_id": stage["stage_id"], **payload_preflight(stage)}
                           for stage in request_spec["stages"]]}
    args = request_spec["arguments"]
    target = bound_path(args["image_path"])
    refs = [bound_path(x) for x in args["reference_images"]]
    client = LocalOnly({"model": args["model"]})
    try:
        client.generate_image_with_references(target, refs, args["prompt"], model=args["model"])
    except Prepared:
        return client.preflight
    raise AssertionError("Preflight unexpectedly returned a provider response")


def compile_conditions(proposal_path, verification_path, output, *, object_index=0,
                       rgb_review_path=None, review_frames=(0, 275, 276, 278, 315, 330, 498)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt, arrays, qa, art, camera, asset, mesh, adapter, body, bindings = _load(
        proposal_path, verification_path, object_index)
    n, timestamps = len(arrays["timestamps_seconds"]), arrays["timestamps_seconds"]
    states = arrays["object_states"][:, object_index]
    root = np.asarray(receipt["cabinet_position_m"])
    local = states[:, :3]-root
    floor = receipt["randomization"]["cube_support_surface_z_cabinet_m"]
    surfaces = vertical_surface_hits(body, local[0, :2])
    require(np.isclose(surfaces, floor, atol=1e-8).any(), "Declared shelf has no bound body surface")
    tops = surfaces[surfaces > floor+np.ptp(mesh.bounds[:, 2])]
    require(len(tops) > 0 and receipt["randomization"]["shelf"] == "middle"
            and (local[:, 1] < 0).all(), "Physical shelf/door-side relation is unsupported")
    ceiling = float(tops.min())
    require((local[:, 2] >= floor-.01).all() and (local[:, 2] < ceiling).all(),
            "Object state leaves declared shelf band")
    slot_id = f"{bindings['cabinet_adapter']['sha256']}:middle_shelf:behind_door_1"
    counts = qa["object_pixels_512"][:, object_index]
    events = visibility_events(counts, timestamps)
    require(events["sustained_candidate_frame"] is not None,
            "No sustained instance visibility candidate; revealed reference remains unknown")
    old = hidden_state_candidates(qa["object_pixels_512"], qa["projected_entirely_in_frame"])[object_index]
    frames = {}
    selected = sorted(set([0, n-1, int(counts.argmax()), events["first_any_instance_pixels_frame"],
                           events["sustained_candidate_frame"], *review_frames]))
    for i in selected:
        require(type(i) is int and 0 <= i < n, "Requested review frame outside native video")
        _, frame = frame_at(bindings["video"]["path"], i, output/"frames"/f"source_{i:06d}.png")
        frame.update(video=bindings["video"], timestamp_seconds=float(timestamps[i]),
                     measured_state_index=i, alignment=receipt["video_alignment"])
        frames[str(i)] = frame
    review = _review(rgb_review_path, video=bindings["video"], receipt=bindings["receipt"],
                     asset=asset, object_index=object_index, slot_id=slot_id, frames=frames, qa=qa)
    if rgb_review_path:
        bindings["rgb_review"] = binding(rgb_review_path)
    np.savez_compressed(output/"visibility_timeline.npz", frame_index=np.arange(n),
                        timestamps_seconds=timestamps, instance_pixels_512=counts,
                        object_state=states, cabinet_qpos=art["cabinet_qpos"],
                        projected_bounds_224=qa["projected_bounds_224"][:, object_index],
                        visible_bounds_224=qa["visible_bounds_224"][:, object_index])
    body_world = np.asarray(adapter["body_bounds_m"])+root
    body_proj = project_world_bounds(body_world, camera["camera_from_world"],
                                     camera["intrinsic_matrix"], 512, 512)
    conditions = []
    roles = [("initial_hidden", 0), ("instance_visibility_candidate", events["sustained_candidate_frame"])]
    if review["first_observed_confirmed_reference"] is not None:
        roles.append(("reviewed_revealed_reference", review["first_observed_confirmed_reference"]))
    roles.append(("terminal_reference", n-1))
    for role, i in roles:
        vertices = world_vertices(mesh.vertices, states[i])
        hull = project_world_bounds(np.stack([vertices.min(0), vertices.max(0)]),
                                    camera["camera_from_world"], camera["intrinsic_matrix"], 512, 512)
        box = qa["projected_bounds_224"][i, object_index]/224
        extrinsic = np.asarray(camera["camera_from_world"])
        in_camera = vertices @ extrinsic[:, :3].T + extrinsic[:, 3]
        require(np.isfinite(in_camera).all() and (in_camera[:, 2] > camera["near"]).all(),
                "Raw OBJ vertices cross the camera near plane")
        pixels = in_camera @ np.asarray(camera["intrinsic_matrix"]).T
        uv = pixels[:, :2]/pixels[:, 2:3]
        raw_box = np.r_[uv.min(0), uv.max(0)]/[512, 512, 512, 512]
        difference = raw_box-box
        exact_equal = bool(np.array_equal(raw_box, box))
        bodybox = np.asarray(body_proj["box_xyxy_normalized"])
        ratio = (box[2:]-box[:2])/(bodybox[2:]-bodybox[:2])
        conditions.append({"role": role, "frame": frames[str(i)], "object_index": object_index,
            "asset_uid": asset["uid"], "physical_slot_id": slot_id,
            "object_state": states[i].tolist(), "object_position_in_cabinet_m": local[i].tolist(),
            "door_angles_rad": art["cabinet_qpos"][i].tolist(),
            "instance_visible_pixels_512": int(counts[i]),
            "visibility": "occluded_in_source_camera" if counts[i] == 0 else "instance_pixels_present",
            "recorded_QA_cooked_hull_box_normalized": box.tolist(),
            "projected_world_AABB": hull,
            "raw_OBJ_vertices_projected_box_normalized": raw_box.tolist(),
            "raw_OBJ_vs_recorded_QA_comparison": {
                "status": "exact_equal" if exact_equal else "exact_equality_failed",
                "exact_equal": exact_equal,
                "raw_minus_recorded_xyxy_normalized": difference.tolist(),
                "maximum_absolute_difference_normalized": float(abs(difference).max()),
                "maximum_absolute_difference_at_native_512_px": float(abs(difference).max()*512),
                "difference_cause": "unknown",
                "raw_to_recorded_cooked_geometry_correspondence": "unknown_vertices_not_exported",
                "acceptance_tolerance_applied": None,
                "is_generated_identity_test": False},
            "source_visible_object_size_px": None if counts[i] == 0 else
                {"visible_bounds_at_QA_224": qa["visible_bounds_224"][i, object_index].tolist(),
                 "full_extent_confirmed": "unknown", "not_geometric_bbox": True},
            "projected_object_to_full_cabinet_body_width_height_ratio": ratio.tolist(),
            "ratio_geometry_sources": {
                "numerator": "recorded_QA_cooked_hull_box_normalized",
                "denominator": "cabinet_body_projected_bounds from bound body world AABB"},
            "ratio_scope": "source camera geometry only; includes occluded projection; not human RGB measurement",
            "generated_identity_pass": False})
    packet = {"schema": SCHEMA, "status": "prepared_source_conditions" if review[
        "first_observed_confirmed_reference"] is not None else "prepared_pending_source_RGB_review",
        "generation_allowed": False, "training_admitted": False, "bindings": bindings,
        "legacy_execution_id": receipt["id"], "physical_execution_UUID": "unknown_not_exported",
        "source_data_split": receipt["split"], "source_frame_count": n,
        "same_source_derived_samples_must_share_split": True,
        "physical_qualification": {"status": "historical_bound_cabinet_contact_v7_review_passed",
            "review": bindings["verification"], "checks": json.loads(Path(verification_path).read_text())["raw_verifier"]["physical_checks"],
            "scope": "right-door opening, release and withdrawal only; stationary contained object",
            "apple_strict_qualification_inherited": False, "full_insert_remove_close_completed": False,
            "receipt_production_approved": receipt["production_approved"],
            "new_physical_execution_performed": False,
            "real_world_dynamics_calibrated": adapter["cabinet_contact_generation"]["physically_calibrated"]},
        "geometry_provenance": {
            "recorded_QA_cooked_hull": {
                "binding": bindings["visibility"], "field": "projected_bounds_224",
                "normalization": "divide stored coordinates by 224",
                "recorded_method": "VisibilityRecorder.reset: get_first_collision_mesh(to_world_frame=False).convex_hull.vertices; image_metrics: pose and camera projection",
                "recording_code_receipt_pins": {p: h for p, h in receipt["source_sha256"].items()
                                                if p.endswith("/rich_runtime_v2.py")},
                "cooked_vertices_exported": False,
                "independent_reconstruction": "unknown_unavailable_without_recorded_collision_vertices",
                "raw_OBJ_equivalence_verified": False,
                "note": "Source code establishes renderer collision-mesh origin, not the cause of its difference from raw OBJ projection"},
            "projected_world_AABB": {
                "binding": bindings["object_collision"],
                "method": "raw OBJ vertices transformed by measured state, world-axis-aligned min/max, then all eight AABB corners projected",
                "independently_reconstructible_from_exported_data": True,
                "same_as_recorded_QA_cooked_hull": False},
            "raw_OBJ_vertices_projected_box": {
                "binding": bindings["object_collision"],
                "method": "project raw OBJ vertices directly after measured object pose; min/max image coordinates",
                "comparison_is_diagnostic_not_acceptance": True}},
        "object": {"object_index": object_index, "asset_uid": asset["uid"], "asset_name": asset["name"],
            "identity_source": "receipt index plus SHA256-bound visual/collision asset, not color",
            "collision_local_extent_m": np.ptp(mesh.bounds, axis=0).tolist(),
            "collision_local_bounds_m": mesh.bounds.tolist(),
            "pixel_only_identity_verified": False, "maximum_measured_translation_m": float(
                np.linalg.norm(states[:, :3]-states[0, :3], axis=1).max())},
        "slot": {"id": slot_id, "physical_shelf": "middle", "door_index": 1,
            "definition": "cabinet-local negative-Y side of middle shelf behind the right door",
            "shelf_surface_z_cabinet_m": floor, "next_body_surface_z_cabinet_m": ceiling,
            "vertical_free_band_m": ceiling-floor,
            "local_object_height_to_vertical_free_band_ratio": float(np.ptp(mesh.bounds[:,2])/(ceiling-floor)),
            "surface_method": "vertical ray through object XY intersects bound cabinet body triangles",
            "whole_compartment_volume_verified": False, "support_force_evidence": "unknown_not_required_for_this_nonmanipulated_object",
            "legacy_RGB_ROI_alias": "upper_compartment",
            "legacy_alias_mapping": "v2 upper_compartment is a manually named image opening; this is the cavity above the physical middle shelf, not a new slot",
            "legacy_alias_is_independent_geometry_measurement": False},
        "camera": camera, "cabinet_body_projected_bounds": body_proj,
        "visibility_events": events, "legacy_scene_proposals_visibility_candidate": old,
        "visibility_timeline": binding(output/"visibility_timeline.npz"), "source_frames": frames,
        "rgb_observations": review, "conditions": conditions,
        "planned_state_difference": {"preserve": ["same object asset/instance and small cuboid shape",
            "same physical middle-shelf slot", "same local object pose and relative size", "left door closed"],
            "change": ["right door closed to open", "object occluded to revealed", "operator hand releases and withdraws"],
            "forbidden": ["make hidden content visible through closed door", "replace cube with flat grid panel",
                          "move/resize content or cabinet cavity to match colors"]},
        "unknowns": ["exact first RGB recognizability frame", "generated human pixel identity/scale/slot",
            "approved single-operator body plan", "generalization to other assets/layouts/sources",
            "recorded collision vertices and cause of raw OBJ versus QA hull projection difference"],
        "paid_calls": 0, "gpu_used": False}
    request = _request_spec(packet, frames)
    write_json(output/"hidden_conditions.json", packet)
    request["conditions"] = binding(output/"hidden_conditions.json")
    write_json(output/"image_request_spec.json", request)
    write_json(output/"payload_preflight.json", payload_preflight(request))
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--object-index", type=int, default=0)
    parser.add_argument("--rgb-review", type=Path)
    parser.add_argument("--review-frames", type=int, nargs="+", default=[0, 275, 276, 278, 315, 330, 498])
    args = parser.parse_args()
    result = compile_conditions(args.proposal, args.verification, args.output, object_index=args.object_index,
                                rgb_review_path=args.rgb_review, review_frames=tuple(args.review_frames))
    print(json.dumps({"status": result["status"], "generation_allowed": False,
                      "source_frames": len(result["source_frames"]), "paid_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
