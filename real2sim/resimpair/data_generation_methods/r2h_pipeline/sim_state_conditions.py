"""Compile narrow apple-transfer anchor conditions from verified sim evidence.

CPU-only preparation. Object/role identity comes from a bound simulation
execution, not colors or screen-side words. Generated images still require
their own identity, state, scale and single-operator review.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from . import scene_proposals
from .common import file_sha256, write_json
from .scene_proposals import binding, checked_path, frame_at, project_world_bounds


SCHEMA = "r2h_apple_sim_state_conditions_v1"
WORKSPACE = Path(__file__).resolve().parents[2]


def checked_binding(value):
    if not isinstance(value, dict) or len(value.get("sha256", "")) != 64:
        raise ValueError("A complete SHA256 evidence binding is required")
    return checked_path(value).resolve(strict=True)


def world_vertices(vertices, state):
    state = np.asarray(state, dtype=float)
    if state.shape != (13,) or not np.isfinite(state).all():
        raise ValueError("Expected finite object position, quaternion and velocity")
    if abs(np.linalg.norm(state[3:7]) - 1) > .001:
        raise ValueError("Object quaternion is not normalized")
    rotation = Rotation.from_quat(np.r_[state[4:7], state[3]]).as_matrix()
    return np.asarray(vertices) @ rotation.T + state[:3]


def project_center(point, camera):
    e = np.asarray(camera["camera_from_world"], dtype=float)
    k = np.asarray(camera["intrinsic_matrix"], dtype=float)
    q = e[:, :3] @ np.asarray(point) + e[:, 3]
    if not np.isfinite(q).all() or q[2] <= camera["near"]:
        return {"status": "unknown", "reason": "center_behind_near_plane"}
    pixels = k @ q
    uv = pixels[:2] / pixels[2] / [camera["width"], camera["height"]]
    if np.any(uv < 0) or np.any(uv > 1):
        return {"status": "unknown", "reason": "center_outside_image",
                "unclipped_center_xy_normalized": uv.tolist()}
    return {"status": "projected_geometry", "center_xy_normalized": uv.tolist(),
            "pixel_identity_verified": False}


def container_projection(center, vertices, camera):
    # The qualified strict adapter places static bowls with identity rotation.
    points = np.asarray(vertices) + np.asarray(center)
    bounds = np.stack([points.min(0), points.max(0)])
    return {
        "center": project_center(points.mean(0), camera),
        "geometry": project_world_bounds(bounds, camera["camera_from_world"],
                                         camera["intrinsic_matrix"], camera["width"], camera["height"]),
        "pose_source": "qualified static bowl mesh translated by exported container_centers_world",
        "pixel_identity_verified": False,
    }


def camera_role_order(source, destination):
    """Relative screen order, never a reassignment of source/destination roles."""
    if any(x["center"]["status"] != "projected_geometry" for x in (source, destination)):
        return {"status": "unknown", "reason": "container_center_projection_unknown"}
    sx = source["center"]["center_xy_normalized"][0]
    dx = destination["center"]["center_xy_normalized"][0]
    if abs(sx - dx) <= 1e-6:
        return {"status": "unknown", "reason": "no_horizontal_order"}
    return {
        "status": "projected_geometry",
        "source_relative_to_destination": "screen_left" if sx < dx else "screen_right",
        "destination_relative_to_source": "screen_right" if sx < dx else "screen_left",
        "source_center_x": sx, "destination_center_x": dx,
        "identity_rule": "Keep execution object/container IDs; screen-side words are camera-specific.",
        "pixel_identity_verified": False,
    }


def verify_camera_geometry(camera, config):
    """Reconstruct the recorded fixed look-at camera in the OpenCV convention."""
    eye, target = np.asarray(config["eye"], float), np.asarray(config["target"], float)
    forward = target - eye
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError("Degenerate camera look-at direction")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("Camera look-at direction is parallel to up")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    expected = np.c_[rotation, -rotation @ eye]
    actual = np.asarray(camera["camera_from_world"], float)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0,
                               err_msg="Camera calibration disagrees with bound look-at configuration")
    focal = camera["height"] / (2 * np.tan(config["fov_radians"] / 2))
    intrinsic = [[focal, 0, camera["width"] / 2], [0, focal, camera["height"] / 2], [0, 0, 1]]
    np.testing.assert_allclose(camera["intrinsic_matrix"], intrinsic, atol=1e-4, rtol=0,
                               err_msg="Camera intrinsic disagrees with resolution/FOV")
    if not np.isfinite(actual).all() or camera["near"] <= 0 or camera["far"] <= camera["near"]:
        raise ValueError("Camera geometry/clipping invalid")


def verify_video_timeline(video, camera, timestamps):
    """Read every selected camera frame, including the added opposite stream."""
    import av
    pts = []
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            if (frame.width, frame.height) != (camera["width"], camera["height"]):
                raise ValueError("Selected camera video dimensions disagree with calibration")
            if frame.pts is None:
                raise ValueError("Selected camera frame lacks a presentation timestamp")
            pts.append(float(frame.pts * frame.time_base))
    np.testing.assert_allclose(pts, timestamps, atol=1e-5, rtol=0,
                               err_msg="Selected camera full decoded count/PTS disagree with states")


def measured_events(grasp, object_states, selected, vertices, rim_z):
    """Event indices from actual simulator grasp and mesh pose; stage is unused."""
    grasp = np.asarray(grasp)
    states = np.asarray(object_states)
    if grasp.dtype != np.bool_ or grasp.ndim != 3 or states.ndim != 3:
        raise ValueError("Expected boolean [time,arm,object] grasp and measured states")
    if grasp.shape[0] != states.shape[0] or grasp.shape[2] != states.shape[1]:
        raise ValueError("Grasp/state timeline or object identity mismatch")
    held = grasp[:, :, selected].any(1)
    starts = np.flatnonzero(held & ~np.r_[False, held[:-1]])
    ends = np.flatnonzero(~held & np.r_[False, held[:-1]])
    if not len(starts) or not len(ends) or held[0] or held[-1]:
        raise ValueError("Complete unheld → measured grasp → release sequence unavailable")
    bottoms = np.array([world_vertices(vertices, s[selected])[:, 2].min() for s in states])
    clear = np.flatnonzero(held & (bottoms > rim_z))
    if not len(clear):
        raise ValueError("No grasped object frame clears the physical source rim")
    release = ends[ends > clear[0]]
    if not len(release):
        raise ValueError("No measured release after rim-clear lift")
    # Preserve all transitions, even if an episode has multiple grasp intervals.
    return {
        "initial": 0, "grasp_established": int(starts[0]),
        "lift_clear": int(clear[0]), "last_grasp_before_release": int(release[-1] - 1),
        "released": int(release[-1]), "terminal": len(states) - 1,
        "all_grasp_onsets": starts.tolist(), "all_grasp_releases": ends.tolist(),
        "source": "arrays.grasp plus transformed collision-mesh bottom relative to source rim",
        "control_stage_names_used": False,
    }


def geometry_relations(points, centers, profile):
    """Expose the existing bowl verifier's geometry predicates, without contact claims."""
    result = []
    for index, center in enumerate(centers):
        predicates = {
            "inside_opening": bool(np.linalg.norm(points[:, :2] - center[:2], axis=1).max() < .075),
            "below_rim": bool(points[:, 2].min() < profile["rim_z_m"] - .01),
            "not_below_floor": bool(points[:, 2].min() > profile["center_inner_floor_z_m"] - .008),
        }
        result.append({
            "container_index": index, "geometrically_inside": all(predicates.values()),
            "predicates_from_existing_verify_bowl": predicates,
            "contact_or_support_inferred_from_geometry_alone": False,
        })
    return result


def _load_verified(receipt_path, verification_path, camera_id):
    receipt_path, verification_path = Path(receipt_path).resolve(), Path(verification_path).resolve()
    receipt = json.loads(receipt_path.read_text())
    verification = json.loads(verification_path.read_text())
    if (receipt.get("scene") != "strict_apple_clutter"
            or receipt.get("goal_key") != "factory:transfer_apple_occupied_bowl_to_empty_bowl"
            or receipt["randomization"].get("transfer_to_unique_empty_bowl") is not True):
        raise ValueError("Only the qualified strict apple-transfer schema is supported")
    if verification.get("passed") is not True or verification.get("receipt_sha256") != file_sha256(receipt_path):
        raise ValueError("No physical verification bound to this exact receipt")
    if not verification.get("physical_checks") or not all(
            value is True for value in verification["physical_checks"].values()):
        raise ValueError("Physical checks are incomplete or contain a non-pass")
    multiview_path = checked_binding(receipt["multiview"])
    multiview = json.loads(multiview_path.read_text())
    if (multiview.get("schema") != "synchronized_multiview_v1"
            or multiview.get("execution_id") != receipt.get("execution_id")
            or multiview.get("diagnostic_failure_trace") is not False):
        raise ValueError("Expected the same completed multiview execution")
    arrays_path = checked_binding(multiview["shared_arrays"])
    if arrays_path != Path(receipt["arrays"]).resolve():
        raise ValueError("Shared multiview arrays differ from the verified receipt")
    if file_sha256(arrays_path) != receipt["sha256"]["arrays.npz"]:
        raise ValueError("Verified arrays hash differs")
    timeline_path = checked_binding(multiview["timeline"])
    camera = multiview["cameras"][camera_id]
    video = checked_binding(camera["video"])
    contact = checked_binding({"path": receipt["contact_data"], "sha256": receipt["contact_sha256"]})
    with np.load(arrays_path, allow_pickle=False) as data:
        arrays = dict(data)
    with np.load(timeline_path, allow_pickle=False) as data:
        timeline = dict(data)
    with np.load(contact, allow_pickle=False) as data:
        contacts = dict(data)
    n = receipt["length"]
    if multiview["length"] != n or verification["frames"] != n:
        raise ValueError("Verified timeline lengths differ")
    for field in ("frame_index", "state_index", "action_index"):
        np.testing.assert_array_equal(timeline[field], np.arange(n))
    np.testing.assert_array_equal(timeline["timestamps_seconds"], arrays["timestamps_seconds"])
    camera_index = timeline["camera_ids"].tolist().index(camera_id)
    np.testing.assert_array_equal(timeline["camera_frame_index"][:, camera_index], np.arange(n))
    for name, c in multiview["cameras"].items():
        config = (multiview["opposite_camera_config"] if name == "opposite"
                  else receipt["randomization"]["camera"])
        verify_camera_geometry(c, config)
    if np.allclose(multiview["cameras"]["opposite"]["camera_from_world"],
                   multiview["cameras"]["native"]["camera_from_world"], atol=1e-6, rtol=0):
        raise ValueError("Opposite view calibration aliases the native view")
    verify_video_timeline(video, camera, arrays["timestamps_seconds"])
    # Reuse and re-run the unchanged independent physical verifier. Its exact
    # source hash must match the source captured by the physical execution.
    sys.path.insert(0, str(WORKSPACE / "PolicyGen"))
    from examples.baselines.hypericl_data_factory import verify_bowl, real_bowl
    from examples.baselines.hypericl_task_expansion.registry import PROGRAMS
    for module in (verify_bowl, real_bowl):
        path = Path(module.__file__).resolve()
        if receipt["source_sha256"].get(str(path)) != file_sha256(path):
            raise ValueError("Qualified physical/source semantics changed: " + str(path))
    fresh = verify_bowl.verify(receipt_path)
    if fresh["physical_checks"] != verification["physical_checks"]:
        raise ValueError("Fresh physical checks disagree with preserved verification")
    selected, destination = PROGRAMS[receipt["program"]][0]
    source = receipt["randomization"]["source_bowl_index"]
    if (destination != receipt["randomization"]["destination_bowl_index"]
            or source != 0 or destination != 1 or receipt["asset_specs"][selected]["name"] != "apple"):
        raise ValueError("Program and measured task roles disagree")
    if not (len(multiview["object_actor_ids"]) == len(multiview["object_identity"])
            == len(receipt["asset_specs"])):
        raise ValueError("Renderer object identities incomplete")
    for actual, expected in zip(multiview["object_identity"], receipt["asset_specs"]):
        if actual != expected:
            raise ValueError("Multiview asset identity differs from verified receipt")
    # The final measurement is after the last command; its frame was not saved.
    # Recheck the *last recorded pre-command state* before using its RGB anchor.
    recorded_terminal = dict(arrays, final_object_states=arrays["object_states"][-1])
    terminal_passed, terminal_checks = verify_bowl.outcome(receipt, recorded_terminal, contacts)
    if not terminal_passed:
        raise ValueError("Last recorded RGB state has no verified stable terminal: " + str(terminal_checks))
    return dict(
        receipt=receipt, verification=verification, fresh_verification=fresh, multiview=multiview,
        arrays=arrays, timeline=timeline, contacts=contacts, camera=camera, camera_index=camera_index,
        selected=selected, source=source, destination=destination, video=video,
        terminal_checks=terminal_checks,
        bindings={
            "receipt": binding(receipt_path), "verification": binding(verification_path),
            "arrays": binding(arrays_path), "contacts": binding(contact),
            "multiview_manifest": binding(multiview_path), "timeline": binding(timeline_path),
            "selected_camera_video": binding(video), "physical_verifier": binding(verify_bowl.__file__),
            "qualified_static_bowl_implementation": binding(real_bowl.__file__),
        })


def compile_conditions(receipt_path, verification_path, camera_id, output, *, partition):
    """Create a new condition packet, or a preserved blocked result on missing evidence."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    bindings = {"compiler": binding(__file__), "scene_proposals": binding(scene_proposals.__file__)}
    try:
        data = _load_verified(receipt_path, verification_path, camera_id)
        bindings.update(data["bindings"])
        r, a, mv = data["receipt"], data["arrays"], data["multiview"]
        selected, source, destination = data["selected"], data["source"], data["destination"]
        spec = r["asset_specs"][selected]
        vertices = np.asarray(trimesh.load(spec["collision"], force="mesh").vertices)
        bowl_vertices = np.asarray(trimesh.load(r["receptacle"]["collision"], force="mesh").vertices)
        for role, asset in (("selected_object", spec), ("container_asset", r["receptacle"])):
            bindings[role + "_collision"] = binding(asset["collision"])
            bindings[role + "_visual"] = binding(asset["visual"])
        centers, profile = a["container_centers_world"], r["receptacle"]["floor_profile"]
        events = measured_events(a["grasp"], a["object_states"], selected, vertices, profile["rim_z_m"])
        execution = r["execution_id"]
        object_id = f"{execution}/object/{selected}"
        source_id, destination_id = [f"{execution}/container/{i}" for i in (source, destination)]
        camera_roles = {}
        for name, camera in mv["cameras"].items():
            src = container_projection(centers[source], bowl_vertices, camera)
            dst = container_projection(centers[destination], bowl_vertices, camera)
            camera_roles[name] = {
                "source_container": src, "destination_container": dst,
                "relative_order": camera_role_order(src, dst),
            }
        selected_roles = camera_roles[camera_id]
        if selected_roles["relative_order"]["status"] != "projected_geometry":
            raise ValueError("Selected camera cannot resolve source/destination horizontal order")
        roles = {
            "selected_object": {
                "id": object_id, "object_index": selected,
                "renderer_actor_id": mv["object_actor_ids"][selected],
                "asset_uid": spec["uid"], "name": spec["name"],
                "collision": bindings["selected_object_collision"],
                "visual": bindings["selected_object_visual"],
            },
        }
        for label, index, identity in (("source_container", source, source_id),
                                       ("destination_container", destination, destination_id)):
            roles[label] = {
                "id": identity, "container_index": index, "asset_uid": r["receptacle"]["uid"],
                "center_world_m": centers[index].tolist(),
                "renderer_actor_id": None,
                "renderer_actor_id_status": "unknown_not_exported_by_original_recorder",
                "identity_basis": "execution UUID + stable container array index; not color or screen position",
                "collision": bindings["container_asset_collision"],
                "visual": bindings["container_asset_visual"],
            }
        reasons = {}
        for label in ("initial", "grasp_established", "lift_clear",
                      "last_grasp_before_release", "released", "terminal"):
            reasons.setdefault(events[label], []).append(label)
        conditions = []
        for index, labels in sorted(reasons.items()):
            _, frame = frame_at(data["video"], index, output / "frames" / f"{camera_id}_{index:06d}.png")
            frame["timestamp_seconds"] = float(a["timestamps_seconds"][index])
            frame["video"] = bindings["selected_camera_video"]
            points = world_vertices(vertices, a["object_states"][index, selected])
            geometry = project_world_bounds(
                [points.min(0), points.max(0)], data["camera"]["camera_from_world"],
                data["camera"]["intrinsic_matrix"], data["camera"]["width"], data["camera"]["height"])
            c = data["camera_index"]
            visibility = {
                "visible_pixels": int(data["timeline"]["object_pixels"][index, c, selected]),
                "bbox_xyxy_pixels": data["timeline"]["visible_bounds"][index, c, selected].tolist(),
                "source": "simulator instance segmentation, not generated-image identity",
            }
            ratios = {}
            for label in ("source_container", "destination_container"):
                bowl_geometry = selected_roles[label]["geometry"]
                if geometry["status"] == bowl_geometry["status"] == "candidate":
                    ob, bb = [np.asarray(x["box_xyxy_normalized"]) for x in (geometry, bowl_geometry)]
                    ratios[label] = {
                        "status": "projected_geometry_reference",
                        "object_to_container_bbox_width": float((ob[2] - ob[0]) / (bb[2] - bb[0])),
                        "object_to_container_bbox_area": float(np.prod(ob[2:] - ob[:2]) / np.prod(bb[2:] - bb[:2])),
                        "use": "same-camera reference only; compare generated object after role/pose mapping",
                        "not_a_pixel_measurement_or_acceptance_threshold": True,
                    }
                else:
                    ratios[label] = {"status": "unknown", "reason": "complete_projection_unavailable"}
            conditions.append({
                "labels": labels, "frame": frame, "object_id": object_id,
                "object_pose_wxyz_world": a["object_states"][index, selected, :7].tolist(),
                "simulator_grasped_by_arm_indices": np.flatnonzero(a["grasp"][index, :, selected]).tolist(),
                "geometric_container_relations": geometry_relations(points, centers, profile),
                "aggregate_bowl_support_force_N": float(data["contacts"]["support_force_N"][index, selected]),
                "support_scope": "Force aggregates all bowls; target attribution also needs location geometry.",
                "object_projection": geometry, "simulator_visibility": visibility,
                "relative_size_reference": ratios,
                "generated_image_identity_verified": False,
                "generated_image_scale_verified": False,
            })
        initial = next(x for x in conditions if "initial" in x["labels"])
        terminal = next(x for x in conditions if "terminal" in x["labels"])
        if not initial["geometric_container_relations"][source]["geometrically_inside"]:
            raise ValueError("Initial selected object has no source-container geometric relation")
        if not terminal["geometric_container_relations"][destination]["geometrically_inside"]:
            raise ValueError("Terminal image has no destination-container geometric relation")
        order = selected_roles["relative_order"]
        delta = {
            "remove": [{"relation": "inside", "subject": object_id, "container": source_id}],
            "add": [{"relation": "inside_and_stably_supported", "subject": object_id, "container": destination_id}],
            "keep": [object_id, source_id, destination_id],
            "preserve_nonselected_objects": [
                {"id": f"{execution}/object/{i}", "asset_uid": s["uid"]}
                for i, s in enumerate(r["asset_specs"]) if i != selected],
            "source_final_occupancy": "selected object absent",
            "destination_initial_occupancy": "empty of recorded movable task objects",
            "destination_final_occupancy": "same selected object; no duplication, enlargement or substitution",
            "screen_language": {
                "selected_camera": camera_id,
                "source_container": order["source_relative_to_destination"],
                "destination_container": order["destination_relative_to_source"],
                "screen_side_is_not_semantic_identity": True,
            },
            "edit_instruction": (
                f"In this {camera_id} view, move the same selected apple from the "
                f"{order['source_relative_to_destination']} source bowl to the "
                f"{order['destination_relative_to_source']} destination bowl. "
                "The source bowl must no longer contain the apple. Preserve both bowl instances, "
                "their positions, the apple's identity and source-referenced relative size, and all "
                "nonselected objects. The terminal apple is released and supported inside the "
                "destination. Preserve the separately approved single-person body/hand plan. "
                "Do not apply screen directions from another camera."),
            "instruction_is_condition_plan_not_verified_generated_image": True,
        }
        result = {
            "schema": SCHEMA, "status": "prepared_for_condition_image_review",
            "scope": "verified strict apple transfer only; automatic source-state condition preparation",
            "execution_id": execution, "experiment_partition": partition, "source_data_split": r["split"],
            "bindings": bindings, "selected_camera": camera_id, "camera_calibration": data["camera"],
            "roles": roles, "camera_role_projections": camera_roles, "measured_events": events,
            "conditions": conditions, "state_delta": delta,
            "terminal_evidence": {
                "stored_verification": data["verification"],
                "fresh_unmodified_verifier_result": data["fresh_verification"],
                "last_recorded_frame_checks": data["terminal_checks"],
                "terminal_frame_index": len(a["object_states"]) - 1,
                "post_last_command_final_state_used_as_rgb": False,
                "support_attribution": "Original verifier: destination geometry + aggregate bowl support + released stable history.",
                "true_target_bowl_contact_ID_exported": False,
            },
            "required_generated_condition_reviews": [
                "Bind and approve one-person body/left-right hand ownership plan before image generation.",
                "Verify same-camera source/destination role mapping in the human initial image.",
                "Verify same object identity, shape, proportions and source-empty/destination-filled terminal state.",
                "Verify contact/release visibility and unknowns before any video submission.",
            ],
            "generation_allowed": False, "api_calls": 0,
            "generated_image_identity_verified": False, "generated_image_scale_verified": False,
            "training_admitted": False, "control_stage_names_used_as_contact_gt": False,
            "color_used_as_identity": False,
            "unknowns": [
                "Container renderer actor IDs were not exported; stable execution/container IDs are used.",
                "Projected collision geometry and simulator visibility do not establish human-image identity.",
                "Human body plan and generated initial/terminal condition images are not supplied or approved.",
            ],
        }
    except (ValueError, KeyError, IndexError, AssertionError, OSError) as error:
        result = {
            "schema": SCHEMA, "status": "blocked_missing_or_inconsistent_evidence",
            "error_type": type(error).__name__, "reason": str(error),
            "bindings": bindings, "experiment_partition": partition, "selected_camera": camera_id,
            "generation_allowed": False, "api_calls": 0, "training_admitted": False,
        }
    frozen = output / "frozen_sources"
    frozen.mkdir()
    for name in ("compiler", "scene_proposals"):
        path = checked_binding(bindings[name])
        (frozen / path.name).write_bytes(path.read_bytes())
    write_json(output / "conditions.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--camera", required=True, choices=["zed2i", "native", "opposite"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--partition", required=True, choices=["development", "validation"])
    args = parser.parse_args()
    result = compile_conditions(args.receipt, args.verification, args.camera, args.output,
                                partition=args.partition)
    print(json.dumps({k: result[k] for k in ("schema", "status", "generation_allowed", "api_calls")},
                     ensure_ascii=False))
    raise SystemExit(0 if result["status"] == "prepared_for_condition_image_review" else 2)


if __name__ == "__main__":
    main()
