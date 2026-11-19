"""Evidence-grounded scene proposals; geometry first, VLM correction second.

No task-category heuristics, historical manual ROI reads, generation or training.
Projected geometry is not visible geometry. Matching and VLM outputs remain
candidates until separately validated for the intended downstream operation.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from .common import file_sha256, json_sha256, probe_video, write_json


def binding(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def checked_path(value):
    if isinstance(value, str):
        return Path(value)
    path = Path(value["path"])
    if value.get("sha256") and file_sha256(path) != value["sha256"]:
        raise ValueError("Evidence binding changed: " + str(path))
    return path


def verify_proposal_bindings(proposal):
    """Reject media/geometry replacements before any paid review or reuse."""
    for value in proposal["bindings"].values():
        checked_path(value)
    for value in proposal["source_keyframes"]:
        checked_path(value)
    if proposal.get("target"):
        checked_path(proposal["target"]["video"])
        checked_path(proposal["target"]["frame"])
    return True


def validate_box(box):
    a = np.asarray(box, dtype=float)
    if a.shape != (4,) or not np.isfinite(a).all():
        raise ValueError("box must contain four finite normalized coordinates")
    if not (0 <= a[0] < a[2] <= 1 and 0 <= a[1] < a[3] <= 1):
        raise ValueError("box must be nondegenerate xyxy within [0,1]")
    return a.tolist()


def project_world_bounds(bounds, camera_from_world, intrinsic, width, height):
    """Project all eight world-axis-aligned corners; no occlusion assumption."""
    b = np.asarray(bounds, float)
    e, k = np.asarray(camera_from_world, float), np.asarray(intrinsic, float)
    if b.shape != (2, 3) or e.shape != (3, 4) or k.shape != (3, 3):
        raise ValueError("invalid bounds/calibration dimensions")
    if not all(np.isfinite(x).all() for x in (b, e, k)) or np.any(b[0] > b[1]):
        raise ValueError("invalid bounds/calibration values")
    points = np.array(list(itertools.product(*zip(b[0], b[1]))))
    camera = points @ e[:, :3].T + e[:, 3]
    if np.any(camera[:, 2] <= .01):
        return {"status": "unknown", "reason": "geometry_crosses_camera_near_plane"}
    pixels = camera @ k.T
    uv = pixels[:, :2] / pixels[:, 2:3]
    box = np.r_[uv.min(0), uv.max(0)] / [width, height, width, height]
    if np.any(box < 0) or np.any(box > 1):
        return {"status": "unknown", "reason": "projected_geometry_not_fully_in_frame",
                "unclipped_box_xyxy_normalized": box.tolist()}
    return {"status": "candidate", "box_xyxy_normalized": validate_box(box),
            "visible_surface_verified": False}


def hidden_state_candidates(pixel_counts, projected_in_frame, *, sustained_frames=3):
    """Visible-instance segmentation, not object labels, drives reveal detection."""
    counts, in_frame = np.asarray(pixel_counts), np.asarray(projected_in_frame)
    if counts.ndim != 2 or counts.shape != in_frame.shape or len(counts) < 2:
        raise ValueError("visibility arrays must be matching [time, object]")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("invalid visibility counts")
    proposals = []
    for obj in range(counts.shape[1]):
        threshold = max(4, int(np.ceil(float(counts[:, obj].max()) * .02)))
        visible = (counts[:, obj] >= threshold) & in_frame[:, obj]
        initial_hidden = counts[0, obj] == 0 and bool(in_frame[0, obj])
        starts = [i for i in range(len(counts)-sustained_frames+1)
                  if visible[i:i+sustained_frames].all()]
        revealed = initial_hidden and bool(starts)
        proposals.append({
            "object_index": obj, "initial_visible_pixels": int(counts[0, obj]),
            "maximum_visible_pixels": int(counts[:, obj].max()),
            "final_visible_pixels": int(counts[-1, obj]),
            "initial_projection_in_frame": bool(in_frame[0, obj]),
            "hidden_then_revealed": bool(revealed),
            "first_sustained_reveal_frame": starts[0] if revealed else None,
            "maximum_visibility_frame": int(counts[:, obj].argmax()),
            "visibility_threshold_pixels": threshold,
            "sustained_frames": sustained_frames,
            "hidden_cause": "occlusion_candidate" if initial_hidden else "not_established",
            "required_condition": "include_revealed_state_visual_anchor" if revealed else None,
            "claim_scope": "simulator_instance_visibility; not human semantic correspondence",
        })
    return proposals


def frame_at(video, index, destination):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot decode frame {index}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), image):
        raise ValueError("Cannot write decoded frame")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, {**binding(destination), "frame_index_0based": int(index),
                   "decoded_RGB_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                   "width": image.shape[1], "height": image.shape[0]}


def register_images(source, target):
    """SIFT + RANSAC homography with explicit support metrics, not confidence."""
    gray_a, gray_b = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (source, target)]
    detector = cv2.SIFT_create(nfeatures=3000)
    ka, da = detector.detectAndCompute(gray_a, None)
    kb, db = detector.detectAndCompute(gray_b, None)
    result = {"method": "SIFT_Lowe0.72_RANSAC", "status": "unknown",
              "metrics_are_not_calibrated_probabilities": True}
    if da is None or db is None or min(len(da), len(db)) < 2:
        return {**result, "reason": "insufficient_features"}
    matches = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [a for pair in matches if len(pair) == 2 for a, b in [pair]
            if a.distance < .72*b.distance]
    result["ratio_test_matches"] = len(good)
    if len(good) < 8:
        return {**result, "reason": "insufficient_correspondences"}
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(9127)
    h, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.)
    if h is None or mask is None:
        return {**result, "reason": "homography_failed"}
    mask = mask.ravel().astype(bool)
    prediction = cv2.perspectiveTransform(pa.reshape(-1, 1, 2), h).reshape(-1, 2)
    errors = np.linalg.norm(prediction-pb, axis=1)
    support = pa[mask]
    coverage = float(cv2.contourArea(cv2.convexHull(support)) / (source.shape[0]*source.shape[1]))
    result.update(homography=h.tolist(), inlier_count=int(mask.sum()),
                  inlier_fraction=float(mask.mean()), source_hull_fraction=coverage,
                  median_inlier_error_px=float(np.median(errors[mask])),
                  source_inlier_pixels=pa[mask].tolist(),
                  target_inlier_pixels=pb[mask].tolist())
    if mask.sum() < 8 or mask.mean() < .35 or coverage < .005:
        return {**result, "reason": "insufficient_spatial_or_geometric_support"}
    return {**result, "status": "candidate",
            "reason": "registration_candidate_requires_local_semantic_verification"}


def transfer_box(box, registration, source_shape, target_shape, *, padding=.007):
    validate_box(box)
    if registration.get("status") != "candidate":
        return {"status": "unknown", "reason": "registration_unresolved"}
    sh, sw = source_shape[:2]
    th, tw = target_shape[:2]
    x0, y0, x1, y1 = box
    corners = np.float32([[x0*sw, y0*sh], [x1*sw, y0*sh],
                          [x1*sw, y1*sh], [x0*sw, y1*sh]])
    h = np.asarray(registration["homography"])
    q = cv2.perspectiveTransform(corners[None], h)[0]
    raw = np.r_[q.min(0), q.max(0)] / [tw, th, tw, th]
    if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1):
        return {"status": "unknown", "reason": "transferred_geometry_outside_image"}
    out = np.clip(raw + [-padding, -padding, padding, padding], 0, 1)
    points = np.asarray(registration["source_inlier_pixels"]) / [sw, sh]
    center = np.array([(x0+x1)/2, (y0+y1)/2])
    nearest = float(np.linalg.norm(points-center, axis=1).min())
    return {"status": "candidate", "box_xyxy_normalized": validate_box(out),
            "nearest_registration_support_normalized": nearest,
            "semantic_identity_verified": False,
            "requires_vlm_correction": True,
            "source": "calibrated_sim_geometry_then_image_registration"}


def stable_background_box(frames, exclusions):
    """Find textured, photometrically stable source regions. No semantic claim."""
    height, width = frames[0].shape[:2]
    grays = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in frames]
    delta = np.max(np.abs(np.stack(grays).astype(float)-grays[0]), axis=0)
    candidates = []
    side = max(16, round(min(width, height)*.10))
    for y in range(side, height-side, side//2):
        for x in range(side, width-side, side//2):
            box = [x/width, y/height, (x+side)/width, (y+side)/height]
            if any(not(box[2] < e[0] or box[0] > e[2] or box[3] < e[1] or box[1] > e[3])
                   for e in exclusions):
                continue
            crop = grays[0][y:y+side, x:x+side]
            pts = cv2.goodFeaturesToTrack(crop, 40, .02, 3)
            count = 0 if pts is None else len(pts)
            d = float(np.quantile(delta[y:y+side, x:x+side], .8))
            if count >= 6 and d <= 20:
                candidates.append((count/(1+d), box, count, d))
    if not candidates:
        return {"status": "unknown", "reason": "no_textured_stable_patch"}
    _, box, count, delta80 = max(candidates)
    return {"status": "candidate", "box_xyxy_normalized": box,
            "feature_count": count, "temporal_delta_80pct_8bit": delta80,
            "source": "multiframe_photometric_stability_and_texture",
            "semantic_static_background_verified": False}


def build_proposals(input_path, output, *, target_video=None):
    """Prepare replayable candidates from an existing collection receipt."""
    input_path, output = Path(input_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    job = json.loads(input_path.read_text())
    robot = job["robot"]
    receipt_path = checked_path(robot["receipt"])
    receipt = json.loads(receipt_path.read_text())
    arrays_path = checked_path(robot.get("trajectory", robot["trajectory_path"]))
    video = checked_path(robot["video"])
    media = probe_video(video)
    arrays = np.load(arrays_path, allow_pickle=False)
    n = len(arrays["timestamps_seconds"])
    if media["frames"] != n:
        raise ValueError("Video and array frame counts differ")
    if not np.allclose(arrays["timestamps_seconds"], np.arange(n)/media["fps"], atol=1e-5, rtol=0):
        raise ValueError("Video/array alignment is not supported")
    sources = {"input": binding(input_path), "receipt": binding(receipt_path),
               "arrays": binding(arrays_path), "video": media}
    unknowns, hidden, rois, keyframes = [], [], [], {0: ["initial"], n-1: ["terminal"]}
    qa_binding = robot.get("qa_only_files", {}).get("qa_visibility")
    qa = None
    if qa_binding:
        qa_path = checked_path(qa_binding)
        sources["visibility"] = binding(qa_path)
        qa = np.load(qa_path, allow_pickle=False)
        required_visibility = {"object_pixels_512", "projected_entirely_in_frame",
                               "projected_bounds_224"}
        missing = sorted(required_visibility.difference(qa.files))
        if missing:
            unknowns.append("Missing visibility export fields: " + ", ".join(missing))
            qa = None
    if qa is not None:
        hidden = hidden_state_candidates(qa["object_pixels_512"], qa["projected_entirely_in_frame"])
        for item in hidden:
            index = item["object_index"]
            specs = receipt.get("asset_specs", [])
            item["asset_name"] = specs[index].get("name") if index < len(specs) else None
            item["asset_uid"] = specs[index].get("uid") if index < len(specs) else None
            item["source"] = "simulator_instance_segmentation_counts"
            item["initial_projected_box_xyxy_normalized"] = (qa["projected_bounds_224"][0,index]/224).tolist()
            if item["hidden_then_revealed"]:
                for key in ["first_sustained_reveal_frame", "maximum_visibility_frame"]:
                    keyframes.setdefault(item[key], []).append(f"object_{index}:{key}")
            box = item["initial_projected_box_xyxy_normalized"]
            if qa["object_pixels_512"][0,index] > 0 and np.all(np.asarray(box) >= 0) and np.all(np.asarray(box) <= 1):
                rois.append({"id": f"object_{index}", "role": "visible_object",
                             "asset_name": item["asset_name"], "asset_uid": item["asset_uid"],
                             "box_xyxy_normalized": validate_box(box),
                             "source": "simulator_projected_collision_hull",
                             "visible_pixels": item["initial_visible_pixels"]})
        for event_index, event in enumerate(receipt.get("events", [])):
            if not isinstance(event, dict) or "handle_bounds_world_m" not in event:
                continue
            if not {"camera_from_world", "intrinsic_512"}.issubset(qa.files):
                unknowns.append("Handle projection needs exported camera calibration")
                continue
            result = project_world_bounds(event["handle_bounds_world_m"], qa["camera_from_world"],
                                          qa["intrinsic_512"], 512, 512)
            if result["status"] == "candidate":
                rois.append({"id": f"handle_{event_index}", "role": "interaction_target",
                             **result, "source": "initial_handle_mesh_world_bounds_receipt_projection",
                             "geometry_time": "skill_initial_state",
                             "event_pointer": f"/events/{event_index}/handle_bounds_world_m"})
    else:
        unknowns.append("No instance visibility/calibration export: hidden states and geometry ROIs unknown")
    # State transitions use measured arrays, never stage-name strings.
    for field in ["grasp"]:
        if field in arrays:
            value = np.asarray(arrays[field]).reshape(n, -1).any(axis=1)
            for t in (np.flatnonzero(value[1:] != value[:-1])+1).tolist():
                keyframes.setdefault(t, []).append(f"measured_{field}_state_transition")
    qa_files = robot.get("qa_only_files", {})
    if "articulation_data" in qa_files:
        path = checked_path(qa_files["articulation_data"])
        sources["articulation"] = binding(path)
        art = np.load(path, allow_pickle=False)
        if "contact_step_peaks_N" in art:
            force = np.asarray(art["contact_step_peaks_N"])
            signal = force[:, 0] > .1
            for t in (np.flatnonzero(signal[1:] != signal[:-1])+1).tolist()[:12]:
                keyframes.setdefault(t, []).append("sim_active_door_contact_threshold_crossing_0.1N")
    indices = sorted(keyframes)
    images, frames = {}, []
    for index in indices:
        image, frame = frame_at(video, index, output/f"source_frame{index:06d}.png")
        frame.update(timestamp_seconds=float(arrays["timestamps_seconds"][index]),
                     reasons=keyframes[index])
        images[index] = image
        frames.append(frame)
    initial = images[0]
    # Source background must remain unchanged across the clip, not only the initial image.
    sampled = [initial, images[n-1]]
    for index in [n//3, 2*n//3]:
        im, _ = frame_at(video, index, output/f"stability_frame{index:06d}.png")
        sampled.append(im)
    exclusions = [r["box_xyxy_normalized"] for r in rois]
    background = stable_background_box(sampled, exclusions)
    if background["status"] == "candidate":
        rois.append({"id": "fixed_background", "role": "background_reference", **background})
    else:
        unknowns.append("No automatic stable-background ROI")
    target = None
    actor = {"status": "unknown", "reason": "No measured TCP poses"}
    if "tcp_world" in arrays:
        xyz = np.asarray(arrays["tcp_world"])[:, :, :3, 3]
        displacement = np.linalg.norm(xyz-xyz[0:1], axis=-1).max(axis=0)
        active_slots = np.flatnonzero(displacement > .03).tolist()
        actor = {
            "status": "candidate", "source": "measured_simulator_TCP_displacement",
            "maximum_displacement_from_initial_m": displacement.tolist(),
            "active_arm_slots_above_0.03m": active_slots,
            "number_of_moving_arms": len(active_slots),
            "minimum_active_hands_proposal": len(active_slots),
            "maximum_visible_hands": None,
            "limitations": "Moving arms do not establish people count or a maximum visible-hand budget. Static support hands may be necessary.",
        }
    if target_video:
        target_video = Path(target_video)
        target_media = probe_video(target_video)
        target_image, target_frame = frame_at(target_video, 0, output/"target_frame000000.png")
        registration = register_images(initial, target_image)
        transfers = [{"id": r["id"], "role": r["role"],
                      **transfer_box(r["box_xyxy_normalized"], registration, initial.shape, target_image.shape)}
                     for r in rois]
        target = {"video": target_media, "frame": target_frame,
                  "registration": registration, "roi_candidates": transfers}
        annotated = target_image.copy()
        for r in transfers:
            if r["status"] == "candidate":
                x0,y0,x1,y1 = r["box_xyxy_normalized"]
                p0=(round(x0*annotated.shape[1]),round(y0*annotated.shape[0]))
                p1=(round(x1*annotated.shape[1]),round(y1*annotated.shape[0]))
                cv2.rectangle(annotated,p0,p1,(0,150,255),2)
                cv2.putText(annotated,r["id"],p0,cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
        cv2.imwrite(str(output/"target_proposals_overlay.jpg"),annotated)
    proposal = {
        "schema": "r2h_scene_proposals_v1", "bindings": sources,
        "coordinate_convention": "xyxy normalized to each image; origin top-left; x right, y down",
        "frame_interval_convention": "zero based; source times from measured arrays",
        "hidden_state_proposals": hidden, "source_keyframes": frames,
        "actor_plan_proposal": actor,
        "source_roi_candidates": rois, "target": target, "unknowns": unknowns,
        "manual_roi_used": False, "category_keyword_rules_used": False,
        "control_stage_names_used_as_contact_gt": False,
        "semantic_correspondence_approved": False, "physical_contact_approved": False,
        "training_admitted": False,
        "limitations": ["Simulator object index follows collector object order, not universal semantic identity.",
                        "Generated video can alter geometry; homography is only a candidate mapping.",
                        "Visibility and contact events belong to the simulator timebase, not generated human frames.",
                        "Stable texture does not prove a semantic background object or long-term identity."],
    }
    write_json(output/"proposal.json", proposal)
    request = correction_request(proposal)
    write_json(output/"vlm_correction_request.json", request)
    return proposal


def correction_request(proposal):
    if proposal["target"] is None:
        return {"required": False, "reason": "No generated target video supplied"}
    target = proposal["target"]
    return {
        "schema": "r2h_scene_correction_request_v1", "required": True,
        "proposal_sha256": json_sha256(proposal),
        "target_frame_sha256": target["frame"]["sha256"],
        "images": [{"label": "SOURCE_SIM_INITIAL", **proposal["source_keyframes"][0]},
                   {"label": "TARGET_GENERATED_INITIAL", **target["frame"]}],
        "roi_ids": [x["id"] for x in proposal["source_roi_candidates"]],
        "instruction": (
            "Inspect source and target initial images. Source geometry gives candidate ROIs, not truth "
            "about the edited image. For each ROI id verify the same visible interaction object or a "
            "fixed textured background patch. Return keep/correct/unknown; correct requires normalized "
            "xyxy and visible evidence. Never infer a hidden handle/occluded object. Background must "
            "not be human/robot or moving door/target object. Do not claim physical contact or training "
            "approval. Cite visible details; if scale or identity is unclear choose unknown."),
        "roi_candidates": target["roi_candidates"],
        "source_roi_candidates": proposal["source_roi_candidates"],
        "actor_plan_proposal": proposal["actor_plan_proposal"],
    }


def apply_correction(proposal, patch):
    """Immutable evidence patch: cannot change raw proposal, task labels or GT."""
    if patch.get("proposal_sha256") != json_sha256(proposal):
        raise ValueError("Correction belongs to a different proposal")
    if proposal["target"] is None or patch.get("target_frame_sha256") != proposal["target"]["frame"]["sha256"]:
        raise ValueError("Correction target image binding differs")
    allowed = {x["id"]: x for x in proposal["target"]["roi_candidates"]}
    decisions = patch.get("decisions", [])
    if {d["roi_id"] for d in decisions} != set(allowed) or len(decisions) != len(allowed):
        raise ValueError("Patch must decide each ROI exactly once")
    accepted = []
    for d in decisions:
        if d.get("status") not in {"keep", "correct", "unknown"} or not d.get("reason", "").strip():
            raise ValueError("Correction requires explicit status and evidence")
        original = allowed[d["roi_id"]]
        if d["status"] == "keep":
            if original["status"] != "candidate":
                raise ValueError("Cannot keep an unknown geometry proposal")
            box = validate_box(original["box_xyxy_normalized"])
        elif d["status"] == "correct":
            box = validate_box(d.get("box_xyxy_normalized"))
        else:
            box = None
        accepted.append({"id": d["roi_id"], "role": original["role"],
                         "status": "vlm_reviewed_candidate" if box else "unknown",
                         "box_xyxy_normalized": box, "reason": d["reason"]})
    return {"schema": "r2h_scene_corrected_rois_v1",
            "raw_proposal_sha256": json_sha256(proposal), "patch_sha256": json_sha256(patch),
            "target_frame": copy.deepcopy(proposal["target"]["frame"]),
            "roi_candidates": accepted, "training_admitted": False,
            "physical_contact_approved": False,
            "claim_scope": "VLM-reviewed 2D location candidates; not tracked-contact evidence"}
