"""Independent CPU review of compiled conditions and real-source counterexamples.

Does not run simulation, edit the compiler, invoke APIs, or mutate source data.
Negative fixtures change copies of actual g00 evidence with consistent hashes,
so rejection must follow semantics/timeline checks rather than a stale digest.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from data_generation_methods.r2h_pipeline.common import file_sha256, read_json, write_json


def bind(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def verify(bound):
    path = Path(bound["path"]).resolve(strict=True)
    if file_sha256(path) != bound["sha256"]:
        raise ValueError("Bound evidence changed")
    return path


def review_packet(packet_path):
    packet = read_json(packet_path)
    if packet["status"] != "prepared_for_condition_image_review" or packet["generation_allowed"] is not False:
        raise ValueError("Expected prepared conditions with generation still blocked")
    for bound in packet["bindings"].values():
        verify(bound)
    receipt = read_json(verify(packet["bindings"]["receipt"]))
    with np.load(verify(packet["bindings"]["arrays"]), allow_pickle=False) as data:
        arrays = dict(data)
    with np.load(verify(packet["bindings"]["contacts"]), allow_pickle=False) as data:
        contacts = dict(data)
    selected = packet["roles"]["selected_object"]["object_index"]
    source = packet["roles"]["source_container"]["container_index"]
    target = packet["roles"]["destination_container"]["container_index"]
    n = len(arrays["object_states"])
    assert (source, target) == (0, 1) and selected in (0, 1)
    assert packet["terminal_evidence"]["terminal_frame_index"] == n - 1
    assert packet["terminal_evidence"]["post_last_command_final_state_used_as_rgb"] is False
    mesh = np.asarray(trimesh.load(verify(packet["bindings"]["selected_object_collision"]),
                                   force="mesh").vertices)
    camera = packet["camera_calibration"]
    k, e = np.asarray(camera["intrinsic_matrix"]), np.asarray(camera["camera_from_world"])
    centers = arrays["container_centers_world"]
    uv = []
    for center in centers:
        point = e[:, :3] @ center + e[:, 3]
        pixel = k @ point
        uv.append(pixel[:2] / pixel[2])
    actual_order = "screen_left" if uv[source][0] < uv[target][0] else "screen_right"
    assert packet["state_delta"]["screen_language"]["source_container"] == actual_order
    reviewed = []
    for condition in packet["conditions"]:
        frame = condition["frame"]
        path = verify(frame)
        video = verify(frame["video"])
        index = frame["frame_index_0based"]
        assert type(index) is int and 0 <= index < n
        assert frame["timestamp_seconds"] == arrays["timestamps_seconds"][index]
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, original = cap.read()
        cap.release()
        stored = cv2.imread(str(path))
        assert ok and np.array_equal(stored, original)
        assert hashlib.sha256(cv2.cvtColor(original, cv2.COLOR_BGR2RGB).tobytes()).hexdigest() == frame["decoded_RGB_sha256"]
        state = arrays["object_states"][index, selected]
        assert np.array_equal(condition["object_pose_wxyz_world"], state[:7].astype(float))
        grasp = np.flatnonzero(arrays["grasp"][index, :, selected]).tolist()
        assert condition["simulator_grasped_by_arm_indices"] == grasp
        assert condition["aggregate_bowl_support_force_N"] == contacts["support_force_N"][index, selected]
        rotation = Rotation.from_quat(np.r_[state[4:7], state[3]]).as_matrix()
        points = mesh @ rotation.T + state[:3]
        for relation in condition["geometric_container_relations"]:
            center = centers[relation["container_index"]]
            profile = receipt["receptacle"]["floor_profile"]
            independent = (
                np.linalg.norm(points[:, :2] - center[:2], axis=1).max() < .075
                and points[:, 2].min() < profile["rim_z_m"] - .01
                and points[:, 2].min() > profile["center_inner_floor_z_m"] - .008
            )
            assert relation["geometrically_inside"] == bool(independent)
        reviewed.append({"frame_index": index, "labels": condition["labels"],
                         "actual_rgb_exact_match": True, "state_and_grasp_exact_match": True})
    initial = next(x for x in packet["conditions"] if "initial" in x["labels"])
    terminal = next(x for x in packet["conditions"] if "terminal" in x["labels"])
    assert initial["geometric_container_relations"][source]["geometrically_inside"]
    assert terminal["geometric_container_relations"][target]["geometrically_inside"]
    assert not terminal["simulator_grasped_by_arm_indices"]
    return {
        "packet": bind(packet_path), "compiler": packet["bindings"]["compiler"],
        "execution_id": packet["execution_id"], "frames": n,
        "selected_camera": packet["selected_camera"], "source_screen_order_recomputed": actual_order,
        "condition_frames": reviewed, "passed_local_binding_and_state_checks": True,
        "human_generated_condition_acceptance": "unknown_not_provided",
    }


def negative_fixture(source_packet, case, folder):
    packet = read_json(source_packet)
    original_receipt = verify(packet["bindings"]["receipt"])
    receipt = deepcopy(read_json(original_receipt))
    verification = deepcopy(read_json(verify(packet["bindings"]["verification"])))
    manifest = deepcopy(read_json(verify(packet["bindings"]["multiview_manifest"])))
    with np.load(verify(packet["bindings"]["arrays"]), allow_pickle=False) as values:
        arrays = {k: v.copy() for k, v in values.items()}
    with np.load(verify(packet["bindings"]["contacts"]), allow_pickle=False) as values:
        contacts = {k: v.copy() for k, v in values.items()}
    selected = packet["roles"]["selected_object"]["object_index"]
    folder.mkdir(parents=True, exist_ok=False)
    if case == "swapped_source_target":
        receipt["randomization"]["source_bowl_index"] = 1
        receipt["randomization"]["destination_bowl_index"] = 0
    elif case == "terminal_frame_with_source_occupancy":
        arrays["object_states"][-10:, selected, :7] = arrays["object_states"][0, selected, :7]
    elif case == "grasp_absent_despite_lift_phase":
        arrays["grasp"][:, :, selected] = False
    elif case == "terminal_still_grasped":
        arrays["grasp"][-1, :, selected] = True
    elif case == "terminal_support_missing":
        contacts["support_force_N"][-20:, selected] = 0
    elif case == "camera_role_direction_copied_across_views":
        manifest["cameras"]["opposite"]["camera_from_world"] = manifest["cameras"]["native"]["camera_from_world"]
    elif case == "wrong_media_or_frame_binding":
        video = folder / "short_opposite.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (512, 512))
        assert writer.isOpened()
        for value in (0, 80, 160):
            writer.write(np.full((512, 512, 3), value, dtype=np.uint8))
        writer.release()
        manifest["cameras"]["opposite"]["video"] = bind(video)
    else:
        raise ValueError(case)
    arrays_path, contacts_path = folder / "arrays.npz", folder / "bowl_contact.npz"
    np.savez_compressed(arrays_path, **arrays)
    np.savez_compressed(contacts_path, **contacts)
    receipt["arrays"] = str(arrays_path)
    receipt["sha256"]["arrays.npz"] = file_sha256(arrays_path)
    receipt["contact_data"] = str(contacts_path)
    receipt["contact_sha256"] = file_sha256(contacts_path)
    manifest["shared_arrays"] = bind(arrays_path)
    fingerprint = hashlib.sha256()
    for field in manifest["dynamic_hash_fields"]:
        fingerprint.update(np.ascontiguousarray(arrays[field]).tobytes())
    manifest["dynamic_trajectory_sha256"] = fingerprint.hexdigest()
    manifest_path = folder / "multiview_manifest.json"
    write_json(manifest_path, manifest)
    receipt["multiview"] = bind(manifest_path)
    # Original relative recording payloads remain immutable; local links allow
    # the real verifier to resolve unchanged bytes beside the copied receipt.
    for name in receipt["sha256"]:
        if name != "arrays.npz":
            (folder / name).symlink_to((original_receipt.parent / name).resolve(strict=True))
    receipt_path = folder / "receipt.json"
    write_json(receipt_path, receipt)
    verification["receipt_sha256"] = file_sha256(receipt_path)
    verification_path = folder / "verification_claim.json"
    write_json(verification_path, verification)
    write_json(folder / "counterexample_provenance.json", {
        "kind": "synthetic_mutation_of_real_accepted_evidence_not_new_simulation",
        "case": case, "source_packet": bind(source_packet),
        "claimed_verification_is_intentionally_rebound_to_test_fresh_checks": True,
        "source_data_mutated": False,
    })
    return receipt_path, verification_path


def run(development, validation, output, protocol):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixed = read_json(protocol)
    positive = [review_packet(p) for p in (development, validation)]
    assert positive[0]["compiler"]["sha256"] == positive[1]["compiler"]["sha256"]
    from data_generation_methods.r2h_pipeline.sim_state_conditions import compile_conditions
    assert file_sha256(Path(compile_conditions.__code__.co_filename)) == positive[0]["compiler"]["sha256"]
    negative = []
    for item in fixed["fixed_negative_controls"]:
        case = item["id"]
        folder = output / "counterexamples" / case
        receipt, verification = negative_fixture(development, case, folder / "inputs")
        result = compile_conditions(receipt, verification, "opposite", folder / "compiler_output",
                                    partition="development")
        blocked = result["status"] == "blocked_missing_or_inconsistent_evidence"
        negative.append({"case": case, "blocked": blocked, "status": result["status"],
                         "reason": result.get("reason"), "result": bind(folder / "compiler_output/conditions.json")})
    result = {
        "schema": "independent_sim_state_condition_audit_v1",
        "protocol": bind(protocol), "reviewer_source": bind(__file__),
        "positive_packets": positive, "real_source_counterexamples": negative,
        "all_negative_controls_blocked": all(x["blocked"] for x in negative),
        "api_calls": 0, "gpu_runs": 0, "generated_human_images": 0,
        "training_admitted": False,
        "limitations": [
            "Negative fixtures are controlled mutations of g00 evidence, not newly failed physics runs.",
            "g01 was historically used for strict planning and shares a human source; only compiler-layout generalization is tested.",
            "Simulation conditions do not verify edited human images or establish general object-identity detection.",
        ],
    }
    write_json(output / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("development", "validation", "output", "protocol"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    report = run(args.development, args.validation, args.output, args.protocol)
    print(json.dumps({"positive_packets": len(report["positive_packets"]),
                      "all_negative_controls_blocked": report["all_negative_controls_blocked"]}))
    raise SystemExit(0 if report["all_negative_controls_blocked"] else 2)
