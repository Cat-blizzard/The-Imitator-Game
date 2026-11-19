"""Versioned independent audit: restore measured grasp windows for native 3-D traces.

All historical synchronization and exact-baseline gates remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import uuid

import av
import numpy as np

FIELDS = ("native_commands", "states", "object_states", "timestamps_seconds")
CAMERAS = ("zed2i", "native", "opposite")


def bind(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify(bound):
    if not isinstance(bound, dict) or bind(bound["path"]) != bound:
        raise ValueError("Evidence file changed or is missing")
    return Path(bound["path"])


def require(value, reason):
    if not value:
        raise ValueError(reason)


def longest_true_run(mask):
    longest = current = 0
    for flag in mask:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def measured_grasp_mask(grasp, *, frame_count, object_count, object_index, robot_count=None, axes=None):
    """Native Recorder grasp is [frame, robot, object], not [frame, robot].

    A legacy two-axis trace requires an explicit [frame, object] declaration;
    guessing its second dimension can silently measure the wrong entity.
    """
    grasp = np.asarray(grasp)
    require(type(object_index) is int and 0 <= object_index < object_count, "Invalid target object index")
    require(grasp.ndim in (2, 3) and grasp.shape[0] == frame_count,
            "Grasp trace has wrong frame count or rank")
    require(np.issubdtype(grasp.dtype, np.bool_) or np.issubdtype(grasp.dtype, np.number),
            "Grasp trace must contain native boolean observations")
    require(np.isfinite(grasp).all() and np.isin(grasp, [0, 1]).all(),
            "Grasp trace contains invalid/unknown values")
    if grasp.ndim == 3:
        require(axes is None or axes == ["frame", "robot", "object"],
                "Explicit grasp axes conflict with native frame/robot/object layout")
        require(grasp.shape[2] == object_count and grasp.shape[1] > 0,
                "Grasp object/robot dimensions differ from scene")
        require(robot_count is None or grasp.shape[1] == robot_count,
                "Grasp robot count differs from state trace")
        return grasp[:, :, object_index].astype(bool).any(axis=1)
    require(axes == ["frame", "object"] and grasp.shape[1] == object_count,
            "Ambiguous two-axis grasp trace: explicit frame/object axes required")
    return grasp[:, object_index].astype(bool)


def audit(attempt, baseline):
    """No generator imports; recompute hashes, decode every PTS, inspect arrays."""
    attempt, baseline = Path(attempt), Path(baseline)
    result = json.loads((attempt / "result.json").read_text())
    require(result.get("passed") is True, "Physical attempt failed; cannot audit as a qualified episode")
    receipt_path = verify(result["receipt"])
    receipt = json.loads(receipt_path.read_text())
    metadata = json.loads((attempt / "input.json").read_text())
    manifest_path = attempt / "episode/multiview/manifest.json"
    require(verify(receipt["multiview"]).resolve() == manifest_path.resolve(),
            "Receipt binds a different multiview manifest")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["schema"] == "synchronized_multiview_v1", "Wrong manifest version")
    require(str(uuid.UUID(manifest["execution_id"])) == metadata["execution_id"],
            "Execution identity mismatch")
    require(receipt["execution_id"] == manifest["execution_id"], "Receipt execution identity mismatch")
    require(not manifest["diagnostic_failure_trace"], "Diagnostic trace is not a successful execution")
    require(manifest["pre_command_capture"] is True and manifest["same_observation_all_cameras"] is True,
            "Recorder must declare the shared pre-command observation")
    with np.load(verify(manifest["shared_arrays"]), allow_pickle=False) as data:
        arrays = dict(data)
    with np.load(verify(manifest["timeline"]), allow_pickle=False) as data:
        timeline = dict(data)
    with np.load(baseline / "episode/arrays.npz", allow_pickle=False) as data:
        prior = dict(data)
    with np.load(attempt / "episode/qa_visibility.npz", allow_pickle=False) as data:
        legacy_qa = dict(data)
    with np.load(baseline / "episode/qa_visibility.npz", allow_pickle=False) as data:
        prior_qa = dict(data)
    n = manifest["length"]
    require(n == receipt["length"] == len(arrays["states"]) > 0, "State/receipt/manifest length mismatch")
    require(tuple(timeline["camera_ids"]) == CAMERAS, "Missing/extra/reordered camera streams")
    for name in ("frame_index", "state_index", "action_index"):
        np.testing.assert_array_equal(timeline[name], np.arange(n), err_msg=name)
    np.testing.assert_array_equal(timeline["camera_frame_index"], np.repeat(np.arange(n)[:, None], 3, axis=1))
    require(timeline["env_step_index"].shape == (n,), "Missing environment step evidence")
    np.testing.assert_array_equal(np.diff(timeline["env_step_index"]), np.ones(n-1))
    np.testing.assert_array_equal(timeline["timestamps_seconds"], arrays["timestamps_seconds"])
    np.testing.assert_allclose(arrays["timestamps_seconds"], np.arange(n) / manifest["fps"], atol=1e-12, rtol=0)
    require(timeline["raw_rgb_sha256"].shape == (n, 3), "Raw observation hash shape mismatch")
    require(all(len(str(x)) == 64 and set(str(x)) <= set("0123456789abcdef")
                for x in timeline["raw_rgb_sha256"].flat), "Malformed raw RGB hashes")
    objects = len(manifest["object_identity"])
    require(timeline["object_pixels"].shape == (n, 3, objects), "Pixel evidence shape mismatch")
    require(timeline["visible_bounds"].shape == (n, 3, objects, 4), "Bounds shape mismatch")
    for camera_index, name in enumerate(("object_pixels_224", "object_pixels_512")):
        np.testing.assert_array_equal(timeline["object_pixels"][:, camera_index], legacy_qa[name])
    np.testing.assert_array_equal(timeline["visible_bounds"][:, 0], legacy_qa["visible_bounds_224"])
    motion_hash = hashlib.sha256()
    differences = {}
    equality = {}
    for key in FIELDS:
        require(np.isfinite(arrays[key]).all(), "Nonfinite shared trace")
        motion_hash.update(np.ascontiguousarray(arrays[key]).tobytes())
        require(arrays[key].shape == prior[key].shape, "Baseline dynamic shape changed: " + key)
        differences[key] = float(np.max(np.abs(arrays[key].astype(float) - prior[key].astype(float))))
        # Retain the original exact gate. Report remaining independent checks
        # even when it fails; never round or replace it with a loose tolerance.
        equality[key] = bool(np.array_equal(arrays[key], prior[key]))
    require(motion_hash.hexdigest() == manifest["dynamic_trajectory_sha256"], "Incorrect dynamic fingerprint")
    np.testing.assert_array_equal(arrays["stage"], prior["stage"])
    decode = {}
    for name in CAMERAS:
        camera = manifest["cameras"][name]
        path = verify(camera["video"])
        intrinsic = np.asarray(camera["intrinsic_matrix"])
        extrinsic = np.asarray(camera["camera_from_world"])
        require(intrinsic.shape == (3, 3) and extrinsic.shape == (3, 4), "Calibration shape mismatch")
        require(np.isfinite(intrinsic).all() and np.isfinite(extrinsic).all(), "Nonfinite calibration")
        require(intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0 and 0 < camera["near"] < camera["far"],
                "Invalid camera geometry")
        np.testing.assert_allclose(intrinsic[2], [0, 0, 1], atol=1e-6, rtol=0)
        rotation = extrinsic[:, :3]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-5, rtol=0)
        require(abs(np.linalg.det(rotation) - 1) < 1e-5, "Camera rotation has wrong handedness")
        if name in ("zed2i", "native"):
            size = "224" if name == "zed2i" else "512"
            np.testing.assert_array_equal(extrinsic, prior_qa["camera_from_world"])
            np.testing.assert_array_equal(intrinsic, prior_qa["intrinsic_" + size])
        stamps = []
        with av.open(str(path)) as movie:
            for frame in movie.decode(video=0):
                require((frame.width, frame.height) == (camera["width"], camera["height"]), "Video size changed")
                require(frame.pts is not None and frame.time_base is not None, "Missing media timestamp")
                stamps.append(float(frame.pts * frame.time_base))
        np.testing.assert_allclose(stamps, arrays["timestamps_seconds"], atol=1e-5, rtol=0)
        decode[name] = {"decoded_frames": len(stamps), "PTS_max_error_seconds":
                        float(np.max(np.abs(stamps - arrays["timestamps_seconds"]))),
                        "width": camera["width"], "height": camera["height"], "file_bytes": path.stat().st_size}
    require(not np.allclose(manifest["cameras"]["native"]["camera_from_world"],
                            manifest["cameras"]["opposite"]["camera_from_world"], atol=1e-6, rtol=0),
            "New camera duplicates original viewpoint")
    arm = metadata["proposal"]["active_arm"]
    require(manifest["object_identity"][arm]["name"] == "apple", "Target identity mismatch")
    windows = {str(stage): arrays["stage"] == stage for stage in dict.fromkeys(arrays["stage"])}
    if "grasp" in arrays:
        windows["measured_grasp_true"] = measured_grasp_mask(
            arrays["grasp"], frame_count=n, object_count=objects, object_index=int(arm),
            robot_count=arrays["poses"].shape[1] if "poses" in arrays else None,
            axes=manifest.get("grasp_axes"))
    visibility = {}
    for stage, mask in windows.items():
        rows = {}
        for camera_index, name in enumerate(CAMERAS):
            count = timeline["object_pixels"][:, camera_index, arm]
            selected = count[mask]
            if not len(selected):
                continue
            # 16 pixels is inherited diagnostic evidence, never universal quality truth.
            rows[name] = {
                "frames": len(selected), "min_pixels": int(selected.min()),
                "frames_ge16": int((selected >= 16).sum()),
                "fraction_ge16": float((selected >= 16).mean()),
                "longest_visible_run_frames": longest_true_run(mask & (count >= 16)),
            }
        if rows:
            eligible = ["native", "opposite"]  # Compare like-resolution views only.
            selected_view = max(eligible, key=lambda name: (
                rows[name]["fraction_ge16"], rows[name]["longest_visible_run_frames"], rows[name]["min_pixels"]))
            visibility[stage] = {"cameras": rows, "recommended_reference_camera": selected_view}
    return {
        "schema": "independent_multiview_audit_v2", "passed": all(equality.values()),
        "auditor": bind(__file__), "manifest": bind(manifest_path),
        "baseline_receipt": bind(baseline / "episode/receipt.json"),
        "execution_id": manifest["execution_id"], "frames": n, "camera_streams": 3, "distinct_viewpoints": 2,
        "synchronization_checks_passed": True,
        "historical_bit_identity_passed": all(equality.values()),
        "shared_dynamic_trace_matches_baseline_exactly": all(equality.values()),
        "dynamic_exact_equality": equality,
        "dynamic_trajectory_sha256": motion_hash.hexdigest(), "dynamic_max_abs_differences": differences,
        "original_cameras_preserved": True, "decoded_streams": decode, "event_visibility": visibility,
        "measured_grasp_axes": ["frame", "robot", "object"] if "grasp" in arrays and arrays["grasp"].ndim == 3 else manifest.get("grasp_axes"),
        "measured_grasp_evidence_present": "grasp" in arrays,
        "recorded_executions": 1,
        "additional_exact_fingerprints_vs_baseline": int(not all(equality.values())),
        "additional_behavior_coverage_claimed": False,
        "limitations": [
            "CPU audit of artifacts and recorder evidence, not an external physical clock measurement.",
            "Raw RGB hashes are encoder-input evidence; lossy H264 decoded pixels cannot be hash-compared to them.",
            "Controller stages are event windows, not independent contact ground truth.",
            "Segmentation area diagnoses observation coverage; does not prove grasp/contact is visually interpretable.",
            "Candidate applies to static-camera apple scenes; no other task mechanism is qualified.",
            "Historical exact mismatch keeps the overall audit failed, even if differences are near zero.",
        ],
        "quality_release_approved": False, "training_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output is immutable")
    report = audit(args.attempt, args.baseline)
    source = Path(__file__).read_bytes()
    source_path = args.output.parent / (report["auditor"]["sha256"] + "_audit_multiview_v2.py")
    if not source_path.exists():
        with source_path.open("xb") as stream:
            stream.write(source)
    report["auditor_snapshot"] = bind(source_path)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in (
        "passed", "frames", "camera_streams", "distinct_viewpoints", "dynamic_max_abs_differences")}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
