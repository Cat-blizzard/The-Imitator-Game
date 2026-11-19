"""Independent read-only arithmetic/decode audit of hidden source conditions.

Does not import the candidate compiler, classify generated identity, or submit a
provider request. Pixel recognition remains a separately attributed observation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import av
import cv2
import numpy as np
import trimesh


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rotation(q):
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                     [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                     [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])


def project_bounds(points, camera):
    e = np.asarray(camera["camera_from_world"])
    p = np.asarray(points) @ e[:, :3].T + e[:, 3]
    assert (p[:, 2] > camera["near"]).all()
    pixel = p @ np.asarray(camera["intrinsic_matrix"]).T
    xy = pixel[:, :2] / pixel[:, 2, None] / [camera["width"], camera["height"]]
    return np.r_[xy.min(axis=0), xy.max(axis=0)]


def vertical_ray_hits(triangles, xy):
    """Moller-Trumbore intersections, independently of compiler's 2-D solver."""
    origin, direction = np.r_[xy, -10.0], np.array([0., 0., 1.])
    a = triangles[:, 0]
    edge1, edge2 = triangles[:, 1] - a, triangles[:, 2] - a
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    usable = abs(determinant) > 1e-12
    inverse = np.zeros(len(triangles))
    inverse[usable] = 1 / determinant[usable]
    s = origin - a
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inverse * (q @ direction)
    distance = inverse * np.einsum("ij,ij->i", edge2, q)
    usable &= (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8) & (distance > 0)
    return np.unique(np.round(origin[2] + distance[usable], 10))


def audit(packet_path, output):
    started = time.monotonic()
    packet_path, output = Path(packet_path), Path(output)
    if output.exists():
        raise ValueError("Audit results are immutable; choose another output")
    p = read(packet_path)
    if p["schema"] != "r2h_hidden_cabinet_content_conditions_v2":
        raise ValueError("This audit revision requires explicit raw-versus-recorded geometry fields")
    bindings = p["bindings"]
    checks = []

    def check(name, ok, **detail):
        checks.append({"name": name, "passed": bool(ok), **detail})

    for name, bound in bindings.items():
        check("binding:" + name, sha(bound["path"]) == bound["sha256"])
    receipt = read(bindings["receipt"]["path"])
    arrays = dict(np.load(bindings["arrays"]["path"], allow_pickle=False))
    qa = dict(np.load(bindings["visibility"]["path"], allow_pickle=False))
    art = dict(np.load(bindings["articulation"]["path"], allow_pickle=False))
    mesh = trimesh.load(bindings["object_collision"]["path"], force="mesh", process=False)
    body = trimesh.load(bindings["cabinet_body_collision"]["path"], force="mesh", process=False)
    selected = p["object"]["object_index"]
    states = arrays["object_states"][:, selected]
    n = len(states)
    counts = qa["object_pixels_512"][:, selected]
    expected_first = int(np.flatnonzero(counts > 0)[0])
    expected_sustained = next(i for i in range(n - 2) if min(counts[i:i+3]) >= 16)
    check("visibility_events", p["visibility_events"]["first_any_instance_pixels_frame"] == expected_first
          and p["visibility_events"]["sustained_candidate_frame"] == expected_sustained
          and p["visibility_events"]["first_rgb_recognizable_frame"] == "unknown",
          first=expected_first, sustained=expected_sustained)
    extent = np.ptp(mesh.vertices, axis=0)
    check("collision_extent_m", np.allclose(extent, p["object"]["collision_local_extent_m"], atol=1e-10, rtol=0),
          measured=extent.tolist())
    check("asset_uid", receipt["asset_specs"][selected]["uid"] == p["object"]["asset_uid"])
    check("stationary_untouched", float(np.linalg.norm(states[:, :3] - states[0, :3], axis=1).max()) < .01
          and not arrays["grasp"][:, :, selected].any())
    local = states[:, :3] - np.asarray(receipt["cabinet_position_m"])
    hits = vertical_ray_hits(body.triangles, local[0, :2])
    floor = p["slot"]["shelf_surface_z_cabinet_m"]
    ceiling = float(hits[hits > floor + extent[2]].min())
    check("physical_slot", np.isclose(hits, floor, atol=1e-8).any()
          and np.isclose(ceiling, p["slot"]["next_body_surface_z_cabinet_m"], atol=1e-8)
          and receipt["randomization"]["shelf"] == p["slot"]["physical_shelf"]
          and (local[:, 1] < 0).all(), ray_hits_z=hits.tolist())
    camera = p["camera"]
    check("camera_matches_measured_QA", np.allclose(camera["camera_from_world"], qa["camera_from_world"], atol=1e-8)
          and np.allclose(camera["intrinsic_matrix"], qa["intrinsic_512"], atol=1e-8))
    wanted = {int(k): v for k, v in p["source_frames"].items()}
    decoded, selected_decodes, pts_errors = 0, {}, []
    with av.open(bindings["video"]["path"]) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            decoded += 1
            if i >= n or frame.pts is None or abs(float(frame.pts * stream.time_base)
                                                 - arrays["timestamps_seconds"][i]) >= 1e-6:
                pts_errors.append(i)
            if i in wanted:
                rgb = frame.to_ndarray(format="rgb24")
                bound = wanted[i]
                png = cv2.cvtColor(cv2.imread(bound["path"]), cv2.COLOR_BGR2RGB)
                selected_decodes[i] = bool(np.array_equal(rgb, png))
                check("frame:" + str(i), selected_decodes[i]
                      and sha(bound["path"]) == bound["sha256"]
                      and hashlib.sha256(rgb.tobytes()).hexdigest() == bound["decoded_RGB_sha256"]
                      and bound["timestamp_seconds"] == float(arrays["timestamps_seconds"][i]))
    check("complete_decode_PTS", decoded == n == p["source_frame_count"] and not pts_errors,
          decoded=decoded, pts_errors=pts_errors)
    strict_projection_comparisons = []
    for c in p["conditions"]:
        i = c["frame"]["frame_index_0based"]
        vertices = mesh.vertices @ rotation(states[i, 3:7]).T + states[i, :3]
        box = project_bounds(vertices, camera)
        aabb = np.array(list(itertools.product(*zip(vertices.min(axis=0), vertices.max(axis=0)))))
        aabb_box = project_bounds(aabb, camera)
        recorded_box = c["recorded_QA_cooked_hull_box_normalized"]
        raw_box = c["raw_OBJ_vertices_projected_box_normalized"]
        comparison = c["raw_OBJ_vs_recorded_QA_comparison"]
        exact_equal = bool(np.array_equal(box, recorded_box))
        max_difference = float(np.max(np.abs(box - recorded_box)))
        strict_projection_comparisons.append({"role": c["role"], "exact_equal": exact_equal,
            "maximum_absolute_difference_normalized": max_difference,
            "maximum_absolute_difference_at_native_512_px": max_difference * 512,
            "difference_cause": "unknown", "acceptance_tolerance_applied": None})
        check("condition:" + c["role"], np.allclose(c["object_state"], states[i], atol=0, rtol=0)
              and np.allclose(c["door_angles_rad"], art["cabinet_qpos"][i], atol=0, rtol=0)
              and c["instance_visible_pixels_512"] == int(counts[i])
              and np.array_equal(recorded_box, qa["projected_bounds_224"][i, selected] / 224)
              and np.allclose(box, raw_box, atol=2e-6, rtol=0)
              and np.allclose(aabb_box, c["projected_world_AABB"]["box_xyxy_normalized"], atol=2e-6, rtol=0)
              and c["generated_identity_pass"] is False
              and comparison["exact_equal"] == exact_equal
              and comparison["difference_cause"] == "unknown"
              and comparison["acceptance_tolerance_applied"] is None
              and np.isclose(comparison["maximum_absolute_difference_normalized"],
                             max_difference, atol=1e-10, rtol=0),
              raw_OBJ_projection_recomputed=True, stored_QA_binding_checked=True,
              strict_raw_equals_stored_QA=exact_equal)
    spec = read(packet_path.parent / "image_request_spec.json")
    preflight = read(packet_path.parent / "payload_preflight.json")
    stages = {x["stage_id"]: x for x in spec["stages"]}
    payloads = {x["stage_id"]: x for x in preflight["stages"]}
    for label, stage in stages.items():
        args = stage["arguments"]
        images = [args["image_path"], *args["reference_images"]]
        check("payload_images:" + label,
              [x["sha256"] for x in images] == [x["sha256"] for x in payloads[label]["image_parts"]]
              and payloads[label]["transport_calls"] == 0 and not payloads[label]["generation_success"])
    check("closed_initial_no_future_image", stages["closed_initial"]["arguments"]["reference_images"] == []
          and stages["closed_initial"]["arguments"]["image_path"]["frame_index_0based"] == 0)
    check("preparation_not_generation_acceptance", p["generation_allowed"] is False
          and p["training_admitted"] is False and spec["generation_allowed"] is False
          and spec["downstream_video_path"]["video_generation_allowed"] is False
          and p["physical_qualification"]["apple_strict_qualification_inherited"] is False)
    result = {"schema": "independent_hidden_conditions_arithmetic_decode_v2",
              "packet": {"path": str(packet_path.resolve()), "sha256": sha(packet_path)},
              "implementation": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
              "source_binding_decode_arithmetic_passed": all(c["passed"] for c in checks), "checks": checks,
              "raw_OBJ_equals_recorded_QA_projection": all(c["exact_equal"] for c in strict_projection_comparisons),
              "strict_projection_comparisons": strict_projection_comparisons,
              "previous_audit": "root_arithmetic_decode_audit_01.json retained four failed equality checks; frozen audit source retained alongside it",
              "checks_count": len(checks), "decoded_frames": decoded,
              "png_frames_redecoded": len(selected_decodes), "new_paid_calls": 0, "new_gpu_runs": 0,
              "generated_identity_or_human_structure_accepted": False,
              "new_physical_acceptance_performed": False,
              "limits": ["Source arithmetic and binding audit; not a visual identity detector",
                         "Raw OBJ projection is not a reproduction of the unexported runtime collision hull; strict comparison remains failed",
                         "RGB observations are separately attributed and not inferred from QA pixels"],
              "wall_seconds": time.monotonic() - started}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return {k: result[k] for k in ["source_binding_decode_arithmetic_passed", "raw_OBJ_equals_recorded_QA_projection",
                                  "checks_count", "decoded_frames", "png_frames_redecoded", "wall_seconds"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.packet, args.output), indent=2))
