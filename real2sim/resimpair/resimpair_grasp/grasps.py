"""Finite CPU antipodal proposals in object coordinates, not qualified grasps."""
from dataclasses import asdict
import sys
import time
import numpy as np
import trimesh

from .common import ROOT, binding, digest, read, resolve, write

GRIPPER = {"id": "panda_wristcam", "tcp": "panda_hand_tcp", "closing_axis": "+Y",
           "approach_axis": "+Z", "max_opening_m": .077, "min_opening_m": .012,
           "minimum_tcp_above_table_m": .025, "finger_tip_depth_from_tcp_m": .008,
           "physical_collision_guard_required": True}
GRIPPER["geometry_bindings"] = [binding(ROOT / "PolicyGen/mani_skill/assets/robots/panda" / p) for p in [
    "panda_v3.urdf", "franka_description/meshes/collision/hand.stl", "franka_description/meshes/collision/finger.stl"]]


def backend():
    # The vendored package's numpy/trimesh fallback runs without Open3D.
    path = ROOT / "third_party/grasp_sampler/src"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from grasp_sampler.antipodal import antipodal_grasps
    from grasp_sampler.types import GraspConfig, ObjMesh
    return antipodal_grasps, GraspConfig, ObjMesh


def panda_frame(pose):
    """Library +X closing -> Panda +Y closing; preserve TCP and approach."""
    result = np.asarray(pose, float).copy()
    closing = result[:3, 0].copy()
    # Enforce top-down instead of allowing a tilted finger to strike the table.
    if abs(closing[2]) > .06:
        return None
    closing[2] = 0.
    closing /= np.linalg.norm(closing)
    approach = np.array([0., 0., -1.])
    result[:3, :3] = np.column_stack([np.cross(closing, approach), closing, approach])
    return result


def generate(descriptor_binding, output, seed=0):
    d = read(resolve(descriptor_binding))
    if not d["roles"]["rigid_manipulation_candidate"]:
        raise ValueError("Geometry did not qualify as manipulation candidate")
    sampler, Config, Obj = backend()
    cfg = Config(gripper_max_width=.077, antipodal_min_width=.012,
        antipodal_samples=1000, antipodal_max_grasps=160,
        approach_facets=((0., 0., -1.),), cone_deg=3., directed_spread=1,
        contact_geometry_policy="mesh", friction_coef=.5, finger_len=.04)
    upstream = read(ROOT / "third_party/grasp_sampler/UPSTREAM.json")
    signature = {"descriptor": descriptor_binding, "gripper": GRIPPER,
                 "config": asdict(cfg), "seed": seed, "upstream_commit": upstream["commit"],
                 "adapter_sha256": binding(__file__)["sha256"]}
    cache_key = digest(signature)
    output = __import__("pathlib").Path(output) / (cache_key + ".json")
    if output.exists():
        return binding(output)
    start = time.monotonic()
    visual = trimesh.load(resolve(d["visual"]), force="scene", process=True).to_geometry()
    # Sampling the actual visual surface prevents filling cup handles/holes by
    # an implicit convex proxy. Cooking approximation has a separate metric.
    visual = trimesh.Trimesh(vertices=visual.vertices, faces=visual.faces, process=True)
    if len(visual.faces) > 8000:
        visual = visual.simplify_quadric_decimation(face_count=8000)
    obj = Obj(visual, visual.extents, visual.bounding_box_oriented.primitive.transform,
              visual.bounding_box_oriented.primitive.extents, "irregular")
    raw = sampler(obj, cfg, np.random.default_rng(seed))
    candidates, excluded = [], {"table_clearance": 0, "tilted_closing": 0, "duplicate": 0}
    seen = set()
    for proposal in raw:
        pose = panda_frame(proposal.pose)
        if pose is None:
            excluded["tilted_closing"] += 1; continue
        if pose[2, 3] + d["half_size"][2] < GRIPPER["minimum_tcp_above_table_m"]:
            excluded["table_clearance"] += 1; continue
        # Preserve several positions and yaw alternatives, including the
        # equivalent 180-degree closing orientation for joint-limit avoidance.
        for flip in [False, True]:
            p = pose.copy()
            if flip:
                p[:3, :2] *= -1
            yaw = float(np.arctan2(p[1, 1], p[0, 1]))
            bucket = (*np.round(p[:3, 3] / .006).astype(int), int(np.round(yaw / .12)))
            if bucket in seen:
                excluded["duplicate"] += 1; continue
            seen.add(bucket)
            radial = float(np.linalg.norm(p[:2, 3]))
            top_depth = d["half_size"][2] - p[2, 3]
            # Prefer opening margin as well as centering/depth. An almost fully
            # open jaw is sensitive to tracking and mesh errors; this ranking is
            # geometric and shared across all assets, not a case-specific pose.
            score = radial + abs(top_depth - min(.028, d["dimensions_m"][2] / 2)) * .5 + .25 * proposal.width
            candidates.append({"tcp_in_asset": p.tolist(), "width_m": proposal.width,
                "proposal": "mesh_antipodal", "geometric_rank_cost": score,
                "contact_geometry": "visual_mesh_simplified_max8000_faces",
                "physical_status": "unknown", "flip": flip})
    candidates.sort(key=lambda x: x["geometric_rank_cost"])
    candidates = candidates[:48]
    for i, c in enumerate(candidates):
        c["id"] = f"g{i:03d}"
    write(output, {"schema": "resimpair_mesh_grasp_cache_v1", "key": cache_key,
          "signature": signature, "raw_proposals": len(raw), "rejected": excluded,
          "candidates": candidates, "elapsed_seconds": time.monotonic() - start,
          "physical_qualification": "unknown", "api_calls": 0,
          "scope": "mesh contact geometry only; runtime IK, full-arm/carry collision, measured grasp/lift/place still required"})
    return binding(output)


def load(cache_binding, descriptor_binding):
    cache = read(resolve(cache_binding))
    if cache["signature"]["descriptor"] != descriptor_binding or cache["signature"]["gripper"] != GRIPPER:
        raise ValueError("Cache belongs to a different mesh/scale/gripper")
    resolve(cache["signature"]["descriptor"])
    for geometry in GRIPPER["geometry_bindings"]:
        resolve(geometry)
    for c in cache["candidates"]:
        T = np.asarray(c["tcp_in_asset"])
        if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]) or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6) or np.linalg.det(T[:3, :3]) < .999:
            raise ValueError("Invalid cached rigid transform")
    return cache
