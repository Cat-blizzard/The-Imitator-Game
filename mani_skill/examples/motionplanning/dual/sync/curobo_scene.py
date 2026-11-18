"""Read-only, portable Panda/PhysX snapshots for the optional cuRobo backend.

All poses are XYZ+WXYZ in metres. Robot collision spheres are link-local;
world obstacles and the TCP are expressed in the selected robot base frame.
The seven arm joints retain controller order; the two measured finger joints
are locked in a nine-joint input cspace which cuRobo subsequently reduces.
This exporter never steps an environment or changes a physical state. It may
create a content-addressed derived URDF with independent physical fingers. Its
conservative sphere model does not replace the existing exact paired checker.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import os
import tempfile
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

from ._arrays import as_float_array as _array
from .curobo_spheres import coarse_convex_spheres as _cover_convex, tight_convex_spheres


ARM_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
FINGER_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
EXPORT_VERSION = "tig-curobo-actual-physx-independent-fingers-with-base-v2"
PAYLOAD_LINK = "tig_payload"
FINGER_LINKS = ("panda_leftfinger", "panda_rightfinger")
SOURCE_TABLE = "scene::table::shape_0"
FINGER_SPHERE_PITCH_M = .005
HAND_POSITIVE_Z_PADDING_M = .002


def _cache_root(cache_dir: str | Path | None) -> Path:
    """Keep derived planner assets outside the source tree by default."""
    if cache_dir is None:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "mani_skill" / "curobo"
    else:
        root = Path(cache_dir).expanduser()
    return root.resolve()


def _finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = _array(value)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def _positive(value: Any, label: str, *, zero: bool = False) -> np.ndarray:
    result = _array(value)
    if not np.isfinite(result).all() or not ((result >= 0).all() if zero else (result > 0).all()):
        raise ValueError(f"Invalid {label}")
    return result


def _single(value: Any, width: int, label: str) -> np.ndarray:
    result = _array(value)
    if result.shape == (1, width):
        result = result[0]
    return _finite(result, (width,), label)


def _matrix(pose: Any) -> np.ndarray:
    """Convert a single SAPIEN/ManiSkill pose or a strict XYZ+WXYZ[7]."""
    if hasattr(pose, "p") and hasattr(pose, "q"):
        p, q = _single(pose.p, 3, "pose XYZ"), _single(pose.q, 4, "pose WXYZ")
    else:
        vector = _finite(pose, (7,), "XYZ+WXYZ pose")
        p, q = vector[:3], vector[3:]
    if not math.isclose(float(np.linalg.norm(q)), 1., rel_tol=0, abs_tol=1e-5):
        raise ValueError("Pose quaternion must be normalized")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = p
    return result


def _pose(matrix: Any) -> list[float]:
    matrix = _finite(matrix, (4, 4), "rigid transform")
    if (not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), rtol=0, atol=1e-8)
            or not math.isclose(float(np.linalg.det(matrix[:3, :3])), 1., abs_tol=1e-8)):
        raise ValueError("Transform is not SE(3)")
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
    if q[0] < 0:
        q = -q
    return np.r_[matrix[:3, 3], q].tolist()


def _transform_vertices(vertices: Any, transform: Any) -> np.ndarray:
    vertices = _array(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Vertices must be nonempty finite [N,3]")
    transform = _matrix(_pose(transform))
    result = vertices @ transform[:3, :3].T + transform[:3, 3]
    assert result.shape == vertices.shape and np.isfinite(result).all()
    return result


def _mesh_arrays(vertices: Any, faces: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = _array(vertices)
    original_faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
            or not np.isfinite(vertices).all() or original_faces.ndim != 2
            or original_faces.shape[1] != 3 or len(original_faces) == 0
            or original_faces.dtype.kind not in "iu"):
        raise ValueError("Mesh requires finite vertices[N>=4,3] and integer faces[M,3]")
    faces = original_faces.astype(np.int64)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Mesh face index outside vertex array")
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces)


def _kind(shape: Any) -> str:
    kind = type(shape).__name__
    accepted = {"PhysxCollisionShapeBox", "PhysxCollisionShapeSphere", "PhysxCollisionShapePlane",
                "PhysxCollisionShapeConvexMesh", "PhysxCollisionShapeTriangleMesh",
                "PhysxCollisionShapeCapsule", "PhysxCollisionShapeCylinder"}
    if kind not in accepted:
        raise ValueError(f"Unsupported collision shape: {kind}")
    return kind


def _shape_mesh(shape: Any) -> tuple[np.ndarray, np.ndarray, str]:
    """Shape-local mesh; curved primitive proxies conservatively contain solids."""
    kind = _kind(shape)
    if kind == "PhysxCollisionShapeBox":
        half = _finite(_positive(shape.half_size, "box half size"), (3,), "box half size")
        mesh = trimesh.creation.box(extents=2 * half)
        policy = "exact_box"
    elif kind in ("PhysxCollisionShapeConvexMesh", "PhysxCollisionShapeTriangleMesh"):
        vertices = _array(shape.vertices)
        scale = _finite(_positive(shape.scale, "mesh scale"), (3,), "mesh scale")
        faces = np.asarray(shape.get_triangles(), dtype=np.int64).reshape(-1, 3)
        vertices, faces = _mesh_arrays(vertices * scale, faces)
        return vertices, faces, "actual_scaled_native_collision_mesh"
    elif kind == "PhysxCollisionShapeSphere":
        radius = float(_positive(shape.radius, "sphere radius"))
        # A circumscribed cube is used only for the world mesh representation;
        # the robot model uses one exact sphere instead.
        mesh = trimesh.creation.box(extents=[2 * radius] * 3)
        policy = "conservative_sphere_bounding_cube"
    elif kind == "PhysxCollisionShapeCylinder":
        radius = float(_positive(shape.radius, "cylinder radius"))
        half = float(_positive(shape.half_length, "cylinder half length"))
        mesh = trimesh.creation.cylinder(radius=radius / math.cos(math.pi / 32), height=2 * half, sections=32)
        # Trimesh longitudinal Z -> PhysX longitudinal X.
        rotation = Rotation.from_euler("y", math.pi / 2).as_matrix()
        mesh.vertices = np.asarray(mesh.vertices) @ rotation.T
        policy = "conservative_circumscribed_32gon_cylinder"
    elif kind == "PhysxCollisionShapeCapsule":
        radius = float(_positive(shape.radius, "capsule radius"))
        half = float(_positive(shape.half_length, "capsule half length", zero=True))
        mesh = trimesh.creation.box(extents=[2 * (half + radius), 2 * radius, 2 * radius])
        policy = "conservative_capsule_bounding_box"
    else:
        raise ValueError("Infinite plane needs the explicit bounded halfspace proxy")
    vertices, faces = _mesh_arrays(mesh.vertices, mesh.faces)
    return vertices, faces, policy


def _shape_spheres(shape: Any, pitch_m: float, *, tight: bool = False,
                   hand_envelope: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cover the full solid in link coordinates; tight fits occupied cells.

    The Panda hand uses an owner-link positive-Z envelope, towards its TCP.
    Apply the collision shape's local pose before fitting this envelope so a
    rotated shape cannot accidentally constrain its own unrelated local Z.
    """
    if hand_envelope and not tight:
        raise ValueError("The hand envelope requires a tight full-solid cover")
    local = _matrix(shape.get_local_pose())
    if _kind(shape) == "PhysxCollisionShapeSphere":
        centers = local[None, :3, 3]
        radius = float(_positive(shape.radius, "sphere radius"))
        policy = "exact_single_sphere"
    else:
        vertices, _, mesh_policy = _shape_mesh(shape)
        if tight:
            if hand_envelope:
                vertices = _transform_vertices(vertices, local)
            values, proof = tight_convex_spheres(vertices, pitch_m,
                positive_z_padding_m=HAND_POSITIVE_Z_PADDING_M if hand_envelope else None,
                max_depth=2)
            if not hand_envelope:
                values[:, :3] = _transform_vertices(values[:, :3], local)
            return [{"center": row[:3].tolist(), "radius": float(row[3])} for row in values], {
                **proof, "geometry_policy": mesh_policy, "shape_local_pose_wxyz": _pose(local),
                "adaptive_rule_task_independent": hand_envelope,
                "envelope_frame": "owner_link" if hand_envelope else None}
        frozen, radius = _coarse_cover_cached(
            np.ascontiguousarray(vertices, dtype=np.float64).tobytes(), len(vertices), float(pitch_m))
        centers = np.asarray(frozen, dtype=np.float64)
        centers = _transform_vertices(centers, local)
        policy = "convex_solid_grid_cell_circumspheres;" + mesh_policy
    spheres = [{"center": c.tolist(), "radius": radius} for c in centers]
    return spheres, {"sphere_count": len(spheres), "coverage_policy": policy,
                     "sphere_radius_m": radius, "shape_local_pose_wxyz": _pose(local)}


@lru_cache(maxsize=128)
def _coarse_cover_cached(vertices_bytes: bytes, count: int, pitch_m: float
                         ) -> tuple[tuple[tuple[float, ...], ...], float]:
    vertices = np.frombuffer(vertices_bytes, dtype=np.float64).reshape(count, 3)
    centers, radius = _cover_convex(vertices, pitch_m)
    return tuple(tuple(float(x) for x in row) for row in centers), float(radius)


def attached_shape_spheres(shape: Any, pitch_m: float, tcp_to_object_wxyz: Any
                           ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit a complete held solid with a tighter envelope towards the hand.

    ``tcp_to_object_wxyz`` maps actor-local points to the measured TCP frame at
    the first confirmed grasp. Return spheres in the ORIGINAL actor frame for
    the attachment manager. A half-turn about TCP X maps negative TCP Z onto
    the existing tight fitter's positive Z. Its 2 mm envelope bounds only the
    conservative approximation; no physical geometry or collision is ignored.
    The fitted actor-local model remains a full-solid cover after any motion.
    """
    tcp_from_actor = _matrix(tcp_to_object_wxyz)
    if _kind(shape) == "PhysxCollisionShapeSphere":
        return _shape_spheres(shape, pitch_m)
    local = _matrix(shape.get_local_pose())
    vertices, _, mesh_policy = _shape_mesh(shape)
    fit_from_actor = np.diag([1., -1., -1., 1.]) @ tcp_from_actor
    fitted_vertices = _transform_vertices(vertices, fit_from_actor @ local)
    values, proof = tight_convex_spheres(fitted_vertices, pitch_m,
        positive_z_padding_m=HAND_POSITIVE_Z_PADDING_M, max_depth=2)
    values[:, :3] = _transform_vertices(values[:, :3], np.linalg.inv(fit_from_actor))
    return [{"center": row[:3].tolist(), "radius": float(row[3])} for row in values], {
        **proof, "geometry_policy": mesh_policy, "shape_local_pose_wxyz": _pose(local),
        "fit_frame": "TCP rotated 180 degrees about X",
        "tight_envelope_direction": "negative TCP Z towards hand",
        "output_frame": "actor_local", "tcp_to_object_wxyz": _pose(tcp_from_actor),
        "task_or_seed_dependent": False}


def _plane_proxy(shape_to_base: np.ndarray) -> tuple[list[float], list[float]]:
    # PhysX outward plane normal is +X. The proxy occupies local X [-4,0].
    offset = np.eye(4)
    offset[0, 3] = -2.
    return _pose(shape_to_base @ offset), [4., 10., 10.]


def _put_world_shape(world: dict[str, Any], name: str, shape: Any,
                     entity_to_base: np.ndarray) -> dict[str, Any]:
    if not name or any(name in group for group in world.values()):
        raise ValueError("World collision names must be unique")
    kind = _kind(shape)
    shape_to_base = entity_to_base @ _matrix(shape.get_local_pose())
    if kind == "PhysxCollisionShapePlane":
        pose, dims = _plane_proxy(shape_to_base)
        world["cuboid"][name] = {"pose": pose, "dims": dims}
        policy = "bounded_halfspace_slab_local_X_minus4_to0_tangent10m"
    elif kind == "PhysxCollisionShapeBox":
        dims = 2 * _finite(_positive(shape.half_size, "box half size"), (3,), "box half size")
        world["cuboid"][name] = {"pose": _pose(shape_to_base), "dims": dims.tolist()}
        policy = "exact_box"
    else:
        vertices, faces, policy = _shape_mesh(shape)
        world["mesh"][name] = {"pose": _pose(shape_to_base),
                                "vertices": vertices.tolist(), "faces": faces.tolist()}
    return {"name": name, "type": kind, "geometry_policy": policy,
            "shape_pose_base_wxyz": _pose(shape_to_base)}


def _srdf_ignores(path: Path, collision_links: list[str]) -> tuple[dict[str, list[str]], int]:
    allowed = set(collision_links)
    result: dict[str, set[str]] = {}
    pair_count = 0
    seen: set[tuple[str, str]] = set()
    for entry in ET.parse(path).getroot().findall("disable_collisions"):
        first, second = entry.attrib["link1"], entry.attrib["link2"]
        if not first or not second or first == second:
            raise ValueError("Malformed SRDF disabled collision pair")
        pair = tuple(sorted((first, second)))
        if pair in seen:
            continue
        seen.add(pair)
        pair_count += 1
        if first in allowed and second in allowed:
            result.setdefault(first, set()).add(second)
            result.setdefault(second, set()).add(first)
    return {name: sorted(values) for name, values in sorted(result.items())}, pair_count


@lru_cache(maxsize=16)
def _urdf_metadata(path: Path, joint_names: tuple[str, ...], source: bytes
                   ) -> tuple[tuple[float, ...], tuple[tuple[str, str], ...]]:
    root = ET.fromstring(source)
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    velocities = []
    for name in joint_names:
        if name not in joints or joints[name].find("limit") is None:
            raise ValueError(f"URDF joint lacks limits: {name}")
        velocity = float(joints[name].find("limit").attrib["velocity"])
        if not math.isfinite(velocity) or velocity <= 0:
            raise ValueError(f"Invalid URDF joint velocity: {name}")
        velocities.append(velocity)
    hashes = {str(path): hashlib.sha256(source).hexdigest()}
    for mesh in root.findall(".//collision/geometry/mesh"):
        filename = mesh.attrib["filename"]
        if filename.startswith("package://"):
            filename = filename[len("package://"):]
        asset = (path.parent / filename).resolve()
        if not asset.is_file():
            raise ValueError(f"Missing real Panda collision asset: {asset}")
        hashes[str(asset)] = hashlib.sha256(asset.read_bytes()).hexdigest()
    return tuple(velocities), tuple(hashes.items())


def _independent_finger_urdf(original: Path, *, output_root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Derive an immutable planner-only model of SAPIEN's two physical fingers.

    TIG exposes both finger joints as independent active state variables, even
    though its input URDF contains a mimic annotation. V2 enforces that URDF
    annotation and forbids separately locking the mimic joint. Removing exactly
    that annotation lets the planner lock the two actual measured states. No
    axes, origins, limits, collision shapes or physical files are changed.
    """
    original = original.resolve()
    destination, metadata = _independent_finger_urdf_cached(
        original, original.read_bytes(), _cache_root(output_root))
    return destination, json.loads(metadata)


@lru_cache(maxsize=16)
def _independent_finger_urdf_cached(original: Path, before: bytes,
                                  output_root: Path) -> tuple[Path, str]:
    def parse(content: bytes) -> ET.Element:
        return ET.fromstring(content, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    tree = parse(before)
    joints = {joint.attrib["name"]: joint for joint in tree.findall("joint")}
    if any(name not in joints for name in FINGER_NAMES):
        raise ValueError("Original Panda URDF lacks its two independent physical finger joints")
    joint = joints[FINGER_NAMES[1]]
    mimics = joint.findall("mimic")
    if (len(mimics) != 1 or mimics[0].attrib.get("joint") != FINGER_NAMES[0]
            or float(mimics[0].attrib.get("multiplier", "1")) != 1.
            or float(mimics[0].attrib.get("offset", "0")) != 0.):
        raise ValueError("Unexpected Panda finger mimic annotation; derivation refuses ambiguity")
    other_mimics = [entry for parent in tree.findall("joint") for entry in parent.findall("mimic")
                   if parent.attrib["name"] != FINGER_NAMES[1]]
    if other_mimics:
        raise ValueError("Derived Panda model may remove only the known second-finger mimic")
    removed = dict(mimics[0].attrib)
    joint.remove(mimics[0])
    rewritten_meshes = []
    for mesh in tree.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        relative = filename[len("package://"):] if filename.startswith("package://") else filename
        path = (original.parent / relative).resolve()
        if not path.is_file():
            raise ValueError(f"Original Panda mesh is missing: {path}")
        mesh.attrib["filename"] = str(path)
        rewritten_meshes.append({"original": filename, "absolute": str(path),
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    payload = ET.tostring(tree, encoding="utf-8", xml_declaration=True) + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    destination = output_root / "panda-independent-fingers" / digest / "panda_v3.urdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete file atomically; simultaneous left/right exports may
    # prepare the same immutable model. The downloaded physical file is intact.
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        staging = Path(handle.name)
        handle.write(payload)
    try:
        try:
            os.link(staging, destination)
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise RuntimeError("Content-addressed derived URDF identity was changed")
    finally:
        staging.unlink()
    # Compare canonical parsed semantics after undoing only the declared edits.
    restored = parse(payload)
    restored_joints = {node.attrib["name"]: node for node in restored.findall("joint")}
    ET.SubElement(restored_joints[FINGER_NAMES[1]], "mimic", removed)
    for mesh, change in zip(restored.findall(".//mesh"), rewritten_meshes, strict=True):
        mesh.attrib["filename"] = change["original"]
    def semantics(element: ET.Element) -> Any:
        # Whitespace and serialization order are not URDF semantics. Sibling
        # element order is retained except the restored mimic child position.
        return (element.tag, tuple(sorted(element.attrib.items())),
                (element.text or "").strip(), sorted((semantics(child) for child in element), key=repr))
    if semantics(restored) != semantics(parse(before)):
        raise RuntimeError("Derived URDF changed undeclared semantics")
    if original.read_bytes() != before:
        raise RuntimeError("Original physical URDF changed during planner-model derivation")
    return destination, json.dumps({"policy": "planner_only_remove_finger2_mimic_and_absolutize_mesh_paths_v1",
        "reason": "actual SAPIEN state has nine independent active joints; preserve both measured finger values instead of forcing equality",
        "original_urdf": str(original), "original_sha256": hashlib.sha256(before).hexdigest(),
        "derived_urdf": str(destination), "derived_sha256": digest,
        "removed_mimic": {"joint_name": FINGER_NAMES[1], "attributes": removed},
        "mesh_path_rewrites": rewritten_meshes, "other_parsed_urdf_semantics_unchanged": True,
        "physical_urdf_modified": False}, allow_nan=False)


def _robot_cfg(path: Path, spheres: dict[str, list[dict[str, Any]]],
               ignores: dict[str, list[str]], q9: Any, velocities: Any) -> dict[str, Any]:
    """Build the official V2 inner RobotCfg dictionary with measured lock state."""
    names = list(ARM_NAMES + FINGER_NAMES)
    q9 = _finite(q9, (9,), "nine-joint default configuration")
    velocities = _finite(_positive(velocities, "URDF velocities"), (9,), "URDF velocities")
    links = list(spheres)
    if not path.is_absolute() or not links or any(not values for values in spheres.values()):
        raise ValueError("An absolute actual URDF and nonempty collision sphere links are required")
    result = {"kinematics": {"format_version": 2.,
        "urdf_path": str(path), "asset_root_path": str(path.parent),
        "base_link": "panda_link0", "tool_frames": ["panda_hand_tcp"],
        "collision_link_names": links, "collision_spheres": spheres,
        "collision_sphere_buffer": 0., "use_global_cumul": True,
        "self_collision_ignore": ignores,
        "self_collision_buffer": {name: 0. for name in links},
        "lock_joints": {name: float(q9[names.index(name)]) for name in FINGER_NAMES},
        "cspace": {"joint_names": names, "default_joint_position": q9.tolist(),
            "null_space_weight": [1.] * 9, "cspace_distance_weight": [1.] * 9,
            "velocity_scale": (.9 / velocities).tolist(),
            "max_acceleration": .9, "max_jerk": 500., "position_limit_clip": 0.}},
        "load_dynamics": False}
    json.dumps(result, allow_nan=False)
    return result


def reserve_payload_config(robot_cfg: dict[str, Any], capacity: int) -> dict[str, Any]:
    """Copy a physical model and reserve disabled spheres on a fixed TCP link.

    The payload has zero nominal mass/inertia here. A sphere fit is collision
    geometry, not a calibrated dynamics model. The caller attaches measured
    geometry later; only the legitimate payload/finger pairs are ignored.
    """
    if type(capacity) is not int or not 1 <= capacity <= 100_000:
        raise ValueError("Payload capacity must be an integer in [1,100000]")
    result = copy.deepcopy(robot_cfg)
    kin = result["kinematics"]
    links = kin["collision_link_names"]
    if (not isinstance(links, list) or len(set(links)) != len(links)
            or any(link not in links for link in FINGER_LINKS)
            or PAYLOAD_LINK in links or PAYLOAD_LINK in kin.get("collision_spheres", {})):
        raise ValueError("Expected an unextended physical robot with both finger links")
    if kin.get("extra_links") or kin.get("extra_collision_spheres"):
        raise ValueError("Existing extra links/spheres cannot be overwritten")
    if kin.get("tool_frames") != ["panda_hand_tcp"]:
        raise ValueError("Payload must be fixed to panda_hand_tcp")
    ignores = kin.setdefault("self_collision_ignore", {})
    if PAYLOAD_LINK in ignores or any(PAYLOAD_LINK in values for values in ignores.values()):
        raise ValueError("Reserved payload name already occurs in the collision policy")
    links.append(PAYLOAD_LINK)
    kin["collision_spheres"][PAYLOAD_LINK] = [
        {"center": [0., 0., 0.], "radius": -1.} for _ in range(capacity)]
    kin["extra_collision_spheres"] = {PAYLOAD_LINK: capacity}
    kin["extra_links"] = {PAYLOAD_LINK: {
        "parent_link_name": "panda_hand_tcp", "link_name": PAYLOAD_LINK,
        "joint_name": "tig_payload_fixed_joint", "joint_type": "FIXED",
        "fixed_transform": [0., 0., 0., 1., 0., 0., 0.],
        "link_mass": 0., "link_inertia": [0.] * 6}}
    ignores[PAYLOAD_LINK] = list(FINGER_LINKS)
    for link in FINGER_LINKS:
        ignores.setdefault(link, []).append(PAYLOAD_LINK)
    kin.setdefault("self_collision_buffer", {})[PAYLOAD_LINK] = 0.
    json.dumps(result, allow_nan=False)
    return result


def resize_payload_config(robot_cfg: dict[str, Any], capacity: int) -> dict[str, Any]:
    """Resize only disabled storage on an already reserved payload link."""
    if type(capacity) is not int or not 1 <= capacity <= 100_000:
        raise ValueError("Payload capacity must be an integer in [1,100000]")
    result = copy.deepcopy(robot_cfg)
    kin = result["kinematics"]
    previous = kin.get("extra_collision_spheres", {}).get(PAYLOAD_LINK)
    spheres = kin.get("collision_spheres", {}).get(PAYLOAD_LINK, [])
    if (type(previous) is not int or len(spheres) != previous
            or any(sphere != {"center": [0., 0., 0.], "radius": -1.} for sphere in spheres)):
        raise ValueError("Expected the disabled payload storage from reserve_payload_config")
    kin["collision_spheres"][PAYLOAD_LINK] = [
        {"center": [0., 0., 0.], "radius": -1.} for _ in range(capacity)]
    kin["extra_collision_spheres"][PAYLOAD_LINK] = capacity
    return result


def _entity_shapes(entity: Any) -> list[Any]:
    import sapien.physx as physx
    bodies = [c for c in entity.components if isinstance(c, physx.PhysxRigidBaseComponent)]
    if len(bodies) > 1 or (bodies and isinstance(bodies[0], physx.PhysxArticulationLinkComponent)):
        raise ValueError("Invalid nonrobot rigid-body registration")
    return list(bodies[0].get_collision_shapes()) if bodies else []


def export_curobo_snapshot(env: Any, arm_index: int, *, sphere_pitch_m: float = .025,
                           cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Export one read-only, fully JSON-compatible selected-arm planning snapshot.

    Input: the actual single CPU-physics dual-Panda TIG environment, canonical
    arm index 0=left or 1=right. Output: ``robot_cfg`` is the RobotConfig inner
    dictionary with its ``kinematics`` member; ``world`` is WorldConfig.from_dict
    compatible. q_start/joint_names contain seven arm joints in controller
    order; base/TCP poses have seven XYZ+WXYZ entries. Peer geometry is measured,
    hence both independently planned paths still require paired FCL checking.
    """
    if isinstance(arm_index, bool) or not isinstance(arm_index, int) or arm_index not in (0, 1):
        raise ValueError("arm_index must be integer 0=left or 1=right")
    pitch = float(sphere_pitch_m)
    if not math.isfinite(pitch) or pitch <= 0:
        raise ValueError("Sphere pitch must be finite and positive")
    from .scene_collision import scene_entities
    raw = getattr(env, "unwrapped", env)
    if getattr(raw, "num_envs", None) != 1 or len(raw.agent.agents) != 2:
        raise ValueError("A single dual-Panda environment is required")
    agents = tuple(raw.agent.agents)
    if hasattr(raw, "left_agent") and (agents[0] is not raw.left_agent or agents[1] is not raw.right_agent):
        raise ValueError("Physical left/right agent order changed")
    agent = agents[arm_index]
    robot = agent.robot
    accepted_robots = {instance for entry in agents for instance in entry.robot._objs}
    for articulation in raw.scene.articulations.values():
        instances = list(articulation._objs)
        if len(instances) != 1 or instances[0] not in accepted_robots:
            raise ValueError("External or batched articulation cannot be omitted from cuRobo world")
    for link in raw.scene.px.get_articulation_link_components():
        if link.articulation not in accepted_robots:
            raise ValueError("Unregistered external articulation cannot be omitted from cuRobo world")
    path = Path(agent.urdf_path).resolve()
    srdf = path.with_suffix(".srdf")
    if path.name != "panda_v3.urdf" or not path.is_file() or not srdf.is_file():
        raise ValueError("This exporter requires the actual PandaWristCam v3 URDF and SRDF")
    full_names = [joint.get_name() for joint in robot.get_active_joints()]
    arm_names = list(agent.controller.controllers["arm"].config.joint_names)
    if tuple(arm_names) != ARM_NAMES or len(full_names) != 9 or set(full_names) != set(ARM_NAMES + FINGER_NAMES):
        raise ValueError("Unexpected actual Panda controller/joint mapping")
    q = _single(robot.get_qpos(), 9, "actual Panda joint state")
    q_by_name = dict(zip(full_names, q))
    cspace_names = list(ARM_NAMES + FINGER_NAMES)
    q9 = [float(q_by_name[name]) for name in cspace_names]
    base_world = _matrix(robot.pose)
    world_base = np.linalg.inv(base_world)
    coverage: dict[str, Any] = {"all_covered": False, "snapshot_only": True,
        "world_frame": "selected_panda_link0", "sphere_pitch_m": pitch,
        "robot_sphere_policy": "complete_convex_solid_with_base_tight_hand_fine_fingers_v2",
        "finger_sphere_pitch_m": min(pitch, FINGER_SPHERE_PITCH_M),
        "hand_sphere_refinement": {"frame": "panda_hand", "positive_z_padding_m": HAND_POSITIVE_Z_PADDING_M,
            "maximum_refinement_depth": 2, "subdivision_axes": ["X", "Y"],
            "all_actual_hand_solids_covered": True, "task_or_seed_dependent": False},
        "active_robot_shapes": [], "world_shapes": [], "skipped_visual_entities": [],
        "active_robot_exceptions": [], "peer_is_measured_snapshot": True,
        "paired_exact_collision_recheck_required": True,
        "infinite_halfspace_external_guard_required": True,
        "plane_proxy_scope": {"depth_m": 4., "tangent_extent_m": 10.,
            "selected_robot_reach_bound_m": 2., "external_infinite_halfspace_guard_remains_required": True},
        "physical_state_changed": False, "environment_steps": 0}
    spheres: dict[str, list[dict[str, Any]]] = {}
    actual_links = list(robot.get_links())
    for link in actual_links:
        bodies = list(link._objs)
        if len(bodies) != 1:
            raise ValueError("Only single-instance physical links are supported")
        shapes = list(bodies[0].get_collision_shapes())
        name = link.get_name()
        for index, shape in enumerate(shapes):
            shape_pitch = min(pitch, FINGER_SPHERE_PITCH_M) if name in FINGER_LINKS else pitch
            values, proof = _shape_spheres(shape, shape_pitch,
                tight=name in ("panda_link0", "panda_link5", "panda_link6", "panda_hand"),
                hand_envelope=name == "panda_hand")
            spheres.setdefault(name, []).extend(values)
            coverage["active_robot_shapes"].append({"link": name, "shape_index": index,
                                                    "type": _kind(shape),
                                                    "sphere_pitch_m": shape_pitch, **proof})
    if not spheres or any(name not in spheres for name in (
            "panda_link0", "camera_link", "panda_hand", *FINGER_LINKS)):
        raise ValueError("Panda base, hand, camera and both actual finger geometries must be covered")
    collision_links = list(spheres)
    ignores, disabled_pairs = _srdf_ignores(srdf, collision_links)
    velocities, hashes = _urdf_metadata(path, tuple(cspace_names), path.read_bytes())
    source_hashes = dict(hashes)
    source_hashes[str(srdf)] = hashlib.sha256(srdf.read_bytes()).hexdigest()
    world: dict[str, Any] = {"cuboid": {}, "mesh": {}}
    # scene_entities enumerates registered actors and raw physics entities and
    # rejects batched/GPU physics. External articulations were checked above.
    entities = scene_entities(raw)
    for actor_name, entity in entities.items():
        shapes = _entity_shapes(entity)
        if not shapes:
            coverage["skipped_visual_entities"].append(str(actor_name))
        for index, shape in enumerate(shapes):
            name = f"scene::{actor_name}::shape_{index}"
            coverage["world_shapes"].append(_put_world_shape(world, name, shape, world_base @ _matrix(entity.pose)))
    peer = agents[1 - arm_index].robot
    for link in peer.get_links():
        bodies = list(link._objs)
        if len(bodies) != 1:
            raise ValueError("Peer link must represent one physical body")
        for index, shape in enumerate(bodies[0].get_collision_shapes()):
            name = f"peer::{link.get_name()}::shape_{index}"
            row = _put_world_shape(world, name, shape, world_base @ _matrix(link.pose))
            row["peer_link"] = link.get_name()
            coverage["world_shapes"].append(row)
    planner_path, derivation = _independent_finger_urdf(path, output_root=_cache_root(cache_dir))
    source_hashes[str(planner_path)] = derivation["derived_sha256"]
    cfg = _robot_cfg(planner_path, spheres, ignores, q9, velocities)
    coverage.update(all_covered=True, robot_collision_link_count=len(spheres),
        robot_collision_sphere_count=sum(len(values) for values in spheres.values()),
        srdf_disabled_pair_count=disabled_pairs, actual_fingers_locked_at_measured_position=True,
        world_shape_count=len(coverage["world_shapes"]),
        native_scene_entity_count=len(entities), planner_urdf_derivation=derivation,
        measured_finger_difference_m=float(q_by_name[FINGER_NAMES[1]] - q_by_name[FINGER_NAMES[0]]),
        planner_locks_match_both_independent_measured_fingers_exactly=True,
        base_collision_spheres_included=True,
        fixed_mount_world_contact={"link": "panda_link0", "obstacle": SOURCE_TABLE,
            "scope": "world collision only; self-collision with moving links remains enabled",
            "exact_checker_retained": True})
    result = {"schema": EXPORT_VERSION, "arm_index": arm_index,
        "robot_cfg": cfg, "world": world, "base_pose_wxyz": _pose(base_world),
        "joint_names": arm_names, "q_start": [float(q_by_name[name]) for name in arm_names],
        "tcp_pose_base_wxyz": _pose(world_base @ _matrix(agent.tcp.pose)),
        "coverage": coverage, "source_sha256": source_hashes}
    # Reject any numpy scalar, NaN, or other opaque native object at the boundary.
    json.dumps(result, allow_nan=False)
    return result


def self_test() -> dict[str, Any]:
    """Small CPU contract check; no simulator or cuRobo import is required."""
    from types import SimpleNamespace
    pose = [1., -2., .5, math.sqrt(.5), 0., 0., math.sqrt(.5)]
    assert np.allclose(_matrix(_pose(_matrix(pose))), _matrix(pose))
    shape = type("PhysxCollisionShapeBox", (), {})()
    shape.half_size = np.array([.01, .015, .02])
    shape.get_local_pose = lambda: SimpleNamespace(p=np.zeros(3), q=np.array([1., 0., 0., 0.]))
    spheres, _ = _shape_spheres(shape, .025)
    geometry = {name: copy.deepcopy(spheres) for name in ("panda_link0", *FINGER_LINKS)}
    cfg = _robot_cfg(Path(tempfile.gettempdir()) / "panda-fixture.urdf", geometry, {},
                     [0.] * 7 + [.031, .027], [2.] * 9)
    before = json.dumps(cfg, sort_keys=True)
    reserved = reserve_payload_config(cfg, 64)
    assert json.dumps(cfg, sort_keys=True) == before
    assert reserved["kinematics"]["collision_spheres"][PAYLOAD_LINK][0]["radius"] < 0
    assert cfg["kinematics"]["lock_joints"] == dict(zip(FINGER_NAMES, [.031, .027]))
    return {"status": "passed", "sphere_shape": [len(spheres), 4],
            "joint_state_shape": [9], "payload_capacity": 64, "environment_steps": 0}


if __name__ == "__main__":
    print(json.dumps(self_test(), allow_nan=False))
