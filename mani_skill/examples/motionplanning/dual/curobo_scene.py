"""Read-only TIG Panda/PhysX snapshots for cuRoboV2-Fast 0.9.0.

All poses are XYZ+WXYZ in metres. Robot collision spheres are link-local;
world obstacles and the TCP are expressed in the selected robot base frame.
The seven arm joints retain controller order; the two measured finger joints
are locked in a nine-joint input cspace which cuRobo subsequently reduces.
This exporter never steps an environment or changes a physical state. It may
create a content-addressed derived URDF with independent physical fingers. Its
conservative sphere model does not replace the existing exact paired checker.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

from .curobo_sphere_cover import coarse_convex_spheres as _cover_convex, tight_convex_spheres


ARM_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
FINGER_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
EXPORT_VERSION = "tig-v3-actual-physx-curobo-v2-independent-fingers-snapshot-v3"


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


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


def _shape_spheres(shape: Any, pitch_m: float, *,
                   tight: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cover the full solid in link coordinates; tight fits occupied cells."""
    local = _matrix(shape.get_local_pose())
    if _kind(shape) == "PhysxCollisionShapeSphere":
        centers = local[None, :3, 3]
        radius = float(_positive(shape.radius, "sphere radius"))
        policy = "exact_single_sphere"
    else:
        vertices, _, mesh_policy = _shape_mesh(shape)
        if tight:
            values, proof = tight_convex_spheres(vertices, pitch_m)
            values[:, :3] = _transform_vertices(values[:, :3], local)
            return [{"center": row[:3].tolist(), "radius": float(row[3])} for row in values], {
                **proof, "geometry_policy": mesh_policy, "shape_local_pose_wxyz": _pose(local)}
        centers, radius = _cover_convex(vertices, pitch_m)
        centers = _transform_vertices(centers, local)
        policy = "convex_solid_grid_cell_circumspheres;" + mesh_policy
    spheres = [{"center": c.tolist(), "radius": radius} for c in centers]
    return spheres, {"sphere_count": len(spheres), "coverage_policy": policy,
                     "sphere_radius_m": radius, "shape_local_pose_wxyz": _pose(local)}


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
            "shape_pose_base_wxyz": _pose(shape_to_base),
            "geometry_sha256": hashlib.sha256(json.dumps(
                world["cuboid"].get(name, world["mesh"].get(name)), sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()}


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


def _urdf_metadata(path: Path, joint_names: list[str]) -> tuple[list[float], dict[str, str]]:
    root = ET.parse(path).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    velocities = []
    for name in joint_names:
        if name not in joints or joints[name].find("limit") is None:
            raise ValueError(f"URDF joint lacks limits: {name}")
        velocity = float(joints[name].find("limit").attrib["velocity"])
        if not math.isfinite(velocity) or velocity <= 0:
            raise ValueError(f"Invalid URDF joint velocity: {name}")
        velocities.append(velocity)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    for mesh in root.findall(".//collision/geometry/mesh"):
        filename = mesh.attrib["filename"]
        if filename.startswith("package://"):
            filename = filename[len("package://"):]
        asset = (path.parent / filename).resolve()
        if not asset.is_file():
            raise ValueError(f"Missing real Panda collision asset: {asset}")
        hashes[str(asset)] = hashlib.sha256(asset.read_bytes()).hexdigest()
    return velocities, hashes


def _independent_finger_urdf(original: Path, *, output_root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Derive an immutable planner-only model of SAPIEN's two physical fingers.

    TIG exposes both finger joints as independent active state variables, even
    though its input URDF contains a mimic annotation. V2 enforces that URDF
    annotation and forbids separately locking the mimic joint. Removing exactly
    that annotation lets the planner lock the two actual measured states. No
    axes, origins, limits, collision shapes or physical files are changed.
    """
    original = original.resolve()
    before = original.read_bytes()
    tree = ET.fromstring(before)
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
    if output_root is None:
        from infrastructure.tig_runtime import resolve_runtime_paths
        paths = resolve_runtime_paths()
        # Preserve the established workspace location for the default cache;
        # an explicitly selected cache also owns its generated planner models.
        output_root = (paths.workspace_root / "infrastructure/backends"
                       if paths.cache_root == paths.workspace_root / "infrastructure/tmp"
                       else paths.cache_root) / "tig-derived-models/panda-v3-independent-fingers"
    destination = output_root.resolve() / digest / "panda_v3.urdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if destination.read_bytes() != payload:
            raise RuntimeError("Content-addressed derived URDF identity was changed")
    # Compare canonical parsed semantics after undoing only the declared edits.
    restored = ET.fromstring(payload)
    restored_joints = {node.attrib["name"]: node for node in restored.findall("joint")}
    ET.SubElement(restored_joints[FINGER_NAMES[1]], "mimic", removed)
    for mesh, change in zip(restored.findall(".//mesh"), rewritten_meshes, strict=True):
        mesh.attrib["filename"] = change["original"]
    def semantics(element: ET.Element) -> Any:
        # Whitespace and serialization order are not URDF semantics. Sibling
        # element order is retained except the restored mimic child position.
        return (element.tag, tuple(sorted(element.attrib.items())),
                (element.text or "").strip(), sorted((semantics(child) for child in element), key=repr))
    if semantics(restored) != semantics(ET.fromstring(before)):
        raise RuntimeError("Derived URDF changed undeclared semantics")
    if original.read_bytes() != before:
        raise RuntimeError("Original physical URDF changed during planner-model derivation")
    return destination, {"policy": "planner_only_remove_finger2_mimic_and_absolutize_mesh_paths_v1",
        "reason": "actual SAPIEN state has nine independent active joints; preserve both measured finger values instead of forcing equality",
        "original_urdf": str(original), "original_sha256": hashlib.sha256(before).hexdigest(),
        "derived_urdf": str(destination), "derived_sha256": digest,
        "removed_mimic": {"joint_name": FINGER_NAMES[1], "attributes": removed},
        "mesh_path_rewrites": rewritten_meshes, "other_parsed_urdf_semantics_unchanged": True,
        "physical_urdf_modified": False}


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


def export_curobo_snapshot(env: Any, arm_index: int, *, sphere_pitch_m: float = .025) -> dict[str, Any]:
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
        "robot_sphere_policy": "conservative_convex_solid_grid_cell_with_tight_link5_v1",
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
        if name == "panda_link0":
            coverage["active_robot_exceptions"].append({"link": name, "shape_count": len(shapes),
                "reason": "fixed table mount excluded only from selected-arm cuRobo active spheres; exact external checker and peer world retain it"})
            continue
        for index, shape in enumerate(shapes):
            values, proof = _shape_spheres(shape, pitch, tight=name == "panda_link5")
            spheres.setdefault(name, []).extend(values)
            coverage["active_robot_shapes"].append({"link": name, "shape_index": index,
                                                    "type": _kind(shape), **proof})
    if not spheres or "camera_link" not in spheres or any(name not in spheres for name in ("panda_hand", "panda_leftfinger", "panda_rightfinger")):
        raise ValueError("Panda hand, camera and both actual finger geometries must be covered")
    collision_links = list(spheres)
    ignores, disabled_pairs = _srdf_ignores(srdf, collision_links)
    velocities, source_hashes = _urdf_metadata(path, cspace_names)
    source_hashes[str(srdf)] = hashlib.sha256(srdf.read_bytes()).hexdigest()
    for source in (Path(__file__), Path(__file__).with_name("curobo_sphere_cover.py")):
        source_hashes[str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
    world: dict[str, Any] = {"cuboid": {}, "mesh": {}}
    # scene_entities enumerates registered actors and raw physics entities and
    # rejects batched/GPU physics. External articulations were checked above.
    entities = scene_entities(raw)
    import sapien.physx as physx
    for actor_name, entity in entities.items():
        bodies = [component for component in entity.components if isinstance(component, physx.PhysxRigidBaseComponent)]
        if len(bodies) > 1 or (bodies and isinstance(bodies[0], physx.PhysxArticulationLinkComponent)):
            raise ValueError("Invalid nonrobot rigid-body registration")
        shapes = list(bodies[0].get_collision_shapes()) if bodies else []
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
    planner_path, derivation = _independent_finger_urdf(path)
    source_hashes[str(planner_path)] = derivation["derived_sha256"]
    cfg = _robot_cfg(planner_path, spheres, ignores, q9, velocities)
    coverage.update(all_covered=True, robot_collision_link_count=len(spheres),
        robot_collision_sphere_count=sum(len(values) for values in spheres.values()),
        srdf_disabled_pair_count=disabled_pairs, actual_fingers_locked_at_measured_position=True,
        world_shape_count=len(coverage["world_shapes"]),
        native_scene_entity_count=len(entities), planner_urdf_derivation=derivation,
        measured_finger_difference_m=float(q_by_name[FINGER_NAMES[1]] - q_by_name[FINGER_NAMES[0]]),
        planner_locks_match_both_independent_measured_fingers_exactly=True)
    result = {"schema": EXPORT_VERSION, "arm_index": arm_index,
        "robot_cfg": cfg, "world": world, "base_pose_wxyz": _pose(base_world),
        "joint_names": arm_names, "q_start": [float(q_by_name[name]) for name in arm_names],
        "tcp_pose_base_wxyz": _pose(world_base @ _matrix(agent.tcp.pose)),
        "coverage": coverage, "source_sha256": source_hashes}
    # Reject any numpy scalar, NaN, or other opaque native object at the boundary.
    json.dumps(result, allow_nan=False)
    return result


def self_test() -> dict[str, Any]:
    """CPU geometry and frame contracts, including interior sphere coverage."""
    from types import SimpleNamespace as NS
    checks: list[str] = []
    def reject(call: Any) -> None:
        try:
            call()
        except (ValueError, TypeError, RuntimeError):
            return
        raise AssertionError("Malformed geometry accepted")
    def fixture(kind: str, **fields: Any) -> Any:
        cls = type(kind, (), {})
        obj = cls()
        for name, value in fields.items():
            setattr(obj, name, value)
        return obj
    identity = NS(p=np.zeros(3), q=np.array([1., 0, 0, 0]))
    angle = math.pi / 2
    translated = NS(p=np.array([.13, -.2, .07]), q=np.array([math.cos(angle / 2), 0, 0, math.sin(angle / 2)]))
    box = fixture("PhysxCollisionShapeBox", half_size=np.array([.05, .03, .02]), get_local_pose=lambda: translated)
    vertices, faces, _ = _shape_mesh(box)
    spheres, proof = _shape_spheres(box, .025)
    centers = np.asarray([sphere["center"] for sphere in spheres])
    radius = spheres[0]["radius"]
    dense = np.array(list(itertools.product(np.linspace(-.05, .05, 15), np.linspace(-.03, .03, 13), np.linspace(-.02, .02, 11))))
    dense = _transform_vertices(dense, _matrix(translated))
    for chunk in np.array_split(dense, 13):
        distance = np.linalg.norm(chunk[:, None, :] - centers[None, :, :], axis=2).min(axis=1)
        assert np.all(distance <= radius + 1e-12)
    assert radius == math.sqrt(3) * .025 / 2 and proof["sphere_count"] == len(spheres)
    checks.append("rotated translated box volume fully covered")
    tetra_vertices = np.array([[0., 0, 0], [.073, 0, 0], [0, .057, 0], [0, 0, .091]])
    tetra_faces = ConvexHull(tetra_vertices).simplices
    mesh = fixture("PhysxCollisionShapeConvexMesh", vertices=tetra_vertices, scale=np.array([2., 1., .5]),
                   get_triangles=lambda: tetra_faces, get_local_pose=lambda: identity)
    scaled, _, _ = _shape_mesh(mesh)
    assert np.array_equal(scaled, tetra_vertices * [2., 1., .5])
    covered, coarse_mesh_proof = _shape_spheres(mesh, .025)
    covered_centers = np.asarray([value["center"] for value in covered])
    rng = np.random.default_rng(412)
    weights = rng.dirichlet(np.ones(4), 1000)
    interior = weights @ scaled
    assert np.all(np.linalg.norm(interior[:, None] - covered_centers[None], axis=2).min(axis=1) <= covered[0]["radius"] + 1e-12)
    checks.append("scaled convex mesh interior sphere coverage")
    tight_values, tight_proof = _shape_spheres(mesh, .025, tight=True)
    tight_array = np.array([[*row["center"], row["radius"]] for row in tight_values])
    tight_gaps = np.linalg.norm(interior[:, None] - tight_array[None, :, :3], axis=2) - tight_array[None, :, 3]
    assert tight_proof["complete_solid_coverage"] and np.max(tight_gaps.min(axis=1)) <= 1e-8
    assert _shape_spheres(mesh, .025) == (covered, coarse_mesh_proof)
    # Cached covers are shape-local; every export owns its transformed array.
    moved = fixture("PhysxCollisionShapeConvexMesh", vertices=tetra_vertices, scale=np.array([2., 1., .5]),
                    get_triangles=lambda: tetra_faces, get_local_pose=lambda: translated)
    moved_values, _ = _shape_spheres(moved, .025, tight=True)
    assert np.allclose([r["center"] for r in moved_values],
        _transform_vertices(tight_array[:, :3], _matrix(translated)), rtol=0, atol=1e-12)
    assert np.array_equal([r["radius"] for r in moved_values], tight_array[:, 3])
    checks.append("tight complete-solid shape cover preserves rotated local frames and cached geometry")
    sphere = fixture("PhysxCollisionShapeSphere", radius=.032, get_local_pose=lambda: translated)
    exact, _ = _shape_spheres(sphere, .025)
    assert len(exact) == 1 and exact[0]["radius"] == .032 and np.allclose(exact[0]["center"], translated.p)
    checks.append("exact sphere translation")
    world: dict[str, Any] = {"cuboid": {}, "mesh": {}}
    base = _matrix([-.6, -.35, 0., 1, 0, 0, 0])
    row = _put_world_shape(world, "box", box, np.linalg.inv(base))
    assert np.allclose(world["cuboid"]["box"]["dims"], [.1, .06, .04])
    assert np.allclose(_matrix(row["shape_pose_base_wxyz"]), np.linalg.inv(base) @ _matrix(translated))
    _put_world_shape(world, "mesh", mesh, np.linalg.inv(base))
    assert world["mesh"]["mesh"]["vertices"] == scaled.tolist()
    plane = fixture("PhysxCollisionShapePlane", get_local_pose=lambda: identity)
    _put_world_shape(world, "plane", plane, np.eye(4))
    slab = world["cuboid"]["plane"]
    assert slab["dims"] == [4., 10., 10.] and slab["pose"][:3] == [-2., 0., 0.]
    tcp = base @ _matrix([.4, .1, .23, math.sqrt(.5), 0, math.sqrt(.5), 0])
    assert np.allclose(_matrix(_pose(np.linalg.inv(base) @ tcp)), np.linalg.inv(base) @ tcp)
    checks.append("world-to-base TCP mesh box and plane transforms")
    here = Path(__file__).resolve()
    asset = here.parents[3] / "assets" / "robots" / "panda" / "panda_v3.urdf"
    velocities, hashes = _urdf_metadata(asset, list(ARM_NAMES + FINGER_NAMES))
    ignores, count = _srdf_ignores(asset.with_suffix(".srdf"), ["panda_hand", "camera_link", "panda_leftfinger", "panda_rightfinger"])
    assert len(velocities) == 9 and all(value > 0 for value in velocities) and len(hashes) > 9
    assert "camera_link" in ignores["panda_hand"] and "panda_hand" in ignores["camera_link"] and count > 30
    checks.append("real v3 URDF assets velocities and symmetric SRDF ignore pairs")
    cfg = _robot_cfg(asset, {"camera_link": exact}, {}, [0.] * 7 + [.021, .022], velocities)
    kin = cfg["kinematics"]
    assert kin["format_version"] == 2. and kin["tool_frames"] == ["panda_hand_tcp"]
    assert "ee_link" not in kin and "retract_config" not in kin["cspace"]
    assert kin["lock_joints"] == {"panda_finger_joint1": .021, "panda_finger_joint2": .022}
    assert kin["cspace"]["joint_names"] == list(ARM_NAMES + FINGER_NAMES)
    assert np.allclose(np.asarray(kin["cspace"]["velocity_scale"]) * velocities, .9)
    assert cfg["load_dynamics"] is False and kin["cspace"]["max_acceleration"] == .9
    checks.append("V2 format tool frame measured finger locks and nine-joint limit scaling")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="tig-curobo-finger-derivation-") as temporary:
        original_sha = hashlib.sha256(asset.read_bytes()).hexdigest()
        derived, proof = _independent_finger_urdf(asset, output_root=Path(temporary))
        assert derived.is_file() and proof["other_parsed_urdf_semantics_unchanged"]
        assert proof["original_sha256"] == original_sha
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == original_sha
        parsed = ET.parse(derived).getroot()
        assert not parsed.findall(".//mimic")
        assert all(Path(mesh.attrib["filename"]).is_absolute() for mesh in parsed.findall(".//mesh"))
        again, again_proof = _independent_finger_urdf(asset, output_root=Path(temporary))
        assert again == derived and again_proof["derived_sha256"] == proof["derived_sha256"]
        independent = _robot_cfg(derived, {"camera_link": exact}, {}, [0.] * 7 + [.021, .022], velocities)
        assert independent["kinematics"]["lock_joints"][FINGER_NAMES[0]] != independent["kinematics"]["lock_joints"][FINGER_NAMES[1]]
        # Exercise the default-path branch from a module too shallow for the
        # former parents[7] lookup, rather than bypassing it with output_root.
        import os
        from unittest.mock import patch
        from infrastructure import tig_runtime
        expected_payload = derived.read_bytes()
        shallow = Path(temporary) / "checkout"
        shallow_module = shallow / "curobo_scene.py"
        assert len(shallow_module.resolve().parents) < 8
        workspace, cache = Path(temporary) / "dependencies", Path(temporary) / "cache"
        suffix = Path("tig-derived-models/panda-v3-independent-fingers")
        cases = [
            ({}, None, shallow / "infrastructure/backends" / suffix),
            ({"TIG_WORKSPACE": str(workspace)}, None, workspace / "infrastructure/backends" / suffix),
            ({"TIG_WORKSPACE": str(workspace), "TIG_CACHE_DIR": str(workspace / "infrastructure/tmp")},
             None, workspace / "infrastructure/backends" / suffix),
            ({"TIG_WORKSPACE": str(workspace), "TIG_CACHE_DIR": str(cache)}, None, cache / suffix),
            ({"TIG_WORKSPACE": str(workspace), "TIG_CACHE_DIR": str(cache)},
             Path(temporary) / "explicit", Path(temporary) / "explicit"),
        ]
        for environment, override, root in cases:
            with patch.dict(os.environ, environment, clear=True), \
                    patch.dict(globals(), {"__file__": str(shallow_module)}), \
                    patch.object(tig_runtime, "__file__", str(shallow / "infrastructure/tig_runtime.py")):
                shallow_derived, shallow_proof = _independent_finger_urdf(asset, output_root=override)
            assert shallow_derived == root / proof["derived_sha256"] / "panda_v3.urdf"
            assert shallow_derived.read_bytes() == expected_payload
            assert shallow_proof["derived_sha256"] == proof["derived_sha256"]
            assert shallow_proof["other_parsed_urdf_semantics_unchanged"]
        assert hashlib.sha256(asset.read_bytes()).hexdigest() == original_sha
        derived.write_bytes(b"tampered")
        reject(lambda: _independent_finger_urdf(asset, output_root=Path(temporary)))
    checks.append("independent-finger derived URDF semantic equivalence absolute assets immutable identity")
    checks.append("shallow checkout workspace/cache/default/explicit derived paths preserve exact URDF bytes")
    reject(lambda: _matrix([0, 0, 0, 0, 0, 0, 0]))
    reject(lambda: _matrix([0, 0, float("nan"), 1, 0, 0, 0]))
    reject(lambda: _pose(np.zeros((4, 4))))
    reject(lambda: _cover_convex(vertices, 0.))
    reject(lambda: _cover_convex(vertices, float("nan")))
    reject(lambda: _cover_convex(np.zeros((4, 3)), .025))
    reject(lambda: _mesh_arrays(vertices, np.array([[0, 1, 1000]])))
    reject(lambda: _kind(NS()))
    reject(lambda: _robot_cfg(asset, {}, {}, np.zeros(9), velocities))
    reject(lambda: _robot_cfg(asset, {"x": exact}, {}, np.zeros(8), velocities))
    reject(lambda: _put_world_shape(world, "box", box, np.eye(4)))
    reject(lambda: export_curobo_snapshot(None, True))
    reject(lambda: export_curobo_snapshot(None, 2))
    reject(lambda: export_curobo_snapshot(None, 0, sphere_pitch_m=float("inf")))
    checks.append("invalid pose geometry index pitch unknownshape duplicate fail closed")
    json.dumps(world, allow_nan=False)
    return {"status": "passed", "schema": EXPORT_VERSION, "checks": checks,
        "shapes": {"box_vertices": list(vertices.shape), "box_faces": list(faces.shape),
                   "covered_grid_centers": list(centers.shape), "convex_test_points": list(interior.shape)},
        "environment_steps": 0, "gpu_used": False, "native_snapshot_tested": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(self_test(), indent=2, allow_nan=False))
