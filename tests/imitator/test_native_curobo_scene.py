"""Portable geometry snapshots without constructing a simulator or CUDA model."""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from imitator.planning._tig_sync import curobo_scene as scene
from imitator.planning._tig_sync import curobo_spheres as spheres


def pose(p=(0., 0., 0.), q=(1., 0., 0., 0.)):
    return NS(p=np.array(p, dtype=float), q=np.array(q, dtype=float))


def box(half=(.01, .01, .01), local=None):
    shape = type("PhysxCollisionShapeBox", (), {})()
    shape.half_size = np.asarray(half, dtype=float)
    shape.local_pose = local or pose()
    shape.get_local_pose = lambda: shape.local_pose
    return shape


def panda_files(tmp_path: Path):
    root = ET.Element("robot", name="panda")
    for name in ("panda_link0", "camera_link", "panda_hand", *scene.FINGER_LINKS):
        ET.SubElement(root, "link", name=name)
    for name in (*scene.ARM_NAMES, *scene.FINGER_NAMES):
        joint = ET.SubElement(root, "joint", name=name, type="prismatic" if "finger" in name else "revolute")
        ET.SubElement(joint, "limit", lower="-3", upper="3", velocity="2", effort="10")
        if name == scene.FINGER_NAMES[1]:
            ET.SubElement(joint, "mimic", joint=scene.FINGER_NAMES[0], multiplier="1", offset="0")
    root.insert(0, ET.Comment("Fixture copyright attribution retained"))
    path = tmp_path / "panda_v3.urdf"
    ET.ElementTree(root).write(path)
    path.with_suffix(".srdf").write_text(
        '<robot><disable_collisions link1="panda_link0" link2="panda_link1" reason="Adjacent"/></robot>')
    return path


def robot(base_pose, fingers):
    native = object()
    links = []
    for name in ("panda_link0", "camera_link", "panda_hand", *scene.FINGER_LINKS):
        body = NS(get_collision_shapes=lambda: [box()])
        links.append(NS(_objs=[body], get_name=lambda n=name: n, pose=base_pose))
    state = np.array([0.] * 7 + list(fingers))
    return NS(_objs=[native], pose=base_pose, get_qpos=lambda: state[None].copy(),
              get_active_joints=lambda: [NS(get_name=lambda n=n: n)
                  for n in (*scene.ARM_NAMES, *scene.FINGER_NAMES)],
              get_links=lambda: links)


def test_solid_cover_includes_interior_and_owns_cached_values():
    cube = np.array(list(itertools.product([-.02, .02], repeat=3)))
    transform = np.array([[1., .3, 0.], [.1, 1., .2], [0., .1, .8]])
    vertices = cube @ transform.T
    points = np.array(list(itertools.product(np.linspace(-.02, .02, 7), repeat=3))) @ transform.T
    values, proof = spheres.tight_convex_spheres(vertices, .025)
    gap = np.min(np.linalg.norm(points[:, None] - values[None, :, :3], axis=2)
                 - values[None, :, 3], axis=1)
    assert proof["complete_solid_coverage"] and np.max(gap) <= 1e-8
    before = spheres._tight_convex_cover_cached.cache_info()
    mutable, metadata = spheres.tight_convex_spheres(vertices, .025)
    mutable[:] = np.nan
    metadata.clear()
    fresh, fresh_metadata = spheres.tight_convex_spheres(vertices, .025)
    assert np.array_equal(fresh, values) and fresh_metadata == proof
    assert spheres._tight_convex_cover_cached.cache_info().hits >= before.hits + 2


def test_shape_local_pose_refreshes_while_solid_fit_is_cached():
    shape = box()
    first, _ = scene._shape_spheres(shape, .025, tight=True)
    before = spheres._tight_convex_cover_cached.cache_info()
    shape.local_pose = pose((.13, -.2, .07), (2**-.5, 0., 0., 2**-.5))
    second, _ = scene._shape_spheres(shape, .025, tight=True)
    expected = scene._transform_vertices([s["center"] for s in first], scene._matrix(shape.local_pose))
    assert np.allclose([s["center"] for s in second], expected)
    assert [s["radius"] for s in second] == [s["radius"] for s in first]
    assert spheres._tight_convex_cover_cached.cache_info().hits == before.hits + 1


def test_hand_envelope_uses_owner_frame_and_covers_rotated_full_solid():
    shape = box((.012, .006, .003),
                pose((.017, -.019, .023), (2**-.5, 0., 2**-.5, 0.)))
    values, proof = scene._shape_spheres(shape, .025, tight=True, hand_envelope=True)
    array = np.array([s["center"] + [s["radius"]] for s in values])
    vertices, _, _ = scene._shape_mesh(shape)
    owner_vertices = scene._transform_vertices(vertices, scene._matrix(shape.local_pose))
    assert proof["envelope_frame"] == "owner_link"
    assert proof["adaptive_rule_task_independent"]
    assert proof["source_solid_max_positive_z_m"] == pytest.approx(owner_vertices[:, 2].max())
    assert np.max(array[:, 2] + array[:, 3]) <= owner_vertices[:, 2].max() + .002 + 1e-12
    assert proof["maximum_allowed_refinement_depth"] == 2
    points = np.array(list(itertools.product(*[
        np.linspace(-half, half, 5) for half in shape.half_size])))
    points = scene._transform_vertices(points, scene._matrix(shape.local_pose))
    gap = np.min(np.linalg.norm(points[:, None] - array[None, :, :3], axis=2)
                 - array[None, :, 3], axis=1)
    assert np.max(gap) <= 1e-8
    before = spheres._tight_convex_cover_cached.cache_info()
    again, _ = scene._shape_spheres(shape, .025, tight=True, hand_envelope=True)
    assert again == values
    assert spheres._tight_convex_cover_cached.cache_info().hits == before.hits + 1
    with pytest.raises(ValueError, match="requires a tight"):
        scene._shape_spheres(shape, .025, hand_envelope=True)


def test_attached_cover_removes_proxy_overlap_without_ignoring_either_solid():
    hand = box((.03, .03, .01))
    target = box((.03, .015, .005))
    relation = [.0, .0, .0195, 1., .0, .0, .0]
    hand_values, _ = scene._shape_spheres(hand, .025, tight=True, hand_envelope=True)
    original, _ = scene._shape_spheres(target, .0125, tight=True)
    refined, proof = scene.attached_shape_spheres(target, .0125, relation)

    def minimum_gap(values):
        spheres = np.array([s["center"] + [s["radius"]] for s in values])
        spheres[:, :3] = scene._transform_vertices(spheres[:, :3], scene._matrix(relation))
        hand_spheres = np.array([s["center"] + [s["radius"]] for s in hand_values])
        return np.min(np.linalg.norm(spheres[:, None, :3] - hand_spheres[None, :, :3], axis=2)
                      - spheres[:, None, 3] - hand_spheres[None, :, 3])

    # The physical boxes have 4.5 mm clearance; the original cell spheres overlap.
    assert minimum_gap(original) < 0
    assert minimum_gap(refined) > 0
    assert proof["complete_solid_coverage"]
    assert proof["output_frame"] == "actor_local"
    assert proof["positive_z_padding_m"] == .002
    points = np.array(list(itertools.product(*[
        np.linspace(-half, half, 5) for half in target.half_size])))
    values = np.array([s["center"] + [s["radius"]] for s in refined])
    gap = np.min(np.linalg.norm(points[:, None] - values[None, :, :3], axis=2)
                 - values[None, :, 3], axis=1)
    assert np.max(gap) <= 1e-8


def test_independent_finger_asset_is_portable_cached_and_preserves_source(tmp_path, monkeypatch):
    path = panda_files(tmp_path)
    original = path.read_bytes()
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    derived, metadata = scene._independent_finger_urdf(path)
    assert derived.is_relative_to(tmp_path / "xdg" / "mani_skill" / "curobo")
    assert path.read_bytes() == original
    assert b"Fixture copyright attribution retained" in derived.read_bytes()
    joints = {j.get("name"): j for j in ET.parse(derived).getroot().findall("joint")}
    assert joints[scene.FINGER_NAMES[1]].find("mimic") is None
    assert metadata["other_parsed_urdf_semantics_unchanged"]
    timestamp = derived.stat().st_mtime_ns
    again, again_metadata = scene._independent_finger_urdf(path)
    assert again == derived and again_metadata == metadata and derived.stat().st_mtime_ns == timestamp
    again_metadata.clear()
    assert scene._independent_finger_urdf(path)[1] == metadata


def test_snapshot_retains_base_world_peer_and_independent_fingers(tmp_path, monkeypatch):
    from imitator.planning._tig_sync import scene_collision
    urdf = panda_files(tmp_path)
    left_robot = robot(pose((-.6, -.35, 0.)), (.031, .027))
    right_robot = robot(pose((.6, -.35, 0.), (0., 0., 0., 1.)), (.026, .024))
    controller = NS(controllers={"arm": NS(config=NS(joint_names=scene.ARM_NAMES))})
    agents = [NS(robot=r, urdf_path=str(urdf), controller=controller,
                 tcp=NS(pose=pose((.2, .1, .3)))) for r in (left_robot, right_robot)]
    table = NS(pose=pose((0., 0., -.05)), shapes=[box((1., .6, .05))])
    target = NS(pose=pose((.2, .1, .04)), shapes=[box()])
    raw = NS(num_envs=1, agent=NS(agents=agents), left_agent=agents[0], right_agent=agents[1],
             scene=NS(articulations={str(i): NS(_objs=r._objs) for i, r in enumerate((left_robot, right_robot))},
                      px=NS(get_articulation_link_components=lambda: [])))
    monkeypatch.setattr(scene_collision, "scene_entities", lambda _: {"table": table, "pill": target})
    monkeypatch.setattr(scene, "_entity_shapes", lambda entity: entity.shapes)
    snapshots = [scene.export_curobo_snapshot(raw, i, cache_dir=tmp_path / "cache") for i in (0, 1)]
    from veritask.data.native_curobo import robot_topology, world_topology
    # Selecting the other Panda changes measured base-relative poses and finger
    # locks, but retains fixed robot/world storage for a sequential profile.
    assert robot_topology(snapshots[0]["robot_cfg"]) == robot_topology(snapshots[1]["robot_cfg"])
    assert world_topology(snapshots[0]["world"]) == world_topology(snapshots[1]["world"])
    for index, result in enumerate(snapshots):
        kin = result["robot_cfg"]["kinematics"]
        assert kin["collision_spheres"]["panda_link0"]
        assert result["coverage"]["active_robot_exceptions"] == []
        assert result["coverage"]["base_collision_spheres_included"]
        assert result["coverage"]["finger_sphere_pitch_m"] == .005
        for name in scene.FINGER_LINKS:
            assert max(s["radius"] for s in kin["collision_spheres"][name]) == pytest.approx(3**.5 * .005 / 2)
            rows = [r for r in result["coverage"]["active_robot_shapes"] if r["link"] == name]
            assert rows and all(r["sphere_pitch_m"] == .005 for r in rows)
        hand_rows = [r for r in result["coverage"]["active_robot_shapes"] if r["link"] == "panda_hand"]
        assert hand_rows and all(r["envelope_frame"] == "owner_link" for r in hand_rows)
        assert scene.SOURCE_TABLE in result["world"]["cuboid"]
        assert "peer::panda_link0::shape_0" in result["world"]["cuboid"]
        expected = (.031, .027) if index == 0 else (.026, .024)
        assert kin["lock_joints"] == dict(zip(scene.FINGER_NAMES, expected))
        base_inverse = np.linalg.inv(scene._matrix(agents[index].robot.pose))
        assert np.allclose(scene._matrix(result["tcp_pose_base_wxyz"]),
                           base_inverse @ scene._matrix(agents[index].tcp.pose))
        assert result["coverage"]["environment_steps"] == 0
        json.dumps(result, allow_nan=False)
    before = spheres._tight_convex_cover_cached.cache_info()
    target.pose = pose((.3, .2, .05))
    updated = scene.export_curobo_snapshot(raw, 0, cache_dir=tmp_path / "cache")
    assert updated["world"]["cuboid"]["scene::pill::shape_0"]["pose"] != snapshots[0]["world"]["cuboid"]["scene::pill::shape_0"]["pose"]
    assert spheres._tight_convex_cover_cached.cache_info().misses == before.misses


def test_payload_reservation_preserves_base_and_only_allows_own_fingers():
    cfg = scene._robot_cfg(Path("/tmp/panda-test.urdf"),
        {name: [{"center": [0., 0., 0.], "radius": .01}]
         for name in ("panda_link0", "panda_hand", *scene.FINGER_LINKS)},
        {"panda_link0": ["panda_link1"]}, [0.] * 7 + [.031, .027], [2.] * 9)
    original = copy.deepcopy(cfg)
    reserved = scene.reserve_payload_config(cfg, 64)
    assert cfg == original
    kin = reserved["kinematics"]
    assert len(kin["collision_spheres"][scene.PAYLOAD_LINK]) == 64
    assert all(x["radius"] < 0 for x in kin["collision_spheres"][scene.PAYLOAD_LINK])
    assert kin["self_collision_ignore"][scene.PAYLOAD_LINK] == list(scene.FINGER_LINKS)
    assert kin["self_collision_ignore"]["panda_link0"] == ["panda_link1"]
    assert "panda_hand" not in kin["self_collision_ignore"][scene.PAYLOAD_LINK]
    assert kin["extra_links"][scene.PAYLOAD_LINK]["link_mass"] == 0
    with pytest.raises(ValueError, match="unextended|extra"):
        scene.reserve_payload_config(reserved, 64)
    resized = scene.resize_payload_config(reserved, 128)
    assert reserved["kinematics"]["extra_collision_spheres"][scene.PAYLOAD_LINK] == 64
    assert resized["kinematics"]["extra_collision_spheres"][scene.PAYLOAD_LINK] == 128
    assert len(resized["kinematics"]["collision_spheres"][scene.PAYLOAD_LINK]) == 128
    # Resizing changes storage only; every ordinary sphere and collision
    # permission stays exactly as exported by the physical model.
    resized["kinematics"]["collision_spheres"][scene.PAYLOAD_LINK] = kin["collision_spheres"][scene.PAYLOAD_LINK]
    resized["kinematics"]["extra_collision_spheres"][scene.PAYLOAD_LINK] = 64
    assert resized == reserved
    altered = copy.deepcopy(reserved)
    altered["kinematics"]["collision_spheres"][scene.PAYLOAD_LINK][0]["radius"] = .01
    with pytest.raises(ValueError, match="disabled payload storage"):
        scene.resize_payload_config(altered, 128)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_pitch_rejected_before_export(bad):
    with pytest.raises(ValueError, match="pitch"):
        scene.export_curobo_snapshot(None, 0, sphere_pitch_m=bad)


def test_module_self_checks():
    assert spheres.self_test()["status"] == "passed"
    assert scene.self_test()["status"] == "passed"
