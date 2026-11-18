"""Robot import contracts that do not require SAPIEN/GPU initialization."""
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from mani_skill.agents.robots.parallel_gripper import ARXX5, Piper, UR5WSG

_SPEC = importlib.util.spec_from_file_location("prepare_embodiment_assets", Path(__file__).parents[2] / "scripts" / "prepare_embodiment_assets.py")
prepare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare)


@pytest.mark.parametrize("robot_cls", [Piper, ARXX5, UR5WSG])
def test_named_joint_keyframe_and_static(robot_cls):
    agent = object.__new__(robot_cls)
    # This fixture covers named keyframes/controller values without physics.
    # The native coupled coordinate has separate real CPU dynamics coverage.
    agent.embodiment_profile = replace(robot_cls.embodiment_profile,
                                       coupled_gripper_joints=None)
    names = list(reversed(robot_cls.arm_joint_names + robot_cls.gripper_joint_names))
    qvel = torch.zeros((2, 8))
    qvel[0, 0] = 5  # Finger speed must not be confused with arm speed.
    agent.scene = SimpleNamespace(device=torch.device("cpu"))
    agent.robot = SimpleNamespace(
        get_active_joints=lambda: [SimpleNamespace(name=name) for name in names],
        get_qvel=lambda: qvel,
    )
    agent._after_loading_articulation()
    values = dict(zip(names, agent.keyframes["rest"].qpos))
    assert [values[name] for name in robot_cls.arm_joint_names] == list(robot_cls.rest_arm_qpos)
    assert [values[name] for name in robot_cls.gripper_joint_names] == [robot_cls.gripper_open_qpos] * 2
    assert agent.is_static().tolist() == [True, True]
    qvel[1, names.index(robot_cls.arm_joint_names[0])] = 0.3
    assert agent.is_static().tolist() == [True, False]
    configs = agent._controller_configs["pd_joint_pos"]
    assert configs["arm"].normalize_action is False
    assert configs["gripper"].normalize_action is True
    assert configs["gripper"].lower < configs["gripper"].upper


def test_import_preserves_ur5_finger_geometry_and_removes_stale_srdf(tmp_path):
    source = tmp_path / "source" / "assets" / "embodiments" / "ur5-wsg"
    (source / "meshes").mkdir(parents=True)
    (source / "meshes" / "finger.stl").write_text("mesh fixture")
    original = '''<robot name="fixture">
    <link name="wsg_50_base_link"/><link name="left"/><link name="right"/>
    <link name="body"><visual><geometry><mesh filename="package://fixture/meshes/finger.stl"/></geometry></visual></link>
    <joint name="base_joint_gripper_left" type="prismatic">
    <parent link="wsg_50_base_link"/><child link="left"/>
    <axis xyz="1 0 0"/><limit lower="-0.055" upper="-0.0027"/>
    </joint></robot>'''
    (source / "ur5_wsg_gripper.urdf").write_text(original)
    (source / "ur5.srdf").write_text('''<robot name="fixture">
    <disable_collisions link1="left" link2="right" reason="Default"/>
    <disable_collisions link1="left" link2="stale_robotiq" reason="Default"/>
    </robot>''')
    destination = prepare.prepare_robot("ur5_wsg", tmp_path / "source", tmp_path / "output")
    tree = ET.parse(destination / "robot.urdf")
    joint = tree.find("joint[@name='base_joint_gripper_left']")
    new_axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
    limit = joint.find("limit")
    assert float(limit.get("lower")) == 0.0027
    assert float(limit.get("upper")) == 0.055
    for q_original in (-0.055, -0.03, -0.0027):
        assert np.allclose(new_axis * -q_original, [q_original, 0, 0])
    assert tree.find("link[@name='grasp_tcp']") is not None
    pairs = ET.parse(destination / "robot.srdf").findall("disable_collisions")
    assert len(pairs) == 1
    assert (source / "ur5_wsg_gripper.urdf").read_text() == original
    assert tree.find(".//mesh").get("filename") == "meshes/finger.stl"
    assert prepare.prepare_robot("ur5_wsg", tmp_path / "source", tmp_path / "output") == destination


def test_resource_resolution_rejects_escape(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "private.stl").write_text("not an asset")
    with pytest.raises(ValueError):
        prepare._resolve_resource(source, "../private.stl")


@pytest.mark.parametrize("robot_cls", [Piper, ARXX5, UR5WSG])
def test_grasp_requires_two_real_contacts(robot_cls):
    agent = object.__new__(robot_cls)
    transform = torch.eye(4).unsqueeze(0).expand(2, -1, -1)
    pose = SimpleNamespace(to_transformation_matrix=lambda: transform)
    left, right = SimpleNamespace(pose=pose), SimpleNamespace(pose=pose)
    agent.finger1_link, agent.finger2_link = left, right
    forces = [torch.tensor(axis).expand(2, -1).clone() for axis in robot_cls.finger_open_directions]
    # First sample has two valid contacts; the other has only one.
    forces[1][1] = 0
    agent.scene = SimpleNamespace(
        device=torch.device("cpu"),
        get_pairwise_contact_forces=lambda link, obj: forces[0 if link is left else 1],
    )
    assert agent.is_grasping(object()).tolist() == [True, False]
    # Mere two-sided contact with the wrong force direction is not a grasp.
    forces[0][0].neg_()
    assert agent.is_grasping(object()).tolist() == [False, False]
