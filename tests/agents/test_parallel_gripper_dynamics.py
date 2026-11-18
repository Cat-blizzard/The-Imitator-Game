"""Real CPU physics for configured coupled opening coordinates; no renderer/GPU."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest
import sapien
import torch

from mani_skill.agents.robots.parallel_gripper import Piper, UR5WSG


def _native_finger_pair(agent_cls, coupled: bool):
    """Two outward-moving prismatic siblings with the real agent loading hook.

    The small articulation isolates the opening coordinate from arm motion,
    collision geometry, meshes and rendering. Both physical joints remain
    mobile within their original travel. The hook under test creates
    the native PhysX tendon; the test never sets later states or copies qpos.
    """
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(.001)
    builder = scene.create_articulation_builder()
    root = builder.create_link_builder()
    root.set_name("gripper_base")
    root.set_mass_and_inertia(1., sapien.Pose(), [.01]*3)
    for index in range(2):
        link = builder.create_link_builder(root)
        link.set_name(agent_cls.finger_link_names[index])
        link.set_joint_name(agent_cls.gripper_joint_names[index])
        link.set_mass_and_inertia(.05, sapien.Pose(), [.00001]*3)
        # Opposite parent-frame axes give equal outward travel for equal q.
        orientation = [1., 0., 0., 0.] if index == 0 else [0., 0., 0., 1.]
        link.set_joint_properties("prismatic", [[agent_cls.gripper_closed_qpos,
                                                agent_cls.gripper_open_qpos]],
            sapien.Pose([0., (index-.5)*.04, 0.], orientation), sapien.Pose())
    articulation = builder.build(fix_root_link=True)
    joints = articulation.get_active_joints()
    agent = object.__new__(agent_cls)
    agent.scene = NS(device=torch.device("cpu"))
    agent.arm_joint_names, agent.rest_arm_qpos = [], ()
    agent.embodiment_profile = (agent_cls.embodiment_profile if coupled else
        replace(agent_cls.embodiment_profile, coupled_gripper_joints=None))
    wrapped_joints = {joint.name: NS(
        get_parent_link=lambda joint=joint: NS(_objs=[joint.parent_link]),
        get_child_link=lambda joint=joint: NS(_objs=[joint.child_link])) for joint in joints}
    agent.robot = NS(_objs=[articulation], active_joints_map=wrapped_joints,
                    get_active_joints=lambda: [NS(name=joint.name) for joint in joints])
    agent._after_loading_articulation()
    articulation.set_qpos([.0195, .0205])
    articulation.set_qvel([0., 0.])
    for joint in joints:
        joint.set_drive_properties(agent_cls.gripper_stiffness, agent_cls.gripper_damping,
                                   agent_cls.gripper_force_limit)
    return scene, articulation, joints, agent


def _step_with_load(scene, articulation, joints, target: float, load: tuple[float, float]):
    for joint in joints:
        joint.set_drive_target(target)
    measured = []
    for _ in range(500):
        articulation.set_qf(load)
        scene.step()
        measured.append(articulation.get_qpos().copy())
    result = np.asarray(measured)
    assert result.shape == (500, 2) and np.isfinite(result).all()
    return result[-100:]


@pytest.mark.parametrize("load", [(12., -12.), (-12., 12.)])
@pytest.mark.parametrize("agent_cls", [Piper, UR5WSG])
def test_coupling_rejects_saturated_differential_load_without_locking_opening(agent_cls, load):
    # Opposing generalized loads exceed the original 10N drive force on each
    # finger. With independent saturated drives the aperture centre can drift.
    assert agent_cls.gripper_force_limit == 10.
    assert max(abs(value) for value in load) > agent_cls.gripper_force_limit
    independent = _native_finger_pair(agent_cls, False)
    uncoupled_q = _step_with_load(*independent[:3], .02, load)
    uncoupled_difference = float(np.max(np.abs(uncoupled_q[:, 0]-uncoupled_q[:, 1])))
    assert uncoupled_difference > .03
    coupled = _native_finger_pair(agent_cls, True)
    held = _step_with_load(*coupled[:3], .02, load)
    opened = _step_with_load(*coupled[:3], .032, load)
    closed = _step_with_load(*coupled[:3], .008, load)
    for target, measured in ((.02, held), (.032, opened), (.008, closed)):
        difference = float(np.max(np.abs(measured[:, 0]-measured[:, 1])))
        assert difference < .0004
        assert difference < uncoupled_difference/50
        np.testing.assert_allclose(measured.mean(axis=1), target, rtol=0, atol=.0001)
    assert float(opened.mean()-closed.mean()) > .023
    assert independent[3].gripper_tendon_count == 0
    assert coupled[3].gripper_tendon_count == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
