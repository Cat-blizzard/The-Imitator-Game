"""Native CPU rotational coupling retains a mobile single opening coordinate."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
import sapien
from mani_skill.agents.robots.xarm6.xarm6_robotiq import XArm6Robotiq


def _pair(coupled: bool, timestep: float):
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(timestep)
    builder = scene.create_articulation_builder()
    root = builder.create_link_builder()
    root.set_name("base")
    root.set_mass_and_inertia(1., sapien.Pose(), [.01]*3)
    names = XArm6Robotiq.embodiment_profile.coupled_gripper_joints
    for i, name in enumerate(names):
        link = builder.create_link_builder(root)
        link.set_name(name+"_link")
        link.set_joint_name(name)
        link.set_mass_and_inertia(.05, sapien.Pose(), [.0001]*3)
        link.set_joint_properties("revolute", [[0., .81]],
            sapien.Pose([0., (i-.5)*.06, 0.], [1., 0., 0., 0.] if i else [0., 0., 0., 1.]),
            sapien.Pose())
    native = builder.build(fix_root_link=True)
    joints = native.get_active_joints()
    agent = object.__new__(XArm6Robotiq)
    agent.robot = NS(_objs=[native], active_joints_map={j.name: NS(
        get_parent_link=lambda j=j: NS(_objs=[j.parent_link]),
        get_child_link=lambda j=j: NS(_objs=[j.child_link])) for j in joints})
    if coupled:
        agent._couple_outer_knuckles()
        assert agent.gripper_tendon_count == 1
    native.set_qpos([.19, .21])
    for joint in joints:
        joint.set_drive_properties(100., 1., XArm6Robotiq.gripper_force_limit)
    return scene, native, joints


def _move(pair, target, load):
    scene, native, joints = pair
    for j in joints: j.set_drive_target(target)
    trace = []
    for _ in range(2000):
        native.set_qf(load)
        scene.step()
        trace.append(native.get_qpos().copy())
    trace = np.asarray(trace)
    assert trace.shape == (2000, 2) and np.isfinite(trace).all()
    return trace[-100:]


@pytest.mark.parametrize("load", [(.15, -.15), (-.15, .15)])
@pytest.mark.parametrize("timestep", [.001, .01])
def test_outer_coupling_resists_asymmetric_load_and_preserves_opening(load, timestep):
    assert XArm6Robotiq.gripper_force_limit == .1
    independent = _move(_pair(False, timestep), .3, load)
    assert np.max(np.abs(np.diff(independent, axis=1))) > .1
    coupled = _pair(True, timestep)
    for target in (.3, .6, .1):
        measured = _move(coupled, target, load)
        assert np.max(np.abs(np.diff(measured, axis=1))) < .001
        np.testing.assert_allclose(measured.mean(axis=1), target, atol=.002, rtol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
