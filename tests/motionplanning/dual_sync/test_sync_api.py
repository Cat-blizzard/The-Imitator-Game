"""Guard actual ManiSkill multi-agent and CPU backend API representations."""
from types import SimpleNamespace

import pytest

from mani_skill.examples.motionplanning.dual.sync.pill_task import TASK_ID
from mani_skill.examples.motionplanning.dual.sync.solve import solve


class ResetReached(Exception):
    """Stop before creating task geometry or advancing a simulation."""


def environment(backend, right_mode="pd_joint_pos"):
    raw = SimpleNamespace(
        spec=SimpleNamespace(id=TASK_ID), num_envs=1,
        robot_uids=("panda_wristcam", "panda_wristcam"),
        # MultiAgent exposes a dictionary here; actual controller modes belong
        # to its individual agents.
        control_mode={"panda_wristcam-0": "pd_joint_pos", "panda_wristcam-1": right_mode},
        agent=SimpleNamespace(agents=[SimpleNamespace(control_mode="pd_joint_pos"),
                                      SimpleNamespace(control_mode=right_mode)]),
        backend=SimpleNamespace(sim_backend=backend),
        control_timestep=.05, sim_timestep=.01,
    )

    def reset(**kwargs):
        raise ResetReached

    return SimpleNamespace(unwrapped=raw, reset=reset)


@pytest.mark.parametrize("backend", ["cpu", "physx_cpu"])
def test_original_cpu_names_and_multiagent_control_dictionary(backend):
    with pytest.raises(ResetReached):
        solve(environment(backend), seed=401)


def test_wrong_individual_controller_is_rejected_before_reset():
    with pytest.raises(ValueError, match="pd_joint_pos"):
        solve(environment("physx_cpu", right_mode="pd_joint_delta_pos"), seed=401)
