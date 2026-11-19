"""ManiSkill 3.0.1 single-arm Panda tabletop tasks wrapped for imsim (E29b).

Each wrapper subclasses the ManiSkill task env unchanged (scene, randomisation, benchmark predicate `evaluate()`) and only
adds what imsim's harness needs:
  * accepts the `dual_config` keyword imitator.make passes (must be 'panda');
  * `self.dual.mount_poses()` (the Panda base pose; the harness and the v3.2 aftermath read it);
  * `imsim_obj` / `imsim_half` (the main object and a nominal half size; the v3.2 judge does not use them);
  * a longer TimeLimit (scripted experts + the evaluator's settle and aftermath run past ManiSkill's 50-100 steps).
The benchmark predicate (info['success'], the card's 'bench' column) is ManiSkill's own evaluate(), unchanged.

Registered ids: Imsim<Task>-v1 for PickCube, StackCube, PushCube, PullCube, PlaceSphere, StackPyramid, LiftPegUpright,
PlugCharger, PegInsertionSide (robot 'panda_wristcam', the only one ManiSkill supports there), PullCubeTool
(ManiSkill ids without 'Imsim').
"""
from __future__ import annotations

import sapien

import importlib
import importlib.util

from mani_skill.utils.registration import register_env

MAX_STEPS = 3000


class _SingleMount:
    name = "panda"

    def __init__(self, env):
        self.env = env

    def mount_poses(self):
        p = self.env.agent.robot.pose
        return [sapien.Pose(p.p[0].cpu().numpy().astype(float), p.q[0].cpu().numpy().astype(float))]


class _ImsimPanda:
    """mixin: imsim bookkeeping on top of a ManiSkill Panda task env"""
    IMSIM_OBJ = "cube"
    IMSIM_HALF = 0.02

    def _imsim_init(self, dual_config):
        if dual_config not in (None, "panda"):
            raise ValueError(f"{type(self).__name__} supports only the stock ManiSkill Panda (got {dual_config!r})")
        self.dual = _SingleMount(self)

    @property
    def imsim_obj(self):
        return getattr(self, self.IMSIM_OBJ)

    @property
    def imsim_half(self):
        return self.IMSIM_HALF


def _wrap(base, name: str, obj: str, half: float, robot: str = "panda"):
    class W(_ImsimPanda, base):
        IMSIM_OBJ, IMSIM_HALF = obj, half

        def __init__(self, *args, dual_config: str = "panda", **kwargs):
            self._imsim_init(dual_config)
            kwargs.setdefault("robot_uids", robot)
            super().__init__(*args, **kwargs)

    W.__name__ = W.__qualname__ = f"Imsim{name}Env"
    register_env(f"Imsim{name}-v1", max_episode_steps=MAX_STEPS)(W)
    return W


# The stock ManiSkill tabletop tasks are not part of the Imitator fork of ManiSkill (it keeps only the Imitator
# tasks). They are wrapped only when the installed ManiSkill provides them (e.g. ManiSkill 3.0.1); the topological
# tasks (envs/topo_tasks.py) only need _ImsimPanda / MAX_STEPS and work either way.
_STOCK = [  # (module, class, name, object attr, nominal half size, robot)
    ("pick_cube", "PickCubeEnv", "PickCube", "cube", 0.02, "panda"),
    ("stack_cube", "StackCubeEnv", "StackCube", "cubeA", 0.02, "panda"),
    ("push_cube", "PushCubeEnv", "PushCube", "obj", 0.02, "panda"),
    ("pull_cube", "PullCubeEnv", "PullCube", "obj", 0.02, "panda"),
    ("place_sphere", "PlaceSphereEnv", "PlaceSphere", "obj", 0.02, "panda"),
    ("stack_pyramid", "StackPyramidEnv", "StackPyramid", "cubeC", 0.02, "panda"),
    ("lift_peg_upright", "LiftPegUprightEnv", "LiftPegUpright", "peg", 0.025, "panda"),
    ("plug_charger", "PlugChargerEnv", "PlugCharger", "charger", 0.02, "panda"),
    # E27's held-out tasks. PegInsertionSide supports only 'panda_wristcam' (the Panda + a wrist camera).
    ("peg_insertion_side", "PegInsertionSideEnv", "PegInsertionSide", "peg", 0.02, "panda_wristcam"),
    ("pull_cube_tool", "PullCubeToolEnv", "PullCubeTool", "cube", 0.02, "panda"),
]
STOCK_AVAILABLE: dict[str, bool] = {}


def stock_task_available(module: str) -> bool:
    return importlib.util.find_spec(f"mani_skill.envs.tasks.tabletop.{module}") is not None


for _mod, _cls, _name, _obj, _half, _robot in _STOCK:
    STOCK_AVAILABLE[_name] = stock_task_available(_mod)
    if STOCK_AVAILABLE[_name]:
        _base = getattr(importlib.import_module(f"mani_skill.envs.tasks.tabletop.{_mod}"), _cls)
        globals()[f"Imsim{_name}Env"] = _wrap(_base, _name, _obj, _half, robot=_robot)
