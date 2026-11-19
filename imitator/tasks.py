"""imsim task registry: one TaskSpec per task (env id, embodiments, contracts, scripted behaviours, recorder).

    import imitator
    imitator.list_tasks()          # names
    imitator.get_task(name)        # TaskSpec

Behaviour names are the contract's counterexample names (role control / counterexample comes from `expected`); aliases
map short names (expert, hold, cage, ...) to them.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable

HERE = os.path.dirname(os.path.abspath(__file__))
CONTRACTS = os.path.join(HERE, "contracts")
SOFT_CONTRACTS = os.path.join(HERE, "soft", "contracts")
GRIPPER_CONFIGS = ("dual_ur5_wsg", "dual_arx_x5", "dual_piper", "dual_xarm6_robotiq")
ALL_DUAL = GRIPPER_CONFIGS + ("dual_xarm6_allegro",)


def _ref(s: str):
    """'module:attr' -> object (lazy import)"""
    mod, attr = s.split(":")
    return getattr(importlib.import_module(mod), attr)


@dataclass
class TaskSpec:
    name: str
    env_id: str
    module: str                                   # imported by make() (registers env_id with ManiSkill)
    kind: str                                     # 'rigid' | 'soft'
    embodiments: tuple
    default_embodiment: str
    contracts: dict                               # contract name -> json path
    default_contract: str
    extra_contracts: tuple = ()                   # judged on the same rollouts by default (e.g. v3 next to v3.1)
    scripts: dict = field(default_factory=dict)   # behaviour -> ('module:function', kwargs)
    aliases: dict = field(default_factory=dict)
    recorder: str = ""                            # 'module:function(env) -> TraceRecorderV3'
    object_attr: str = "cube"
    half_attr: str = "cube_half"
    derive: str | None = None                     # 'module:function(contract) -> derive(traces)'
    reset_options: Callable[[int], dict] | None = None
    ground_truth: str | None = None               # 'module:function(rec, log) -> dict' run after the verdict
    batched_script: str | None = None             # GPU batched scripted behaviours (evaluate(sim='gpu'))
    max_policy_steps: int = 600
    has_cloth: bool = False
    gpu_only_cloth: bool = False                  # Newton cloth needs CUDA even when the robot runs on PhysX CPU
    description: str = ""
    env_defaults: dict = field(default_factory=dict)
    # data engine (E29a, imitator/data/engine.py); all optional, the defaults keep every existing task working
    data_expert: str = "expert"                   # behaviour (or alias) imitator.generate runs as the expert
    data_script: str | None = None                # 'module:function(env, **kw)' expert for imitator.generate when the
                                                  # `scripts` entry is not an env driver (cloth tasks)
    phys_exclude: tuple = ()                      # actor names the reset physical uncertainty must not move / rescale
    cloth_pose_aware: bool = False                # the expert reads the cloth's actual pose (cloth pose offsets allowed)
    judge: str | None = None                      # 'module:function(rec, contract, derive=None) -> v31.judge_rec-style
                                                  # dict' (settle + aftermath + verdict); None = v31.judge_rec

    def behaviour(self, name: str) -> str:
        return self.aliases.get(name, name)

    def script(self, name: str):
        fn, kw = self.scripts[self.behaviour(name)]
        f = _ref(fn)
        return lambda env: f(env, **kw)

    def contract_path(self, name: str | None = None) -> str:
        name = name or self.default_contract
        if name in self.contracts:
            return self.contracts[name]
        if os.path.exists(name):
            return name
        raise KeyError(f"{self.name}: unknown contract {name!r} (known: {sorted(self.contracts)})")

    def load_contract(self, name: str | None = None):
        from .verify import v31  # noqa: F401  (registers the aftermath metric before validation)
        from veritask.spec.contract import TaskContract
        return TaskContract.load(self.contract_path(name))

    def describe(self) -> dict:
        return dict(name=self.name, env_id=self.env_id, kind=self.kind, embodiments=list(self.embodiments),
                    default_embodiment=self.default_embodiment, contracts=sorted(self.contracts),
                    default_contract=self.default_contract, behaviours=sorted(self.scripts),
                    aliases=dict(self.aliases), gpu_batched=bool(self.batched_script), has_cloth=self.has_cloth,
                    description=self.description)


def _side_by_seed(seed: int) -> dict:
    return {"side": "left" if int(seed) % 2 == 0 else "right"}      # E19 protocol: even seeds left arm, odd right


TASKS: dict[str, TaskSpec] = {}

TASKS["DualCubePlace-v1"] = TaskSpec(
    name="DualCubePlace-v1", env_id="ImsimDualTabletop-v1", module="imitator.envs.dual_tabletop", kind="rigid",
    embodiments=ALL_DUAL, default_embodiment="dual_ur5_wsg",
    contracts={"dual_cube_place_v3.1": os.path.join(CONTRACTS, "dual_cube_place_v3.1.json"),
               "dual_cube_place_v3": os.path.join(CONTRACTS, "dual_cube_place_v3.json")},
    default_contract="dual_cube_place_v3.1", extra_contracts=("dual_cube_place_v3",),
    scripts={"scripted_place": ("imitator.planning.scripted:pick_place", dict(variant="release")),
             "hold_without_release": ("imitator.planning.scripted:pick_place", dict(variant="hold")),
             "cage_no_contact": ("imitator.planning.scripted:pick_place", dict(variant="cage")),
             "no_op": ("imitator.planning.scripted:no_op", {})},
    aliases={"expert": "scripted_place", "release": "scripted_place", "hold": "hold_without_release",
             "cage": "cage_no_contact"},
    recorder="imitator.verify.v31:make_recorder_v31", object_attr="cube", half_attr="cube_half",
    reset_options=_side_by_seed, ground_truth="imitator.harness:cube_lift_probe",
    batched_script="imitator.planning.batched:pick_place_batched",
    description="E19 smoke task: one cube, one target square, either arm (the arm on the cube's side) places it.")

TASKS["DualHandover-v1"] = TaskSpec(
    name="DualHandover-v1", env_id="DualHandover-v1", module="imitator.envs.dual_handover", kind="rigid",
    embodiments=GRIPPER_CONFIGS, default_embodiment="dual_ur5_wsg",
    contracts={"imsim_dual_handover_v2": os.path.join(CONTRACTS, "imsim_dual_handover_v2.json"),
               "dual_handover_v1": os.path.join(CONTRACTS, "dual_handover_v1.json")},
    default_contract="imsim_dual_handover_v2",       # E29b revision (v3.2 signals); v1 needs recorder make_recorder
    scripts={b: ("imitator.planning.handover:run", dict(variant=b))
             for b in ("handover", "no_handover", "both_hold", "drop_and_push", "table_transfer", "no_op")},
    aliases={"expert": "handover"},
    recorder="imitator.verify.handover:make_recorder_v32", object_attr="block", half_attr="block_half",
    derive="imitator.verify.handover:make_derive", max_policy_steps=3000, judge="imitator.verify.v32:judge",
    description="Left arm picks a 4x18x4 cm block (72 g, E29b) from the left zone, hands it to the right arm in mid-air, "
                "the right arm places it in the right target square, both let go.")

TASKS["DualFoldTowel-v1"] = TaskSpec(
    name="DualFoldTowel-v1", env_id="DualFoldTowel-v1", module="imitator.envs.dual_fold_towel", kind="soft",
    # E29b: dual_xarm6_robotiq removed (E23: no cloth support; E29b pilot with an empirical cloth-close command: 0/4,
    # the released towel stays caught in the opened Robotiq linkage and is lifted by the retreat)
    embodiments=("dual_ur5_wsg", "dual_arx_x5", "dual_piper"), default_embodiment="dual_ur5_wsg",
    contracts={"dual_towel_fold_v1": os.path.join(SOFT_CONTRACTS, "dual_towel_fold_v1.json"),
               "dual_towel_fold_v2": os.path.join(SOFT_CONTRACTS, "dual_towel_fold_v2.json")},
    default_contract="dual_towel_fold_v2",     # E29b revision (E24c fixes: 5 s cap, position-based speed, retreat)
    scripts={b: ("imitator.planning.fold:run", dict(behaviour=b))
             for b in ("expert", "no_grip", "drop_midway", "drag_on_ground")},
    has_cloth=True, gpu_only_cloth=True, max_policy_steps=400,
    cloth_pose_aware=True,                     # E29b: planning.fold.corner_points reads the towel's actual corners
    description="30x30 cm Newton VBD towel; each arm grasps one far corner and folds it onto the near corner on its side "
                "(half fold); one-way PhysX -> Newton bridge (E20 route A).")

TASKS["FoldTowel-v1"] = TaskSpec(
    name="FoldTowel-v1", env_id="FoldTowel-v1", module="imitator.envs.fold_towel", kind="soft",
    embodiments=("panda_cloth",), default_embodiment="panda_cloth",
    contracts={"towel_fold_diagonal_v3": os.path.join(SOFT_CONTRACTS, "towel_fold_diagonal_v3.json")},
    default_contract="towel_fold_diagonal_v3",
    scripts={b: ("imitator.soft.expert:towel_waypoints", dict(mode=b)) for b in ("expert", "no_grip", "drop_midway", "drag_on_ground")},
    has_cloth=True, gpu_only_cloth=True, max_policy_steps=200,
    data_script="imitator.soft.expert:run_towel", cloth_pose_aware=True,       # E29a R1.2 (run_towel annotates phases)
    description="E20 single-arm Panda diagonal towel fold (route A).")


# ================================================================================================ E29b task expansion
# Skill tags for the task table (research/results/e29_platform/tasks/task_table.md).
TASK_SKILLS: dict[str, str] = {"DualCubePlace-v1": "pick-and-place (either arm)",
                               "DualHandover-v1": "bimanual mid-air handover",
                               "DualFoldTowel-v1": "bimanual cloth fold (Newton VBD)",
                               "FoldTowel-v1": "cloth diagonal fold (Newton VBD)"}

# ------------------------------------------------------------------------------------------------ ManiSkill Panda (E29b)
PANDA = ("panda",)


def _stock_available(ms_id: str) -> bool:
    """the ManiSkill stock task behind an Imsim<Task> wrapper exists in the installed mani_skill (not in the fork)"""
    import importlib.util
    import re as _re
    mod = _re.sub(r"(?<!^)(?=[A-Z])", "_", ms_id.split("-")[0]).lower()
    return importlib.util.find_spec(f"mani_skill.envs.tasks.tabletop.{mod}") is not None


def _panda(name: str, ms_id: str, contract: str, fn: str, behaviours, recorder: str, skill: str, desc: str,
           aliases: dict | None = None, max_steps: int = 1500):
    if not _stock_available(ms_id):
        return None
    TASKS[name] = TaskSpec(
        name=name, env_id=f"Imsim{ms_id}", module="imitator.envs.panda_ms", kind="rigid", embodiments=PANDA,
        default_embodiment="panda", contracts={contract: os.path.join(CONTRACTS, contract + ".json")},
        default_contract=contract,
        scripts={b: (f"imitator.planning.panda_ms:{fn}", dict(variant=b)) for b in behaviours},
        aliases=dict(aliases or {}), recorder=f"imitator.verify.panda_tasks:{recorder}", object_attr="imsim_obj",
        half_attr="imsim_half", max_policy_steps=max_steps, description=desc, judge="imitator.verify.v32:judge")
    TASK_SKILLS[name] = skill


_panda("PickCube-v1", "PickCube-v1", "imsim_pick_cube_v1", "pick_cube",
       ("expert", "drop_at_goal", "place_under_goal", "near_miss", "no_op"), "pick_cube", "pick and hold at a 3-D goal",
       "ManiSkill PickCube-v1 (Panda): grasp the cube and hold it still at the goal point (no release).")
_panda("StackCube-v1", "StackCube-v1", "imsim_stack_cube_v1", "stack_cube",
       ("expert", "hold_on_top", "cage", "press_on_top", "beside", "no_op"), "stack_cube", "stack (place on object)",
       "ManiSkill StackCube-v1 (Panda): put cube A on cube B, let go, move away.")
_panda("PushCube-v1", "PushCube-v1", "imsim_push_cube_v1", "push_cube",
       ("expert", "carry", "push_and_press", "short", "no_op"), "push_cube", "non-prehensile push",
       "ManiSkill PushCube-v1 (Panda): push the cube along the table into the goal disc, move away.")
_panda("PullCube-v1", "PullCube-v1", "imsim_pull_cube_v1", "pull_cube",
       ("expert", "carry", "hold_above", "pull_and_press", "short", "no_op"), "pull_cube", "non-prehensile pull",
       "ManiSkill PullCube-v1 (Panda): pull the cube along the table towards the robot into the goal disc, move away.")
_panda("PlaceSphere-v1", "PlaceSphere-v1", "imsim_place_sphere_v1", "place_sphere",
       ("expert", "hold_in_bin", "cage", "beside", "no_op"), "place_sphere", "place into a container",
       "ManiSkill PlaceSphere-v1 (Panda): put the ball into the small bin, let go, move away.")
_panda("StackPyramid-v1", "StackPyramid-v1", "imsim_stack_pyramid_v1", "stack_pyramid",
       ("expert", "hold_top", "c_on_a_only", "pair_only", "cage_top", "no_op"), "stack_pyramid",
       "multi-object assembly (3-cube pyramid)",
       "ManiSkill StackPyramid-v1 (Panda): put cube A next to cube B, cube C on top of both, let go, move away.",
       max_steps=2500)
_panda("LiftPegUpright-v1", "LiftPegUpright-v1", "imsim_lift_peg_upright_v1", "lift_peg_upright",
       ("expert", "hold_upright", "tilted_release", "lift_flat", "no_op"), "lift_peg_upright", "reorient (stand upright)",
       "ManiSkill LiftPegUpright-v1 (Panda): stand the lying peg upright on the table, let go, move away.")

_panda("PlugCharger-v1", "PlugCharger-v1", "imsim_plug_charger_v1", "plug_charger",
       ("expert", "hold_plugged", "release_short", "cage_plugged", "no_op"), "plug_charger", "insertion (tight, 0.5 mm)",
       "ManiSkill PlugCharger-v1 (Panda): plug the charger into the receptacle, let go, move away.")

# E27's held-out tasks, added after E27 finished (coordinator request); E27's protocol-v2 contracts are the reference
_panda("PegInsertionSide-v1", "PegInsertionSide-v1", "imsim_peg_insertion_side_v1", "peg_insertion_side",
       ("expert", "hold_inserted", "cage_inserted", "partial_insert", "release_at_entrance", "no_op"), "peg_insertion_side",
       "insertion (side hole, 3 mm clearance)",
       "ManiSkill PegInsertionSide-v1 (Panda + wrist camera): insert the peg's head into the side hole of the box, let go, "
       "move away.")
_panda("PullCubeTool-v1", "PullCubeTool-v1", "imsim_pull_cube_tool_v1", "pull_cube_tool",
       ("expert", "hold_tool", "cage_tool", "pull_short", "pull_by_gripper", "no_op"), "pull_cube_tool",
       "tool use (hook and pull)",
       "ManiSkill PullCubeTool-v1 (Panda): use the L-shaped tool to pull the out-of-reach cube towards the robot, let go "
       "of the tool, move away.")

# ------------------------------------------------------------------------------------------------ E29b: dual-arm tasks
# (added below by the dual-task modules; keep this marker)

# --- E29b fork B1: DualLiftBox-v1, DualStack-v1
B1_CONFIGS = GRIPPER_CONFIGS
TASKS["DualLiftBox-v1"] = TaskSpec(
    name="DualLiftBox-v1", env_id="DualLiftBox-v1", module="imitator.envs.dual_lift_box", kind="rigid",
    embodiments=B1_CONFIGS, default_embodiment="dual_ur5_wsg",
    contracts={"imsim_dual_lift_box_v1": os.path.join(CONTRACTS, "imsim_dual_lift_box_v1.json")},
    default_contract="imsim_dual_lift_box_v1",
    scripts={b: ("imitator.planning.dual_lift_box:run", dict(variant=b))
             for b in ("expert", "single_arm_center", "one_end_lift", "low_lift", "drop_at_end", "no_op")},
    recorder="imitator.verify.dual_tasks_b1:lift_box", object_attr="box", half_attr="box_half", max_policy_steps=2000,
    judge="imitator.verify.v32:judge",
    description="Both arms grasp the two ends of a 4 x 36 x 4 cm box and lift it together, level, at least 5 cm off the "
                "table, and hold it still (no release).")
TASK_SKILLS["DualLiftBox-v1"] = "bimanual cooperative lift (held, level)"

TASKS["DualStack-v1"] = TaskSpec(
    name="DualStack-v1", env_id="DualStack-v1", module="imitator.envs.dual_stack", kind="rigid",
    embodiments=B1_CONFIGS, default_embodiment="dual_ur5_wsg",
    contracts={"imsim_dual_stack_v1": os.path.join(CONTRACTS, "imsim_dual_stack_v1.json")},
    default_contract="imsim_dual_stack_v1",
    scripts={b: ("imitator.planning.dual_stack:run", dict(variant=b))
             for b in ("expert", "one_arm_does_both", "hold_top", "side_by_side", "cage", "no_op")},
    recorder="imitator.verify.dual_tasks_b1:stack", object_attr="cubeB", half_attr="cube_half",
    derive="imitator.verify.dual_tasks_b1:make_stack_derive", max_policy_steps=3000, judge="imitator.verify.v32:judge",
    description="The left arm puts the red cube on the green square, the right arm puts the blue cube on top of it; "
                "both let go and move away.")
TASK_SKILLS["DualStack-v1"] = "bimanual sequential stacking (one cube per arm)"

# --- E29b fork B2: DualPegInHole-v1, DualDrawer-v1
# PiPER is not supported by DualPegInHole (its grasp lets the held block pivot 4-10 deg and the insertion poses are not
# IK-reachable on most seeds); xArm6 + Robotiq is not supported by DualDrawer (its grasp slips off the handle / knob
# under the drawer's 5 N pull). See research/results/e29_platform/tasks/STATUS.md ([B2] entries).
TASKS["DualPegInHole-v1"] = TaskSpec(
    name="DualPegInHole-v1", env_id="DualPegInHole-v1", module="imitator.envs.dual_peg_in_hole", kind="rigid",
    embodiments=("dual_ur5_wsg", "dual_arx_x5", "dual_xarm6_robotiq"), default_embodiment="dual_ur5_wsg",
    contracts={"imsim_dual_peg_in_hole_v1": os.path.join(CONTRACTS, "imsim_dual_peg_in_hole_v1.json")},
    default_contract="imsim_dual_peg_in_hole_v1",
    scripts={b: ("imitator.planning.dual_peg_in_hole:run", dict(variant=b))
             for b in ("expert", "hold_peg", "peg_at_entrance", "block_on_table", "no_op")},
    recorder="imitator.verify.dual_tasks_b2:peg_in_hole", object_attr="peg", half_attr="cube_half", max_policy_steps=3000,
    judge="imitator.verify.v32:judge", phys_exclude=("block", "peg"),
    description="The left arm lifts a holed block by its handle and holds it up; the right arm inserts a standing peg into "
                "the block's blind hole and lets go; the left arm keeps holding the block.")
TASK_SKILLS["DualPegInHole-v1"] = "bimanual in-air insertion (hold + insert)"

TASKS["DualDrawer-v1"] = TaskSpec(
    name="DualDrawer-v1", env_id="DualDrawer-v1", module="imitator.envs.dual_drawer", kind="rigid",
    embodiments=("dual_ur5_wsg", "dual_arx_x5", "dual_piper"), default_embodiment="dual_ur5_wsg",
    contracts={"imsim_dual_drawer_v1": os.path.join(CONTRACTS, "imsim_dual_drawer_v1.json")},
    default_contract="imsim_dual_drawer_v1",
    scripts={b: ("imitator.planning.dual_drawer:run", dict(variant=b))
             for b in ("expert", "pull_without_stabilising", "open_small", "hold_handle", "no_op")},
    recorder="imitator.verify.dual_tasks_b2:drawer", object_attr="cabinet_link", half_attr="cube_half",
    max_policy_steps=3000, judge="imitator.verify.v32:judge", phys_exclude=("cabinet_art",),
    description="A free-standing cabinet with a sticky drawer: the left arm holds the cabinet still by its top knob, the "
                "right arm pulls the drawer open >= 8 cm; both let go and move away.")
TASK_SKILLS["DualDrawer-v1"] = "bimanual stabilise-and-open (articulated drawer)"

# ------------------------------------------------------------------------------------------------ E29b: cloth tasks
# (added below by the cloth-task modules; keep this marker)
CLOTH_GRIPPERS = ("dual_ur5_wsg", "dual_arx_x5", "dual_piper")   # prismatic parallel grippers (cloth close command)
TASKS["DualClothCoverBox-v1"] = TaskSpec(
    name="DualClothCoverBox-v1", env_id="DualClothCoverBox-v1", module="imitator.envs.dual_cloth_cover", kind="soft",
    embodiments=CLOTH_GRIPPERS, default_embodiment="dual_ur5_wsg",
    contracts={"dual_cloth_cover_box_v1": os.path.join(SOFT_CONTRACTS, "dual_cloth_cover_box_v1.json"),
               "dual_cloth_cover_box_v2": os.path.join(SOFT_CONTRACTS, "dual_cloth_cover_box_v2.json")},
    default_contract="dual_cloth_cover_box_v2",           # E29b revision: v1 accepted short_fold in 4/60 worlds
    scripts={b: ("imitator.planning.fold:run", dict(behaviour=b))
             for b in ("expert", "short_fold", "drag_on_ground", "hold_cloth", "no_grip")},
    has_cloth=True, gpu_only_cloth=True, max_policy_steps=400,
    description="E29b: 30x30 cm Newton VBD towel with a 6x6x2 cm box resting on its near half; both arms lift the far "
                "corners and lay the far half over the box (covered top face), release, retreat (route A).")
TASK_SKILLS["DualClothCoverBox-v1"] = "bimanual cloth covering of an object (Newton VBD)"


def get_task(name: str) -> TaskSpec:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]


def list_tasks() -> list[str]:
    return sorted(TASKS)


# ================================================================================================ card D2 (appended)
# Topological tasks (research/results/d2_topology/): success = a topological relation (threaded / hung by the handle),
# judged by veritask's topology trace metrics, plus an evaluator-run proof test declared in the contract (schema 0.4).
# The env's evaluate() is the benchmark-style distance predicate (card predicate (a)), not the contract.
def _topo(name: str, contract: str, fn: str, behaviours, recorder: str, skill: str, desc: str, obj: str,
          aliases: dict | None = None, max_steps: int = 2000):
    TASKS[name] = TaskSpec(
        name=name, env_id=f"Imsim{name}", module="imitator.envs.topo_tasks", kind="rigid", embodiments=PANDA,
        default_embodiment="panda", contracts={contract: os.path.join(CONTRACTS, contract + ".json")},
        default_contract=contract,
        scripts={b: (f"imitator.planning.topo_tasks:{fn}", dict(variant=b)) for b in behaviours},
        aliases=dict(aliases or {}), recorder=f"imitator.verify.topo_tasks:{recorder}", object_attr=obj,
        half_attr="imsim_half", max_policy_steps=max_steps, description=desc, judge="imitator.verify.proof_test:judge")
    TASK_SKILLS[name] = skill


_topo("RingOnPeg-v1", "imsim_ring_on_peg_v1", "ring_on_peg", ("expert", "on_top", "beside", "held", "no_op"),
      "ring_on_peg", "thread a ring over a peg (topological)",
      "D2: put the torus ring (core radius 4.5 cm) over the 6 cm peg, let go, move away; judged by disk crossings of the "
      "peg axis through the ring + a push / table-shake proof test.", obj="ring")
_topo("MugOnHook-v1", "imsim_mug_on_hook_v1", "mug_on_hook",
      ("expert", "expert_shallow", "on_tip", "held_near", "beside", "dropped_below", "no_op"),
      "mug_on_hook", "hang a mug by its handle (topological)",
      "D2: hang the mug on the rack's hook by its handle, let go, move away; judged by disk crossings of the hook through "
      "the handle + a hook-shake proof test.", obj="mug")
