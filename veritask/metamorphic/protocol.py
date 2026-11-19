"""Metamorphic test protocol for benchmark success predicates (card B1, 突破方向与实验构想.md section 4).

A success predicate is tested by *metamorphic relations*: after a transformation of the episode whose effect on a
genuine completion is known, the benchmark's OWN predicate is evaluated again.

  item  transformation (applied by the evaluator, never by the policy)                expected for a real completion
  M0    from the initial state do nothing (arm held, gripper at its initial command)   NOT success
        until the episode horizon
  M1    hold: keep the arm target and the gripper command of the test point, 2 s      still success
  M2    release: keep the arm target, command the gripper fully open, 2 s             still success   (place-type only)
  M3    retreat: continue from M2; move the TCP 10 cm up along world +z with the     still success   (place-type only)
        gripper open, then hold 1 s
  M4    continue: the policy keeps acting for 5 s (only policies we own)              still success
  M5    nudge: continue from M3; add a horizontal velocity change dv (calibrated per  still success   (place-type only)
        benchmark: an object resting on the table moves < 1 cm) to each manipulated
        object, then hold 2 s
  M6    freeze: goal objects fixed at their initial pose, the same actions replayed  NOT success
  M7    half step: physics timestep halved (control rate unchanged), the same actions verdict unchanged
        replayed

Test points: 'first' = the first control step at which the predicate is true; 'end' = the last step of the episode
(tests at 'end' only if the predicate is true there).  The *primary* test point of a benchmark is where its official
evaluation credits success ('first' for any-step scoring, 'end' for final-step scoring).

Everything here is benchmark-agnostic; durations are converted to control steps with the adapter's control rate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

PROTOCOL_VERSION = "b1-metamorphic-1.0"

ITEMS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7")
POST_ITEMS = ("M1", "M2", "M3", "M5", "M4")          # applied at a test point
EPISODE_ITEMS = ("M0", "M6", "M7")                   # applied from the initial state
PLACE_ONLY = ("M2", "M3", "M5")                      # card B1 step 2: M2/M3 (and M5, which follows them) place-type only
EXPECT_SUCCESS = {"M1": True, "M2": True, "M3": True, "M4": True, "M5": True, "M0": False, "M6": False}
TASK_CLASSES = ("place", "hold", "event")
TEST_POINTS = ("first", "end")


@dataclass
class Protocol:
    hold_s: float = 2.0             # M1
    release_s: float = 2.0          # M2
    retreat_height_m: float = 0.10  # M3: TCP displacement along world +z
    retreat_speed_mps: float = 0.10  # M3: commanded TCP speed of the lift (servo reference moves at this speed)
    retreat_timeout_s: float = 3.0  # M3: lift phase cap (then the wait starts regardless; 'reached' is recorded)
    retreat_tol_m: float = 0.01     # M3: TCP within this distance of the lift target counts as reached
    retreat_wait_s: float = 1.0     # M3
    continue_s: float = 5.0         # M4
    nudge_dv_mps: float | None = None   # M5: calibrated per benchmark (None = M5 not run)
    nudge_wait_s: float = 2.0       # M5
    nudge_calib_max_disp_m: float = 0.01   # M5 calibration criterion (median displacement of a resting object)
    half_step_scale: float = 0.5    # M7
    contact_force_eps_n: float = 0.05      # arm-object contact threshold (N); adapters may use impulse equivalents
    items: tuple = ITEMS
    test_points: tuple = TEST_POINTS
    version: str = PROTOCOL_VERSION
    notes: dict = field(default_factory=dict)

    def steps(self, seconds: float, control_freq: float) -> int:
        return max(1, int(round(seconds * control_freq)))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["items"] = list(self.items); d["test_points"] = list(self.test_points)
        return d


def applicable(item: str, task_class: str, has_policy: bool = False) -> bool:
    """Whether a post-success item applies to a task class (card B1: M2/M3 only for place-type tasks)."""
    if item in PLACE_ONLY:
        return task_class == "place"
    if item == "M4":
        return has_policy
    return True
