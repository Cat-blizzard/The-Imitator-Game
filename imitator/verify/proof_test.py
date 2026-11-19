"""Proof test for imsim ManiSkill tasks (card D2): the simulator binding of the optional protocol stage declared in a
contract's "proof_test" block (veritask contract schema 0.4; semantics and verdict in veritask/adapters/protocol_v3.py,
section "D2 additions").

Order of the evaluator's stages:  policy -> settle (hold A) -> judged window -> retreat aftermath -> PROOF TEST.
The proof test
  1. clears the robot: the arm is moved (teleported, zero velocity) to a configuration out of the workspace, gripper
     open, and held there (Panda: joint 1 at +90 deg, arm raised over its base; TCP about 0.8 m above the table);
  2. lets the scene come to rest (up to 2 s, until every judged object's speed <= 2 cm/s for 10 steps) and snapshots it;
  3. for every declared test, restores the snapshot and records: stage 0 pre-settle (pre_settle_seconds), stage 1 the
     disturbance, stage 2 rest (rest_seconds); traces go under 'pt:<test>:<signal>' (+ 'pt:<test>:pt_stage');
  4. judges every test with veritask.adapters.protocol_v3.evaluate_proof_tests and leaves the scene in the snapshot.
Disturbances act on scene objects, never through the robot. Kinematic objects (racks, pegs, bins, the table) are driven
by per-physics-substep kinematic targets; a dynamic target of a driven test is made kinematic for the test. Pushes and
pulls act on dynamic objects (velocity change / force at the centre of mass, every substep).

  lift_tilt    params lift_m, tilt_deg, lift_seconds, tilt_seconds, hold_seconds, tilt_toward (scene object whose
               side goes down; default: world +x side) | tilt_axis [x, y, z]. Min-jerk lift, then min-jerk tilt about
               a horizontal axis through the container origin; the container stays lifted and tilted during the rest.
  shake        params amplitude_m, freq_hz, duration_seconds, axis [x, y, z], axis_frame 'world' | 'target',
               ramp_seconds (raised-cosine envelope at both ends; default 0.25 s). x(t) = A e(t) sin(2 pi f t).
  push         params delta_v_mps, direction [x, y, z] (world; projected horizontal), coast_seconds (default 0.5).
               A velocity change applied at the centre of mass in one physics substep (impulse = mass x delta_v).
  pull         params force_N, duration_seconds, axis [x, y, z], axis_frame 'world' | 'target'.
  table_shake  params accel_mps2, freq_hz, duration_seconds, axis [x, y, z], fixtures [kinematic objects moved with the
               table], ramp_seconds. Amplitude = accel / (2 pi f)^2.

    from imitator.verify import proof_test as PT
    j = PT.judge(rec, contract)            # v32.judge + proof test; j['proof'], j['success_with_proof']
"""
from __future__ import annotations

import math

import numpy as np

from . import v32

CLEAR_QPOS_PANDA = (np.pi / 2, -0.6, 0.0, -1.2, 0.0, 0.9, np.pi / 4)
CLEAR_SECONDS, CLEAR_SPEED, CLEAR_QUIET = 2.0, 0.02, 10


# ================================================================================================ helpers
def resolve(u, name: str):
    if name == "table":
        t = v32.table_actor(u)
        if t is None:
            raise KeyError("this env has no table actor")
        return t
    a = getattr(u, name, None)
    if a is None:
        raise KeyError(f"proof test target {name!r} is not an attribute of {type(u).__name__}")
    return a


def body(actor):
    return actor._bodies[0]


def pose_of(actor):
    import sapien
    o = actor._objs[0]
    p = o.entity_pose if hasattr(o, "entity_pose") else o.pose
    return sapien.Pose(np.asarray(p.p, float), np.asarray(p.q, float))


def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def envelope(t: float, T: float, ramp: float) -> float:
    if ramp <= 0:
        return 1.0
    if t < ramp:
        return 0.5 - 0.5 * math.cos(math.pi * t / ramp)
    if t > T - ramp:
        return 0.5 - 0.5 * math.cos(math.pi * max(T - t, 0.0) / ramp)
    return 1.0


def _axis(params: dict, actor, key: str = "axis", default=(1.0, 0.0, 0.0)) -> np.ndarray:
    a = np.asarray(params.get(key, default), float)
    if params.get(f"{key}_frame", "world") == "target":
        R = pose_of(actor).to_transformation_matrix()[:3, :3]
        a = R @ a
    return a / np.linalg.norm(a)


def _quat_axis_angle(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[math.cos(ang / 2), math.sin(ang / 2) * axis]


def _qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


# ================================================================================================ drivers
class Driver:
    """per-substep disturbance; t = time since the disturbance started, at the END of the upcoming substep"""
    duration = 0.0

    def __init__(self, u, test: dict):
        self.u, self.test, self.params = u, test, dict(test.get("params") or {})
        self.actor = resolve(u, test["target"])
        self.restore = []

    def begin(self):
        pass

    def substep(self, t: float):
        pass

    def end(self):
        for f in self.restore:
            f()

    def _make_kinematic(self, actor):
        b = body(actor)
        if not getattr(b, "kinematic", True):
            b.kinematic = True
            self.restore.append(lambda b=b: setattr(b, "kinematic", False))


class LiftTilt(Driver):
    def begin(self):
        p = self.params
        self.lift, self.tilt = float(p["lift_m"]), math.radians(float(p["tilt_deg"]))
        self.T1, self.T2, self.T3 = float(p["lift_seconds"]), float(p["tilt_seconds"]), float(p["hold_seconds"])
        self.duration = self.T1 + self.T2 + self.T3
        self._make_kinematic(self.actor)
        self.P0 = pose_of(self.actor)
        if "tilt_axis" in p:
            k = np.asarray(p["tilt_axis"], float)
        else:
            d = np.array([1.0, 0.0, 0.0])
            tw = p.get("tilt_toward")
            if tw:
                off = pose_of(resolve(self.u, tw)).p - self.P0.p
                off[2] = 0.0
                if np.linalg.norm(off) > 1e-3:
                    d = off / np.linalg.norm(off)
            k = np.cross([0.0, 0.0, 1.0], d)                 # rotating +tilt about z x d lowers the d side
        self.k = k / np.linalg.norm(k)
        self.info = dict(tilt_axis=self.k.round(4).tolist())

    def pose(self, t):
        import sapien
        dz = self.lift * min_jerk(t / self.T1) if self.T1 > 0 else self.lift
        ang = self.tilt * (min_jerk((t - self.T1) / self.T2) if self.T2 > 0 else float(t >= self.T1))
        q = _qmul(_quat_axis_angle(self.k, ang), np.asarray(self.P0.q, float))
        return sapien.Pose(np.asarray(self.P0.p, float) + [0, 0, dz], q)

    def substep(self, t):
        body(self.actor).set_kinematic_target(self.pose(min(t, self.duration)))


class Shake(Driver):
    def begin(self):
        p = self.params
        self.A, self.f, self.duration = float(p["amplitude_m"]), float(p["freq_hz"]), float(p["duration_seconds"])
        self.ramp = float(p.get("ramp_seconds", 0.25))
        self._make_kinematic(self.actor)
        self.P0 = pose_of(self.actor)
        self.a = _axis(p, self.actor)
        self.info = dict(axis=self.a.round(4).tolist(), peak_accel_mps2=round(self.A * (2 * math.pi * self.f) ** 2, 3))

    def offset(self, t):
        t = min(t, self.duration)
        return self.A * envelope(t, self.duration, self.ramp) * math.sin(2 * math.pi * self.f * t)

    def substep(self, t):
        import sapien
        body(self.actor).set_kinematic_target(sapien.Pose(np.asarray(self.P0.p) + self.offset(t) * self.a, self.P0.q))


class TableShake(Shake):
    def begin(self):
        p = self.params
        self.f, self.duration = float(p["freq_hz"]), float(p["duration_seconds"])
        self.A = float(p["accel_mps2"]) / (2 * math.pi * self.f) ** 2
        self.ramp = float(p.get("ramp_seconds", 0.25))
        self.a = _axis(p, self.actor)
        self.actors = [self.actor] + [resolve(self.u, n) for n in p.get("fixtures", [])]
        for a in self.actors:
            self._make_kinematic(a)
        self.P0s = [pose_of(a) for a in self.actors]
        self.info = dict(axis=self.a.round(4).tolist(), amplitude_m=round(self.A, 6),
                         fixtures=list(p.get("fixtures", [])))

    def substep(self, t):
        import sapien
        off = self.offset(t) * self.a
        for a, P0 in zip(self.actors, self.P0s):
            body(a).set_kinematic_target(sapien.Pose(np.asarray(P0.p) + off, P0.q))


class Push(Driver):
    def begin(self):
        p = self.params
        d = np.asarray(p.get("direction", [1.0, 0.0, 0.0]), float)
        d[2] = 0.0
        self.d = d / np.linalg.norm(d)
        self.dv = float(p["delta_v_mps"])
        self.duration = float(p.get("coast_seconds", 0.5))
        self.done = False
        b = body(self.actor)
        self.info = dict(direction=self.d.round(4).tolist(), mass_kg=float(b.mass),
                         impulse_Ns=round(float(b.mass) * self.dv, 6))

    def substep(self, t):
        if not self.done:
            b = body(self.actor)
            b.set_linear_velocity(np.asarray(b.get_linear_velocity(), float) + self.dv * self.d)
            self.done = True


class Pull(Driver):
    def begin(self):
        p = self.params
        self.F, self.duration = float(p["force_N"]), float(p["duration_seconds"])
        self.a = _axis(p, self.actor)
        self.info = dict(axis=self.a.round(4).tolist())

    def substep(self, t):
        if t <= self.duration + 1e-9:
            b = body(self.actor)
            com = (pose_of(self.actor) * b.cmass_local_pose).p
            b.add_force_at_point(self.F * self.a, np.asarray(com, float))


DRIVERS = {"lift_tilt": LiftTilt, "shake": Shake, "push": Push, "pull": Pull, "table_shake": TableShake}


# ================================================================================================ robot clearing
def clear_action(rec) -> np.ndarray:
    """teleport the (single) Panda out of the workspace and return the hold action there (pd_joint_pos, gripper open)"""
    import torch
    u = rec.env.unwrapped
    if hasattr(u.agent, "agents"):
        raise NotImplementedError("proof test: robot clearing is implemented for the single ManiSkill Panda only")
    ag = u.agent
    if ag.control_mode != "pd_joint_pos":
        rec.hold_action_A()                                   # switches to pd_joint_pos (restored at the next reset)
    q = np.r_[np.asarray(CLEAR_QPOS_PANDA, float), 0.04, 0.04]
    ag.robot.set_qpos(torch.tensor(q, dtype=torch.float32)[None])
    ag.robot.set_qvel(torch.zeros((1, len(q)), dtype=torch.float32))
    ag.controller.reset()
    return np.r_[np.asarray(CLEAR_QPOS_PANDA, float), 1.0].astype(np.float32)


# ================================================================================================ running
def _install(u, driver: Driver, dt: float):
    state = dict(k=0)
    had = vars(u).get("_before_simulation_step")              # an instance-level hook installed by someone else
    orig = u._before_simulation_step

    def before():
        orig()
        state["k"] += 1
        driver.substep(state["k"] * dt)
    u._before_simulation_step = before

    def uninstall():
        if state.get("off"):
            return
        state["off"] = True
        if had is not None:
            u._before_simulation_step = had
        elif "_before_simulation_step" in vars(u):
            del u._before_simulation_step                     # back to the class method
    return uninstall


def _steps(seconds: float, hz: float) -> int:
    return int(round(float(seconds) * hz))


def run_proof_tests(rec, contract) -> dict | None:
    """Run the contract's proof tests on a CPU recorder after v32.judge. Returns dict(result, traces, info) or None
    when the contract declares no proof test. The recorder's judged traces are left unchanged."""
    from veritask.adapters import protocol_v3 as P
    blk = P.proof_test_block(contract)
    if not blk:
        return None
    u = rec.env.unwrapped
    hz = float(u.control_freq)
    dt = 1.0 / float(u.sim_freq)
    saved = {k: list(v) for k, v in rec.traces.items()}
    names = list(saved)
    act = clear_action(rec)
    rec.hold_action = act
    rec._settling = True
    info: dict = dict(clear_steps=0)
    try:
        # 2. post-clear rest (not judged)
        rec.set_traces({k: [] for k in names})
        quiet = 0
        for k in range(_steps(CLEAR_SECONDS, hz)):
            rec.step(act)
            info["clear_steps"] = k + 1
            sp = rec.traces.get("objs_speed")
            quiet = quiet + 1 if (sp and sp[-1] <= CLEAR_SPEED) else 0
            if quiet >= CLEAR_QUIET:
                break
        info["clear_quiet"] = quiet >= CLEAR_QUIET
        snap = u.get_state_dict()
        out: dict = {}
        for test in blk["tests"]:
            u.set_state_dict(snap)
            rec.set_traces({k: [] for k in names})
            stage = []
            drv = DRIVERS[test["kind"]](u, test)
            for _ in range(_steps(blk["pre_settle_seconds"], hz)):
                rec.step(act); stage.append(0.0)
            drv.begin()
            n_dist = max(1, int(math.ceil(drv.duration * hz - 1e-9)))
            uninstall = _install(u, drv, dt)
            try:
                for _ in range(n_dist):
                    rec.step(act); stage.append(1.0)
                uninstall()
                for _ in range(_steps(test.get("rest_seconds", 1.0), hz)):      # driven objects stay where they are
                    rec.step(act); stage.append(2.0)
            finally:
                uninstall()
                drv.end()                                                       # dynamic targets made kinematic: back
            for k, v in rec.traces.items():
                out[P.pt_key(test["name"], k)] = list(v)
            out[P.pt_key(test["name"], P.PT_STAGE_SIGNAL)] = stage
            info[test["name"]] = dict(getattr(drv, "info", {}), n_steps=len(stage), n_disturb=n_dist)
        u.set_state_dict(snap)
    finally:
        rec._settling = False
        rec.set_traces(saved)
    res = P.evaluate_proof_tests(contract, out, control_freq=hz)
    return dict(result=res, traces=out, info=info)


def summary(res: dict | None, combined) -> dict | None:
    if res is None:
        return None
    r = res["result"]
    return dict(success=r["success"], with_proof=combined, binding_test=r["binding_test"], info=res["info"],
                tests=[{k: t[k] for k in ("name", "kind", "target", "success", "status", "terms", "margins",
                                          "binding_term", "n_steps")} for t in r["tests"]])


def judge(rec, contract, derive=None) -> dict:
    """v32.judge (settle hold A + retreat aftermath + verdict) followed by the proof test when the contract declares
    one and the judged window was found. Adds j['proof'] (summary), j['success_with_proof'] (main AND proof); the
    proof summary is also stored as j['aftermath']['proof_test'] and the proof traces are merged into
    j['_aft_traces'] (so imitator.evaluate writes them to episodes.jsonl and the trace file)."""
    from veritask.adapters import protocol_v3 as P
    j = v32.judge(rec, contract, derive=derive)
    if not P.proof_test_block(contract):
        j["proof"], j["success_with_proof"] = None, j["success"]
        return j
    if (j.get("window") or {}).get("status") != "ok":
        res = None
        j["proof"] = dict(success=None, with_proof=P.combine_with_proof(j["success"], dict(declared=True, success=None)),
                          status="skipped: no judged window")
    else:
        res = run_proof_tests(rec, contract)
        j["proof"] = summary(res, P.combine_with_proof(j["success"], res["result"]))
    j["success_with_proof"] = j["proof"]["with_proof"]
    j["aftermath"] = dict(j.get("aftermath") or {}, proof_test=j["proof"])
    if res is not None:
        j["_aft_traces"] = dict(j.get("_aft_traces") or {}, **res["traces"])
    return j
