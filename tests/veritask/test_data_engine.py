"""Tests for veritask.data (data engine v0).

Fast tests use a toy 1-D session (no simulator). One optional smoke test runs the real ManiSkill PokeCube binding when
mani_skill is importable (run it with research/envs/msup/bin/python).
usage: python -B tests/test_data_engine.py            (or: python -m pytest -q tests/test_data_engine.py)
"""
import copy, json, os, shutil, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import REPO  # noqa: E402
ROOT = os.path.join(REPO, "veritask")   # package dir (tasks/ lives here)
sys.path.insert(0, REPO)
from veritask.spec.contract import TaskContract, OutcomeTerm
from veritask.data import (Snapshot, Perturbation, Trajectory, ContractCertifier, ProgressGraph, ProgressNode, DataEngine,
                           PRIMITIVES, validate_ops, term_monotonicity, contract_progress, JointNoisePerturber, ShovePerturber,
                           DropPerturber, TargetOffsetPerturber, ProgramPerturber, AIProgramProposer, NullCorrector,
                           ReplayStagesCorrector, StageAwareCorrector, SamplingMPCCorrector, ExpertSource, WeakPolicySource,
                           EpisodeWriter, finalize)
from veritask.data.engine import SEG_CORRECT
from _paths import DATA, REPO, TMP as _TMP, repo_path  # noqa: E402,F401
TMP = _TMP

TMP = os.path.join(_TMP, "test_data_engine")
PROGRAMS = os.path.join(ROOT, "data/examples/perturbation_programs.json")


# ============================================================================== toy simulator
def toy_contract():
    """Slide a block to x = 1 with the pusher (never with the hand). Bench: |x - 1| < 0.1 at the end (looser)."""
    return TaskContract(task_id="toy_v1", instruction="push the block to x=1 with the tool", require_released=False, outcome=[
        OutcomeTerm(metric="trace.final", op="<", threshold=0.05, args=dict(signal="err", window=3)),
        OutcomeTerm(metric="trace.count_true", op="<=", threshold=0, args=dict(signal="hand_touch")),
        OutcomeTerm(metric="trace.count_true", op=">=", threshold=1, args=dict(signal="tool_touch"))])


class ToySession:
    """1-D block at x; 'tool' moves the block toward the goal; perturbation primitives move x directly."""
    task_id = "toy_v1"

    def __init__(self):
        self.contract = toy_contract(); self.plan_calls = []; self.backend_name = "toy"
        self.graph = ProgressGraph([ProgressNode("at_goal", "push", term_key="trace.final:err"),
                                    ProgressNode("tool_used", "push", term_key="trace.count_true:tool_touch")])
        self.plans = {"expert": ["approach", "push", "rest"]}

    # --- engine.Session
    def reset(self, seed):
        self.seed, self.x, self.step_count, self.ctx, self.stage, self.segment = seed, 0.0, 0, {}, "start", 0
        self.traces = {k: [] for k in ("err", "hand_touch", "tool_touch", "bench_success")}; self.log = dict(action=[], seg=[])

    def _step(self, a=0.0, tool=False, hand=False):
        self.x += a; self.step_count += 1
        err = abs(self.x - 1.0)
        for k, v in (("err", err), ("hand_touch", float(hand)), ("tool_touch", float(tool)), ("bench_success", float(err < 0.1))):
            self.traces[k].append(v)
        self.log["action"].append([a]); self.log["seg"].append(self.segment)

    def snapshot(self, stage, source, **meta):
        return Snapshot(task=self.task_id, seed=self.seed, step=self.step_count, stage=stage, source=source,
                        sim_state=dict(x=self.x), traces=copy.deepcopy(self.traces), log=copy.deepcopy(self.log),
                        ctx=dict(self.ctx), meta=dict(objects=["block"], holding={0: None}, targets=[], active_arms=[0], **meta))

    def restore(self, s):
        self.x, self.step_count, self.traces, self.log, self.ctx = s.sim_state["x"], s.step, copy.deepcopy(s.traces), copy.deepcopy(s.log), dict(s.ctx)

    def objects(self):
        return ["block"]

    def holding(self):
        return {0: None}

    def settle(self, n):
        for _ in range(n):
            self._step()

    def hold(self, steps):
        self.settle(steps)

    def begin_segment(self, seg):
        self.segment = seg

    def episode_arrays(self):
        return {"action": np.asarray(self.log["action"], np.float32), "segment": np.asarray(self.log["seg"], np.int8),
                **{f"trace.{k}": np.asarray(v, np.float32) for k, v in self.traces.items()}}

    def set_backend(self, name):
        return "toy"

    # --- primitives used by the perturbations below
    def shove(self, obj, vx=0.0, vy=0.0, vz=0.0, wx=0.0, wy=0.0, wz=0.0, settle=0):
        self.x += vx; self.settle(int(settle))

    def set_gripper(self, arm, cmd, steps=1):
        self.settle(int(steps))

    # --- plans
    def plan_stages(self, plan):
        return list(self.plans[plan])

    def run_plan(self, plan, start_stage=None, ctx=None, noise=None, rng=None, strict=False, on_boundary=None, until_segment_end=False):
        names = self.plans[plan]; i0 = names.index(start_stage) if start_stage else 0; ran = []
        for st in names[i0:]:
            if on_boundary:
                on_boundary(st)
            self.stage = st; ran.append(st)
            if st == "push":
                self.plan_calls.append(dict(backend="toy", ok=True, ms=0.1))
                for _ in range(20):
                    d = 1.0 - self.x
                    if abs(d) < 0.01:
                        break
                    self._step(a=float(np.clip(d, -0.1, 0.1)), tool=True)
                if noise:  # weak policy: stops early
                    self.x -= 0.2
            else:
                self._step()
        return dict(ok=True, stages=ran, failed=[], failed_stage=None, faults=[dict(kind="early", step=1)] if noise else [])

    def progress(self):
        return self.graph.infer(self.contract, self.traces, {})


# ============================================================================== tests
def test_certifier_categories():
    c = ContractCertifier(toy_contract(), released_signal=None)
    base = dict(hand_touch=[0, 0, 0], tool_touch=[0, 1, 0])
    assert c.certify(dict(base, err=[0.01] * 3, bench_success=[1, 1, 1])).category == "certified"
    assert c.certify(dict(base, err=[0.08] * 3, bench_success=[1, 1, 1])).category == "false_recovery"
    assert c.certify(dict(base, err=[0.01] * 3, bench_success=[1, 1, 0])).category == "contract_only"
    assert c.certify(dict(base, err=[0.5] * 3, bench_success=[0, 0, 0])).category == "both_fail"
    v = c.certify(dict(err=[0.01] * 3, hand_touch=[0, 1, 0], tool_touch=[0, 1, 0], bench_success=[1, 1, 1]))
    assert v.category == "false_recovery" and v.binding_term == "trace.count_true:hand_touch", v


def test_monotonicity_and_progress():
    t = toy_contract().outcome
    assert [term_monotonicity(x) for x in t] == ["state", "latch_false", "latch_true"]
    # the real contracts: PokeCube robot-cube contact is an invariant, PressJuicer knob depth latches true
    pc = TaskContract.load(os.path.join(ROOT, "tasks/poke_cube_v1.json"))
    pj = TaskContract.load(os.path.join(ROOT, "tasks/press_juicer_v1.json"))
    assert term_monotonicity(pc.outcome[1]) == "latch_false" and term_monotonicity(pj.outcome[0]) == "latch_true"
    prog = contract_progress(toy_contract(), dict(err=[0.5, 0.02], hand_touch=[0, 1], tool_touch=[0, 0]))
    assert prog["trace.final:err"]["holds"] and prog["trace.count_true:hand_touch"]["dead"]
    g = ProgressGraph([ProgressNode("at_goal", "push", term_key="trace.final:err")])
    assert g.infer(toy_contract(), dict(err=[0.5], hand_touch=[1], tool_touch=[0]), {})["abstain"]
    assert g.infer(toy_contract(), dict(err=[0.5], hand_touch=[0], tool_touch=[0]), {})["stage"] == "push"
    assert g.infer(toy_contract(), dict(err=[0.01], hand_touch=[0], tool_touch=[1]), {})["stage"] == "done"
    g2 = ProgressGraph([ProgressNode("upright", "no_skill", term_key="trace.final:err", or_aux="held")])
    assert g2.infer(toy_contract(), dict(err=[0.5], hand_touch=[0], tool_touch=[0]), {"held": True})["stage"] == "settle"


def test_validate_ops_and_programs():
    assert validate_ops([dict(op="shove", args=dict(obj="cube", vx=0.3, settle=5))], ["cube"]) == []
    assert validate_ops([dict(op="shove", args=dict(obj="cube", vx=9.0))], ["cube"])          # out of range
    assert validate_ops([dict(op="teleport", args={})], ["cube"])                              # unknown op
    assert validate_ops([dict(op="offset_pose", args=dict(obj="ghost", dx=0.01))], ["cube"])   # unknown object
    snap = Snapshot("toy_v1", 0, 10, "transport", "expert", ctl=dict(grip=[-1.0]),
                    meta=dict(objects=["cup_1"], holding={0: "cup_1"}, targets=["tray"], all_targets=["tray"], active_arms=[0]))
    props = ProgramPerturber(PROGRAMS).propose(snap, np.random.default_rng(0))
    assert {p.kind for p in props} == {"program:colleague_moves_target", "program:grasp_slip_with_nudge", "program:arm_bumped_up_and_aside"}
    slip = next(p for p in props if p.kind.endswith("slip_with_nudge"))
    assert slip.ops[1]["args"]["obj"] == "cup_1" and slip.ops[0]["args"]["arm"] == 0
    snap.meta["holding"] = {0: None}; snap.meta["targets"] = []
    assert {p.kind for p in ProgramPerturber(PROGRAMS).propose(snap, np.random.default_rng(0))} == {"program:arm_bumped_up_and_aside"}
    ok, rej = AIProgramProposer.parse('{"programs": [{"name": "a", "ops": [{"op": "hold", "args": {"steps": 5}}]},'
                                      ' {"name": "b", "ops": [{"op": "hold", "args": {"steps": 500}}]}]}', ["cube"])
    assert [p["name"] for p in ok] == ["a"] and len(rej) == 1
    assert "offset_pose" in AIProgramProposer.build_prompt("x", {"outcome": []}, {}, ["cube"])
    try:
        AIProgramProposer().generate("x", {}, {}, []); raise AssertionError("expected NotImplementedError")
    except NotImplementedError:
        pass


def test_builtin_perturbers():
    rng = np.random.default_rng(1)
    snap = Snapshot("t", 0, 5, "lift", "expert", ctl=dict(grip=[-1.0, 1.0]),
                    meta=dict(objects=["cup_1", "cup_2"], holding={0: "cup_1", 1: None}, targets=["tray"], active_arms=[0]))
    j = JointNoisePerturber().propose(snap, rng); assert len(j) == 3 and all(not validate_ops(p.ops, []) for p in j)
    s = ShovePerturber().propose(snap, rng); assert [p.params["obj"] for p in s] == ["cup_2"]   # held object skipped
    d = DropPerturber().propose(snap, rng); assert len(d) == 1 and d[0].ops[0]["args"]["cmd"] == 1.0 and d[0].ops[1]["args"]["cmd"] == -1.0
    t = TargetOffsetPerturber().propose(snap, rng); assert len(t) == 1 and 0.02 <= t[0].params["dist"] <= 0.06
    for p in j + s + d + t:
        assert not validate_ops(p.ops, ["cup_1", "cup_2", "tray"]), p


def test_engine_end_to_end_toy():
    shutil.rmtree(TMP, ignore_errors=True)
    env = ToySession(); cert = ContractCertifier(env.contract, released_signal=None)

    class Shove:
        name = "shove"

        def propose(self, snap, rng):
            return [Perturbation("shove", "object", [dict(op="shove", args=dict(obj="block", vx=v, settle=2))], dict(v=v))
                    for v in (-0.5, 0.07)] if snap.stage == "rest" else []

    class ToyL2(StageAwareCorrector):
        layer = "L2"; backend = "toy"
    rows = []
    eng = DataEngine(env, [ExpertSource(plan="expert", certifier=cert), WeakPolicySource(plan="expert", certifier=cert)],
                     [Shove()], [NullCorrector(), ReplayStagesCorrector(plan="expert"), ToyL2(plan="expert")], cert,
                     writer=EpisodeWriter(TMP, shard=0), settle_steps=3, per_perturber=2, log=rows.append)
    led = eng.run([0], rng_seed=0).summary()
    att = [r for r in rows if r.get("kind") == "attempt"]
    by = {(r["perturbation"]["params"].get("v"), r["source"], r["corrector"]): r for r in att}
    # big shove: L0 cannot fix it, L1 (resume at 'rest') does nothing, L2 infers 'push' and recovers
    assert by[(-0.5, "expert", "L0_hold")]["verdict"]["category"] == "both_fail"
    assert by[(-0.5, "expert", "L1_replay_expert")]["verdict"]["category"] == "both_fail"
    assert by[(-0.5, "expert", "L2_stage_toy")]["verdict"]["category"] == "certified"
    # small shove: bench (|err| < 0.1) still passes, contract (< 0.05) fails -> a false recovery for L0/L1
    assert by[(0.07, "expert", "L0_hold")]["verdict"]["category"] == "false_recovery"
    assert by[(0.07, "expert", "L2_stage_toy")]["verdict"]["category"] == "certified"
    # weak policy stops 0.2 short: its own failure state becomes a snapshot with the identity perturbation
    weak = [r for r in att if r["source"] == "weak"]
    assert weak and all(r["perturbation"]["kind"] == "weak_policy" for r in weak)
    assert any(r["corrector"] == "L2_stage_toy" and r["verdict"]["category"] == "certified" for r in weak)
    a = led["ALL"]
    assert a["attempts"] == len(att) and a["false_recovery"] >= 2 and a["certified"] >= 3
    info = finalize(TMP, dict(task="toy"))
    assert info["total_episodes"] == a["certified"] and info["total_rejected"] == a["false_recovery"]
    ep = [json.loads(l) for l in open(f"{TMP}/meta/episodes.jsonl")]
    z = np.load(os.path.join(TMP, ep[0]["file"]))
    assert len(z["action"]) == ep[0]["length"] and (z["segment"] == SEG_CORRECT).any() or ep[0]["corrector"] == "L0_hold"
    shutil.rmtree(TMP, ignore_errors=True)


def test_mpc_stub():
    try:
        SamplingMPCCorrector().correct(None, None, None); raise AssertionError
    except NotImplementedError:
        pass


def test_sim_smoke_pokecube():
    """Optional: real ManiSkill PokeCube through the session (skipped unless mani_skill 3.0.1 is importable)."""
    try:
        import importlib.util
        import mani_skill  # noqa: F401
        import mplib  # noqa: F401
        if importlib.util.find_spec("mani_skill.envs.tasks.tabletop.poke_cube") is None:
            raise ImportError("stock PokeCube-v1 is not in this mani_skill (the Imitator fork keeps only its own tasks)")
    except Exception:
        print("skip test_sim_smoke_pokecube (run with research/envs/msup/bin/python)"); return
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from veritask.data.maniskill.session import ManiSkillSession
    from veritask.data.maniskill.tasks import PokeCube
    sess = ManiSkillSession(PokeCube()); cert = ContractCertifier(sess.contract, released_signal=None)
    src = ExpertSource(plan="expert", certifier=cert)
    snaps = src.rollout(sess, 0, np.random.default_rng(0))
    assert src.last["verdict"]["contract_success"] and src.last["verdict"]["bench_success"], src.last
    snap = next(s for s in snaps if s.stage == "p_align")
    rows = []
    eng = DataEngine(sess, [], [], [NullCorrector(), StageAwareCorrector(plan="recover")], cert, log=rows.append)
    eng.run_attempt(snap, DropPerturber().propose(snap, np.random.default_rng(0))[0])
    cats = {r["corrector"]: r["verdict"]["category"] if r["verdict"] else r["reason"] for r in rows}
    assert cats["L0_hold"] == "both_fail" and cats["L2_stage_mplib"] == "certified", cats
    print("sim smoke:", cats)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f(); print("ok", k)
