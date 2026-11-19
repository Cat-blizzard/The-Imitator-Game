"""imitator.evaluate(..., sim='gpu', num_envs=N) for rigid tasks with a batched script (DualCubePlace-v1): GPU PhysX, N envs,
batched scripted policy, batched hold-A settle, batched v3.1 retreat aftermath, per-env protocol-v3 verdicts (v3.1 and
v3 on the same rollout), evidence rows, throughput and GPU memory of the process."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .tasks import _ref


def _hold_targets(ex):
    for ag, t in zip(ex.agents, ex.targets):
        idx = [i for i, j in enumerate(ag.robot.active_joints) if j.name in ag.record.arm_joints]
        t[:, :ag.record.dof_arm] = ag.robot.get_qpos()[:, idx].cpu().numpy()


def settle_batched(rec, ex, contract) -> list[dict]:
    from veritask.adapters.protocol_v3 import SettleConfig, find_window
    u = rec.unwrapped
    cfg = SettleConfig.from_contract(contract, "A")
    K, W, N = cfg.steps(u.control_freq)
    n_pol = rec.n_steps()
    _hold_targets(ex)
    rec._settling = True
    st = [dict(status="incomplete")] * rec.N
    try:
        for k in range(N):
            ex.step()
            if k >= K + W - 1:
                st = [find_window(rec.traces[e], cfg, n_pol, u.control_freq) for e in range(rec.N)]
                if all(s["status"] in ("ok", "not_quiescent", "missing_signal") for s in st):
                    break
    finally:
        rec._settling = False
    return st


def aftermath_batched(rec, ex, cfg: dict, windows: list[dict]) -> list[dict]:
    from .verify import v31
    u = rec.unwrapped
    dt = float(u.control_timestep)
    bases = u.dual.mount_poses()
    half = float(u.cube_half)
    T = u.cube.pose.to_transformation_matrix().cpu().numpy().astype(float)
    p0 = T[:, :3, 3].copy()
    n0 = rec.n_steps()
    plans, info = {}, [dict(near=[], failed=[]) for _ in range(rec.N)]
    for e in range(rec.N):
        if windows[e].get("status") != "ok":
            continue
        for i, ag in enumerate(ex.agents):
            if v31.min_gripper_distance(ag, T[e], half, env_idx=e) <= cfg["near_m"]:
                info[e]["near"].append(i)
                q = v31.plan_retreat(ag, bases[i], dt, cfg, env_idx=e)
                if q is None:
                    info[e]["failed"].append(i)
                else:
                    plans[(e, i)] = q
    n_ret = max([len(q) for q in plans.values()] + [int(round(cfg["retreat_s"] / dt))])
    n = n_ret + int(round(cfg["hold_s"] / dt))
    disp = np.zeros(rec.N)
    rec._settling = True
    try:
        for k in range(n):
            for (e, i), q in plans.items():
                ex.targets[i][e, :ex.agents[i].record.dof_arm] = q[min(k, len(q) - 1)]
            ex.step()
            p = u.cube.pose.p.cpu().numpy().astype(float)
            disp = np.maximum(disp, np.linalg.norm(p - p0, axis=1))
    finally:
        rec._settling = False
    for e in range(rec.N):
        aft = {v31.AFT_PREFIX + k: list(v[n0:]) for k, v in rec.traces[e].items()}
        for k in list(rec.traces[e]):
            rec.traces[e][k] = rec.traces[e][k][:n0]
        info[e].update(value=(float(disp[e]) if windows[e].get("status") == "ok" and not info[e]["failed"] else None),
                       steps=n, aft=aft)
    return info


class _RecEnv:
    """BatchExecutor drives the recorder (records every step) through this thin adapter"""

    def __init__(self, rec):
        self.rec = rec
        self.unwrapped = rec.unwrapped

    def step(self, a):
        return self.rec.step(a)


def evaluate_batched(spec, emb, policy_or_script, seeds, contract=None, num_envs: int = 16, out: str = "", card=False,
                     experiment: str = "E23", verbose: bool = True, env_kwargs=None, save_traces: bool = True, tag: str = ""):
    from .api import make
    from .harness import _jsonable, _write_rows, evidence_row, role_of, write_card
    from .planning.batched import BatchExecutor
    from .soft.gpu_mem import GpuMemPoller
    from .verify import v31
    from .verify.batched import BatchedRecorder
    from veritask.adapters.base import save_traces as _save
    from veritask.audit import evidence as E
    if not spec.batched_script:
        raise ValueError(f"{spec.name}: no batched script (GPU evaluation is implemented for DualCubePlace-v1)")
    C0 = spec.load_contract(contract)
    extra = [spec.load_contract(c) for c in spec.extra_contracts]
    behaviour = spec.behaviour(policy_or_script)
    variant = dict(scripted_place="release", hold_without_release="hold", cage_no_contact="cage", no_op="no_op")[behaviour]
    poller = GpuMemPoller(os.getpid()).start()
    t_build = time.time()
    env = make(spec.name, emb, num_envs=num_envs, sim="gpu", render="none", **(env_kwargs or {}))
    u = env.unwrapped
    rec = BatchedRecorder(env)
    rec.unwrapped = u
    t_build = time.time() - t_build
    script = _ref(spec.batched_script)
    rows, ev = [], {c.task_id: [] for c in [C0] + extra}
    perf = []
    for seed in seeds:
        t0 = time.time()
        rec.reset(seed=int(seed))
        rec.timing = dict(substep_query_s=0.0, probe_s=0.0, sim_s=0.0)
        logs = script(_RecEnv(rec), variant=variant)
        n_pol = rec.n_steps()
        t_pol = time.time() - t0
        ex = BatchExecutor(_RecEnv(rec))
        last = rec.last_action
        for i, k in enumerate(ex.keys):                  # keep the policy's last gripper command
            ex.targets[i][:] = last[k].cpu().numpy()
        wins = settle_batched(rec, ex, C0)
        cfg = v31.retreat_cfg(C0)
        aft = aftermath_batched(rec, ex, cfg, wins)
        wall = time.time() - t0
        steps = rec.n_steps() + aft[0]["steps"]
        perf.append(dict(seed=int(seed), num_envs=num_envs, steps=steps, wall_s=wall, policy_wall_s=t_pol,
                         env_steps_per_s=steps * num_envs / wall, sim_only_env_steps_per_s=steps * num_envs / max(rec.timing["sim_s"], 1e-9),
                         probe_s=rec.timing["probe_s"], sim_s=rec.timing["sim_s"]))
        for e in range(num_envs):
            tr = rec.traces[e]
            extra_m = {v31.RETREAT_METRIC: aft[e]["value"]} if aft[e]["value"] is not None else {}
            judged = {C0.task_id: v31.judge_traces(C0, tr, float(u.control_freq), extra_m)}
            for c in extra:
                judged[c.task_id] = v31.judge_traces(c, tr, float(u.control_freq), extra_m if v31.retreat_cfg(c) else {})
            executed = logs[e].get("status") == "ok"
            for c in [C0] + extra:
                j = judged[c.task_id]
                role, expected = role_of(c, behaviour)
                row = dict(task=spec.name, embodiment=emb, sim="gpu", num_envs=num_envs, contract=c.task_id,
                           behaviour=behaviour, role=role, expected=expected, seed=int(seed), env_index=e,
                           plan_status=logs[e].get("status"), executed=executed, arm=logs[e].get("arm"),
                           policy_steps=n_pol, verdict=j["success"], released=j["released"], binding=j["binding_term"],
                           protocol_reject=j["protocol_reject"], terms=j["terms"], margin=j["margin"],
                           window_status=j["info"].get("window_status"),
                           bench_final=j["bench"].get("bench_success_final"), bench_any=j["bench"].get("bench_success_any"),
                           aftermath={k: v for k, v in aft[e].items() if k != "aft"},
                           correct=(None if expected is None or not executed or j["success"] is None
                                    else bool(j["success"]) == expected), tag=tag,
                           log={k: v for k, v in logs[e].items()})
                rows.append(row)
                if executed:
                    ev[c.task_id].append(evidence_row(c, row, "sapien-physx-gpu", experiment, os.path.join(out, "episodes.jsonl")))
            if save_traces:
                _save(os.path.join(out, "traces", f"{behaviour}_s{seed}_e{e}{tag}.npz"), {**tr, **aft[e]["aft"]},
                      meta=_jsonable(dict(task=spec.name, embodiment=emb, behaviour=behaviour, seed=int(seed), env=e,
                                          contract=C0.task_id, control_freq=float(u.control_freq),
                                          extra_measurements=extra_m, sim="gpu", num_envs=num_envs)))
        if verbose:
            acc = sum(1 for r in rows if r["seed"] == int(seed) and r["contract"] == C0.task_id and r["verdict"])
            exe = sum(1 for r in rows if r["seed"] == int(seed) and r["contract"] == C0.task_id and r["executed"])
            print(json.dumps(dict(task=spec.name, embodiment=emb, behaviour=behaviour, seed=int(seed), num_envs=num_envs,
                                  executed=exe, accepted_v31=acc, perf=perf[-1])), flush=True)
    peak = poller.stop()
    _write_rows(os.path.join(out, "episodes.jsonl"), rows)
    with open(os.path.join(out, "perf.jsonl"), "a") as f:
        for p in perf:
            f.write(json.dumps(dict(p, behaviour=behaviour, embodiment=emb, gpu_peak_mib=peak, build_s=t_build,
                                    sim_freq=float(u.sim_freq), tag=tag)) + "\n")
    cards = {}
    for c in [C0] + extra:
        p = os.path.join(out, f"evidence_{c.task_id}.jsonl")
        E.save(p, (E.load(p) if os.path.exists(p) else []) + ev[c.task_id])
        if card:
            cards[c.task_id] = write_card(c, p, os.path.join(out, f"card_{c.task_id}.md"))["level"]
    env.close()
    return dict(rows=rows, out_dir=out, cards=cards, perf=perf, gpu_peak_mib=peak)
