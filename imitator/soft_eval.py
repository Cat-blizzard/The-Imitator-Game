"""Evaluation of the cloth tasks (DualFoldTowel-v1, FoldTowel-v1) for imitator.evaluate: N parallel worlds per batch,
scripted behaviour per world, evaluator hold-A settle, protocol-v3 verdict per world from per-step traces.

ClothTraceRecorder keeps one row of per-world signals per control step (every tensor in env.cloth_signals(), plus
towel_footprint_ratio from veritask's scipy convex hull, the benchmark-style flag and the phase), not the vertices
(except at reset and at the end, for figures). Verdicts: veritask.adapters.protocol_v3.evaluate_traces_v3 on each world's
traces (window located from the traces alone, so they are recomputable from the saved .npz).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

PHASE = "phase"
BENCH = "bench_success"


class ClothTraceRecorder:
    def __init__(self, n: int, control_freq: float):
        self.n, self.hz = n, float(control_freq)
        self.rows: list[dict] = []
        self.v_init = None
        self.v_end = None

    def set_init(self, q):
        self.v_init = q.detach().cpu().numpy().astype(np.float64)

    def record(self, sig: dict, q, bench, phase: int):
        from veritask.outcomes.metrics import footprint_ratio
        row = {}
        for k, v in sig.items():
            if hasattr(v, "detach"):
                row[k] = v.detach().float().cpu().numpy().astype(np.float64).reshape(-1)
        x = q.detach().cpu().numpy().astype(np.float64)
        fr = np.full(self.n, np.inf)
        for w in range(self.n):
            if np.isfinite(x[w]).all():
                try:
                    fr[w] = footprint_ratio(x[w], self.v_init[w])
                except Exception:
                    pass
        row["towel_footprint_ratio"] = fr
        row[BENCH] = np.asarray(bench.detach().cpu().numpy() if hasattr(bench, "detach") else bench, float).reshape(-1)
        row[PHASE] = np.full(self.n, float(phase))
        self.rows.append(row)
        self.v_end = x

    def traces(self, w: int) -> dict:
        keys = self.rows[0].keys()
        return {k: np.array([r[k][w] for r in self.rows]) for k in keys}

    @property
    def n_policy(self) -> int:
        return int(sum(1 for r in self.rows if r[PHASE][0] < 0.5))


def _settle(env, rec, contract, max_extra: int | None = None):
    from veritask.adapters.protocol_v3 import SettleConfig, find_window
    u = env.unwrapped
    cfg = SettleConfig.from_contract(contract, "A")
    K, W, N = cfg.steps(u.control_freq)
    n_pol = rec.n_policy
    u.hold_robot()
    for k in range(max_extra or N):
        _, _, _, _, info = env.step(None)
        rec.record(u.cloth_signals(), u.cloth.particle_q(), info["success"], 1)
        if k >= K + W - 1 and k % 5 == 0:
            st = [find_window(rec.traces(w), cfg, n_pol, u.control_freq)["status"] for w in range(rec.n)]
            if all(s in ("ok", "not_quiescent") for s in st):
                break


def run_fold_batch(env, behaviours: list[str], contract=None, frames_every: int = 0, camera=None) -> dict:
    """DualFoldTowel-v1: plan + run one behaviour per world, settle; returns dict(rec, plan_info, frames, seconds)"""
    from .planning.fold import plan_batch
    u = env.unwrapped
    n = u.num_envs
    assert len(behaviours) == n
    t0 = time.time()
    Q, G, info = plan_batch(u, behaviours)
    P = info.pop("_phases", None)            # E29b: per-step phase codes per arm (planning/fold.py)
    t_plan = time.time() - t0
    keys = list(u.single_action_space.spaces.keys()) if hasattr(u, "single_action_space") else list(u.action_space.spaces.keys())
    rec = ClothTraceRecorder(n, u.control_freq)
    rec.set_init(u.cloth.particle_q())
    frames = []
    dev = u.device
    t1 = time.time()
    from .data.phase import CONTACT, FREE, set_phase
    for k in range(len(Q[0])):
        if P is not None:                    # phase of world 0's behaviour; every world's codes on the env
            for i in (0, 1):
                set_phase(env, CONTACT if P[i][k, 0] else FREE, arm=i)
            u._imsim_phase_worlds = [P[i][k] for i in (0, 1)]
        act = {keys[i]: torch.as_tensor(np.concatenate([Q[i][k], G[i][k]], -1), device=dev) for i in (0, 1)}
        if n == 1:
            act = {kk: v[0] for kk, v in act.items()}
        _, _, _, _, inf = env.step(act)
        rec.record(u.cloth_signals(), u.cloth.particle_q(), inf["success"], 0)
        if frames_every and k % frames_every == 0 and camera is not None:
            frames.append((k, camera()))
    t_pol = time.time() - t1
    if contract is not None:
        _settle(env, rec, contract)
    if frames_every and camera is not None:
        frames.append((len(rec.rows), camera()))
    return dict(rec=rec, plan_info=info, frames=frames, seconds=dict(plan=t_plan, policy=t_pol, total=time.time() - t0),
                policy_steps=len(Q[0]))


PATCH_M = 0.05      # cloth aftermath: particles within 5 cm of a retracting gripper are the ones it could pull


def cloth_aftermath(env, cfg: dict) -> dict:
    """E29b retreat aftermath for the dual cloth tasks (all worlds at once, after the settle): in every world, every arm
    whose gripper (any gripper-link collision point, imitator.verify.v32.ImsimView) is within cfg['near_m'] of a cloth
    particle backs its TCP off by cfg['retreat_m'] along -z_tcp over cfg['retreat_s'] (pinocchio CLIK, no collision
    check) with its last gripper command, then everything holds cfg['hold_s']. Returns per world the largest displacement
    of the particles within PATCH_M of a near arm's gripper at the start of the aftermath (0 when no arm is near: nothing
    is retracted, and the rest of the towel's slow creep is not a consequence of the robot; None for a world where a near
    arm's retreat could not be planned), the per-step value (trace 'aft:cloth_disp_max'), and which arms were near."""
    import sapien
    from scipy.spatial import cKDTree
    from transforms3d.quaternions import mat2quat
    from .planning.mplib_planner import ArmPlanner
    from .verify import v32
    u = env.unwrapped
    n = u.num_envs
    dt = float(u.control_timestep)
    vs = v32.views(u)
    bases = u.dual.mount_poses()
    q0 = u.cloth.particle_q().clone()
    Q0 = q0.detach().cpu().numpy()
    near = [[] for _ in range(n)]
    failed = [False] * n
    patch = np.zeros(Q0.shape[:2], bool)                 # (n, V) particles near a retracting gripper
    n_ret = max(2, int(round(cfg["retreat_s"] / dt)))
    paths = {}
    for i, v in enumerate(vs):
        planner = None
        for w in range(n):
            G = v.gripper_points(env_idx=w)
            if not len(G) or not np.isfinite(Q0[w]).all():
                continue
            if float(cKDTree(Q0[w]).query(G)[0].min()) > cfg["near_m"]:
                continue
            near[w].append(i)
            patch[w] |= cKDTree(G).query(Q0[w])[0] <= PATCH_M
            if planner is None:
                planner = ArmPlanner(v.ag, bases[i], control_dt=dt, table=False, env_idx=w)
            planner.env_idx = w
            T = v.tcp_T(env_idx=w)
            goal = sapien.Pose(T[:3, 3] - T[:3, 2] * cfg["retreat_m"], mat2quat(T[:3, :3]))
            res = planner.cartesian(goal, speed=cfg["retreat_m"] / cfg["retreat_s"], min_steps=n_ret)
            if res["status"] != "Success":
                failed[w] = True
            else:
                paths[(w, i)] = res["position"]
    n_steps = max([len(p) for p in paths.values()] + [n_ret]) + int(round(cfg["hold_s"] / dt))
    disp = np.zeros(n)
    per_step = []
    mask = torch.as_tensor(patch, device=q0.device)
    for k in range(n_steps):
        for i, ag in enumerate(u.agent.agents):
            arm = ag.controller.controllers["arm"]
            tq = arm._target_qpos.clone()
            changed = False
            for (w, ii), P in paths.items():
                if ii == i:
                    tq[w] = torch.as_tensor(P[min(k, len(P) - 1)], dtype=tq.dtype, device=tq.device)
                    changed = True
            if changed:
                arm._target_qpos = tq
                arm.set_drive_targets(tq)
        if u.gpu_sim_enabled:
            u.scene.px.gpu_apply_articulation_target_position()
        env.step(None)
        d = (torch.linalg.norm((u.cloth.particle_q() - q0).float(), dim=-1) * mask).amax(1)
        d = torch.nan_to_num(d, nan=1e3).detach().cpu().numpy().astype(float)
        per_step.append(d)
        disp = np.maximum(disp, d)
    per_step = np.asarray(per_step)
    return dict(value=[None if failed[w] else float(disp[w]) for w in range(n)], near=near, failed=failed,
                steps=n_steps, per_step=per_step)


def judge_world(contract, tr: dict, hz: float, extra: dict | None = None) -> dict:
    from veritask.adapters.protocol_v3 import evaluate_traces_v3
    v, bench, info = evaluate_traces_v3(contract, tr, control_freq=hz, extra_measurements=dict(extra or {}))
    return dict(success=v["success"], released=v.get("released"), binding_term=v.get("binding_term"),
                terms={t["metric"]: t["value"] for t in v["terms"]},
                margin=min((t["norm_margin"] for t in v["terms"] if t.get("norm_margin") is not None), default=None),
                bench=bench, info=info, protocol_reject=v.get("protocol_reject"))


def evaluate_soft(spec, emb, policy_or_script, seeds, contract=None, num_envs: int = 1, out: str = "", card=False,
                  experiment: str = "E23", verbose: bool = True, env_kwargs=None, tag: str = "") -> dict:
    from .api import make
    from .harness import _jsonable, _write_rows, evidence_row, role_of, write_card
    from veritask.adapters.base import save_traces
    from veritask.audit import evidence as E
    if spec.name == "FoldTowel-v1":
        return evaluate_fold_towel_e20(spec, policy_or_script, seeds, contract, num_envs, out, card, experiment, verbose,
                                       tag)
    # dual-arm cloth tasks (DualFoldTowel-v1 and the E29b cloth tasks): batched worlds, planning/fold.py planner
    C = spec.load_contract(contract)
    if policy_or_script == "all":
        BEHAVIOURS = tuple(spec.scripts)
        if num_envs % len(BEHAVIOURS):
            raise ValueError(f"'all' needs num_envs divisible by {len(BEHAVIOURS)}")
        behaviours = [b for b in BEHAVIOURS for _ in range(num_envs // len(BEHAVIOURS))]
    else:
        behaviours = [spec.behaviour(policy_or_script)] * num_envs
    sim = "gpu" if num_envs > 1 else "cpu"
    env = make(spec.name, emb, num_envs=num_envs, sim=sim, render="none", **(env_kwargs or {}))
    u = env.unwrapped
    rows, ev = [], []
    for seed in seeds:
        env.reset(seed=int(seed))
        r = run_fold_batch(env, behaviours, contract=C)
        rec = r["rec"]
        from .verify.v31 import AFT_PREFIX, RETREAT_METRIC, retreat_cfg
        rcfg = retreat_cfg(C)
        aft = cloth_aftermath(env, rcfg) if rcfg is not None else None
        for w in range(num_envs):
            tr = rec.traces(w)
            extra = {}
            if aft is not None and aft["value"][w] is not None:
                extra[RETREAT_METRIC] = aft["value"][w]
            j = judge_world(C, tr, u.control_freq, extra)
            if aft is not None:
                tr = dict(tr)
                tr[AFT_PREFIX + "cloth_disp_max"] = aft["per_step"][:, w]
            b = behaviours[w]
            role, expected = role_of(C, b)
            n_pol = rec.n_policy
            row = dict(task=spec.name, embodiment=emb, sim=sim, contract=C.task_id, behaviour=b, role=role,
                       expected=expected, seed=int(seed), env_index=w, executed=True, plan_status="ok",
                       policy_steps=n_pol, verdict=j["success"], released=j["released"], binding=j["binding_term"],
                       protocol_reject=j["protocol_reject"], terms=j["terms"], margin=j["margin"],
                       window_status=j["info"].get("window_status"), window=j["info"].get("window"),
                       bench_final=j["bench"].get("bench_success_final"), bench_any=j["bench"].get("bench_success_any"),
                       bench_after_settle=j["bench"].get("bench_success_after_settle"),
                       lift_max_L=float(np.max(tr["towel_lift_L"])), lift_max_R=float(np.max(tr["towel_lift_R"])),
                       nonfinite=bool(np.max(tr["cloth_nonfinite"]) > 0), max_speed=float(np.max(tr["cloth_speed_max"])),
                       min_particle_z_mm=1000 * float(np.min(tr["cloth_zmin"])),
                       max_fmax_N=float(np.max(tr["robot_cloth_fmax"])), seconds=r["seconds"], tag=tag,
                       aftermath=None if aft is None else dict(value=aft["value"][w], near=aft["near"][w],
                                                               failed=aft["failed"][w], steps=aft["steps"]),
                       correct=None if expected is None or j["success"] is None else bool(j["success"]) == expected,
                       plan=_jsonable({k: [dict((kk, vv) for kk, vv in a.items() if kk != "R") for a in v]
                                       for k, v in r["plan_info"].items()}) if w == 0 else None)
            rows.append(row)
            ev.append(evidence_row(C, row, "sapien-physx-gpu+newton-vbd" if sim == "gpu" else "sapien-physx-cpu+newton-vbd",
                                   experiment, os.path.join(out, "episodes.jsonl")))
            save_traces(os.path.join(out, "traces", f"{b}_s{seed}_w{w}{tag}.npz"), tr,
                        meta=_jsonable(dict(task=spec.name, embodiment=emb, behaviour=b, seed=int(seed), world=w,
                                            contract=C.task_id, control_freq=float(u.control_freq),
                                            extra_measurements=extra)))
        np.savez_compressed(os.path.join(out, "traces", f"verts_s{seed}{tag}.npz"), init=rec.v_init.astype(np.float32),
                            end=rec.v_end.astype(np.float32), behaviours=np.array(behaviours))
        if verbose:
            from collections import Counter
            acc = Counter((rw["behaviour"], rw["verdict"]) for rw in rows if rw["seed"] == int(seed))
            print(json.dumps(dict(task=spec.name, embodiment=emb, seed=int(seed), verdicts={f"{k[0]}:{k[1]}": v for k, v in acc.items()},
                                  seconds=r["seconds"])), flush=True)
    _write_rows(os.path.join(out, "episodes.jsonl"), rows)
    p = os.path.join(out, f"evidence_{C.task_id}.jsonl")
    old = E.load(p) if os.path.exists(p) else []
    E.save(p, old + ev)
    cards = {}
    if card:
        cards[C.task_id] = write_card(C, p, os.path.join(out, f"card_{C.task_id}.md"))["level"]
    env.close()
    return dict(rows=rows, out_dir=out, cards=cards)


def evaluate_fold_towel_e20(spec, policy_or_script, seeds, contract, num_envs, out, card, experiment, verbose, tag):
    """FoldTowel-v1 (E20 single Panda, route A) through E20's run_contract_episode"""
    from .api import make
    from .envs.fold_towel import run_contract_episode
    from .harness import _write_rows, evidence_row, role_of, write_card
    from veritask.audit import evidence as E
    C = spec.load_contract(contract)
    beh = ["expert", "no_grip", "drop_midway", "drag_on_ground"]
    behaviours = [b for b in beh for _ in range(max(1, num_envs // 4))] if policy_or_script == "all" else \
        [spec.behaviour(policy_or_script)] * num_envs
    env = make(spec.name, "panda_cloth", num_envs=len(behaviours), sim="gpu", render="none")
    rows, ev = [], []
    for seed in seeds:
        res, rec = run_contract_episode(env, behaviours, seed=int(seed))
        for w, r in enumerate(res):
            role, expected = role_of(C, behaviours[w])
            row = dict(task=spec.name, embodiment="panda_cloth", sim="gpu", contract=C.task_id, behaviour=behaviours[w],
                       role=role, expected=expected, seed=int(seed), env_index=w, executed=True, verdict=r["v3_success"],
                       binding=r["v3_binding"], terms=r["v3_terms"], bench_final=None, tag=tag)
            rows.append(row)
            ev.append(evidence_row(C, row, "sapien-physx-gpu+newton-vbd", experiment, os.path.join(out, "episodes.jsonl")))
    _write_rows(os.path.join(out, "episodes.jsonl"), rows)
    p = os.path.join(out, f"evidence_{C.task_id}.jsonl")
    E.save(p, (E.load(p) if os.path.exists(p) else []) + ev)
    if card:
        write_card(C, p, os.path.join(out, f"card_{C.task_id}.md"))
    env.close()
    return dict(rows=rows, out_dir=out, cards={})
