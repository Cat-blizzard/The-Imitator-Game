"""imitator.evaluate() (module imitator.harness): run episodes, record the traces the contract needs, run the protocol-v3 hold-A settle (+ the v3.1
retreat aftermath when the contract has it), score the contract and the benchmark-style flag, write evidence rows
(veritask EvidenceRow) and optionally a credibility card.

    res = imitator.evaluate("DualHandover-v1", "dual_ur5_wsg", "handover", seeds=range(8))
    res = imitator.evaluate("DualCubePlace-v1", "dual_piper", "hold", n_episodes=8, card=True)
    res = imitator.evaluate("DualCubePlace-v1", "dual_ur5_wsg", "expert", sim="gpu", num_envs=16)   # batched, GPU PhysX
    res = imitator.evaluate("DualFoldTowel-v1", "dual_arx_x5", "drag_on_ground", num_envs=8)          # cloth (GPU)

policy_or_script
  str          a scripted behaviour of the task (the contract's counterexample names, or an alias such as 'expert')
  callable     a closed-loop policy: action = policy(obs) is stepped up to TaskSpec.max_policy_steps (or `max_steps`)
               or until it returns None; mark a callable with `imitator.script` to receive the (recorded) env instead and
               drive it yourself: script(env) -> dict log
Outputs (out_dir, default $IMITATOR_OUT/eval/<task>/<embodiment>/, IMITATOR_OUT defaults to ./results):
  episodes.jsonl                one row per episode and contract (verdict, terms, margins, bench flags, aftermath, ...)
  evidence_<task_id>.jsonl      veritask EvidenceRow per executed episode (role from the contract's counterexamples)
  traces/<behaviour>_s<seed>.npz  judged traces + aftermath ('aft:' keys); verdicts are recomputable from these files
  card_<task_id>.md             credibility card over all evidence rows in the file (card=True)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import numpy as np

from .tasks import _ref, get_task

from .paths import out_dir

DEFAULT_OUT = out_dir("eval")


def script(fn):
    """decorator: fn(env) drives the recorded env itself (instead of policy(obs) -> action)"""
    fn.imsim_script = True
    return fn


def role_of(contract, behaviour: str) -> tuple[str, bool | None]:
    for c in contract.counterexamples:
        if c.name == behaviour:
            return ("control" if c.expected else "counterexample"), bool(c.expected)
    return "natural", None


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    return repr(x)


def _write_rows(path: str, rows: list[dict]):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_jsonable(r), ensure_ascii=False) + "\n")


def evidence_row(contract, row: dict, engine: str, experiment: str, source: str):
    from veritask.audit.evidence import EvidenceRow
    role, _ = role_of(contract, row["behaviour"])
    return EvidenceRow(contract=contract.task_id, behaviour=row["behaviour"], role=role, verdict=row["verdict"],
                       bench=row.get("bench_final"), margin=row.get("margin"), binding_term=row.get("binding"),
                       engine=engine, seed=row.get("seed"), experiment=experiment, source=source,
                       extra=dict(embodiment=row.get("embodiment"), task=row.get("task"), sim=row.get("sim"),
                                  bench_semantics="benchmark-style info['success'] at the policy's last step",
                                  executed=row.get("executed"), env=row.get("env_index")))


def write_card(contract, evidence_path: str, out_md: str) -> dict:
    from veritask.audit import evidence as E
    from veritask.audit.card import render_markdown, summarize
    rows = E.load(evidence_path)
    s = summarize(contract, rows)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(render_markdown(contract, s))
    return s


# ------------------------------------------------------------------------------------------------ ground truth probes
def cube_lift_probe(rec, log: dict) -> dict | None:
    """E19's lift probe (after the verdict): the acting arm rises 5 cm with its last gripper command; held = cube rose
    >= 3 cm. Ground truth for 'released' (not independent of the v3.1 retreat term, which is the same kind of test)."""
    from .planning import scripted as S
    arm = log.get("arm")
    if arm is None or log.get("status") != "ok":
        return None
    try:
        return S.lift_probe(rec, arm, last_action=rec.last_action)
    except Exception as e:  # noqa: BLE001
        return dict(status=f"error {e!r}"[:200])


# ------------------------------------------------------------------------------------------------ rigid, CPU, 1 env
def _run_policy(rec, policy, max_steps: int) -> dict:
    obs, _ = rec.reset_out if hasattr(rec, "reset_out") else (None, None)
    n = 0
    for _ in range(max_steps):
        a = policy(obs)
        if a is None:
            break
        obs, _, term, trunc, info = rec.step(a)
        n += 1
    return dict(status="ok", steps=n, arm=None)


def evaluate(task: str, embodiment: str | None = None, policy_or_script="expert", n_episodes: int | None = None,
             seeds=None, contract: str | None = None, sim: str = "cpu", num_envs: int = 1, out_dir: str | None = None,
             card: bool = False, extra_contracts=None, ground_truth: bool = True, save_traces: bool = True,
             experiment: str = "E23", max_steps: int | None = None, env_kwargs: dict | None = None,
             verbose: bool = True, tag: str = "") -> dict:
    spec = get_task(task)
    emb = embodiment or spec.default_embodiment
    seeds = list(seeds) if seeds is not None else list(range(n_episodes or 1))
    out = out_dir or os.path.join(DEFAULT_OUT, spec.name, emb)
    os.makedirs(os.path.join(out, "traces"), exist_ok=True)
    if spec.kind == "soft":
        from .soft_eval import evaluate_soft
        return evaluate_soft(spec, emb, policy_or_script, seeds, contract, num_envs=max(num_envs, 1), out=out, card=card,
                             experiment=experiment, verbose=verbose, env_kwargs=env_kwargs, tag=tag)
    if sim == "gpu":
        from .batched import evaluate_batched
        return evaluate_batched(spec, emb, policy_or_script, seeds, contract, num_envs=num_envs, out=out, card=card,
                                experiment=experiment, verbose=verbose, env_kwargs=env_kwargs, save_traces=save_traces,
                                tag=tag)
    from .api import make
    from .verify import v31
    from veritask.adapters.base import save_traces as _save
    C0 = spec.load_contract(contract)
    extra = [spec.load_contract(c) for c in (extra_contracts if extra_contracts is not None else spec.extra_contracts)
             if spec.contract_path(c) != spec.contract_path(contract)]
    contracts = [C0] + extra
    env = make(spec.name, emb, sim="cpu", render="none", obs="state", **(env_kwargs or {}))
    rec = _ref(spec.recorder)(env)
    u = env.unwrapped
    bases = u.dual.mount_poses()
    is_name = isinstance(policy_or_script, str)
    behaviour = spec.behaviour(policy_or_script) if is_name else getattr(policy_or_script, "__name__", "policy")
    run_script = spec.script(policy_or_script) if is_name else None
    derives = {c.task_id: (_ref(spec.derive)(c) if spec.derive else None) for c in contracts}
    results, ev_rows = [], {c.task_id: [] for c in contracts}
    for seed in seeds:
        t0 = time.time()
        try:
            opts = spec.reset_options(seed) if spec.reset_options else None
            rec.reset_out = rec.reset(seed=int(seed), options=opts)
            if run_script is not None:
                log = run_script(rec)
            elif getattr(policy_or_script, "imsim_script", False):
                log = policy_or_script(rec)
            else:
                log = _run_policy(rec, policy_or_script, max_steps or spec.max_policy_steps)
            n_pol = len(rec.traces["phase"])
            obj, half = getattr(u, spec.object_attr), getattr(u, spec.half_attr)
            j0 = v31.judge_rec(rec, C0, obj, half, bases, derive=derives[C0.task_id])
            judged = {C0.task_id: j0}
            for c in extra:
                ex_m = dict(j0["extra"]) if v31.retreat_cfg(c) is not None else {}
                judged[c.task_id] = v31.judge_traces(c, rec.traces, float(u.control_freq), ex_m, derive=derives[c.task_id])
            gt = None
            tr_all = {k: list(v) for k, v in rec.traces.items()}
            if j0.get("_aft_traces"):
                tr_all.update(j0["_aft_traces"])
            if ground_truth and spec.ground_truth:
                gt = _ref(spec.ground_truth)(rec, log)
            executed = log.get("status") == "ok"
            for c in contracts:
                j = judged[c.task_id]
                role, expected = role_of(c, behaviour)
                row = dict(task=spec.name, embodiment=emb, sim="cpu", contract=c.task_id, behaviour=behaviour, role=role,
                           expected=expected, seed=int(seed), plan_status=log.get("status"), executed=executed,
                           arm=log.get("arm"), policy_steps=n_pol, verdict=j["success"], released=j["released"],
                           binding=j["binding_term"], protocol_reject=j["protocol_reject"], terms=j["terms"],
                           margins=j["margins"], margin=j["margin"],
                           window_status=j["info"].get("window_status"), window=j["info"].get("window"),
                           bench_final=j["bench"].get("bench_success_final"), bench_any=j["bench"].get("bench_success_any"),
                           bench_after_settle=j["bench"].get("bench_success_after_settle"),
                           aftermath=j0.get("aftermath"), ground_truth=gt, seconds=round(time.time() - t0, 1),
                           correct=(None if expected is None or not executed or j["success"] is None
                                    else bool(j["success"]) == expected),
                           log={k: v for k, v in log.items() if k not in ("final_targets",)}, tag=tag)
                results.append(row)
                if executed:
                    ev_rows[c.task_id].append(evidence_row(c, row, "sapien-physx-cpu", experiment,
                                                           os.path.join(out, "episodes.jsonl")))
            if save_traces:
                meta = dict(task=spec.name, embodiment=emb, behaviour=behaviour, seed=int(seed), contract=C0.task_id,
                            control_freq=float(u.control_freq), extra_measurements=j0["extra"],
                            aftermath={k: v for k, v in (j0.get("aftermath") or {}).items() if k != "obj_path"})
                _save(os.path.join(out, "traces", f"{behaviour}_s{seed}{tag}.npz"), tr_all, meta=_jsonable(meta))
            if verbose:
                print(json.dumps(_jsonable({k: results[-len(contracts)].get(k) for k in
                                            ("task", "embodiment", "behaviour", "seed", "plan_status", "verdict", "binding",
                                             "bench_final", "seconds")} | {"aftermath": (j0.get("aftermath") or {}).get("value"),
                                                                             "v_extra": [judged[c.task_id]["success"] for c in extra]})),
                      flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results.append(dict(task=spec.name, embodiment=emb, behaviour=behaviour, seed=int(seed), error=repr(e)[:400],
                                tag=tag))
    _write_rows(os.path.join(out, "episodes.jsonl"), results)
    from veritask.audit import evidence as E
    cards = {}
    for c in contracts:
        p = os.path.join(out, f"evidence_{c.task_id}.jsonl")
        old = E.load(p) if os.path.exists(p) else []
        E.save(p, old + ev_rows[c.task_id])
        if card:
            cards[c.task_id] = write_card(c, p, os.path.join(out, f"card_{c.task_id}.md"))["level"]
    env.close()
    return dict(rows=results, out_dir=out, cards=cards)


def summarize_rows(rows: list[dict], contract_id: str | None = None) -> dict:
    """accepted / executed per behaviour (and bench flags) for one contract"""
    out = {}
    for r in rows:
        if "error" in r or (contract_id and r.get("contract") != contract_id):
            continue
        d = out.setdefault(r["behaviour"], dict(n=0, executed=0, accepted=0, unknown=0, bench=0))
        d["n"] += 1
        if r.get("executed"):
            d["executed"] += 1
            if r.get("verdict") is None:
                d["unknown"] += 1
            elif r["verdict"]:
                d["accepted"] += 1
            d["bench"] += int(bool(r.get("bench_final")))
    return out
