"""python -m imitator {list, run, eval, card, generate, certify, devmodel}

  list                         tasks (behaviours, contracts, embodiments) and embodiments
  run  TASK [-e EMB] [-s SCRIPT] [--seed N] [--render auto|none|sapien_cpu|newton] [--frames PNG]
                               one scripted episode with rendering (keyframe strip) and the benchmark-style flag
  eval TASK [-e EMB] [-s SCRIPT] [--episodes N | --seeds 0-7,9] [--contract NAME] [--sim cpu|gpu] [--num-envs N]
            [--out DIR] [--card]
                               contract verdicts (+ v3 next to v3.1 for the cube task), evidence rows, optional card
  card CONTRACT EVIDENCE.jsonl [--out CARD.md]
                               credibility card (veritask.audit.card) from an evidence ledger
  card DATASET_DIR             dataset card (card.md + card.json) of an imitator.generate dataset
  generate TASK [-e EMB] [--preset mp|dart|dart_dir|phys|dart_hi_phys|dart_dir_nophys|dart_dir_white|dart_dir_freeonly]
           [--spec SPEC.json] [--devmodel DEV.json] [--scale S] [--seeds 0-49] [--workers K] [--out DIR] [--rgb]
           [--keep-rejected] [--no-certify] [--num-envs B] [--sim cpu|gpu] [--contract NAME] [--overwrite]
                               contract-certified dataset (ManiSkill3 trajectory HDF5 + JSON + card)
  certify DATASET TASK [-e EMB] [--contract NAME] [--workers K] [--out PREFIX] [--max-episodes N] [--tol-m M]
                               replay a ManiSkill3 trajectory dataset, judge every episode, report diverging replays
  devmodel TASK [-e EMB] --out DEV.json (--policy module:callable --expert DATASET [--n 20] | --rollouts R.json) [--workers K]
           [--calibrate [--scales 1,0.5,0.25,0.125,0.0625] [--pilot-seeds 900000-900011] [--threshold 0.5]]
                               E26 deviation model of a learner (+ s_t calibration)
"""
from __future__ import annotations

import argparse
import json
import sys


def _seeds(s: str):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m imitator", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("task"); p.add_argument("-e", "--embodiment"); p.add_argument("-s", "--script", default="expert")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--render", default="auto")
    p.add_argument("--frames"); p.add_argument("--every", type=int, default=25)
    p = sub.add_parser("eval")
    p.add_argument("task"); p.add_argument("-e", "--embodiment"); p.add_argument("-s", "--script", default="expert")
    p.add_argument("--episodes", type=int); p.add_argument("--seeds"); p.add_argument("--contract")
    p.add_argument("--sim", default="cpu", choices=["cpu", "gpu"]); p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--out"); p.add_argument("--card", action="store_true"); p.add_argument("--quiet", action="store_true")
    p = sub.add_parser("card")
    p.add_argument("contract"); p.add_argument("evidence", nargs="?"); p.add_argument("--out")
    p = sub.add_parser("generate")
    p.add_argument("task"); p.add_argument("-e", "--embodiment"); p.add_argument("--preset", default="mp")
    p.add_argument("--spec"); p.add_argument("--devmodel"); p.add_argument("--scale", type=float)
    p.add_argument("--seeds", default="0-9"); p.add_argument("--workers", type=int, default=1); p.add_argument("--out")
    p.add_argument("--rgb", action="store_true"); p.add_argument("--keep-rejected", action="store_true")
    p.add_argument("--no-certify", action="store_true"); p.add_argument("--num-envs", type=int)
    p.add_argument("--sim", choices=["cpu", "gpu"]); p.add_argument("--contract"); p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p = sub.add_parser("certify")
    p.add_argument("dataset"); p.add_argument("task"); p.add_argument("-e", "--embodiment"); p.add_argument("--contract")
    p.add_argument("--workers", type=int, default=1); p.add_argument("--out"); p.add_argument("--max-episodes", type=int)
    p.add_argument("--tol-m", type=float, default=0.01); p.add_argument("--quiet", action="store_true")
    p = sub.add_parser("devmodel")
    p.add_argument("task"); p.add_argument("-e", "--embodiment"); p.add_argument("--out", required=True)
    p.add_argument("--policy"); p.add_argument("--expert"); p.add_argument("--rollouts"); p.add_argument("--n", type=int, default=20)
    p.add_argument("--workers", type=int, default=1); p.add_argument("--calibrate", action="store_true")
    p.add_argument("--scales", default="1,0.5,0.25,0.125,0.0625"); p.add_argument("--pilot-seeds", default="900000-900011")
    p.add_argument("--threshold", type=float, default=0.5); p.add_argument("--preset", default="dart_dir")
    p.add_argument("--calib-out", help="directory of the calibration pilot datasets (default: next to --out)")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        from .api import list_embodiments
        from .tasks import TASKS
        d = dict(tasks={k: v.describe() for k, v in TASKS.items()}, embodiments=list_embodiments(detail=True))
        if a.json:
            print(json.dumps(d, indent=1))
        else:
            for k, v in d["tasks"].items():
                print(f"{k:18s} [{v['kind']}] contracts={v['contracts']} behaviours={v['behaviours']}"
                      f"{' (GPU batched)' if v['gpu_batched'] else ''}\n{'':18s} embodiments={v['embodiments']}")
            print()
            for k, v in d["embodiments"].items():
                print(f"{k:20s} {v['label']}  (GPU physics {v['gpu_sim_freq']} Hz)")
        return 0
    if a.cmd == "run":
        from .viz import run_with_frames
        frames, log, bench = run_with_frames(a.task, a.embodiment or __import__("imsim").get_task(a.task).default_embodiment,
                                             a.script, a.seed, a.every, None if a.render == "auto" else a.render, a.frames)
        print(json.dumps(dict(status=log.get("status"), steps=log.get("steps"), bench_success=bench, frames=len(frames),
                              png=a.frames)))
        return 0
    if a.cmd == "eval":
        import imitator
        seeds = _seeds(a.seeds) if a.seeds else list(range(a.episodes or 1))
        r = imitator.evaluate(a.task, a.embodiment, a.script, seeds=seeds, contract=a.contract, sim=a.sim, num_envs=a.num_envs,
                           out_dir=a.out, card=a.card, verbose=not a.quiet)
        from .harness import summarize_rows
        for cid in sorted({x.get("contract") for x in r["rows"] if x.get("contract")}):
            print(cid, json.dumps(summarize_rows(r["rows"], cid)))
        print("out:", r["out_dir"], "cards:", r.get("cards"))
        return 0
    if a.cmd == "card" and a.evidence is None:
        import os
        from .data.card import write_card as _wc
        if not os.path.exists(os.path.join(a.contract, "dataset.json")):
            ap.error("card DATASET_DIR needs a directory written by imitator.generate (dataset.json); "
                     "for a credibility card pass CONTRACT EVIDENCE.jsonl")
        r = _wc(a.contract)
        c = r["card"]
        print(json.dumps(dict(card=r["md"], json=r["json"], attempted=c["counts"]["attempted"],
                              certified=c["counts"]["certified"], rate=round(c["certification_rate"], 3),
                              causes=c["rejection_causes"])))
        return 0
    if a.cmd == "generate":
        import imitator
        from .data.perturb import PerturbSpec, preset
        if a.spec:
            spec = PerturbSpec.from_dict(json.load(open(a.spec)))
            if a.devmodel:
                spec = spec.replace(devmodel=a.devmodel)
            if a.scale is not None:
                spec = spec.replace(scale=a.scale)
        else:
            kw = {}
            if a.devmodel:
                kw["devmodel"] = a.devmodel
            if a.scale is not None:
                kw["scale"] = a.scale
            spec = preset(a.preset, **kw)
        r = imitator.generate(a.task, a.embodiment, seeds=_seeds(a.seeds), perturb=spec, certify=not a.no_certify,
                           out=a.out, workers=a.workers, rgb=a.rgb, keep_rejected=a.keep_rejected, num_envs=a.num_envs,
                           sim=a.sim, contract=a.contract, overwrite=a.overwrite, verbose=not a.quiet)
        print("out:", r["out"], "card:", r.get("card"))
        return 0
    if a.cmd == "certify":
        import imitator
        r = imitator.certify(a.dataset, a.task, contract=a.contract, embodiment=a.embodiment, workers=a.workers, out=a.out,
                          max_episodes=a.max_episodes, tol_m=a.tol_m, verbose=not a.quiet)
        print("report:", r["out"] + ".md")
        return 0
    if a.cmd == "devmodel":
        import imitator
        rollouts = json.load(open(a.rollouts)) if a.rollouts else None
        dm = imitator.fit_deviation_model(a.task, a.embodiment, rollouts=rollouts, policy=a.policy, expert=a.expert,
                                       n=a.n, workers=a.workers, out=a.out)
        print(json.dumps(dm.summary()))
        if a.calibrate:
            r = imitator.calibrate_scale(a.task, a.embodiment, devmodel=a.out, scales=[float(x) for x in a.scales.split(",")],
                                      seeds=_seeds(a.pilot_seeds), threshold=a.threshold, preset=a.preset,
                                      workers=a.workers, out=a.calib_out)
            print(json.dumps(dict(scale=r["scale"], passed=r["passed"], table=r["table"])))
        return 0
    if a.cmd == "card":
        import os
        from .harness import write_card
        from .verify import v31  # noqa: F401
        from veritask.spec.contract import TaskContract
        path = a.contract
        if not os.path.exists(path):
            from .tasks import TASKS
            for t in TASKS.values():
                if path in t.contracts:
                    path = t.contracts[path]; break
        c = TaskContract.load(path)
        out = a.out or os.path.splitext(a.evidence)[0] + "_card.md"
        s = write_card(c, a.evidence, out)
        print(json.dumps(dict(card=out, level=s["level"], reasons=s["level_reasons"]), ensure_ascii=False))
        return 0


if __name__ == "__main__":
    sys.exit(main())
