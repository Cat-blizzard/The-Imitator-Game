"""Episode output, close to LeRobot conventions (a converter to LeRobotDataset can come later).

<root>/
  meta/info.json                 dataset-level info (fps, keys, contract files, code version) - written by finalize()
  meta/episodes.jsonl            one row per written episode: index, file, length, verdict, margins, perturbation
                                 spec, corrector layer, provenance
  data/chunk-<shard>/episode_<index>.npz          certified episodes
  rejected/chunk-<shard>/episode_<index>.npz      optional: false recoveries (bench pass, contract fail) kept as
                                                  hard negatives for judges / reward models
npz keys (T = control steps of the whole episode, 20 Hz):
  action (T, A)  observation.state (T, Q) robot joint positions   obj.<name> (T, 7) pose [x y z qw qx qy qz]
  segment (T,) 0 prefix / 1 perturbation / 2 correction / 3 settle   stage (T,) index into stage_names
  trace.<signal> (T,) every TraceRecorder probe incl. bench_success  frame_index, timestamp
  observation.image (T, H, W, 3) uint8 only if renders were enabled (off by default)
Parallel workers write disjoint shards (shard = worker id) and disjoint index ranges (worker * 1_000_000 + n).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np


class EpisodeWriter:
    def __init__(self, root: str, shard: int = 0, keep=("certified", "contract_only"), keep_rejected=("false_recovery",),
                 fps: int = 20):
        self.root, self.shard, self.keep, self.keep_rejected, self.fps = root, shard, set(keep), set(keep_rejected), fps
        self.n = 0
        os.makedirs(f"{root}/meta", exist_ok=True)
        self.index_path = f"{root}/meta/episodes_shard{shard:03d}.jsonl"

    def write(self, env, snap, pert, traj, verdict, row) -> str | None:
        cat = verdict.category if verdict is not None else None
        sub = "data" if cat in self.keep else ("rejected" if cat in self.keep_rejected else None)
        if sub is None:
            return None
        idx = self.shard * 1_000_000 + self.n; self.n += 1
        d = f"{self.root}/{sub}/chunk-{self.shard:03d}"; os.makedirs(d, exist_ok=True)
        path = f"{d}/episode_{idx:07d}.npz"
        arr = env.episode_arrays()
        T = len(arr["action"])
        arr.update(frame_index=np.arange(T), timestamp=np.arange(T, dtype=np.float32) / self.fps,
                   episode_index=np.full(T, idx))
        np.savez_compressed(path, **arr)
        meta = dict(episode_index=idx, file=os.path.relpath(path, self.root), split=sub, length=T, task=snap.task,
                    seed=snap.seed, source=snap.source, snapshot=snap.id, snap_stage=snap.stage, snap_step=snap.step,
                    perturbation=pert.spec(), corrector=traj.corrector, layer=traj.layer,
                    correction_steps=[traj.start_step, traj.end_step], correction_stages=traj.stages,
                    verdict=verdict.to_json())
        with open(self.index_path, "a") as f:
            f.write(json.dumps(meta) + "\n")
        return meta["file"]


def finalize(root: str, info: dict) -> dict:
    """Merge shard indices into meta/episodes.jsonl and write meta/info.json."""
    rows = []
    for p in sorted(glob.glob(f"{root}/meta/episodes_shard*.jsonl")):
        rows += [json.loads(l) for l in open(p) if l.strip()]
    rows.sort(key=lambda r: r["episode_index"])
    with open(f"{root}/meta/episodes.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    info = dict(info, total_episodes=sum(r["split"] == "data" for r in rows),
                total_rejected=sum(r["split"] == "rejected" for r in rows),
                total_frames=sum(r["length"] for r in rows if r["split"] == "data"))
    json.dump(info, open(f"{root}/meta/info.json", "w"), indent=1)
    return info
