"""cuRobo planning server for the data engine. Runs ONLY in the separate venv research/envs/curobo (cuRoboV2 0.9.0,
system torch 2.10+cu130); the simulator venvs talk to it through data/curobo_client.py, so SAPIEN and cuRobo never
share a process.

usage: research/envs/curobo/bin/python -B -I -m veritask.data.curobo_server --address <unix socket> [--batch 32]
       (or run this file directly with PYTHONPATH=veritask)
       python /path/to/veritask/data/curobo_server.py --stdio

The optional --stdio native-sync profile serves the original dual-Panda tasks
with the measured robot/scene, independent fingers, held-object geometry and
two persistent approach/transport planners. It uses native_curobo's versioned
JSONL requests; the legacy socket protocol below remains unchanged. Neither
profile imports or steps the simulator. Native candidate acceptance remains
with the shared expert executor, not with this service.

Protocol (multiprocessing.connection, authkey from VERITASK_CUROBO_KEY or b'veritask-e7'):
  {'op': 'ping'}                                   -> {'ok': True, 'gpu_mem_gb': ...}
  {'op': 'plan', 'q_start': [7], 'goal': [7],      -> {'ok', 'positions': [[7] ...] resampled at dt, 'status',
   'world': [{name, dims, pose}], 'path': [[7]..],      'ms' (GPU planning wall time), 'duration_s'}
   (path given -> Cartesian waypoint planning)
   'dt': 0.05, 'vmax': 0.9}
  {'op': 'plan_batch', 'queries': [plan request ...]} -> {'results': [...], 'ms_total', 'ms_per_query'}  (BatchMotionPlanner)
Poses are TCP poses (ManiSkill panda_hand_tcp = panda_hand + 0.1034 m along z) in the ROBOT BASE frame,
quaternions wxyz; world cuboids are in the base frame too. Trajectories are uniformly time-scaled so no joint
exceeds vmax rad/s (mplib in the E1 scripts used 0.9 rad/s), then sampled at the control dt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

TCP_OFFSET = 0.1034


def _tcp_to_hand(goal7):
    from scipy.spatial.transform import Rotation as R
    p, q = np.asarray(goal7[:3], float), np.asarray(goal7[3:], float)
    Rm = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return np.r_[p - Rm[:, 2] * TCP_OFFSET, q]


def _resample(pos, dt_src, dt, vmax):
    pos = np.asarray(pos, float)
    if len(pos) < 2:
        return pos
    v = np.abs(np.diff(pos, axis=0)).max() / dt_src
    scale = max(1.0, v / vmax)
    t_src = np.arange(len(pos)) * dt_src * scale
    t = np.arange(0.0, t_src[-1] + 1e-9, dt)
    if t[-1] < t_src[-1]:
        t = np.r_[t, t_src[-1]]
    return np.stack([np.interp(t, t_src, pos[:, j]) for j in range(pos.shape[1])], 1)


class Server:
    def __init__(self, batch: int = 32):
        import torch
        import curobo._src.runtime as runtime
        runtime.torch_compile = False  # no inductor subprocesses; kernels use NVRTC/Warp + CUDA graphs
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        self.torch = torch
        self.lock = threading.Lock()
        t = time.time()
        self.mp = MotionPlanner(MotionPlannerCfg.create(robot="franka.yml", scene_model="collision_table.yml",
                                                        collision_cache={"cuboid": 8, "mesh": 2}, use_cuda_graph=True))
        self.mp.warmup(enable_graph=True, num_warmup_iterations=3)
        self.batch = batch; self.bmp = None
        self.world_key = {}
        print(f"[curobo_server] planner ready in {time.time() - t:.1f}s; joints {self.mp.joint_names}", flush=True)

    def _batch_planner(self):
        if self.bmp is None:
            from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg
            t = time.time()
            self.bmp = BatchMotionPlanner(MotionPlannerCfg.create(robot="franka.yml", scene_model="collision_table.yml",
                                                                  collision_cache={"cuboid": 8, "mesh": 2},
                                                                  max_batch_size=self.batch, use_cuda_graph=True))
            self.bmp.warmup(enable_graph=False, num_warmup_iterations=2)
            print(f"[curobo_server] batch planner (B={self.batch}) ready in {time.time() - t:.1f}s", flush=True)
        return self.bmp

    def _set_world(self, planner, world, tag):
        from curobo.scene import Cuboid, Scene
        key = hashlib.md5(json.dumps(world, sort_keys=True).encode()).hexdigest()
        if self.world_key.get(tag) == key:
            return
        cubs = [Cuboid(name=c["name"], dims=list(map(float, c["dims"])), pose=list(map(float, c["pose"]))) for c in world]
        if not cubs:  # keep a far-away dummy so the cuboid cache layout stays valid
            cubs = [Cuboid(name="dummy", dims=[0.01, 0.01, 0.01], pose=[5.0, 5.0, -5.0, 1, 0, 0, 0])]
        planner.update_world(Scene(cuboid=cubs))
        if hasattr(planner, "reset_seed"):
            planner.reset_seed()
        self.world_key[tag] = key

    def _goal(self, planner, hand7_list):
        from curobo.types import GoalToolPose
        dev = planner.default_joint_state.position.device
        h = np.asarray(hand7_list, np.float32)  # [B, K, 7]
        pos = self.torch.tensor(h[..., :3], device=dev)[:, :, None, None, :].contiguous()
        quat = self.torch.tensor(h[..., 3:], device=dev)[:, :, None, None, :].contiguous()
        return GoalToolPose(planner.tool_frames, pos, quat)

    def _start(self, planner, qs):
        from curobo.types import JointState
        dev = planner.default_joint_state.position.device
        return JointState.from_position(self.torch.tensor(np.asarray(qs, np.float32), device=dev), joint_names=planner.joint_names)

    def plan(self, req):
        torch = self.torch
        with self.lock:
            self._set_world(self.mp, req.get("world", []), "single")
            hand = _tcp_to_hand(req["goal"])
            start = self._start(self.mp, [req["q_start"]])
            torch.cuda.synchronize(); t = time.time()
            status = "ok"
            try:
                if req.get("path"):
                    # straight-line TCP path given as K waypoints (client interpolates from the measured TCP pose)
                    wps = [_tcp_to_hand(w) for w in req["path"]]
                    res = self.mp.plan_pose_waypoints(self._goal(self.mp, [wps]), start, path_tolerance=(0.01, 0.1),
                                                      final_tolerance=(0.005, 0.05), max_attempts=3)
                else:
                    res = self.mp.plan_pose(self._goal(self.mp, [[hand]]), start, max_attempts=4)
            except Exception as e:  # report, never crash the server
                res, status = None, f"exception:{type(e).__name__}:{str(e)[:160]}"
            torch.cuda.synchronize(); ms = (time.time() - t) * 1000
            ok = res is not None and bool(res.success.any())
            if not ok:
                return dict(ok=False, positions=None, status=status if res is None and status != "ok" else "fail", ms=round(ms, 2))
            tr = res.get_interpolated_plan()
            names = list(tr.joint_names); idx = [names.index(j) for j in self.mp.joint_names]
            pos = tr.position.reshape(-1, len(names))[:, idx].cpu().numpy()
            dt_src = float(tr.dt.reshape(-1)[0])
            out = _resample(pos, dt_src, req.get("dt", 0.05), req.get("vmax", 0.9))
            return dict(ok=True, positions=out.tolist(), status="ok", ms=round(ms, 2), duration_s=round(len(out) * req.get("dt", 0.05), 3),
                        raw_duration_s=round((len(pos) - 1) * dt_src, 3))

    def plan_batch(self, req):
        """Batched pose planning (free-space plan_pose only) for the mplib-vs-cuRobo comparison. All queries in one
        call must share the world. Pads to the allocated batch size."""
        torch = self.torch
        with self.lock:
            bmp = self._batch_planner()
            qs = req["queries"]; B = self.batch
            self._set_world(bmp, qs[0].get("world", []), "batch")
            results, ms_total = [], 0.0
            for i in range(0, len(qs), B):
                chunk = qs[i:i + B]; n = len(chunk)
                pad = chunk + [chunk[-1]] * (B - n)
                goal = self._goal(bmp, [[_tcp_to_hand(q["goal"])] for q in pad])
                start = self._start(bmp, [q["q_start"] for q in pad])
                torch.cuda.synchronize(); t = time.time()
                res = bmp.plan_pose(goal, start, max_attempts=req.get("max_attempts", 4))
                torch.cuda.synchronize(); ms = (time.time() - t) * 1000; ms_total += ms
                succ = res.success.any(dim=-1).reshape(-1).cpu().numpy() if res is not None else np.zeros(B, bool)
                for j in range(n):
                    results.append(dict(ok=bool(succ[j])))
            return dict(results=results, ms_total=round(ms_total, 2), ms_per_query=round(ms_total / max(1, len(qs)), 3))


def serve(address: str, batch: int = 32):
    from multiprocessing.connection import Listener
    srv = Server(batch=batch)
    key = os.environ.get("VERITASK_CUROBO_KEY", "veritask-e7").encode()
    if os.path.exists(address):
        os.remove(address)
    lst = Listener(address, family="AF_UNIX", authkey=key)
    print(f"[curobo_server] listening on {address}", flush=True)

    def handle(conn):
        try:
            while True:
                req = conn.recv()
                op = req.get("op")
                if op == "ping":
                    conn.send(dict(ok=True, gpu_mem_gb=round(srv.torch.cuda.max_memory_reserved() / 1e9, 3)))
                elif op == "plan":
                    conn.send(srv.plan(req))
                elif op == "plan_batch":
                    conn.send(srv.plan_batch(req))
                elif op == "shutdown":
                    conn.send(dict(ok=True)); os._exit(0)
                else:
                    conn.send(dict(ok=False, status=f"unknown op {op}"))
        except EOFError:
            pass
        finally:
            conn.close()
    while True:
        c = lst.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


def main(argv: list[str] | None = None) -> None:
    """Start the legacy socket service or the measured native-Panda profile."""
    ap = argparse.ArgumentParser()
    transport = ap.add_mutually_exclusive_group(required=True)
    transport.add_argument("--address", help="Legacy data-corrector Unix socket")
    transport.add_argument("--stdio", action="store_true",
                           help="Native synchronous Panda JSONL profile; no simulator imports")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args(argv)
    if a.stdio:
        # The optional cuRobo interpreter invokes this file by absolute path.
        # Resolve this checkout without importing the simulator's environment.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from veritask.data.native_curobo import serve_stdio
        serve_stdio()
    else:
        serve(a.address, a.batch)


if __name__ == "__main__":
    main()
