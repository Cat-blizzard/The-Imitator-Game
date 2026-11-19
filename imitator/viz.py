"""Keyframe strips for figures and `python -m imitator run --frames`: run a scripted behaviour with rendering and save every
k-th rendered frame (SAPIEN on lavapipe for rigid CPU tasks, the Newton camera for cloth / GPU-without-Vulkan)."""
from __future__ import annotations

import os

import numpy as np


class FrameGrabber:
    """wraps env.step: every `every` steps (and at the first step) keeps env.render() of env 0"""

    def __init__(self, env, every: int = 25):
        self.env, self.every, self.frames, self.k = env, int(every), [], 0
        self.unwrapped = env.unwrapped

    def grab(self):
        img = self.env.render()
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()
        img = np.asarray(img)
        if img.ndim == 4:
            img = img[0]
        self.frames.append((self.k, img.astype(np.uint8)))

    def step(self, a):
        out = self.env.step(a)
        if self.k % self.every == 0:
            self.grab()
        self.k += 1
        return out

    def __getattr__(self, name):
        return getattr(self.env, name)


def strip(frames, n: int = 6, label: str = "", path: str | None = None, dpi: int = 110):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = np.unique(np.linspace(0, len(frames) - 1, n).round().astype(int))
    fig, axs = plt.subplots(1, len(idx), figsize=(2.6 * len(idx), 2.3))
    axs = np.atleast_1d(axs)
    for ax, i in zip(axs, idx):
        k, img = frames[i]
        ax.imshow(img); ax.set_axis_off(); ax.set_title(f"step {k}", fontsize=8)
    if label:
        fig.suptitle(label, fontsize=9)
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return idx


def run_with_frames(task: str, embodiment: str, behaviour: str, seed: int = 0, every: int = 25, render: str | None = None,
                    out_png: str | None = None, n_frames: int = 6, num_envs: int = 1, env_kwargs: dict | None = None):
    """rigid: CPU env + SAPIEN (auto backend); cloth: Newton camera. Returns (frames, log, bench_success_last)"""
    from .api import make
    from .tasks import get_task
    spec = get_task(task)
    if spec.kind == "soft":
        env = make(task, embodiment, num_envs=num_envs, sim="gpu" if num_envs > 1 else "cpu", render="newton",
                   **(env_kwargs or {}))
        env.reset(seed=seed)
        g = FrameGrabber(env, every)
        g.grab()
        from .soft_eval import run_fold_batch
        r = run_fold_batch(g, [spec.behaviour(behaviour)] * num_envs)
        g.grab()
        log = dict(status="ok", steps=r["policy_steps"])
        bench = bool(env.unwrapped.evaluate()["success"][0])
    else:
        env = make(task, embodiment, sim="cpu", render=render, **(env_kwargs or {}))
        opts = spec.reset_options(seed) if spec.reset_options else None
        env.reset(seed=seed, options=opts)
        g = FrameGrabber(env, every)
        g.grab()
        log = spec.script(behaviour)(g)
        g.grab()
        bench = bool(env.unwrapped.evaluate()["success"][0])
    if out_png:
        strip(g.frames, n_frames, f"{task} / {embodiment} / {behaviour} (seed {seed}; bench flag {bench})", out_png)
    env.close()
    return g.frames, log, bench
