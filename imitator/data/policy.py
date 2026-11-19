"""Policies for the data engine's learner loop (deviation-model rollouts, imitator.certify-style closed loops).

as_policy(policy, layout) accepts a callable: policy(obs_flat (D,) float32) -> flat action (A,) or the env's dict
action; optional reset(seed=...). Imitator 2.0 ships no model code: wrap your own learner (e.g. a diffusion policy
trained on imitator.generate datasets) in such a callable. (1.0 research versions also accepted a checkpoint path of a
small diffusion policy; that loader was removed with the model code.)
"""
from __future__ import annotations

import numpy as np


def as_policy(policy, layout=None):
    if callable(policy):
        return policy
    if isinstance(policy, str) and ":" in policy:          # "package.module:callable_or_factory"
        import importlib
        mod, attr = policy.split(":", 1)
        obj = getattr(importlib.import_module(mod), attr)
        return obj() if isinstance(obj, type) else obj
    if isinstance(policy, str):
        raise TypeError("checkpoint paths are not supported (no model code in imitator 2.0): pass a callable "
                        "policy(obs_flat) -> action that wraps your own model")
    raise TypeError(f"policy must be a callable, got {type(policy)}")


def rollout(env, layout, policy, seed: int, max_steps: int, options=None, spec=None, ep=None, env_states=None):
    """closed-loop rollout on a CPU env; returns dict(obs, actions, arm_qpos, success) as arrays"""
    from . import store
    if ep is not None:
        obs = store.reset_to_episode(env, spec, ep, env_states)
    else:
        obs = store.flat_obs(env.reset(seed=int(seed), options=options)[0])
    if hasattr(policy, "reset"):
        policy.reset(seed=int(seed))
    u = env.unwrapped
    O, A, Q, S = [obs], [], [layout.arm_qpos(u)], []
    for _ in range(int(max_steps)):
        a = policy(obs)
        act = a if isinstance(a, dict) else layout.unflatten_action(a)
        o, _, _, _, info = env.step(act)
        obs = store.flat_obs(o)
        O.append(obs); A.append(layout.flatten_action(act)); Q.append(layout.arm_qpos(u))
        s = info.get("success", False)
        S.append(bool(s[0]) if hasattr(s, "__len__") else bool(s))
    return dict(obs=np.asarray(O), actions=np.asarray(A), arm_qpos=np.asarray(Q), success=np.asarray(S))
