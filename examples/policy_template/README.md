# Adding a policy

The benchmark ships no model code. A policy is evaluated through the interface in
[`imitator/eval/policy.py`](../../imitator/eval/policy.py):

| method | called | returns |
|---|---|---|
| `prepare_for_eval(human_video, robot_obs, human_desc)` | once per episode | — |
| `get_action(obs)` | every `pred_horizon` steps (or every step with temporal aggregation) | action chunk `(B, H, 16)` |
| `clear_cache()` | after each batch of episodes | — |

Write a factory `make_policy(checkpoint=None, device=..., **kwargs) -> (policy, model_config)` and run

```bash
python -m imitator.eval.run --policy my_pkg.my_module:make_policy --checkpoint ckpt.pt \
    --eval-config imitator/eval/lists/eval_all.txt --output-dir results/my_policy --input-mode video_only
```

`policy.py` here is a random policy that exercises the whole loop. The full 1.0 baselines (ACT, Diffusion Policy, VQ-BeT,
UniSkill, XSkill, GR00T, RDT, π0.5, OpenVLA-OFT, HyperICL) with training and evaluation are in `examples/baselines/`.
