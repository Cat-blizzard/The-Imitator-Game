"""Convenience alias for native video ACT through ``imitator.eval.run``.

Example: python -m examples.baselines.act.eval_act_native_video --checkpoint CKPT
  --human-manifest human.json --env-id TwoRobotPickPillToRegions-v1
  --seeds 7000 --output-dir outputs/video --device cpu

There is one public evaluator. This module only supplies the ACT policy factory
and forwards native evaluation options; it does not implement another rollout.
"""
from __future__ import annotations

import argparse
import json
from typing import Sequence


def evaluation_argv(argv: Sequence[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--human-manifest")
    parser.add_argument("--human-role", default="evaluation")
    parser.add_argument("--human-demo-id")
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--video-backend", choices=("torchcodec", "opencv"))
    options, rest = parser.parse_known_args(argv)
    kwargs = {key: value for key, value in vars(options).items() if value is not None}
    return ["--native", "--policy", "examples.baselines.act.native_video_policy:make_policy",
            "--policy-kwargs", json.dumps(kwargs), *rest]


def main(argv: Sequence[str] | None = None):
    from imitator.eval.run import main as evaluate
    return evaluate(evaluation_argv(argv))


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--self-test"]:
        args = evaluation_argv(["--human-role", "development", "--checkpoint", "policy.pt"])
        assert "--native" in args and args[-2:] == ["--checkpoint", "policy.pt"]
        assert json.loads(args[args.index("--policy-kwargs") + 1])["human_role"] == "development"
        print({"shared_evaluator_forwarding": True})
    else:
        main()
