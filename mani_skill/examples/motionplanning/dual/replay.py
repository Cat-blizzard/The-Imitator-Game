"""Replay original TIG actions from their recorded initial state and policy.

Use --input-dir and --output-dir. Public defaults are cross-version replay and
normal output. --replay-mode strict explicitly requires matching runtime source.
Source archive integrity, action compatibility and the recorded quality policy
remain validated in either mode. The legacy validation entry keeps its defaults.
"""
from __future__ import annotations

import json
import sys

from .task_presets import replay_arguments, run_public, self_test


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == ['--self-test']:
        from .validate_curobo_complete_replay import self_test as runtime_self_test
        print(json.dumps({'status': 'passed', 'public_entry': self_test(),
                          'runtime': runtime_self_test()}, allow_nan=False))
        return 0
    if any(argument in ('--help', '-h') for argument in arguments):
        print('Replay recorded dual-arm actions:\n'
            '  python -m mani_skill.examples.motionplanning.dual.replay \\\n'
            '    --input-dir RECORDING --output-dir NEW_DIRECTORY\n\n'
            'Defaults: --replay-mode cross-version --recording-mode normal.\n'
            'Archive integrity and action compatibility remain checked. Source quality policy is retained.\n'
            'Require the same runtime source: --replay-mode strict.\n'
            'Detailed recording/video: --recording-mode diagnostic.\n'
            'Optional task verification: --task TASK. CPU wrapper sanity check: --self-test.')
        return 0
    from .validate_curobo_complete_replay import main as replay_main
    return run_public(replay_arguments(arguments), replay_main)


if __name__ == '__main__':
    raise SystemExit(main())
