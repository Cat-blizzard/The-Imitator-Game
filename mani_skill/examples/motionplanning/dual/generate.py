"""Generate a paired TIG trajectory using the selected task's fixed recipe.

The public entry defaults to task-default-v1 and normal recording. The legacy
validation entry retains its original defaults. Both use the same executor.
Use --manifest FILE --output-dir DIR --worker-lifetime batch|episode for an
explicit batch; defaults and episode fields use the single-run parameter names.
"""
from __future__ import annotations

import json
import sys

from .task_presets import TASK_PRESETS, generation_arguments, has_option, run_public, self_test


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if has_option(arguments, '--manifest'):
        from .generation_batch import main as batch_main
        return batch_main(arguments)
    if arguments == ['--self-test']:
        from .validate_curobo_complete import self_test as runtime_self_test
        print(json.dumps({'status': 'passed', 'public_entry': self_test(),
                          'runtime': runtime_self_test()}, allow_nan=False))
        return 0
    if '--help-all' in arguments:
        from .validate_curobo_complete import build_parser
        parser = build_parser()
        parser.set_defaults(quality_profile='task-default-v1', recording_mode='normal')
        parser.epilog = ('Single runs select missing recipe options by task. Explicit recipe options win. '
            'Manifest batches retain their explicit configuration and do not apply task presets.')
        parser.print_help()
        return 0
    if any(argument in ('--help', '-h') for argument in arguments):
        print('Generate a dual-arm trajectory:\n'
            '  python -m mani_skill.examples.motionplanning.dual.generate \\\n'
            '    --task TASK --backend mplib|curobo --seed SEED --output-dir NEW_DIRECTORY\n\n'
            'Tasks:\n  ' + '\n  '.join(TASK_PRESETS) + '\n\n'
            'Task recipes are selected automatically; explicit advanced options take priority.\n'
            'Default quality: task-default-v1 (retreat and final stability remain selected).\n'
            'Default recording: normal (actions/state/events; no video).\n'
            'Advanced options: --help-all. Diagnostic/video: --recording-mode diagnostic.\n'
            'Batch: --manifest FILE --output-dir DIR --worker-lifetime batch|episode.\n'
            'Use --manifest FILE --help for batch options.')
        return 0
    from .validate_curobo_complete import main as generate_main
    return run_public(generation_arguments(arguments), generate_main)


if __name__ == '__main__':
    raise SystemExit(main())
