"""Generate a paired TIG trajectory with an explicit task, backend and seed.

The public entry defaults to task-default-v1 and normal recording. The legacy
validation entry retains its original defaults. Both use the same executor.
Use --manifest FILE --output-dir DIR --worker-lifetime batch|episode for an
explicit batch; defaults and episode fields use the single-run parameter names.
"""
from __future__ import annotations

import sys

from .validate_curobo_complete import build_parser, main as generate_main


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(argument == '--manifest' or argument.startswith('--manifest=') for argument in arguments):
        from .generation_batch import main as batch_main
        return batch_main(arguments)
    if any(argument in ('--help', '-h') for argument in arguments):
        parser = build_parser()
        parser.epilog = ('Batch: --manifest FILE --output-dir DIR '
            '--worker-lifetime batch|episode. Use --manifest FILE --help for batch options.')
        parser.print_help()
        return 0
    defaults = []
    for flag, value in (('--quality-profile', 'task-default-v1'), ('--recording-mode', 'normal')):
        if not any(argument == flag or argument.startswith(flag+'=') for argument in arguments):
            defaults.extend((flag, value))
    return generate_main(defaults + arguments)


if __name__ == '__main__':
    raise SystemExit(main())
