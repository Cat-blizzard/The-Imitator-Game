"""Generate a paired TIG trajectory with an explicit task, backend and seed.

The public entry defaults to task-default-v1 and normal recording. The legacy
validation entry retains its original defaults. Both use the same executor.
"""
from __future__ import annotations

import sys

from .validate_curobo_complete import main as generate_main


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    defaults = []
    for flag, value in (('--quality-profile', 'task-default-v1'), ('--recording-mode', 'normal')):
        if not any(argument == flag or argument.startswith(flag+'=') for argument in arguments):
            defaults.extend((flag, value))
    return generate_main(defaults + arguments)


if __name__ == '__main__':
    raise SystemExit(main())
