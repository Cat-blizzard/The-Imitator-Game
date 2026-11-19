"""Immutable evidence, explicit runtime state, and portable release snapshots."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

from corl2026_workshop.experiments.task_contract_v3.core import (
    bind, checked, read, replace_state, require, sha, write,
)

import sys
from resimpair_runtime import ROOT as WORKSPACE, RUNS, REGISTRY
DEFAULT_ROOT = RUNS / "default"
PYTHON = Path(sys.executable)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", value),
            "Invalid artifact identifier")
    return value


def write_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == value, "Immutable binary differs: " + str(path))
    else:
        temp = path.with_name(path.name + f".{os.getpid()}.tmp")
        with temp.open("xb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        temp.unlink()


def implementation():
    """Requests bind the code actually imported, including prompt/schema files."""
    root = Path(__file__).parent
    return {p.name: bind(p) for p in sorted(root.glob("*.py"))}


_SOURCE_ACTIVATION = None


def activate():
    global _SOURCE_ACTIVATION
    if _SOURCE_ACTIVATION is not None:
        import sys
        module = sys.modules.get('data_generation_methods')
        if module is not None:
            canonical = Path(_SOURCE_ACTIVATION['source_root']).resolve()
            origins = [Path(p).resolve() for p in getattr(module, '__path__', ())]
            # Namespace packages may list the worktree second. The canonical
            # path must remain first, and every loaded concrete module must
            # still originate from the frozen canonical tree.
            concrete = [Path(m.__file__).resolve() for name, m in sys.modules.items()
                        if name.startswith('data_generation_methods.') and getattr(m, '__file__', None)]
            require(origins and origins[0] == canonical
                    and all(p.is_relative_to(canonical) for p in concrete),
                    'canonical_source_changed_after_activation')
        return dict(_SOURCE_ACTIVATION)
    from corl2026_workshop.experiments.state_anchor_v2.bootstrap import activate as frozen
    _SOURCE_ACTIVATION = frozen()
    return dict(_SOURCE_ACTIVATION)


def verify_implementation(bindings):
    for item in bindings.values():
        checked(item)
