import hashlib
import json
import os
from pathlib import Path

DEFAULT_ROOT = Path('/mnt/sdf/datasets/hypericl_task_expansion_20261015_v1')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bind_gpu(index):
    # Import this before Torch or SAPIEN; renderer enumeration is independent of CUDA.
    from ..hypericl_reference_control.common import bind_gpu as bind
    return bind(index)


def source_pins():
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__).parent / name for name in ('collect.py', 'envs.py', 'assets.py', 'registry.py', 'common.py')]
    paths += [root / p for p in (
        'mani_skill/utils/scene_builder/table/utils.py',
        'mani_skill/utils/scene_builder/table/scene_builder.py',
        'mani_skill/examples/motionplanning/panda/motionplanner.py',
        'mani_skill/assets/robots/panda/panda_v2.urdf',
        'examples/baselines/hypericlzq/eef.py')]
    return {str(p): sha(p) for p in paths}
