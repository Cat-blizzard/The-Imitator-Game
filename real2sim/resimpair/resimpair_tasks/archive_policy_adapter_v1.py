"""Explicit archive-policy opt-in for the existing native background worker.

Original NumPy, rendering code, budget, locks, image/physics rules and default
DEFLATE behavior stay unchanged. A new campaign can bind DEFLATE1 before its
first submission. The policy only replaces the renderer module's NumPy view.
"""
import argparse
from contextlib import contextmanager
import fcntl
from pathlib import Path
import time
from unittest.mock import patch

from resimpair_grasp.common import binding, read, resolve, write
from . import closure_background_replay_v1 as retained
from . import segmentation_archive_v1 as archive

VERSION = 'native_background_archive_policy_adapter_v1'
DEFAULT_POLICY = 'deflate6'
POLICIES = ('deflate6', 'deflate1')
POLICY_FILE = 'archive_policy_v1.json'


@contextmanager
def install(policy=DEFAULT_POLICY, receipts=None):
    """One isolated worker; never assign attributes on the global NumPy module."""
    if policy not in POLICIES:
        raise ValueError('explicit_known_archive_policy_required')
    original = retained.render.np
    receipts = [] if receipts is None else receipts

    class RendererNumpy:
        def __getattr__(self, name):
            return getattr(original, name)

        def savez_compressed(self, path, *args, **arrays):
            if args:
                raise ValueError('renderer_archive_requires_named_arrays')
            if Path(path).name != 'segmentation.npz':
                raise ValueError('only_registered_segmentation_archive_is_adapted')
            if Path(path).exists():
                raise FileExistsError(path)
            started = time.perf_counter()
            if policy == 'deflate6':
                result = original.savez_compressed(path, **arrays)
            else:
                archive.write_npz(path, **arrays)
                result = None
            elapsed = time.perf_counter() - started
            receipts.append({'path': binding(path), 'policy': policy,
                             'write_wall_seconds': elapsed,
                             'members_in_order': list(arrays),
                             'arrays': {name: {'dtype': value.dtype.str,
                                              'shape': list(value.shape)}
                                        for name, value in arrays.items()}})
            return result

    with patch.object(retained.render, 'np', RendererNumpy()):
        yield receipts


def prepare(root, policy=DEFAULT_POLICY):
    """Append immutable policy binding before any existing allocation is spent."""
    root = Path(root).resolve()
    protocol, limits = retained.render.verify(root)
    if policy not in POLICIES:
        raise ValueError('explicit_known_archive_policy_required')
    if protocol.get('closure_background_adapter') != binding(retained.__file__):
        raise ValueError('only_bound_closure_background_worker_supported')
    amendment = {'schema': VERSION, 'policy': policy,
        'adapter': binding(__file__), 'writer': binding(archive.__file__),
        'retained_worker': binding(retained.__file__),
        'retained_renderer': binding(retained.render.__file__),
        'protocol': binding(root / 'protocol.json'),
        'budget': binding(root / 'budget/limits.json'),
        'same_allocation_id': limits['allocation_id'], 'extra_allowance': 0,
        'default_policy_unchanged': DEFAULT_POLICY,
        'mutates_global_numpy': False, 'physics_images_cameras_rules_unchanged': True,
        'scope': 'serialization policy only; whole-video timing benefit unmeasured'}
    with (root / '.supervisor.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if list((root / 'budget/reservations').glob('*.json')) or list((root / 'cases').glob('*/reservation.json')):
            raise ValueError('archive_policy_must_precede_first_submission')
        write(root / POLICY_FILE, amendment)
    return amendment


def verify(root):
    root = Path(root).resolve()
    protocol, limits = retained.render.verify(root)
    amendment = read(root / POLICY_FILE)
    expected = {'adapter': binding(__file__), 'writer': binding(archive.__file__),
        'retained_worker': binding(retained.__file__),
        'retained_renderer': binding(retained.render.__file__),
        'protocol': binding(root / 'protocol.json'), 'budget': binding(root / 'budget/limits.json')}
    if (amendment.get('schema') != VERSION or amendment.get('policy') not in POLICIES
            or amendment.get('same_allocation_id') != limits['allocation_id']
            or any(amendment.get(k) != v for k, v in expected.items())
            or protocol.get('closure_background_adapter') != binding(retained.__file__)):
        raise ValueError('archive_policy_binding_mismatch')
    for ref in expected.values():
        resolve(ref)
    return amendment


def worker(root, case_id):
    root = Path(root).resolve(); amendment = verify(root)
    traces = []
    folder = root / 'cases' / case_id
    # The retained worker still checks original reservation, resource ownership,
    # source states and all frozen scene inputs before rendering.
    entry = {'schema': VERSION, 'policy': binding(root / POLICY_FILE),
             'case': case_id, 'adapter': binding(__file__), 'new_physics': 0, 'API_calls': 0}
    write(folder / 'archive_policy_entry.json', entry)
    failure = None
    try:
        with install(amendment['policy'], traces):
            result = retained.worker(root, case_id)
        if len(traces) != 1:
            raise ValueError('successful_renderer_requires_one_recorded_archive_save')
        return result
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        write(folder / 'archive_policy_result.json', {
            'schema': VERSION, 'entry': binding(folder / 'archive_policy_entry.json'),
            'save_calls': traces, 'failure': failure,
            'worker_result': binding(folder / 'worker_result.json') if (folder / 'worker_result.json').exists() else None,
            'execution_bundle': binding(folder / 'execution_bundle.json') if (folder / 'execution_bundle.json').exists() else None,
            'global_numpy_mutated': False, 'serializer_only': True})


@contextmanager
def dispatch():
    original = retained.render.subprocess

    class RendererSubprocess:
        def __getattr__(self, name):
            return getattr(original, name)

        def Popen(self, command, *args, **kwargs):
            command = list(command)
            if command[1:4] != ['-m', 'resimpair_tasks.scene_quality_render_v3', 'worker']:
                raise ValueError('unexpected_renderer_worker_command')
            command[2] = 'resimpair_tasks.archive_policy_adapter_v1'
            return original.Popen(command, *args, **kwargs)

    with patch.object(retained.render, 'subprocess', RendererSubprocess()):
        yield


def run(root, stage, limit, gate=None):
    verify(root)
    with dispatch():
        # Original supervisor owns every GPU reservation, lock, unknown-submit
        # recovery check, timeout, allowance and independent video screen gate.
        return retained.batch.run(root, stage=stage, limit=limit, gate=gate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare-policy'); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--archive-policy', choices=POLICIES, default=DEFAULT_POLICY)
    p = sub.add_parser('worker'); p.add_argument('--root', type=Path, required=True); p.add_argument('--case', required=True)
    p = sub.add_parser('run'); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=('screen', 'video'), required=True)
    p.add_argument('--limit', type=int, required=True); p.add_argument('--gate', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare-policy': print(prepare(args.root, args.archive_policy))
    elif args.command == 'worker': worker(args.root, args.case)
    else: print(run(args.root, args.stage, args.limit, args.gate))
