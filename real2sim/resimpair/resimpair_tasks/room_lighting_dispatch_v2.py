"""Module-local worker dispatch; preserve shared GPU resource-query subprocesses.

Reuses the original prepared light protocol, its exact three-job allocation and
v1 worker. A bound pre-launch amendment records this supervisor-only repair.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from resimpair_grasp.common import binding, read, resolve, write
from . import room_lighting_replay_v1 as retained

VERSION = 'finite_room_light_module_local_dispatch_v2'


@contextmanager
def dispatch():
    original_module = retained.render.subprocess

    class ModuleLocalSubprocess:
        def __getattr__(self, name):
            return getattr(original_module, name)

        def Popen(self, command, *args, **kwargs):
            command = list(command)
            if command[1:4] != ['-m', 'resimpair_tasks.scene_quality_render_v3', 'worker']:
                raise ValueError('unexpected_renderer_child_command')
            command[2] = 'resimpair_tasks.room_lighting_replay_v1'
            return original_module.Popen(command, *args, **kwargs)

    # Replace the render module's reference, never attributes on subprocess.
    # Other modules keep the real subprocess object for nvidia-smi/ffprobe/etc.
    with patch.object(retained.render, 'subprocess', ModuleLocalSubprocess()):
        yield


def prepare(root):
    root = Path(root).resolve(); protocol, limits = retained.render.verify(root)
    if protocol['finite_light_adapter'] != binding(retained.__file__):
        raise ValueError('original_worker_changed')
    reservations = list((root / 'budget/reservations').glob('*.json'))
    if reservations or list((root / 'cases').glob('*/reservation.json')):
        raise ValueError('dispatch_repair_must_precede_first_submission')
    amendment = {'schema': VERSION, 'supervisor': binding(__file__),
        'retained_worker': binding(retained.__file__), 'protocol': binding(root / 'protocol.json'),
        'budget': binding(root / 'budget/limits.json'),
        'reason': 'v1 patched attributes on the global subprocess module; shared GPU nvidia-smi check_output would hit the worker-only command gate',
        'repair': 'replace only scene_quality_render_v3.subprocess reference with a proxy; all resource modules retain the original subprocess module',
        'new_render_submissions_at_amendment': len(reservations),
        'additional_allowance': 0, 'same_allocation_id': limits['allocation_id'],
        'same_maximum_attempts': limits['maximum_attempts'],
        'same_lineage_cap': limits['maximum_lineage_render_jobs_including_history'],
        'lighting_physics_camera_states_thresholds_unchanged': True}
    write(root / 'dispatch_v2.json', amendment)
    return amendment


def run(root, stage, limit, gate=None):
    root = Path(root).resolve(); protocol, limits = retained.render.verify(root)
    amendment = read(root / 'dispatch_v2.json')
    if (amendment['schema'] != VERSION or amendment['supervisor'] != binding(__file__)
            or amendment['retained_worker'] != binding(retained.__file__)
            or amendment['protocol'] != binding(root / 'protocol.json')
            or amendment['budget'] != binding(root / 'budget/limits.json')
            or amendment['same_allocation_id'] != limits['allocation_id']):
        raise ValueError('dispatch_amendment_binding_mismatch')
    for key in ('supervisor', 'retained_worker', 'protocol', 'budget'):
        resolve(amendment[key])
    with dispatch():
        return retained.batch.run(root, stage=stage, limit=limit, gate=gate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--root', type=Path, required=True)
    p = sub.add_parser('run'); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=('screen', 'video'), required=True)
    p.add_argument('--limit', type=int, required=True); p.add_argument('--gate', type=Path)
    args = parser.parse_args()
    print(prepare(args.root) if args.command == 'prepare' else run(args.root, args.stage, args.limit, args.gate))
