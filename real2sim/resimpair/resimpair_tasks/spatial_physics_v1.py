"""Bounded CPU-only execution of new layouts with the existing native worker.

The first three factory scenes and two goals per scene are frozen before IK.
Rejected requests remain in the denominator. Six attempts maximum, one worker,
no GPU rendering, API calls, physical retries or changed acceptance rules.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from resimpair_grasp.common import ROOT, binding, read, resolve, write

VERSION = 'broad_spatial_cpu_physics_six_v1'
GOALS = ('relocate_selected', 'contain_selected')


def prepare(root, factory, controller_parent):
    from resimpair_pickplace.runner import code
    from resimpair_pickplace.cartesian_reachability import bind_selection
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import core
    root, factory, controller_parent = (Path(p).resolve() for p in (root, factory, controller_parent))
    if root.exists():
        raise ValueError('existing_physics_campaign_resume_only')
    parent = read(controller_parent / 'experiment.json')
    if parent['candidate_motion_override'] != 'bounded_continuation_motion_v3':
        raise ValueError('retained_native_candidate_required')
    rows = read(factory / 'scenes.json')['scenes'][:3]
    if len(rows) != 3:
        raise ValueError('three_predeclared_scenes_required')
    write(root / 'selection.json', {'factory': binding(factory / 'protocol.json'), 'rows': rows,
        'goals_per_scene': GOALS, 'selection': 'first three scene IDs, before arm selection or physics',
        'layout_novelty': 'new seeds; source assets previously exposed, not globally heldout assets'})
    write(root / 'budget' / 'limits.json', {'physical_attempt_limit': 6, 'parallel_workers': 1,
        'per_attempt_timeout_seconds': 180, 'api_limit': 0, 'gpu_jobs': 0,
        'parent_physics_protocol': binding(controller_parent / 'protocol.json'),
        'prior_physical_spent_in_parent_lineage': read(controller_parent / 'protocol.json')['maximum_cumulative']})
    write(root / 'experiment.json', {'schema': VERSION, 'budget': str(root / 'budget'),
        'code': code(), 'legacy_code': core.implementation(), 'gpu': None, 'gpu_uuid': None,
        'timeout_seconds': 180, 'api_limit': 0, 'render_backend': 'none',
        'candidate_motion_override': parent['candidate_motion_override'], 'motion_policy': parent['motion_policy'],
        'release_policy': parent['release_policy'], 'contact_policy': parent['contact_policy'],
        'parent_controller': binding(controller_parent / 'experiment.json'), 'default_promoted': False})
    write(root / 'rules.json', read(controller_parent / 'rules.json'))
    write(root / 'protocol.json', {'schema': VERSION, 'driver': binding(__file__),
        'worker': binding(ROOT / 'resimpair_tasks/headless_physics_v1.py'),
        'selection': binding(root / 'selection.json'), 'rules': binding(root / 'rules.json'),
        'budget': binding(root / 'budget' / 'limits.json'), 'full_acceptance_unchanged': True,
        'full_pair_qualified': False, 'rendered_frames': 0, 'layout_resampling_after_physics': False})
    cases = []
    for row in rows:
        scene_dir = factory / 'scenes' / row['scene_id']
        for goal in GOALS:
            identity = row['scene_id'] + '_' + goal
            folder = root / 'inputs' / identity
            case = {'id': identity, 'execution_id': identity, 'eligible': False,
                'partition': row['partition'], 'source_ids': row['source_ids'], 'split_group': row['source_ids'],
                'source_request': row['request'], 'requested_task_id': goal, 'method': 'cudex',
                'source_coverage': 'synthetic_only'}
            if not row['accepted_geometry']:
                case['preflight_reason'] = row['failure']['reason']
            else:
                scene = read(scene_dir / 'compiled_scene.json')
                task = next(t for t in read(scene_dir / 'compiled_tasks.json')['tasks'] if t['requested_task_id'] == goal)
                write(folder / 'role_intent.json', task)
                selected, planned = bind_selection(scene, task, folder)
                case['shared_scene'] = binding(scene_dir / 'compiled_scene.json')
                case['arm_selection'] = binding(folder / 'arm_selection.json')
                case['eligible'] = selected is not None
                if selected is None:
                    case['preflight_reason'] = 'no_single_arm_source_and_destination_grasp'
                else:
                    write(folder / 'scene.json', selected); write(folder / 'task.json', planned)
                    case.update(scene=core.bind(folder / 'scene.json'), task=core.bind(folder / 'task.json'))
            cases.append(case)
            print({'case': identity, 'eligible': case['eligible']}, flush=True)
    write(root / 'cohort.json', {'cases': cases, 'selection': binding(root / 'selection.json'),
        'no_outcome_based_scene_replacement': True})
    write(root / 'frozen_inputs.json', {'cohort': binding(root / 'cohort.json'),
        'experiment': binding(root / 'experiment.json'), 'protocol': binding(root / 'protocol.json')})
    return {'requested': 6, 'endpoint_eligible': sum(c['eligible'] for c in cases), 'physical_attempts': 0}


def run(root):
    from resimpair_pickplace.runner import verify
    root = Path(root).resolve()
    for bound in read(root / 'frozen_inputs.json').values():
        resolve(bound)
    verify(root)
    protocol = read(root / 'protocol.json')
    for name in ('driver', 'worker', 'selection', 'rules', 'budget'):
        resolve(protocol[name])
    limits = read(root / 'budget' / 'limits.json')
    if limits['physical_attempt_limit'] != 6 or limits['api_limit'] != 0 or limits['gpu_jobs'] != 0:
        raise ValueError('frozen_six_CPU_attempt_limit_required')
    if shutil.disk_usage(root).free < 20 * 2**30:
        raise ValueError('insufficient_disk_space')
    with (root / '.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launched = []
        for case in read(root / 'cohort.json')['cases']:
            folder = root / 'executions' / case['id']
            if (folder / 'terminal.json').exists():
                continue
            if (folder / 'reservation.json').exists():
                raise ValueError('unknown_previous_attempt_recover_without_repeat')
            if not case['eligible']:
                write(folder / 'terminal.json', {'case': case['id'], 'physical_attempts': 0,
                    'returncode': 'preflight_rejected', 'reason': case['preflight_reason'], 'api_calls': 0})
                continue
            used = len(list((root / 'budget' / 'reservations').glob('*.json')))
            if used >= limits['physical_attempt_limit']:
                raise ValueError('physical_attempt_budget_exhausted')
            reservation = {'case': case, 'campaign': str(root), 'started_unix': time.time(),
                'gpu_jobs': 0, 'api_calls': 0, 'protocol': binding(root / 'protocol.json')}
            write(root / 'budget' / 'reservations' / f'{used:04d}.json', reservation)
            write(folder / 'reservation.json', reservation)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
                VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.x86_64.json',
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                PYTHONPATH=str(ROOT), RESIMPAIR_HEADLESS_RESERVATION=str(folder / 'reservation.json'))
            started = time.monotonic()
            with (folder / 'worker.log').open('x') as log:
                child = subprocess.Popen([sys.executable, '-u', '-m', 'resimpair_tasks.headless_physics_v1',
                    'worker', '--root', str(root), '--case', case['id']], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                write(folder / 'process.json', {'pid': child.pid, 'started_unix': time.time()})
                try:
                    rc = child.wait(timeout=limits['per_attempt_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); child.wait()
                    rc = 'timeout'
            terminal = {'case': case['id'], 'physical_attempts': 1, 'returncode': rc,
                'elapsed_seconds': time.monotonic() - started, 'gpu_jobs': 0, 'api_calls': 0}
            write(folder / 'terminal.json', terminal); launched.append(terminal)
            print(terminal, flush=True)
            if rc != 0 or not (folder / 'execution_bundle.json').exists():
                break
        write(root / 'run_sessions' / f'{time.time_ns()}.json', {'rows': launched})
    return {'new_physical_attempts': len(launched), 'rows': launched}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--factory', type=Path)
    parser.add_argument('--controller-parent', type=Path)
    args = parser.parse_args()
    print(prepare(args.root, args.factory, args.controller_parent) if args.command == 'prepare' else run(args.root))
