"""Finite CPU comparison using independently frozen layouts and old acceptance."""
import argparse
from contextlib import nullcontext
import copy
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from resimpair_grasp.common import ROOT, binding, read, resolve, write

VERSION = 'whole_path_comparison_cpu_v1'


def prepare(root, independent):
    from resimpair_pickplace.runner import code
    from resimpair_pickplace.cartesian_reachability import bind_selection
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import core
    from resimpair_pickplace.whole_path_motion_v1 import VERSION as CANDIDATE
    root, independent = Path(root).resolve(), Path(independent).resolve()
    if root.exists():
        raise ValueError('preserve_existing_campaign')
    protocol = read(independent)
    for ref in protocol['frozen_evaluation_dependencies'].values():
        resolve(ref)
    write(root / 'budget/limits.json', {'physical_attempt_limit': protocol['new_physical_attempt_limit'],
        'allocation': protocol['allocation'], 'parallel_workers': 1, 'api_limit': 0, 'gpu_jobs': 0,
        'per_attempt_timeout_seconds': protocol['per_attempt_timeout_seconds'],
        'prior_physics_lineage': protocol['prior_physics_lineage'],
        'independent_protocol': binding(independent)})
    rows = read(resolve(protocol['cases']))['cases']; manifests = []
    for arm, override in [('candidate', CANDIDATE), ('baseline', protocol['baseline_controller'])]:
        out = root / arm
        write(out / 'experiment.json', {'schema': VERSION, 'budget': str(root / 'budget'),
            'code': code(), 'legacy_code': core.implementation(), 'gpu': None, 'gpu_uuid': None,
            'timeout_seconds': protocol['per_attempt_timeout_seconds'], 'api_limit': 0, 'render_backend': 'none',
            'candidate_motion_override': override, 'motion_policy': protocol['baseline_motion_policy'],
            'release_policy': protocol['baseline_release_policy'], 'contact_policy': protocol['baseline_contact_policy'],
            'independent_protocol': binding(independent), 'default_promoted': False})
        write(out / 'rules.json', read(resolve(protocol['original_rules'])))
        write(out / 'protocol.json', {'schema': VERSION, 'driver': binding(__file__),
            'independent_protocol': binding(independent), 'arm': arm,
            'shared_budget': binding(root / 'budget/limits.json'), 'api_calls': 0, 'gpu_jobs': 0})
        cases = []
        for row in rows:
            if arm == 'baseline' and row['experiment_role'] == 'development_regression':
                manifests.append({'case_id': row['case_id'], 'arm': 'historical_baseline',
                                  'directory': row['historical_baseline_execution'], 'case': row['historical_baseline_case']})
                continue
            folder = out / 'inputs' / row['case_id']
            scene = read(resolve(row['compiled_scene']))
            task = next(t for t in read(resolve(row['compiled_tasks']))['tasks'] if t['requested_task_id'] == row['requested_task_id'])
            selected, assigned = bind_selection(scene, task, folder)
            case = {'id': row['case_id'], 'execution_id': row['case_id'], 'eligible': selected is not None,
                'partition': row['dataset_partition'], 'experiment_role': row['experiment_role'],
                'source_ids': row['source_ids'], 'split_group': row['source_ids'], 'source_request': row['source_request'],
                'requested_task_id': row['requested_task_id'], 'method': 'cudex', 'source_coverage': 'synthetic_only',
                'arm_selection': binding(folder / 'arm_selection.json')}
            if selected is not None:
                # Candidate searches the original cache and both arms. The
                # endpoint-assigned task is only its pre-action fallback intent.
                write(folder / 'scene.json', scene if arm == 'candidate' else selected)
                write(folder / 'task.json', assigned)
                case.update(scene=binding(folder / 'scene.json'), task=binding(folder / 'task.json'))
            else:
                case['preflight_reason'] = 'no_single_arm_source_and_destination_grasp'
            write(folder / 'case.json', case); cases.append(case)
            manifests.append({'case_id': row['case_id'], 'arm': arm,
                              'directory': str(out / 'executions' / case['id']), 'case': binding(folder / 'case.json')})
        write(out / 'cohort.json', {'cases': cases, 'outcome_based_replacement': False})
        write(out / 'frozen_inputs.json', {'experiment': binding(out / 'experiment.json'),
              'cohort': binding(out / 'cohort.json'), 'protocol': binding(out / 'protocol.json')})
    write(root / 'comparison_manifest.json', {'rows': manifests})
    return {'new_registered': 14, 'historical_baseline_reused': 6}


def worker(root, case_id):
    from resimpair_runtime import configure
    configure()
    from resimpair_pickplace.runner import verify
    from resimpair_pickplace import environment
    from resimpair_pickplace import whole_path_motion_v1 as candidate
    from resimpair_pickplace import continuation_motion_v3 as baseline
    from resimpair_pickplace.stability_motion import solve
    from resimpair_pickplace.seed_compatibility import seed_receipt
    from resimpair_grasp.runner import native_geometry
    from .headless_physics_v1 import no_render_helpers
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import simulation, planning_protocols
    from corl2026_workshop.experiments.autonomous_pipeline_v4.timing import Accumulator, span
    import torch
    import gymnasium as gym
    root = Path(root).resolve(); spec = verify(root); resolve(read(root / 'protocol.json')['driver'])
    case = next(c for c in read(root / 'cohort.json')['cases'] if c['id'] == case_id)
    directory = root / 'executions' / case_id
    if os.environ.get('RESIMPAIR_HEADLESS_RESERVATION') != str(directory / 'reservation.json'):
        raise ValueError('bounded_supervisor_required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_available():
        raise ValueError('CPU_worker_requires_disabled_CUDA')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    scene, task = read(resolve(case['scene'])), read(resolve(case['task']))
    reset_seed = seed_receipt(scene['seed']); write(directory / 'simulation_seed.json', reset_seed)
    class StateRecorder(simulation.Recorder):
        def reset(self, seed):
            self.obs, _ = self.env.reset(seed=seed, options={'reconfigure': True})
            self.hold(25); self.initial = self.measure()
            self.initial_physics = simulation.array(self.unwrapped.get_state()).copy()
            self.initial_physics_sha = hashlib.sha256(np.ascontiguousarray(self.initial_physics).tobytes()).hexdigest()
            self.physics_rows = []; self.recording = True
            self.motion_planning_agent_participation = False
        def capture(self):
            self.rows.append(self.measure()); self.stages.append(self.stage)
            self.physics_rows.append(simulation.array(self.unwrapped.get_state()).copy())
    performance = Accumulator(); env = rec = None; failure = bundle = None
    is_candidate = spec['candidate_motion_override'] == candidate.VERSION
    def declare():
        effective = copy.deepcopy(rec.scene_spec) if rec else scene
        write(directory / 'effective_scene.json', effective); write(directory / 'effective_task.json', task)
        ids = [o['id'] for o in effective['objects']]
        supports = ids + [s['id'] for s in effective['supports']] + ['table']
        write(directory / 'event_protocol.json', planning_protocols.pick_place(effective, task, ids, supports, simulation.targets(effective, task)))
        write(directory / 'planning_declaration.json', {'event_protocol': binding(directory / 'event_protocol.json'),
              'case': case, 'effective_scene': binding(directory / 'effective_scene.json'),
              'effective_task': binding(directory / 'effective_task.json'), 'scope': 'before_physical_execution',
              'declared_unix': time.time(), 'implementation': spec['code'], 'driver': binding(__file__),
              'LLM_calls': 0})
        return effective
    with no_render_helpers():
        try:
            with span(directory, 'simulation_environment_setup'):
                environment.register(scene, performance, contact_policy=spec['contact_policy'])
                env = gym.make(environment.ENV_ID, sim_backend='physx_cpu', render_backend='none',
                    control_mode='pd_joint_pos', obs_mode='state_dict', reward_mode='none', render_mode=None)
            with span(directory, 'simulation_reset_and_recorders'):
                performance.phase = 'reset'; rec = StateRecorder(env, directory); rec.reset(reset_seed['simulator_reset_seed'])
                native_geometry(rec, directory); environment.native_evidence(rec, directory)
            if is_candidate:
                with span(directory, 'complete_path_selection_before_motion'):
                    task = candidate.prepare(rec, task)
            effective_scene = declare()
            with (candidate.install_for_worker(root) if is_candidate else baseline.install_for_worker(root)):
                with span(directory, 'planning_execution_without_rendering'):
                    performance.phase = 'execution'
                    solve(rec, task, 'cached_antipodal', release_policy=spec['release_policy'], motion_policy=spec['motion_policy'])
        except Exception as error:
            import traceback
            failure = {'type': type(error).__name__, 'message': str(error), 'trace': traceback.format_exc()}
            write(directory / 'planning_failure.json', failure)
        finally:
            if rec is not None and rec.recording:
                if not (directory / 'event_protocol.json').exists():
                    effective_scene = declare()
                performance.phase = 'finalize'; rec.finish()
                bundle = simulation.export(rec, root, case, task, {**effective_scene, 'cameras': {}}, directory, failure)
                if bundle:
                    np.savez_compressed(directory / 'deferred_render_states.npz', physics_states=np.stack(rec.physics_rows),
                                        timestamps_seconds=np.arange(len(rec.rows)) / rec.control_freq)
                    write(directory / 'deferred_render_manifest.json', {'schema': VERSION,
                          'original_scene': binding(directory / 'effective_scene.json'),
                          'planned_cameras': effective_scene['cameras'], 'physics_states': binding(directory / 'deferred_render_states.npz'),
                          'physical_execution_count': 1, 'rendered_frames': 0, 'not_eligible_for_R2H_export': True})
            if env is not None: env.close()
            performance.persist(directory / 'timing_breakdown.json')
    write(directory / 'worker_result.json', {'bundle': bundle, 'failure': failure, 'api_calls': 0, 'gpu_jobs': 0, 'full_pair_qualified': False})
    return {'bundle': bool(bundle), 'failure': failure['message'] if failure else None}


def run(root, arm, limit):
    from resimpair_pickplace.runner import verify
    root = Path(root).resolve(); folder = root / arm
    spec = verify(folder); protocol = read(folder / 'protocol.json'); resolve(protocol['driver'])
    limits = read(resolve(protocol['shared_budget']))
    if not 1 <= limit <= limits['physical_attempt_limit']:
        raise ValueError('bounded_launch_count_required')
    for ref in read(folder / 'frozen_inputs.json').values(): resolve(ref)
    budget = root / 'budget'; launched = []
    with (root / '.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for case in read(folder / 'cohort.json')['cases']:
            directory = folder / 'executions' / case['id']
            if (directory / 'terminal.json').exists(): continue
            if len(launched) >= limit: break
            if (directory / 'reservation.json').exists():
                raise ValueError('unknown_attempt_recover_without_repeat')
            if not case['eligible']:
                write(directory / 'terminal.json', {'case': case['id'], 'physical_attempts': 0,
                      'returncode': 'preflight_rejected', 'reason': case['preflight_reason'], 'api_calls': 0})
                continue
            used = len(list((budget / 'reservations').glob('*.json')))
            if used >= limits['physical_attempt_limit']:
                raise ValueError('shared_physical_budget_exhausted')
            reservation = {'case': case, 'campaign': str(folder), 'started_unix': time.time(),
                           'arm': arm, 'gpu_jobs': 0, 'api_calls': 0, 'protocol': binding(folder / 'protocol.json')}
            write(budget / 'reservations' / f'{used:04d}.json', reservation); write(directory / 'reservation.json', reservation)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
                VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.x86_64.json', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='1', PYTHONPATH=str(ROOT), RESIMPAIR_HEADLESS_RESERVATION=str(directory / 'reservation.json'))
            began = time.monotonic()
            with (directory / 'worker.log').open('x') as stream:
                child = subprocess.Popen([sys.executable, '-u', '-m', 'resimpair_tasks.whole_path_campaign_v1', 'worker',
                    '--root', str(folder), '--case', case['id']], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                write(directory / 'process.json', {'pid': child.pid, 'started_unix': time.time()})
                try: rc = child.wait(timeout=limits['per_attempt_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=8)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                    rc = 'timeout'
            terminal = {'case': case['id'], 'physical_attempts': 1, 'returncode': rc,
                        'elapsed_seconds': time.monotonic() - began, 'gpu_jobs': 0, 'api_calls': 0}
            write(directory / 'terminal.json', terminal); launched.append(terminal); print(terminal, flush=True)
            if rc != 0: break
    return {'new_attempts': len(launched)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--independent', type=Path)
    parser.add_argument('--arm', choices=['baseline', 'candidate'])
    parser.add_argument('--case'); parser.add_argument('--limit', type=int, default=1)
    args = parser.parse_args()
    print(prepare(args.root, args.independent) if args.command == 'prepare' else
          run(args.root, args.arm, args.limit) if args.command == 'run' else worker(args.root, args.case))
