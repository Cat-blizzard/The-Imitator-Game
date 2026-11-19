"""CPU-only execution of frozen scene programs under two selection policies.

The physical skill, mesh guards and evaluator remain the retained versions.
Rolling commitments precede each operation; the final protocol aggregates them.
"""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import time
import numpy as np
from resimpair_grasp.common import ROOT, binding, read, resolve, write

VERSION = 'measured_multitask_worker_v3'


def worker(root, case_id):
    from resimpair_runtime import configure
    configure()
    from resimpair_pickplace.runner import verify
    from resimpair_pickplace import environment
    from resimpair_pickplace import whole_path_motion_v1 as candidate
    from resimpair_pickplace import rolling_pickplace_v3 as rolling
    from resimpair_pickplace import native_mesh_access_v1
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
    if spec['candidate_motion_override'] != candidate.VERSION:
        raise ValueError('retained_whole_path_guard_required')
    mode = spec['execution_selection_policy']
    if mode not in {'initial_state_v1', rolling.VERSION}:
        raise ValueError('unknown_selection_policy')
    is_rolling = mode == rolling.VERSION
    def declare():
        effective = copy.deepcopy(rec.scene_spec) if rec else scene
        # Unexecuted operations may have no effector after a failure. These
        # fallback declarations never qualify a failed rollout as successful.
        for operation in task['operations']:
            if operation['arm'] is None:
                item = next(o for o in scene['objects'] if o['id'] == operation['object'])
                operation['arm'] = int(np.argmin([abs(r[1] - item['position'][1]) for r in scene['robot_roots']]))
        write(directory / 'effective_scene.json', effective); write(directory / 'effective_task.json', task)
        ids = [o['id'] for o in effective['objects']]
        supports = ids + [s['id'] for s in effective['supports']] + ['table']
        write(directory / 'event_protocol.json', planning_protocols.pick_place(effective, task, ids, supports, simulation.targets(effective, task)))
        write(directory / 'planning_declaration.json', {'event_protocol': binding(directory / 'event_protocol.json'),
              'case': case, 'effective_scene': binding(directory / 'effective_scene.json'),
              'effective_task': binding(directory / 'effective_task.json'), 'scope': 'aggregation_of_stepwise_pre_action_commitments' if is_rolling else 'before_physical_execution',
              'execution_selection_policy': mode,
              'operation_commitments': [binding(p) for p in sorted((directory / 'operation_commitments').glob('[0-9][0-9].json'))],
              'declared_unix': time.time(), 'implementation': spec['code'], 'driver': binding(__file__),
              'LLM_calls': 0})
        return effective
    with no_render_helpers(), native_mesh_access_v1.install():
        try:
            with span(directory, 'simulation_environment_setup'):
                environment.register(scene, performance, contact_policy=spec['contact_policy'])
                env = gym.make(environment.ENV_ID, sim_backend='physx_cpu', render_backend='none',
                    control_mode='pd_joint_pos', obs_mode='state_dict', reward_mode='none', render_mode=None)
            with span(directory, 'simulation_reset_and_recorders'):
                performance.phase = 'reset'; rec = StateRecorder(env, directory); rec.reset(reset_seed['simulator_reset_seed'])
                native_geometry(rec, directory); environment.native_evidence(rec, directory)
                # Preserve the common initial state even if all path proposals
                # fail before the first physical action and no bundle exists.
                np.savez_compressed(directory / 'initial_reset_state.npz', physics_state=rec.initial_physics,
                                    object_states=rec.initial['object_states'])
                write(directory / 'initial_reset_evidence.json', {'state': binding(directory / 'initial_reset_state.npz'),
                    'physics_sha256': rec.initial_physics_sha, 'reset_seed': binding(directory / 'simulation_seed.json'),
                    'source_scene': case['scene'], 'frame': 0, 'reset_count': 1})
            if not is_rolling:
                with span(directory, 'complete_path_selection_before_motion'):
                    before = rolling.state_sha(rec)
                    try:
                        task = candidate.prepare(rec, task)
                    finally:
                        if rolling.state_sha(rec) != before:
                            raise RuntimeError('geometry_preview_changed_physical_state')
            effective_scene = copy.deepcopy(rec.scene_spec) if is_rolling else declare()
            with candidate.install_for_worker(root):
                with span(directory, 'planning_execution_without_rendering'):
                    performance.phase = 'execution'
                    if is_rolling:
                        rolling.execute(rec, task, release_policy=spec['release_policy'], motion_policy=spec['motion_policy'])
                    else:
                        solve(rec, task, 'cached_antipodal', release_policy=spec['release_policy'], motion_policy=spec['motion_policy'])
        except Exception as error:
            import traceback
            failure = {'type': type(error).__name__, 'message': str(error), 'trace': traceback.format_exc()}
            write(directory / 'planning_failure.json', failure)
        finally:
            if rec is not None and rec.recording:
                if is_rolling or not (directory / 'event_protocol.json').exists():
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    print(worker(args.root, args.case))
