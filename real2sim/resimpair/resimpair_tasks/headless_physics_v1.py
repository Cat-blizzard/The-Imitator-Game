"""Finite native CPU physics validation with deferred, explicitly absent RGB.

Reuses the exact saved six scenes, tasks, solver, controls and contact policy.
The old full evaluator still fails its two-camera requirement; physical checks
are reported separately. Full simulator states are saved for future rendering.
This does not provide CUDA cuDex optimization or fabricate simulator videos.
"""
import argparse
from contextlib import contextmanager
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

VERSION = 'headless_native_physics_v1'


@contextmanager
def no_render_helpers():
    """Remove the wall helper's unconditional render body, preserving collision.

    The stock helper bypasses ManiSkill's can_render() check. Preserve its
    entity name, pose, rigid component and default-material box exactly.
    """
    import sapien
    from mani_skill.utils.scene_builder.table import utils
    from mani_skill.utils.building.actor_builder import ActorBuilder
    original = utils.create_entity_box
    changed = {}

    def entity_box(scene, pose, half_size, color=None, is_static=False, name='', texture_id=None):
        if scene.can_render():
            raise ValueError('headless_render_device_must_be_disabled')
        entity = sapien.Entity(); entity.set_name(name); entity.set_pose(pose)
        rigid = (sapien.physx.PhysxRigidStaticComponent() if is_static
                 else sapien.physx.PhysxRigidDynamicComponent())
        rigid.attach(sapien.physx.PhysxCollisionShapeBox(half_size=half_size, material=None))
        entity.add_component(rigid)
        for sub_scene in scene.sub_scenes:
            sub_scene.add_entity(entity)
        return entity

    utils.create_entity_box = entity_box
    for name in ('add_box_visual', 'add_sphere_visual', 'add_visual_from_file',
                 'add_capsule_visual', 'add_cylinder_visual'):
        prior = getattr(ActorBuilder, name)
        changed[name] = (name in ActorBuilder.__dict__, prior)
        def skip_visual(self, *args, **kwargs):
            if self.scene.can_render():
                raise ValueError('unexpected_render_enabled_builder')
            return self
        setattr(ActorBuilder, name, skip_visual)
    try:
        yield
    finally:
        utils.create_entity_box = original
        for name, (owned, prior) in changed.items():
            if owned:
                setattr(ActorBuilder, name, prior)
            else:
                delattr(ActorBuilder, name)


def worker(root, case_id):
    from resimpair_runtime import configure
    configure()
    from resimpair_pickplace.runner import verify
    from resimpair_pickplace import environment
    from resimpair_pickplace.continuation_motion_v3 import install_for_worker
    from resimpair_pickplace.stability_motion import solve
    from resimpair_pickplace.seed_compatibility import seed_receipt
    from resimpair_grasp.runner import native_geometry
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import simulation, core, planning_protocols
    from corl2026_workshop.experiments.autonomous_pipeline_v4.timing import Accumulator, span
    import torch
    import gymnasium as gym
    root = Path(root).resolve(); spec = verify(root)
    resolve(read(root/'protocol.json')['driver'])
    case = next(c for c in read(root/'cohort.json')['cases'] if c['id'] == case_id)
    directory = root/'executions'/case_id
    if os.environ.get('RESIMPAIR_HEADLESS_RESERVATION') != str(directory/'reservation.json'):
        raise ValueError('bounded_supervisor_required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_available():
        raise ValueError('CUDA_must_be_disabled_for_headless_native_worker')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    scene, task = core.read(core.checked(case['scene'])), core.read(core.checked(case['task']))
    reset_seed = seed_receipt(scene['seed']); write(directory/'simulation_seed.json', reset_seed)
    ids = [o['id'] for o in scene['objects']]
    supports = ids+[s['id'] for s in scene['supports']]+['table']
    write(directory/'event_protocol.json', planning_protocols.pick_place(scene, task, ids, supports,
        simulation.targets(scene, task)))
    write(directory/'planning_declaration.json', {'event_protocol': core.bind(directory/'event_protocol.json'),
        'case': case, 'scope': 'before_physical_execution', 'declared_unix': time.time(),
        'implementation': spec['code'], 'headless_driver': binding(__file__)})

    class StateRecorder(simulation.Recorder):
        def reset(self, seed):
            self.obs, _ = self.env.reset(seed=seed, options={'reconfigure': True})
            self.hold(25)
            self.initial = self.measure()
            self.initial_physics = simulation.array(self.unwrapped.get_state()).copy()
            self.initial_physics_sha = hashlib.sha256(np.ascontiguousarray(self.initial_physics).tobytes()).hexdigest()
            self.physics_rows = []
            self.recording = True

        def capture(self):
            self.rows.append(self.measure()); self.stages.append(self.stage)
            self.physics_rows.append(simulation.array(self.unwrapped.get_state()).copy())

    performance = Accumulator(); env = rec = None; failure = bundle = None
    with no_render_helpers(), install_for_worker(root):
        try:
            with span(directory, 'simulation_environment_setup'):
                environment.register(scene, performance, contact_policy=spec.get('contact_policy', 'native_default'))
                env = gym.make(environment.ENV_ID, sim_backend='physx_cpu', render_backend='none',
                    control_mode='pd_joint_pos', obs_mode='state_dict', reward_mode='none', render_mode=None)
                if env.unwrapped.scene.can_render() or env.unwrapped.gpu_sim_enabled:
                    raise ValueError('headless_native_CPU_backend_required')
            with span(directory, 'simulation_reset_and_recorders'):
                performance.phase = 'reset'; rec = StateRecorder(env, directory)
                rec.reset(reset_seed['simulator_reset_seed'])
                native_geometry(rec, directory); environment.native_evidence(rec, directory)
            with span(directory, 'planning_execution_without_rendering'):
                performance.phase = 'execution'
                solve(rec, task, 'cached_antipodal', release_policy=spec.get('release_policy', 'legacy_v1'),
                      motion_policy=spec.get('motion_policy', 'legacy_v1'))
        except Exception as error:
            import traceback
            failure = {'type': type(error).__name__, 'message': str(error), 'trace': traceback.format_exc()}
            write(directory/'planning_failure.json', failure)
        finally:
            if rec is not None and rec.recording:
                performance.phase = 'finalize'; rec.finish()
                # Original export with no generated cameras. It still records
                # all physics arrays and original full scene below; absence of
                # images is explicit, and the old camera check remains false.
                export_scene = {**scene, 'cameras': {}}
                bundle = simulation.export(rec, root, case, task, export_scene, directory, failure)
                if bundle:
                    np.savez_compressed(directory/'deferred_render_states.npz',
                        physics_states=np.stack(rec.physics_rows),
                        timestamps_seconds=np.arange(len(rec.rows))/rec.control_freq)
                    write(directory/'deferred_render_manifest.json', {'schema': VERSION,
                        'original_scene': case['scene'], 'planned_cameras': scene['cameras'],
                        'physics_states': binding(directory/'deferred_render_states.npz'),
                        'physical_execution_count': 1, 'rendered_frames': 0,
                        'camera_synchronization': 'unmeasured_pending_render',
                        'not_eligible_for_R2H_export': True})
            if env is not None:
                env.close()
            performance.persist(directory/'timing_breakdown.json')
    write(directory/'worker_result.json', {'schema': VERSION, 'bundle': bundle,
        'failure': failure, 'gpu_jobs': 0, 'api_calls': 0, 'rendered_frames': 0,
        'full_pair_qualified': False})
    return {'bundle_saved': bool(bundle), 'failure': failure['message'] if failure else None}


def prepare(root, parent):
    from resimpair_pickplace.runner import code
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import core
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if root.exists():
        raise ValueError('retain_existing_headless_campaign')
    old = read(parent/'proposed_experiment.json')
    if list((parent/'executions').glob('*/reservation.json')):
        raise ValueError('parent_physics_already_started_do_not_duplicate')
    write(root/'experiment.json', {**old, 'code': code(), 'legacy_code': core.implementation(),
        'schema': VERSION, 'gpu': None, 'gpu_uuid': None,
        'render_backend': 'none', 'parent_protocol': binding(parent/'protocol.json')})
    write(root/'cohort.json', read(parent/'cohort.json'))
    write(root/'rules.json', read(parent/'rules.json'))
    write(root/'protocol.json', {'schema': VERSION, 'driver': binding(__file__),
        'parent': str(parent), 'parent_protocol': binding(parent/'protocol.json'),
        'same_six_input_cases': binding(parent/'cohort.json'),
        'shared_physical_budget': old['budget'], 'maximum_attempts': 6,
        'prior_physical_spent': 679, 'maximum_cumulative': 685,
        'per_attempt_timeout_seconds': 180, 'parallel_workers': 1,
        'controller': 'bounded_continuation_motion_v3',
        'full_acceptance_unchanged': True, 'full_acceptance_expectation': 'false_without_two_camera_media',
        'api_calls': 0, 'gpu_jobs': 0, 'default_promoted': False})
    return {'prepared': 6, 'physical_executions': 0}


def run(root, limit):
    root = Path(root).resolve(); protocol = read(root/'protocol.json'); resolve(protocol['driver'])
    if not 1 <= limit <= 6:
        raise ValueError('bounded_launch_limit')
    parent = Path(protocol['parent'])
    with (parent/'.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if list((parent/'executions').glob('*/reservation.json')):
            raise ValueError('parent_has_started_cases')
        budget = Path(protocol['shared_physical_budget'])
        launched = []
        for case in read(root/'cohort.json')['cases']:
            directory = root/'executions'/case['id']
            if (directory/'terminal.json').exists():
                continue
            if len(launched) >= limit:
                break
            if (directory/'reservation.json').exists():
                raise ValueError('unknown_headless_attempt_recover_without_repeat')
            used = len(list((budget/'reservations').glob('*.json')))
            if used >= protocol['maximum_attempts']:
                raise ValueError('shared_physical_budget_exhausted')
            reservation = {'case': case, 'campaign': str(root), 'started_unix': time.time(),
                'render_backend': 'none', 'gpu': None, 'protocol': binding(root/'protocol.json')}
            write(budget/'reservations'/f'{used:04d}.json', reservation)
            write(directory/'reservation.json', reservation)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
                VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.x86_64.json',
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                PYTHONPATH=str(ROOT), RESIMPAIR_HEADLESS_RESERVATION=str(directory/'reservation.json'))
            started = time.monotonic()
            with (directory/'worker.log').open('x') as stream:
                child = subprocess.Popen([sys.executable, '-u', '-m', 'resimpair_tasks.headless_physics_v1',
                    'worker', '--root', str(root), '--case', case['id']], cwd=ROOT, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                write(directory/'process.json', {'pid': child.pid, 'started_unix': time.time()})
                try:
                    rc = child.wait(timeout=protocol['per_attempt_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); child.wait()
                    rc = 'timeout'
            terminal = {'case': case['id'], 'returncode': rc, 'physical_attempts': 1,
                'elapsed_seconds': time.monotonic()-started, 'gpu_jobs': 0, 'api_calls': 0}
            write(directory/'terminal.json', terminal); launched.append(terminal)
            print(terminal, flush=True)
            if rc != 0 or not (directory/'execution_bundle.json').exists():
                break
        write(root/'snapshots'/f'run_{time.time_ns()}.json', {'rows': launched})
        return {'new_attempts': len(launched), 'rows': launched}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare','run','worker'])
    p.add_argument('--root', type=Path, required=True); p.add_argument('--parent', type=Path)
    p.add_argument('--case'); p.add_argument('--limit', type=int, default=1)
    a = p.parse_args()
    if a.command == 'prepare': print(prepare(a.root, a.parent))
    elif a.command == 'worker': print(worker(a.root, a.case))
    else: print(run(a.root, a.limit))
