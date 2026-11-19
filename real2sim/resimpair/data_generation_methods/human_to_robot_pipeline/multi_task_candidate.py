"""Finite opt-in scene qualification; reuses frozen task experts and verifiers.

Apple retains strict motion. Book/cabinet retain native task-specific checks and
are explicitly not promoted to apple's strict path-collision qualification.
The launcher holds shared GPU locks and accounts failed/unknown submissions.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
import numpy as np
from .planning_contract import WORKSPACE, FROZEN_SPEC, bind, save_new, check_binding
from .candidate_scene import validate_background

BASE = Path('/mnt/sdf/datasets/hypericl_data_factory_20261016_v1')
PYTHON = Path('/mnt/sdf/dataset/zhouxunzhe/zhiqi/PolicyGen/.venv/bin/python')
MAX_ATTEMPTS = 6
CANONICAL_CAMPAIGN_ROOT = WORKSPACE.parent/'simulation'
TIMEOUT = 900
TASKS = {'apple_bowl_transfer', 'closed_book_to_open_box', 'right_cabinet_open_only'}
LOCK_PATHS = [
    Path('/mnt/sdg/datasets/real_sim_human_pipeline/GPU2_candidate_shared.lock'),
    Path('/mnt/sdg/datasets/real_sim_human_pipeline/report_control_20261021/gpu2.lock'),
    BASE/'qualified_render_gpu_2.lock',
    Path('/mnt/sdf/datasets/hypericl_geometry_mechanisms_20261017_v1/GPU2_single_native_child.lock'),
    Path('/mnt/sdf/datasets/hypericl_human_robot_task_library_20261019_v1/generation/GPU_2_human_first.lock'),
]


def validate_case(case):
    if case.get('schema') != 'rsh_scene_candidate_v2':
        raise ValueError('New launches require the source/inventory-bound scene schema v2')
    if case.get('task') not in TASKS or case.get('split') not in {'train', 'validation', 'test'}:
        raise ValueError('Unsupported task/split')
    if not isinstance(case.get('seed'), int) or not case.get('case_id') or '/' in case['case_id']:
        raise ValueError('Invalid immutable attempt identity')
    check_binding(case['human_spec'])
    task = next(t for t in json.loads(Path(case['human_spec']['path']).read_text())['tasks'] if t['id'] == case['task'])
    valid = {w['id'] for w in task['human_windows'] if w['split'] == case['split']}
    if not case.get('human_window_ids') or not set(case['human_window_ids']) <= valid:
        raise ValueError('Human source must stay in its frozen data split')
    validate_background(case.get('background', {}))
    if case['task'] != 'apple_bowl_transfer' and case.get('added_distractors'):
        raise ValueError('New physical clutter is not qualified in native book/cabinet planners')
    if case['task'] == 'apple_bowl_transfer':
        from .structured_randomization import screen_layout
        if not screen_layout(case['proposal'])['coarse_screen_passed']:
            raise ValueError('Apple geometry coarse screen rejected')
        if case['proposal']['split'] != case['split'] or case['proposal']['seed'] != case['seed']:
            raise ValueError('Apple proposal identity differs')
        proposal = case['proposal']
        if case['human_window_ids'] != [proposal.get('human_window_id')]:
            raise ValueError('Nested proposal and candidate human source differ')
        expected = next(w for w in task['human_windows'] if w['id'] == proposal['human_window_id'])
        if proposal.get('human_source_window') != expected or proposal.get('human_video') != expected['video']:
            raise ValueError('Nested source window/video does not match frozen source')
        required_ids = ['source_container','destination_container'] + [x['instance_id'] for x in proposal['added_distractors']]
        if proposal.get('planner_requirements',{}).get('obstacle_instance_ids') != required_ids:
            raise ValueError('Declared obstacle identities differ from actual added objects')
        inventory = [{'instance_id':x['instance_id'],'asset_uid':x['asset_uid'],'category':x['category']} for x in proposal['added_distractors']]
        if len({x['instance_id'] for x in inventory}) != len(inventory):
            raise ValueError('Duplicate physical body identity')
        if case.get('scene_inventory') != {'static_distractor_count':len(inventory),'static_distractors':inventory}:
            raise ValueError('Cross-layer body IDs/categories/count differ')
        if proposal.get('factor_design',{}).get('clutter_count') != len(inventory):
            raise ValueError('Factor metadata clutter count differs from actual added objects')
    elif case.get('scene_inventory') != {'static_distractor_count':0,'static_distractors':[]}:
        raise ValueError('Native book/cabinet scene inventory must declare no added physical clutter')
    return task


def gpu_snapshot():
    value = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=20)
    ident, memory, utilization = [x.strip() for x in value.split(',')]
    owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True, timeout=20)
    return dict(gpu=2, uuid=ident, memory_MiB=int(memory), utilization=int(utilization),
                compute_owners=[x for x in owners.splitlines() if x.split(',')[0].strip() == ident])


def check_idle(snapshot):
    if snapshot['compute_owners'] or snapshot['memory_MiB'] > 512 or snapshot['utilization'] > 10:
        raise RuntimeError('GPU2 occupied; do not disturb owner')


def ledger_reservation(ledger, case, snapshot):
    records = sorted(ledger.glob('*_reserved.json'))
    if len(records) >= MAX_ATTEMPTS:
        raise RuntimeError('Cumulative six-attempt budget exhausted; failed/unknown count')
    if any(json.loads(p.read_text())['case_id'] == case['case_id'] for p in records):
        raise FileExistsError('Previously submitted identity; never retry automatically')
    number = len(records) + 1
    save_new(ledger/f'{number:02d}_reserved.json', dict(case_id=case['case_id'], task=case['task'],
        seed=case['seed'], reserved_unix=time.time(), gpu_snapshot=snapshot,
        maximum_total_attempts=MAX_ATTEMPTS, timeout_seconds=TIMEOUT, paid_calls=0,
        case_sha256=hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(),
        prelaunch_sources=[bind(__file__),bind(Path(__file__).with_name('candidate_scene.py')),
                           bind(Path(__file__).with_name('strict_apple_adapter.py')),
                           bind(Path(__file__).with_name('experimental_apple_adapter.py'))]))
    return number


def launch(case_path, root):
    case_path, root = Path(case_path).resolve(), Path(root).resolve()
    case = json.loads(case_path.read_text()); validate_case(case)
    if root != CANONICAL_CAMPAIGN_ROOT.resolve() or not str(root).startswith('/mnt/sdg/'):
        raise ValueError('Output must use the fixed campaign ledger; changing directories cannot reset budget')
    root.mkdir(parents=True, exist_ok=True)
    ledger = root/'ledger'; ledger.mkdir(exist_ok=True)
    with contextlib.ExitStack() as stack:
        for path in [ledger/'.lock', *LOCK_PATHS]:
            # Historical SDF lock files are read-only descriptors; no historical writes.
            if path.exists():
                handle = path.open('r')
            elif str(path).startswith('/mnt/sdg/'):
                path.parent.mkdir(parents=True, exist_ok=True); handle = path.open('a')
            else:
                raise FileNotFoundError('Missing shared historical lock: '+str(path))
            stack.callback(handle.close)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = gpu_snapshot(); check_idle(snapshot)
        if shutil.disk_usage(root).free < 150*2**30:
            raise RuntimeError('SDG below 150 GiB reserve')
        target = root/'attempts'/case['case_id']
        if target.exists():
            raise FileExistsError('Attempt output already exists')
        number = ledger_reservation(ledger, case, snapshot)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(root/'tmp'),
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        env.pop('CUDA_VISIBLE_DEVICES', None); Path(env['TMPDIR']).mkdir(exist_ok=True)
        env['RSH_LEDGER_RESERVATION'] = str(ledger/f'{number:02d}_reserved.json')
        argv = [str(PYTHON), '-B', '-u', '-m', __spec__.name, 'worker', '--case', str(case_path), '--output', str(target)]
        log = ledger/f'{number:02d}_worker.log'; started = time.time()
        with log.open('x') as stream:
            proc = subprocess.Popen(argv, cwd=WORKSPACE, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            save_new(ledger/f'{number:02d}_process.json', dict(pid=proc.pid, argv=argv, started_unix=started))
            timeout = False
            try:
                proc.wait(timeout=TIMEOUT)
            except subprocess.TimeoutExpired:
                timeout = True
                if proc.poll() is None and os.getpgid(proc.pid) == proc.pid:
                    os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    if proc.poll() is None and os.getpgid(proc.pid) == proc.pid:
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=15)
        record = dict(case_id=case['case_id'], task=case['task'], exit_code=proc.returncode,
            timeout=timeout, wall_seconds=time.time()-started, output=str(target), log=bind(log),
            total_attempts_charged=number, retries=False, paid_calls=0,
            result=bind(target/'result.json') if (target/'result.json').is_file() else None)
        save_new(ledger/f'{number:02d}_terminal.json', record)
        print(json.dumps(record)); return record


def archive_sources(output, files, base_pins):
    pins = dict(base_pins)
    for path in files:
        path = Path(path).resolve(); pins[str(path)] = bind(path)['sha256']
    archive = {}
    directory = output/'source_snapshot'; directory.mkdir()
    for path, digest in pins.items():
        dest = directory/(digest+Path(path).suffix)
        if not dest.exists(): shutil.copyfile(path, dest)
        check_binding({'path':str(dest),'sha256':digest})
        archive[path] = dict(path=str(dest), sha256=digest)
    save_new(output/'source_archive.json', archive)
    return pins


def build_recorder(task):
    from examples.baselines.hypericl_data_factory.rich_runtime_v2 import VisibilityRecorder
    from examples.baselines.hypericl_task_expansion.collect import array, require
    from .multiview_recording import SynchronizedMultiviewMixin
    class Recorder(SynchronizedMultiviewMixin, VisibilityRecorder):
        def measure(self):
            r = super().measure(); b = self.unwrapped
            if task == 'closed_book_to_open_box':
                r.update(box_support_force_N=b.box_support_force.copy(), robot_box_force_N=float(b.robot_box_force))
            else:
                r.update(cabinet_qpos=array(b.cabinet.qpos).reshape(2).copy(), cabinet_qvel=array(b.cabinet.qvel).reshape(2).copy(),
                    door_world=np.stack([array(d.pose.to_transformation_matrix()).reshape(4,4) for d in b.door_links]),
                    door_grasp=np.array([[bool(array(a.is_grasping(d)).item()) for d in b.door_links] for a in b.agent.agents]),
                    contact_step_peaks_N=b.contact_step_peaks.copy(), contact_window_aggregate_N=b.contact_impulse_sums.copy()*b.control_freq,
                    asset_pair_contact_step_peak_N=float(b.asset_pair_contact_step_peak))
            return r
        def image_metrics(self):
            value = super().image_metrics()
            if task == 'right_cabinet_open_only':
                import trimesh
                b=self.unwrapped
                if not hasattr(self,'_door_visual_vertices'):
                    self._door_visual_vertices=np.asarray(trimesh.load(b.cabinet_adapter['doors'][1]['visual'],force='scene').to_mesh().vertices)
                seg=array(self.obs['sensor_data']['zed2i']['segmentation']).reshape(224,224)
                ident=int(array(b.door_links[1].per_scene_id).item()); T=array(b.door_links[1].pose.to_transformation_matrix()).reshape(4,4)
                world=self._door_visual_vertices@T[:3,:3].T+T[:3,3]; camera=world@self.extrinsic[:,:3].T+self.extrinsic[:,3]
                require((camera[:,2]>.01).all(),'Target door crosses camera near plane')
                projected=camera@self.intrinsic.T; uv=projected[:,:2]/projected[:,2:3]; bounds=np.r_[uv.min(0),uv.max(0)]
                value.update(target_door_visible_pixels_224=int((seg==ident).sum()), target_door_projected_bounds_224=bounds,
                    target_door_entirely_in_frame=bool((bounds[:2]>=0).all() and (bounds[2:]<224).all()))
            return value
        def step(self, action):
            started=time.perf_counter(); result=super().step(action)
            if not hasattr(self,'control_timing'): self.control_timing={}
            item=self.control_timing.setdefault(self.stage,dict(steps=0,wall_seconds=0.)); item['steps']+=1; item['wall_seconds']+=time.perf_counter()-started
            if self.recording:
                if task == 'closed_book_to_open_box':
                    require(self.unwrapped.robot_box_force < 2, 'native robot-box contact exceeded 2N')
                else:
                    post=self.measure(); require(post['contact_step_peaks_N'][1]<2 and post['contact_step_peaks_N'][2]<.5, 'Forbidden native contact')
            return result
    return Recorder


def native_worker(case, output):
    sys.path.insert(0, str(WORKSPACE/'PolicyGen'))
    from examples.baselines.hypericl_task_expansion.common import bind_gpu, source_pins
    gpu=bind_gpu(2)
    import gymnasium as gym
    from examples.baselines.hypericl_task_expansion.collect import save, save_diagnostic, require
    from examples.baselines.hypericl_data_factory.rich_runtime_v2 import save_native
    from examples.baselines.hypericl_data_factory.collect_dataset import visibility_quality
    from . import candidate_scene, multiview_recording
    from .candidate_scene import register_views
    from .multiview_recording import save_multiview
    task, seed, split=case['task'],case['seed'],case['split']; execution_id=str(uuid.uuid4())
    output.mkdir(parents=True, exist_ok=False)
    save_new(output/'input.json', dict(case=case, execution_id=execution_id, production_approved=False, training_admitted=False))
    if task=='closed_book_to_open_box':
        from examples.baselines.hypericl_data_factory import next_task_book_pilot as expert, next_task_book_verify as verifier
        gatepath=BASE/'next_task_book_qualification_v1/review_approval.json'
        gate=json.loads(gatepath.read_text()); assets=Path(gate['approved_asset_root'])
        bookpath=assets/'closed_pocket_book/adapter.json'; book=json.loads(bookpath.read_text())
        check_binding(dict(path=str(bookpath),sha256=gate['approved_book_adapter_sha256']))
        check_binding(dict(path=str(assets/'open_cardboard_box/adapter.json'),sha256=gate['approved_box_adapter_sha256']))
        row=dict(uid=book['uid']+'_closed_only',name='closed_pocket_book',source=book['source_url'],license=book['source_license_metadata'],
            adapter=dict(estimated_mass_kg=.08,top_obb_width_fits_panda=True,bounds_m=book['visual']['bounds_m'],
                visual=book['visual']['path'],visual_sha256=book['visual']['sha256'],collision=book['convex_collision']['path'],collision_sha256=book['convex_collision']['sha256']))
        pear=next(r for r in json.loads((BASE/'fruit_adapters_v2/catalog.json').read_text())['assets'] if r['name']=='pear')
        expert.register(assets)
        env_id=register_views('HyperICLClosedBookToBoxDev-v1','CandidateBookViews-v1',case.get('background',{}))
        selected=np.random.default_rng(seed+730001).choice(2,2,replace=False)
        program='pick_first_to_left' if list(selected).index(0)==0 else 'pick_second_to_left'
        env_kwargs=dict(asset_pool=[row,pear],family='book')
        folder=output/'episodes'/str(seed)
        extra_files=[bookpath,assets/'open_cardboard_box/adapter.json']
    else:
        from examples.baselines.hypericl_data_factory import cabinet_open_atomic_produce_v7 as expert, cabinet_open_atomic_production_verify_v1 as verifier, cabinet_open_atomic_skill as skill
        from examples.baselines.hypericl_data_factory.cabinet_robot_pilot import block_pool
        gatepath=BASE/'cabinet_contact_actual_visual_review_v1/review_approval.json'
        gate=json.loads(gatepath.read_text()); adapterpath=BASE/'cabinet_contact_adapter_v7/adapter.json'
        check_binding(dict(path=str(adapterpath),sha256=gate['approved_adapter_sha256']))
        adapter=json.loads(adapterpath.read_text()); pool=block_pool(output)
        expert.register(); env_id=register_views('HyperICLRightDoorOpenAtomic-v6','CandidateCabinetViews-v1',case.get('background',{}))
        program='pick_second_to_left'
        env_kwargs=dict(asset_pool=pool,cabinet_adapter=adapter,cabinet_x=.05,operation='remove_leave_open',shelf='middle',block_offset_y=.035)
        folder=output/'cases'/str(seed)/'episode'; extra_files=[adapterpath,Path(skill.__file__)]
    for path,digest in gate['native_producer_source_pins'].items():
        check_binding(dict(path=path,sha256=digest))
    pins=archive_sources(output,[__file__,candidate_scene.__file__,multiview_recording.__file__,expert.__file__,verifier.__file__,gatepath,*extra_files],
        {**source_pins(),**gate['native_producer_source_pins']})
    Recorder=build_recorder(task); env=rec=None; started=time.perf_counter(); timings={}
    try:
        env=gym.make(env_id,**env_kwargs,view_profile='balanced',data_split=case.get('background_split',split),program=program,
            sim_backend='physx_cpu',render_backend=gpu['render_backend'],control_mode='pd_joint_pos',obs_mode='rgb+segmentation',reward_mode='none',
            render_mode='rgb_array',sensor_configs=dict(shader_pack='rt-fast'))
        rec=Recorder(env); rec.output_root=str(folder.parent); folder.parent.mkdir(parents=True,exist_ok=True)
        rec.reset(seed,program); timings['environment_reset_seconds']=time.perf_counter()-started
        begin=time.perf_counter()
        if task=='closed_book_to_open_box': events=expert.solve(rec); metrics,final=expert.qualify(rec)
        else:
            from examples.baselines.hypericl_data_factory.cabinet_clearance_physics_v5 import shape_settings
            shapes=shape_settings(rec.unwrapped.cabinet_body,rec.unwrapped.door_links)
            require(len(shapes)==53 and all(abs(s['contact_offset_m']-.0002)<1e-10 and s['rest_offset_m']==0 for s in shapes),'Actual cabinet offsets differ')
            robot=[dict(arm=i,link=l.name,index=j,contact_offset_m=float(s.contact_offset),rest_offset_m=float(s.rest_offset)) for i,a in enumerate(rec.unwrapped.agent.agents) for l in a.robot.get_links() for j,s in enumerate(l._bodies[0].get_collision_shapes())]
            save_new(folder.parent/'actual_cooked_shape_settings.json',dict(cabinet=shapes,robot=robot,geometry_and_success_rules_unchanged=True))
            events=[skill.open_door(rec,1,'z_flip'),expert.withdraw(rec)];metrics,final=expert.qualify(rec)
        timings['planning_control_seconds']=time.perf_counter()-begin
        require(metrics['success'],'Unchanged native physical outcome rejected: '+json.dumps(metrics))
        begin=time.perf_counter()
        receipt=save(rec,folder,task,program,split,seed,events,metrics,final,pins)
        (folder/'receipt.json').rename(folder/'receipt.base.json')
        receipt.update(save_native(rec,folder)); receipt.update(save_multiview(rec,folder,execution_id))
        receipt.update(randomization=rec.unwrapped.randomization,initial_measured_states=rec.initial['states'].tolist(),
            initial_measured_object_states=rec.initial['objects'].tolist(),fresh_recorder_per_episode=True,production_approved=False,
            human_pairing_requires_separate_review=True,training_admitted=False,candidate_scene_expansion=True,
            independent_strict_path_qualification=False,source_task_binding=case['human_spec'],human_window_ids=case['human_window_ids'])
        if task=='closed_book_to_open_box':
            quality=visibility_quality(rec,program); require(quality['passed'],'Unchanged book media gate rejected')
            contact=folder/'box_contacts.npz'
            np.savez_compressed(contact,support_force_N=np.stack([r['box_support_force_N'] for r in rec.rows]),robot_box_force_N=np.asarray([r['robot_box_force_N'] for r in rec.rows]),stage=np.asarray(rec.stage_names,dtype='U40'))
            receipt.update(goal_key=gate['allowed_goal_key'],qualification_only=True,receptacle=rec.unwrapped.receptacle_spec,
                image_quality=quality,contact_data=str(contact),contact_sha256=bind(contact)['sha256'])
        else:
            receipt.pop('container_geometry',None)
            receipt.update(goal_key='factory:cabinet_open_right_only',qualification_only=False,full_insert_remove_close_task_completed=False,
                production_gate=bind(gatepath),task='cabinet_open_right_only',active_arm=0,cabinet_adapter=str(adapterpath),
                cabinet_adapter_sha256=bind(adapterpath)['sha256'],cabinet_position_m=rec.unwrapped.cabinet_position.tolist(),
                target_geometry_kind='articulated_cabinet_right_door_not_receptacle',legacy_container_centers_world_is_not_goal_geometry=True,
                qualification_flag_note='False required by unchanged historical cabinet verifier; production_approved and training_admitted remain false')
            joint=folder.parent/'joint_contact_measurements.npz'
            np.savez_compressed(joint,**{k:np.stack([r[k] for r in rec.rows]) for k in ['cabinet_qpos','cabinet_qvel','door_world','door_grasp','contact_step_peaks_N','contact_window_aggregate_N','asset_pair_contact_step_peak_N']},stage=np.asarray(rec.stage_names,dtype='U48'))
            actual=folder.parent/'robot_active_door_contact_records.json'; asset=folder.parent/'asset_pair_contact_records.json'
            save_new(actual,rec.unwrapped.robot_active_door_contact_records); save_new(asset,rec.unwrapped.asset_pair_contact_records)
            visibility=folder/'door_visibility.npz'; pixels=np.asarray([v['target_door_visible_pixels_224'] for v in rec.visibility]);inside=np.asarray([v['target_door_entirely_in_frame'] for v in rec.visibility])
            np.savez_compressed(visibility,visible_pixels_224=pixels,projected_bounds_224=np.asarray([v['target_door_projected_bounds_224'] for v in rec.visibility]),entirely_in_frame=inside,camera_from_world=rec.extrinsic,intrinsic_224=rec.intrinsic,intrinsic_512=rec.native_intrinsic)
            for key,path in [('articulation_data',joint),('actual_contact_data',actual),('asset_contact_data',asset),('actual_shape_settings',folder.parent/'actual_cooked_shape_settings.json'),('door_visibility',visibility)]:
                receipt[key]=str(path);receipt[key+'_sha256']=bind(path)['sha256']
            receipt['target_door_image_quality']=dict(passed=bool(pixels[0]>=24 and pixels[-1]>=16 and inside.mean()>=.95),initial_pixels_224=int(pixels[0]),final_pixels_224=int(pixels[-1]),projected_in_frame_fraction=float(inside.mean()))
        save_new(folder/'receipt.json',receipt); timings['media_export_seconds']=time.perf_counter()-begin
        begin=time.perf_counter(); verification=verifier.verify(folder/'receipt.json');timings['independent_verification_seconds']=time.perf_counter()-begin
        save_new(output/'verification.json',verification)
        result=dict(passed=True,task=task,frames=len(rec.rows),receipt=bind(folder/'receipt.json'),original_independent_verifier=verification,
            independent_strict_path_qualification=False,source_id=case['human_window_ids'],production_approved=False,training_admitted=False)
    except Exception as exc:
        diagnostics=[]
        if rec is not None and rec.rows:
            failed=output/'failure'
            for name,fn in [('base',save_diagnostic),('native',save_native)]:
                try: fn(rec,failed)
                except Exception as err: diagnostics.append(dict(export=name,error=repr(err)))
            try: save_multiview(rec,failed,execution_id,diagnostic=True)
            except Exception as err: diagnostics.append(dict(export='multiview',error=repr(err)))
            fields=['box_support_force_N','robot_box_force_N'] if task=='closed_book_to_open_box' else ['cabinet_qpos','cabinet_qvel','door_grasp','contact_step_peaks_N']
            np.savez_compressed(failed/'task_contacts.npz',**{k:np.asarray([r[k] for r in rec.rows]) for k in fields},timestamps_seconds=np.arange(len(rec.rows))/rec.control_freq)
        result=dict(passed=False,task=task,error=repr(exc),traceback=traceback.format_exc(),stage=rec.stage if rec else 'initialization',
            frames=len(rec.rows) if rec else 0,diagnostic_errors=diagnostics,production_approved=False,training_admitted=False)
    finally:
        timings['total_worker_seconds']=time.perf_counter()-started
        if rec is not None:timings['control_step_timings']=getattr(rec,'control_timing',{})
        save_new(output/'timing.json',timings)
        if env is not None:env.close()
    save_new(output/'result.json',result); return result


def worker(case_path, output):
    case=json.loads(Path(case_path).read_text());validate_case(case)
    reserved = Path(os.environ.get('RSH_LEDGER_RESERVATION', '/missing'))
    if reserved.parent != CANONICAL_CAMPAIGN_ROOT/'ledger' or not reserved.is_file():
        raise RuntimeError('Worker requires a charged reservation from the fixed campaign launcher')
    reservation = json.loads(reserved.read_text())
    expected = hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()
    if reservation['case_sha256'] != expected or reservation['case_id'] != case['case_id']:
        raise ValueError('Worker input differs from charged reservation')
    if Path(output).resolve() != CANONICAL_CAMPAIGN_ROOT/'attempts'/case['case_id']:
        raise ValueError('Worker output differs from charged campaign identity')
    if case['task']=='apple_bowl_transfer':
        from .strict_apple_adapter import run
        from .planning_contract import build_apple_contract
        proposal=dict(case['proposal']); proposal['background']=case.get('background',{})
        start=time.perf_counter();result=run(proposal,build_apple_contract(),output,2,multiview=True)
        save_new(output/'candidate_case_binding.json',dict(case=case,candidate_sources=[bind(__file__),bind(Path(__file__).with_name('candidate_scene.py'))],strict_motion_unchanged=True))
        save_new(output/'timing.json',dict(total_worker_seconds=time.perf_counter()-start))
    else:result=native_worker(case,output)
    print(json.dumps(result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('launch');a.add_argument('--case',type=Path,required=True);a.add_argument('--root',type=Path,required=True)
    a=sub.add_parser('worker');a.add_argument('--case',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='launch':return launch(args.case,args.root)
    result=worker(args.case,args.output);raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
