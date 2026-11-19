"""Bounded local dual-arm qualification; GPU2 UUID + shared lock, no training/API."""
from __future__ import annotations
import argparse
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

from ..human_to_robot_pipeline.planning_contract import bind,save_new

WORKSPACE=Path(__file__).resolve().parents[2]
GPU_UUID='GPU-807bd44d-36ea-31d0-427c-50e264428c67'
LOCK=Path('/mnt/sdg/datasets/real_sim_human_pipeline/report_control_20261021/gpu2.lock')
THRESHOLDS={'minimum_lift_m':.12,'stable_hold_frames':20,'max_linear_velocity_m_s':.03,
    'max_angular_velocity_rad_s':.3,'max_forbidden_contact_N':.5,
    'moving_TCP_speed_m_s':.01,'minimum_simultaneous_motion_seconds':.5}


def register():
    from mani_skill.utils.registration import REGISTERED_ENVS,register_env
    from examples.baselines.hypericl_data_factory.rich_runtime_v2 import register as parent_register
    from examples.baselines.hypericl_task_expansion.collect import array
    parent_register();parent=REGISTERED_ENVS['HyperICLRichAssetRelocation-v2'].cls
    @register_env('PairedFruitLiftCandidate-v1',max_episode_steps=3000)
    class PairedFruitLift(parent):
        def _initialize_episode(self,env_idx,options):
            super()._initialize_episode(env_idx,options)
            self._robots={link._bodies[0].entity:(i,link.get_name())
                for i,agent in enumerate(self.agent.agents) for link in agent.robot.get_links()}
            self._objects={body.entity:i for i,obj in enumerate(self.objects) for body in obj._bodies}
            self._world={body.entity for actor in [self.table_scene.table]+self.bins for body in actor._bodies}
            self.control_contact_peaks={'inter_arm':0.,'wrong_object':0.,'robot_environment':0.,'fruit_fruit':0.}
            self.control_substeps=[]
            self.control_substep_objects=[]
            self.control_substep_finger_forces=[]
        def _before_control_step(self):
            super()._before_control_step()
            self.control_contact_peaks={k:0. for k in self.control_contact_peaks}
            self.control_substeps=[]
            self.control_substep_objects=[]
            self.control_substep_finger_forces=[]
        def _after_simulation_step(self):
            super()._after_simulation_step()
            force={k:0. for k in self.control_contact_peaks}
            finger_forces=np.zeros((2,2))
            for contact in self.scene.get_contacts():
                entities=[x.entity for x in contact.bodies];r=[self._robots.get(x) for x in entities]
                o=[self._objects.get(x) for x in entities]
                magnitude=sum(float(np.linalg.norm(p.impulse)) for p in contact.points)*self.sim_freq
                if all(x is not None for x in r) and r[0][0]!=r[1][0]:force['inter_arm']+=magnitude
                if all(x is not None for x in o) and o[0]!=o[1]:force['fruit_fruit']+=magnitude
                for a,z in [(0,1),(1,0)]:
                    if r[a] is None:continue
                    if o[z] is not None and o[z]!=r[a][0]:force['wrong_object']+=magnitude
                    if o[z] is not None and o[z]==r[a][0] and r[a][1] in ['panda_leftfinger','panda_rightfinger']:
                        finger_forces[r[a][0],['panda_leftfinger','panda_rightfinger'].index(r[a][1])]+=magnitude
                    if entities[z] in self._world and r[a][1]!='panda_link0':force['robot_environment']+=magnitude
            for key,val in force.items():self.control_contact_peaks[key]=max(self.control_contact_peaks[key],val)
            self.control_substeps.append([force[k] for k in self.control_contact_peaks])
            self.control_substep_finger_forces.append(finger_forces)
            self.control_substep_objects.append(np.stack([np.r_[array(o.pose.raw_pose).reshape(7),
                array(o.linear_velocity).reshape(3),array(o.angular_velocity).reshape(3)] for o in self.objects]))


def metrics(rec,mode):
    rows=rec.rows+[rec.measure()]
    objects=np.array([r['objects'] for r in rows]);grasp=np.array([r['grasp'] for r in rows])
    tcp=np.array([r['tcp_world'][:,:3,3] for r in rows]);n=len(rec.rows)
    speed=np.linalg.norm(np.diff(tcp,axis=0),axis=-1)*rec.control_freq
    moving=speed>THRESHOLDS['moving_TCP_speed_m_s']
    height=objects[:,:,2]-objects[0,:,2]
    own_grasp=np.stack([grasp[:,i,i] for i in range(2)],axis=1)
    tail=THRESHOLDS['stable_hold_frames']
    peaks=np.asarray(rec.transition_contacts)
    checks={
        'both_fruits_lifted':bool((height[-tail:]>=THRESHOLDS['minimum_lift_m']).all()),
        'both_native_grasps_sustained':bool(own_grasp[-tail:].all()),
        'both_fruits_stable':bool((np.linalg.norm(objects[-tail:,:,7:10],axis=-1)<THRESHOLDS['max_linear_velocity_m_s']).all()
            and (np.linalg.norm(objects[-tail:,:,10:13],axis=-1)<THRESHOLDS['max_angular_velocity_rad_s']).all()),
        'no_forbidden_contact_at_any_physics_substep':bool(peaks.max(initial=0)<THRESHOLDS['max_forbidden_contact_N']),
        'single_control_clock':bool(len(rec.commands)==len(rec.rows)==len(rec.native_frames)==len(rec.transition_contacts)),
        'simultaneous_motion':bool((moving.all(axis=1).sum()/rec.control_freq)>=THRESHOLDS['minimum_simultaneous_motion_seconds'])
            if mode=='synchronous' else None}
    success=all(v for v in checks.values() if v is not None)
    return {'success':success,'checks':checks,'thresholds':THRESHOLDS,'frames':n,
        'simulated_duration_seconds':n/rec.control_freq,
        'simultaneous_TCP_motion_seconds':float(moving.all(axis=1).sum()/rec.control_freq),
        'simultaneous_fraction_of_any_motion':float(moving.all(axis=1).sum()/max(1,moving.any(axis=1).sum())),
        'arm_moving_seconds':(moving.sum(axis=0)/rec.control_freq).tolist(),
        'final_height_gain_m':height[-1].tolist(),'minimum_last20_height_gain_m':height[-tail:].min(axis=0).tolist(),
        'final_native_grasp':own_grasp[-1].tolist(),'last20_all_native_grasp':own_grasp[-tail:].all(axis=0).tolist(),
        'maximum_physics_substep_forbidden_contact_N':dict(zip(rec.contact_keys,peaks.max(axis=0).tolist())),
        'final_object_linear_velocity_m_s':np.linalg.norm(objects[-1,:,7:10],axis=-1).tolist(),
        'final_object_angular_velocity_rad_s':np.linalg.norm(objects[-1,:,10:13],axis=-1).tolist()},rows[-1]


def run_case(output,seed,mode,gpu_info,hold_strategy):
    from examples.baselines.hypericl_task_expansion.collect import save,save_diagnostic,array
    from examples.baselines.hypericl_data_factory.rich_runtime_v2 import VisibilityRecorder,save_native
    from . import synchronized_pick
    from .synchronized_pick import solve
    import gymnasium as gym
    import cv2
    import mplib.planner
    from ..human_to_robot_pipeline.strict_apple_adapter import robot_dependency_paths
    from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
    output.mkdir(parents=True,exist_ok=False)
    frozen=output/'source_archive';frozen.mkdir()
    sources=[Path(__file__),Path(synchronized_pick.__file__),
        WORKSPACE/'data_generation_methods/human_to_robot_pipeline/strict_motion.py',
        WORKSPACE/'PolicyGen/examples/baselines/hypericl_data_factory/rich_runtime_v2.py',
        WORKSPACE/'PolicyGen/examples/baselines/hypericl_data_factory/rich_collect.py',
        WORKSPACE/'PolicyGen/examples/baselines/hypericl_task_expansion/collect.py',
        Path(mplib.planner.__file__)]
    pins={str(p.resolve()):bind(p)['sha256'] for p in sources}
    for p in sources:shutil.copy2(p,frozen/(pins[str(p.resolve())]+'_'+p.name))
    dependencies=robot_dependency_paths(PandaWristCam.urdf_path)
    save_new(output/'robot_dependencies.json',{'dependencies':[bind(p) for p in dependencies]})
    fruit_path=Path('/mnt/sdf/datasets/hypericl_data_factory_20261016_v1/fruit_adapters_v2/catalog.json')
    fruit=json.loads(fruit_path.read_text())['assets']
    execution=str(uuid.uuid4())
    save_new(output/'input.json',{'seed':seed,'mode':mode,'hold_strategy':hold_strategy,'execution_id':execution,'asset_catalog':bind(fruit_path),
        'source_sha256':pins,'gpu':gpu_info,'thresholds':THRESHOLDS,
        'task':'both respective fruits grasped and lifted0.12m, held stable1s; not transfer/release task',
        'API_calls':0,'training':False,'production_approved':False})
    class Recorder(VisibilityRecorder):
        def __init__(self,env):
            super().__init__(env);self.transition_contacts=[];self.substep_contacts=[]
            self.substep_objects=[];self.substep_finger_forces=[]
            self.contact_keys=['inter_arm','wrong_object','robot_environment','fruit_fruit']
        def step(self,action):
            recorded=self.recording;result=super().step(action)
            if recorded:
                self.transition_contacts.append([self.unwrapped.control_contact_peaks[k] for k in self.contact_keys])
                self.substep_contacts.append(self.unwrapped.control_substeps.copy())
                self.substep_objects.append(self.unwrapped.control_substep_objects.copy())
                self.substep_finger_forces.append(self.unwrapped.control_substep_finger_forces.copy())
            return result
    env=None;rec=None;start=time.perf_counter();solve_started=None
    try:
        env=gym.make('PairedFruitLiftCandidate-v1',asset_pool=fruit,family='fruit',view_profile='balanced',
            data_split='qualification',program='sort_direct',sim_backend='physx_cpu',render_backend=gpu_info['render_backend'],
            control_mode='pd_joint_pos',obs_mode='rgb+segmentation',reward_mode='none',render_mode='rgb_array',
            sensor_configs={'shader_pack':'rt-fast'})
        rec=Recorder(env);rec.reset(seed,'sort_direct')
        initialized=time.perf_counter();solve_started=initialized
        events,timings=solve(rec,output/'plans',mode,hold_strategy)
        solved=time.perf_counter();values,final=metrics(rec,mode)
        receipt=save(rec,output/'episode','dual_fruit_lift',mode,'qualification',seed,events,values,final,pins)
        (output/'episode/receipt.json').rename(output/'episode/receipt.base.json')
        receipt.update(save_native(rec,output/'episode'))
        contact=output/'episode/physics_substep_contacts.npz'
        np.savez_compressed(contact,contact_peak_per_command=np.asarray(rec.transition_contacts),
            contact_each_physics_substep=np.asarray(rec.substep_contacts),contact_keys=np.array(rec.contact_keys),
            object_states_each_physics_substep=np.asarray(rec.substep_objects),
            own_finger_contact_force_each_physics_substep=np.asarray(rec.substep_finger_forces),
            command_timestamps_seconds=np.arange(len(rec.rows))/rec.control_freq,
            control_to_substep_mapping='substeps[k] occur after pre-command state/frame[k] and command[k]',
            final_postcommand_object_states=final['objects'],final_postcommand_grasp=final['grasp'],
            final_postcommand_tcp_world=final['tcp_world'])
        receipt.update(execution_id=execution,contact_trace=bind(contact),two_planners=True,mode=mode,
            randomization=rec.unwrapped.randomization,qualification_only=True,production_approved=False,
            training_admitted=False,strict_apple_qualification_inherited=False)
        save_new(output/'episode/receipt.json',receipt)
        images=[]
        for label,index in [('initial',0),('simultaneous_lift',next((i for i,s in enumerate(rec.stage_names) if s=='lift'),len(rec.rows)//2)),('final',len(rec.rows)-1)]:
            path=output/'episode'/f'{label}.png';cv2.imwrite(str(path),cv2.cvtColor(rec.native_frames[index],cv2.COLOR_RGB2BGR))
            images.append({'label':label,'frame':index,**bind(path)})
        result={'passed':values['success'],'mode':mode,'hold_strategy':hold_strategy,'seed':seed,'execution_id':execution,'metrics':values,
            'timing':{**timings,'initialization_seconds':initialized-start,'solve_and_physics_seconds':solved-initialized,
                      'total_wall_seconds':time.perf_counter()-start},'receipt':bind(output/'episode/receipt.json'),
            'video':bind(output/'episode/video_native512.mp4'),'keyframes':images,
            'independent_physical_execution':True,'new_task_candidate_only':True,'production_approved':False}
    except Exception as error:
        if rec is not None and rec.rows:
            try:
                save_diagnostic(rec,output/'failure');save_native(rec,output/'failure')
                np.savez_compressed(output/'failure/contacts_and_grasp.npz',
                    grasp=np.array([r['grasp'] for r in rec.rows]),contact_peaks=np.asarray(rec.transition_contacts),
                    physics_substeps=np.asarray(rec.substep_contacts),contact_keys=np.asarray(rec.contact_keys),
                    object_states_each_physics_substep=np.asarray(rec.substep_objects),
                    own_finger_contact_force_each_physics_substep=np.asarray(rec.substep_finger_forces))
            except Exception as secondary:save_new(output/'diagnostic_error.json',{'error':repr(secondary)})
        result={'passed':False,'mode':mode,'seed':seed,'error':repr(error),'traceback':traceback.format_exc(),
            'frames':len(rec.rows) if rec else 0,'failure_preserved':True,'total_wall_seconds':time.perf_counter()-start}
    finally:
        if env is not None:env.close()
    save_new(output/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--seed',type=int,default=901021000)
    parser.add_argument('--mode',choices=['synchronous','sequential'],required=True)
    parser.add_argument('--hold-strategy',choices=['full_close','measured_compression'],default='full_close')
    args=parser.parse_args()
    LOCK.parent.mkdir(parents=True,exist_ok=True)
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader'],text=True)
        lines=[x.split(', ') for x in output.strip().splitlines()]
        selected=[x for x in lines if x[0]=='2']
        if len(selected)!=1 or selected[0][1]!=GPU_UUID or int(selected[0][2].split()[0])>500:
            raise RuntimeError('GPU2 identity/idle-memory preflight rejected')
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if GPU_UUID in processes:raise RuntimeError('GPU2 has a compute process; do not start')
        if shutil.disk_usage(LOCK.parent).free<10*1024**3:raise RuntimeError('Insufficient SDG space')
        campaign=LOCK.parent/'dual_arm'
        if not args.output.resolve().is_relative_to(campaign.resolve()):
            raise RuntimeError('Output must remain under the fixed dual-arm campaign')
        protocol=json.loads((campaign/'protocol.json').read_text())
        ledger=campaign/'attempts';ledger.mkdir(exist_ok=True)
        started=list(ledger.glob('*.json'))
        if len(started)>=protocol['max_physical_attempts']:
            raise RuntimeError('Cumulative physical attempt budget exhausted')
        save_new(ledger/f'{len(started)+1:03d}_started.json',{'output':str(args.output.resolve()),
            'seed':args.seed,'mode':args.mode,'started_unix':time.time(),'GPU_UUID':GPU_UUID,
            'timeout_seconds':protocol['per_attempt_timeout_seconds'],'source':bind(__file__)})
        signal.alarm(protocol['per_attempt_timeout_seconds'])
        sys.path.insert(0,str(WORKSPACE/'PolicyGen'))
        from examples.baselines.hypericl_task_expansion.common import bind_gpu
        gpu=bind_gpu(2)
        if gpu['uuid']!=GPU_UUID:raise RuntimeError('GPU mapping changed')
        register()
        result=run_case(args.output,args.seed,args.mode,gpu,args.hold_strategy)
        print(json.dumps(result,indent=2),flush=True)
        return 0 if result['passed'] else 2


if __name__=='__main__':sys.exit(main())
