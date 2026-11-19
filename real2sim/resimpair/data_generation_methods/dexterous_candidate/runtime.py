"""Actual CPU PhysX arm-hand candidate, explicitly isolated GPU2 rendering."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .assets import ASSETS, WORKSPACE, binding, save


def register(urdf):
    import numpy as np
    import torch
    import sapien
    from mani_skill.agents.base_agent import BaseAgent, Keyframe
    from mani_skill.agents.controllers import PDJointPosControllerConfig
    from mani_skill.agents.registration import register_agent
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.building import actors
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.types import SimConfig
    arm = [f'panda_joint{i}' for i in range(1,8)]
    hand = [f'allegro_joint_{i}.0' for i in range(16)]
    home = np.r_[[0,-.5,0,-2.0,0,1.5,.785], np.zeros(12), [.3,0,0,0]]

    @register_agent()
    class CandidatePandaAllegro(BaseAgent):
        uid = 'candidate_panda_allegro_v1'
        urdf_path = str(urdf)
        fix_root_link = True
        load_multiple_collisions = False
        disable_self_collisions = False
        keyframes = {'rest': Keyframe(qpos=home, pose=sapien.Pose())}
        urdf_config = {'_materials': {'hand': {'static_friction': 1.5, 'dynamic_friction': 1.0,
                                              'restitution': 0.0}},
                       'link': {f'allegro_link_{i}.0': {'material':'hand'} for i in range(16)}}

        @property
        def _controller_configs(self):
            return {'pd_joint_pos': PDJointPosControllerConfig(
                arm+hand, None, None, [900]*7+[45]*16, [90]*7+[5]*16,
                [87]*4+[12]*3+[8]*16, normalize_action=False)}

        def _after_init(self):
            self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), 'allegro_base_link')
            self.palm = sapien_utils.get_obj_by_name(self.robot.get_links(), 'allegro_palm')
            self.finger_groups = {name: [sapien_utils.get_obj_by_name(self.robot.get_links(),
                f'allegro_link_{i}.0') for i in indices]
                for name, indices in {'index': range(0,4), 'middle': range(4,8),
                                      'ring': range(8,12), 'thumb': range(12,16)}.items()}

    @register_env('CandidatePandaAllegroContact-v1', max_episode_steps=600)
    class CandidateContactEnv(BaseEnv):
        SUPPORTED_ROBOTS = ['candidate_panda_allegro_v1']

        def __init__(self, *args, **kwargs):
            super().__init__(*args, robot_uids='candidate_panda_allegro_v1', **kwargs)

        @property
        def _default_sim_config(self):
            return SimConfig(sim_freq=240, control_freq=20)

        @property
        def _default_sensor_configs(self):
            return [CameraConfig('overview', sapien_utils.look_at([.75,.70,.65],[-.18,0,.23]),512,512,1.1,.01,10),
                    CameraConfig('close', sapien_utils.look_at([.36,.32,.30],[.035,0,.10]),512,512,1.0,.01,10)]

        def _load_agent(self, options):
            super()._load_agent(options, sapien.Pose(p=[-.55,0,0]))

        def _load_scene(self, options):
            self.table = actors.build_box(self.scene, half_sizes=[.8,.6,.025], color=[.65,.52,.36,1],
                name='table', body_type='static', initial_pose=sapien.Pose(p=[-.15,0,-.025]))
            self.cube = actors.build_cube(self.scene, half_size=.025, color=[.85,.08,.04,1],
                name='task_cube', initial_pose=sapien.Pose(p=[.065,0,.025]))

        def _initialize_episode(self, env_idx, options):
            self.agent.robot.set_pose(sapien.Pose(p=[-.55,0,0]))
            named_home = dict(zip(arm+hand, home))
            actual_order = [j.name for j in self.agent.robot.get_active_joints()]
            self.agent.reset(np.array([named_home[name] for name in actual_order]))
            self.cube.set_pose(sapien.Pose(p=[.065,0,.025]))

        def evaluate(self):
            return {}

        def _get_obs_extra(self, info):
            return {}

    return arm, hand


def run(output, urdf, probe_only=False):
    process_started = time.monotonic()
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    lock_path = output.parents[1]/'gpu2.lock'
    # All dexterous runs live .../report_control_20261021/dexterous/ATTEMPT.
    lock_path = output.parent.parent/'gpu2.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path.insert(0,str(WORKSPACE/'PolicyGen'))
        from examples.baselines.hypericl_task_expansion.common import bind_gpu
        gpu = bind_gpu(2)
        if gpu['uuid'] != 'GPU-807bd44d-36ea-31d0-427c-50e264428c67':
            raise RuntimeError('GPU2 UUID changed')
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name',
                                        '--format=csv,noheader'],text=True)
        if gpu['uuid'] in apps:
            raise RuntimeError('GPU2 has an existing compute process; no launch')
        save(output/'gpu_before.json',{'gpu':gpu,'compute_apps_snapshot':apps,'lock':str(lock_path)})
        import numpy as np
        import gymnasium as gym
        import sapien
        from PIL import Image
        import imageio.v2 as imageio
        from transforms3d.euler import euler2quat
        arm_names, hand_names = register(urdf)
        started = time.monotonic()
        env = None
        try:
            env = gym.make('CandidatePandaAllegroContact-v1', sim_backend='physx_cpu',
                render_backend=gpu['render_backend'], obs_mode='rgb', control_mode='pd_joint_pos',
                reward_mode='none', sensor_configs={'shader_pack':'rt-fast'})
            obs,_ = env.reset(seed=421)
            u = env.unwrapped; robot = u.agent.robot
            names = [j.name for j in robot.get_active_joints()]
            controller_names = [j.name for j in u.agent.controller.joints]
            qlim = robot.get_qlimits()[0].cpu().numpy()
            if len(names)!=23 or set(names)!=set(arm_names+hand_names):
                raise ValueError('Runtime DOF/name mismatch')
            pino = robot.create_pinocchio_model()
            save(output/'runtime_model.json',{'urdf':binding(urdf),'dof':len(names),
                'active_joint_names':names,'controller_joint_names':controller_names,
                'action_dimension':int(env.action_space.shape[-1]),'joint_limits':qlim.tolist(),
                'arm_dof':7,'hand_dof':16,'is_grasping_override_used':False,
                'self_collisions_disabled':u.agent.disable_self_collisions,
                'pino_ik_signature':str(pino.compute_inverse_kinematics.__doc__),
                'implementation':binding(__file__)})
            frames=[]; actions=[]; qs=[]; qvs=[]; objects=[]; tcp=[]; forces=[]; stages=[]
            def capture(stage, action):
                observation = u.get_obs()
                a = observation['sensor_data']['overview']['rgb'][0].cpu().numpy()
                b = observation['sensor_data']['close']['rgb'][0].cpu().numpy()
                frames.append(np.concatenate([a,b],axis=1))
                actions.append(action.copy());qs.append(robot.qpos[0].cpu().numpy().copy())
                qvs.append(robot.qvel[0].cpu().numpy().copy());objects.append(u.cube.pose.raw_pose[0].cpu().numpy().copy())
                tcp.append(u.agent.tcp.pose.raw_pose[0].cpu().numpy().copy());stages.append(stage)
                groups=[]
                for links in u.agent.finger_groups.values():
                    groups.append(sum(float(np.linalg.norm(u.scene.get_pairwise_contact_forces(link,u.cube)[0].cpu().numpy())) for link in links))
                forces.append(groups)
            current = robot.qpos[0].cpu().numpy().copy()
            capture('initialized', current)
            Image.fromarray(frames[0]).save(output/'initial.png')
            if probe_only:
                save(output/'result.json',{'model_loaded':True,'dof':23,'action_dimension':23,
                     'control_trajectory_executed':False,'grasp_success':False,'wall_seconds':time.monotonic()-started})
                return
            link_names = [x.name for x in robot.get_links()]
            tip_index = link_names.index('allegro_base_link')
            mask = np.array([name in arm_names for name in names],dtype=bool)
            arm_idx = [names.index(n) for n in arm_names]
            hand_idx = [names.index(n) for n in hand_names]
            base = robot.pose.sp
            planned=[]
            def ik(point, seed):
                target = base.inv()*sapien.Pose(p=point,q=euler2quat(np.pi,0,0))
                q,ok,error=pino.compute_inverse_kinematics(tip_index,target,initial_qpos=seed,
                active_qmask=mask.astype(np.int32),max_iterations=300)
                planned.append({'target_world_m':point,'success':bool(ok),'error':np.asarray(error).tolist()})
                if not ok:raise RuntimeError('IK failed for declared waypoint')
                return np.asarray(q)
            def move(target,stage,steps):
                nonlocal current
                start=current.copy()
                for i in range(1,steps+1):
                    q=start+(target-start)*(i/steps)
                    if np.any(q<qlim[:,0]-1e-5) or np.any(q>qlim[:,1]+1e-5):
                        raise RuntimeError('Command joint limits violated')
                    action=np.array([q[names.index(n)] for n in controller_names],dtype=np.float32)
                    env.step(action);capture(stage,action)
                current=target.copy()
            home=current.copy()
            pre=ik([0,0,.27],current);move(pre,'approach',70)
            grasp=ik([0,0,.135],current);move(grasp,'descend',45)
            close=current.copy()
            close[hand_idx]=np.array([0,.75,.7,.6,0,.8,.75,.6,0,.8,.75,.6,1.1,.4,.6,.65])
            move(close,'close_fingers',50)
            lift=ik([0,0,.27],current);lift[hand_idx]=close[hand_idx]
            move(lift,'lift',55)
            move(lift,'hold',35)
            opened=current.copy();opened[hand_idx]=home[hand_idx]
            move(opened,'release',25)
            move(opened,'settle',20)
            obj=np.asarray(objects);force=np.asarray(forces);stage=np.asarray(stages)
            speed=np.r_[0,np.linalg.norm(np.diff(obj[:,:3],axis=0),axis=1)*20]
            good=(obj[:,2]>.075)&(force[:,3]>.05)&(force[:,:3].max(axis=1)>.05)&(speed<.05)&(stage=='hold')
            longest=run_length=0
            for val in good:
                run_length=run_length+1 if val else 0;longest=max(longest,run_length)
            np.savez_compressed(output/'trajectory.npz',actions=actions,qpos=qs,qvel=qvs,object_pose=obj,
                tcp_pose=tcp,finger_contact_force_N=force,stages=stage,timestamps=np.arange(len(stage))/20)
            imageio.mimsave(output/'trajectory.mp4',frames,fps=20,codec='libx264',quality=8)
            for tag in ('approach','descend','close_fingers','lift','hold','release','settle'):
                indices=np.flatnonzero(stage==tag)
                Image.fromarray(frames[int(indices[-1])]).save(output/(tag+'.png'))
            save(output/'ik_waypoints.json',planned)
            save(output/'result.json',{'schema':'dexterous_candidate_actual_attempt_v1','model_loaded':True,
                'control_trajectory_executed':True,'grasp_success':longest>=20,
                'acceptance_protocol':binding(output.parent/'acceptance_protocol.json'),
                'stable_multifinger_lift_hold_frames':int(longest),'max_cube_height_m':float(obj[:,2].max()),
                'max_contact_force_by_group_N':force.max(axis=0).tolist(),'finger_groups':['index','middle','ring','thumb'],
                'control_frames':len(stage),'dof':23,'action_dimension':23,'simulated_seconds':(len(stage)-1)/20,
                'wall_seconds':time.monotonic()-started,'total_process_wall_seconds':time.monotonic()-process_started,
                'gpu':gpu,'api_calls':0,
                'limitations':['Open-loop IK waypoints; no independent dense whole-arm collision certificate.',
                    'Fixed hand coupling is a simulation design, not a physical hardware qualification.',
                    'Contact is sampled at control-frame ends; no RGB inference is treated as physical truth.']})
        except BaseException as exc:
            save(output/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),
                                      'wall_seconds':time.monotonic()-started,'api_calls':0})
            raise
        finally:
            if env is not None: env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--urdf',required=True)
    p.add_argument('--probe-only',action='store_true');a=p.parse_args()
    run(a.output,a.urdf,a.probe_only)
