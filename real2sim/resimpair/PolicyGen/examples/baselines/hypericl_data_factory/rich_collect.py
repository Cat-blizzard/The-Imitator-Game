"""Pilot richer assets with reusable, scene-conditioned native Panda experts.

This is a new asset/randomization version of relocation programs, not a claim
of new task semantics. No original environment or collector source is edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from .bootstrap import ROOT,sha,write
from ..hypericl_task_expansion.common import bind_gpu,source_pins


def register():
    import numpy as np
    import sapien
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.scene_builder.table import TableSceneBuilder
    from ..hypericl_task_expansion.envs import TaskExpansionEnv

    @register_env('HyperICLRichAssetRelocation-v1',max_episode_steps=3000)
    class RichRelocationEnv(TaskExpansionEnv):
        def __init__(self,*args,asset_pool,family='bottle',**kwargs):
            self.asset_pool=asset_pool;self.factory_family=family
            super().__init__(*args,scene_name='tableware',**kwargs)

        def _load_scene(self,options):
            seed=int(options.get('layout_seed',0));rng=np.random.default_rng(seed+730001)
            chosen=rng.choice(len(self.asset_pool),size=2,replace=False)
            rows=[self.asset_pool[int(i)]for i in chosen]
            material_rows=json.loads((ROOT.parent/'open_assets/task_factory_20261016_v1/polyhaven_downloads.json').read_text())['assets']
            surfaces=[r for r in material_rows if r.get('kind')=='Diffuse']
            hdris=[r for r in material_rows if r.get('kind')=='hdri']
            surface=surfaces[int(rng.integers(len(surfaces)))];hdri=hdris[int(rng.integers(len(hdris)))]
            self.randomization=dict(schema='rich_asset_relocation_v1',layout_seed=seed,
                asset_uids=[r['uid']for r in rows],source_category=self.factory_family,
                masses_kg=[r['adapter']['estimated_mass_kg']*float(rng.uniform(.85,1.15))for r in rows],
                friction=float(rng.uniform(.65,.95)),ambient_rgb=rng.uniform(.26,.40,3).tolist(),
                diffuse_surface=surface,hdri=hdri,
                camera=dict(eye=(np.array([-.4,0.,.65])+rng.uniform([-.025,-.035,-.025],[.025,.035,.025])).tolist(),
                            target=[0.,0.,.10],fov_radians=float(rng.uniform(1.66,1.78)),fixed_during_episode=True),
                appearance_sampled_independently_of_goal=True,shared_native_layout_randomization='object XY +/-2.5cm, yaw +/-0.28rad; trays XY +/-1.8cm')
            self.scene_name='rich_'+self.factory_family
            self.asset_specs=[]
            for row,mass in zip(rows,self.randomization['masses_kg']):
                a=row['adapter']
                if not a['top_obb_width_fits_panda']:raise ValueError('asset exceeds pilot OBB grasp aperture')
                self.asset_specs.append(dict(name=row['name'],category='objaverse:'+row['uid'],uid=row['uid'],
                    model_id=0,rotation_x_deg=0.,uniform_scale=1.,bounds_rotated_m=a['bounds_m'],
                    source=row['source'],license=row['license'],estimated_mass_kg=mass,
                    visual=a['visual'],collision=a['collision'],
                    files={a['visual']:a['visual_sha256'],a['collision']:a['collision_sha256']},
                    collision_scope='Convex whole-object relocation; not cavity-dependent interaction'))
            self.table_scene=TableSceneBuilder(self,robot_init_qpos_noise=0.);self.table_scene.build()
            self.objects=[];self.bins=[]
            contact=sapien.physx.PhysxMaterial(self.randomization['friction'],self.randomization['friction'],0.)
            for i,spec in enumerate(self.asset_specs):
                b=self.scene.create_actor_builder()
                b.add_visual_from_file(spec['visual'])
                b.add_convex_collision_from_file(spec['collision'],material=contact)
                b.initial_pose=sapien.Pose([0,0,.15]);actor=b.build(f'object_{i}')
                actor.set_mass(spec['estimated_mass_kg']);self.objects.append(actor)
            for j,color in enumerate(([.65,.30,.16,1.],[.12,.32,.62,1.])):
                b=self.scene.create_actor_builder()
                shapes=[([.125,.095,.006],[0,0,.006])]
                shapes += [([.005,.095,.021],[x,0,.021])for x in(-.12,.12)]
                shapes += [([.115,.005,.021],[0,y,.021])for y in(-.09,.09)]
                for half,center in shapes:
                    pose=sapien.Pose(center);b.add_box_collision(pose=pose,half_size=half)
                    b.add_box_visual(pose=pose,half_size=half,material=color)
                b.initial_pose=sapien.Pose(self.goal_centers[j]);self.bins.append(b.build_static(f'container_{j}'))
            # This surface changes RGB only. The original table collider remains.
            material=sapien.render.RenderMaterial();material.set_base_color([1,1,1,1])
            material.set_base_color_texture(sapien.render.RenderTexture2D(surface['path']))
            material.set_roughness(.75)
            b=self.scene.create_actor_builder();b.add_box_visual(half_size=[.65,.65,.00005],material=material)
            b.initial_pose=sapien.Pose([0,0,.0001]);b.build_static('appearance_only_table_surface')

        def _load_lighting(self,options):
            super()._load_lighting(options)
            self.scene.set_ambient_light(self.randomization['ambient_rgb'])
            for scene in self.scene.sub_scenes:scene.set_environment_map(self.randomization['hdri']['path'])

        @property
        def _default_sensor_configs(self):
            c=self.randomization['camera']
            return [CameraConfig('zed2i',sapien_utils.look_at(c['eye'],c['target']),224,224,c['fov_radians'],.01,100)]

        def _initialize_episode(self,env_idx,options):
            from ..hypericlzq.eef import panda_fk,panda_ik
            super()._initialize_episode(env_idx,options)
            # Taller real assets must not overlap the old low home gripper.
            # Change initialization through checked IK before physics settling;
            # recorded manipulation still uses native PD commands exclusively.
            homes=[]
            for agent in self.agent.agents:
                q=agent.robot.qpos.detach().cpu().numpy().reshape(9).copy()
                desired=panda_fk(q[:7]);desired[2,3]+=.16
                arm,ok,_=panda_ik(desired,q[:7])
                if not bool(ok):raise ValueError('raised collision-clear home IK failed')
                q[:7]=arm;agent.reset(q);homes.append(q.tolist())
            self.randomization['initial_robot_joint_positions']=homes
            self.randomization['initial_tcp_raise_m']=.16


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'rich_asset_pilot_v1')
    p.add_argument('--family',choices=['bottle','cup','book'],default='bottle')
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--seed-base',type=int,default=40000000)
    p.add_argument('--layouts',type=int,default=2)
    p.add_argument('--programs',nargs='+',default=['pick_first_to_left','pick_first_to_right'])
    args=p.parse_args();gpu=bind_gpu(args.gpu)
    import gymnasium as gym
    import numpy as np
    from ..hypericl_task_expansion.collect import Recorder,solve,qualify,save,save_diagnostic,require,array
    from ..hypericl_task_expansion.registry import PROGRAMS
    args.root.mkdir(parents=True,exist_ok=False)
    if any(p not in PROGRAMS for p in args.programs):raise ValueError('unknown program')
    review=ROOT/'asset_visual_review_v1.json';adapters=ROOT/'asset_adapters_v2/catalog.json'
    approved={r['uid']for r in json.loads(review.read_text())['assets']if r['approved_for_interaction_test']}
    pool=[r for r in json.loads(adapters.read_text())['assets']if r.get('uid')in approved and
          args.family in r.get('categories',[])and r['adapter']['top_obb_width_fits_panda']]
    if len(pool)<2:raise ValueError('need two visually reviewed grasp-width candidates')
    pins=source_pins();pins[str(Path(__file__).resolve())]=sha(__file__)
    pins[str(review)]=sha(review);pins[str(adapters)]=sha(adapters)
    archive={}
    for filename,digest in pins.items():
        source=Path(filename);dest=args.root/'source_snapshot'/(digest+source.suffix)
        dest.parent.mkdir(exist_ok=True);shutil.copyfile(source,dest)
        if sha(dest)!=digest:raise ValueError('source snapshot hash mismatch')
        archive[filename]=dict(path=str(dest),sha256=digest)
    write(args.root/'source_archive.json',archive)
    write(args.root/'plan.json',dict(schema='rich_asset_relocation_pilot_v1',family=args.family,
        asset_uids=[r['uid']for r in pool],layouts=args.layouts,seed_base=args.seed_base,programs=args.programs,
        source_sha256=pins,qualification_only=True,production_approved=False,
        program_structure=['scene-conditioned grasp/approach','physical grasp and lift validation',
                           'scene-conditioned transfer','place/release','retreat','settle and qualify'],
        per_waypoint_planning='screw interpolation; bounded RRTConnect fallback',
        failures_preserved=True,original_benchmark_modified=False,model_training_started=False))
    register()
    class SettledRecorder(Recorder):
        def hold(self,steps,grippers=(1,1)):
            super().hold(steps,grippers)
            if not self.recording:
                consecutive=0
                for _ in range(120):
                    stable=all(float(np.linalg.norm(array(o.linear_velocity)))<.01 and
                               float(np.linalg.norm(array(o.angular_velocity)))<.10 for o in self.unwrapped.objects)
                    consecutive=consecutive+1 if stable else 0
                    if consecutive>=10:break
                    super().hold(1,grippers)
                require(consecutive>=10,'No physically stable initial pose within bounded settling budget')
    env=gym.make('HyperICLRichAssetRelocation-v1',asset_pool=pool,family=args.family,
        program=args.programs[0],sim_backend='physx_cpu',render_backend=gpu['render_backend'],
        control_mode='pd_joint_pos',obs_mode='rgb',reward_mode='none',render_mode='rgb_array',
        sensor_configs=dict(shader_pack='rt-fast'),human_render_camera_configs=dict(shader_pack='rt-fast'))
    results=[];started=time.time()
    try:
        for seed in range(args.seed_base,args.seed_base+args.layouts):
            for program in args.programs:
                # Recorder.reset restores physics but does not clear its row,
                # frame or command lists. Never share a Recorder across clips.
                recorder=SettledRecorder(env)
                folder=args.root/'episodes'/f'{seed}_{program}'
                try:
                    recorder.reset(seed,program);events=solve(recorder,program)
                    metrics,final=qualify(recorder,program);require(metrics['success'],'physical goal checks failed')
                    receipt=save(recorder,folder,recorder.unwrapped.scene_name,program,'qualification',seed,events,metrics,final,pins)
                    receipt.update(qualification_only=True,production_approved=False,
                        randomization=recorder.unwrapped.randomization,asset_visual_review_sha256=sha(review),
                        initial_measured_states=recorder.initial['states'].tolist(),
                        initial_measured_object_states=recorder.initial['objects'].tolist(),
                        fresh_recorder_per_episode=True)
                    write(folder/'receipt.json',receipt)
                    results.append(dict(seed=seed,program=program,passed=True,receipt=str(folder/'receipt.json')))
                except Exception as exc:
                    if recorder.rows:
                        try:save_diagnostic(recorder,folder)
                        except Exception as diagnostic:write(folder/'diagnostic_failure.json',dict(error=repr(diagnostic)))
                    failure=dict(seed=seed,program=program,passed=False,error=repr(exc),qualification_only=True)
                    write(folder/'failure.json',failure);results.append(failure)
                write(args.root/'state.json',dict(status='qualifying_new_asset_and_background_pipeline',completed=len(results),
                    attempts=args.layouts*len(args.programs),passed=sum(r['passed']for r in results),elapsed_seconds=time.time()-started,
                    results=results,updated_unix=time.time()))
    finally:env.close()
    for path,digest in pins.items():
        if sha(path)!=digest:raise RuntimeError(('pinned source changed',path))
    report=dict(status='pilot_complete_requires_independent_video_and_physics_review',attempted=len(results),
        passed=sum(r['passed']for r in results),failed=sum(not r['passed']for r in results),results=results,
        production_approved=False,elapsed_seconds=time.time()-started)
    write(args.root/'summary.json',report);write(args.root/'state.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
