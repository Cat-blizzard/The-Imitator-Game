"""Native passive-joint/tool experiments on the retained dual-Panda recorder."""
from copy import deepcopy
import os
from pathlib import Path
import time

import numpy as np

from .core import bind, checked, implementation, read, require, write
from .simulation import Recorder, array
from .timing import span, Accumulator


def register(spec, performance):
    import sapien
    from scipy.spatial.transform import Rotation
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.types import SimConfig, SceneConfig
    from .simulation import register as table
    base=deepcopy(spec)
    # Keep the retained table/robot/camera setup. Build new mechanism assets
    # with density so native mass AND inertia agree; set_mass alone does not.
    base["objects"]=[]
    parent=table(base,performance)

    @register_env("ReSimPairNovelMechanisms-v1",max_episode_steps=4000)
    class Novel(parent):
        @property
        def _default_sim_config(self):
            return SimConfig(sim_freq=200,control_freq=20,scene_config=SceneConfig(
                contact_offset=.002,solver_position_iterations=25,solver_velocity_iterations=4))

        def _load_scene(self,options):
            self.scene_spec=base
            super()._load_scene(options)
            by_id={}
            for item in spec["objects"]:
                builder=self.scene.create_actor_builder()
                density=item["geometry"]["density_kg_m3"] or 1000.
                if item["shape"]=="sphere":
                    if item.get('collision_enabled', True):
                        builder.add_sphere_collision(radius=item["radius"],density=density)
                    builder.add_sphere_visual(radius=item["radius"],material=item["rgba"])
                else:
                    for part in item.get("parts",[{"center":[0,0,0],"half_size":item["half_size"]}]):
                        pose=sapien.Pose(part["center"])
                        if item.get('collision_enabled', True):
                            builder.add_box_collision(pose=pose,half_size=part["half_size"],density=density)
                        builder.add_box_visual(pose=pose,half_size=part["half_size"],material=item["rgba"])
                builder.initial_pose=sapien.Pose(item["position"],Rotation.from_euler("z",item["yaw"]).as_quat(scalar_first=True))
                actor=builder.build(item["id"]) if item["kind"]=="movable" else builder.build_static(item["id"])
                if item["shape"]=="sphere" and item["kind"]=="movable":
                    # Native auto-computation floors tiny inertia at 1e-6.
                    # Set the declared uniform-sphere value during construction,
                    # before reset/stepping; no runtime state/force intervention.
                    for body in actor._bodies:body.inertia=[.4*item["mass_kg"]*item["radius"]**2]*3
                by_id[item["id"]]=actor
            self.scene_spec=spec
            self.objects=[by_id[o["id"]] for o in spec["objects"]]
            self.object_ids=[o["id"] for o in spec["objects"]]
            self.articulations=[];self.door_links=[]
            for item in spec["joints"]:
                builder=self.scene.create_articulation_builder()
                root=builder.create_link_builder();root.set_name(item["id"]+"_anchor")
                link=builder.create_link_builder(root);link.set_name(item["id"]);link.set_joint_name(item["id"]+"_joint")
                for part in item["parts"]:
                    pose=sapien.Pose(part["center"])
                    link.add_box_collision(pose=pose,half_size=part["half_size"])
                    link.add_box_visual(pose=pose,half_size=part["half_size"],material=item["rgba"])
                rot=Rotation.align_vectors([item["axis_world"]],[[1.,0.,0.]])[0].as_quat(scalar_first=True)
                link.set_joint_properties(item["joint_type"],[item["limits"]],
                    pose_in_parent=sapien.Pose(item["pivot_world"],rot),pose_in_child=sapien.Pose(q=rot),
                    friction=item["friction"],damping=item["damping"])
                # Inertia and mass are engineering design inputs; no real-object claim.
                link.set_mass_and_inertia(item["mass_kg"],sapien.Pose(),[.00012,.00012,.00012])
                builder.initial_pose=sapien.Pose()
                articulation=builder.build(fix_root_link=True,name=item["id"]+"_assembly")
                for joint in articulation.active_joints:
                    joint.set_drive_properties(stiffness=item["spring_N_or_Nm"],damping=item["damping"])
                    joint.set_drive_target(item["rest_q"])
                    joint.set_friction(item["friction"])
                self.articulations.append(articulation)
                self.door_links.append(next(l for l in articulation.get_links() if l.name==item["id"]))
            self.support_ids=self.object_ids+[s["id"] for s in spec["supports"]]+["table"]
            self.support_bodies={o._bodies[0].entity:i for i,o in enumerate(self.objects+self.support_actors+[self.table_scene.table])}
            self.object_bodies={o._bodies[0].entity:i for i,o in enumerate(self.objects)}
            self.targets=self.objects+self.door_links
            self.target_ids=self.object_ids+[j["id"] for j in spec["joints"]]
            self.target_bodies={o._bodies[0].entity:i for i,o in enumerate(self.targets)}
            n=len(self.objects);nt=len(self.targets)
            self.support_impulses=np.zeros((n,len(self.support_ids)))
            self.robot_object_force=np.zeros((2,n))
            self.robot_contact_impulses=np.zeros((2,nt))
            self.object_contact_impulses=np.zeros((nt,nt))

        def _initialize_episode(self,env_idx,options):
            super()._initialize_episode(env_idx,options)
            for a,item in zip(self.articulations,spec["joints"]):
                a.set_qpos(np.array([item["initial_q"]]));a.set_qvel(np.zeros(1))

        def _before_control_step(self):
            super()._before_control_step()
            self.robot_contact_impulses[:]=0.;self.object_contact_impulses[:]=0.

        def _after_simulation_step(self):
            super()._after_simulation_step()
            for contact in self.scene.get_contacts():
                a,b=[x.entity for x in contact.bodies]
                impulse=sum(float(np.linalg.norm(x.impulse)) for x in contact.points)
                if a in self.target_bodies and b in self.target_bodies:
                    i,j=self.target_bodies[a],self.target_bodies[b]
                    self.object_contact_impulses[i,j]+=impulse;self.object_contact_impulses[j,i]+=impulse
                for first,second in [(a,b),(b,a)]:
                    if first in self.robot_bodies and second in self.target_bodies:
                        self.robot_contact_impulses[self.robot_bodies[first],self.target_bodies[second]]+=impulse
    return Novel


class NovelRecorder(Recorder):
    def _measure(self):
        result=super()._measure();b=self.unwrapped
        result.update(cabinet_qpos=np.array([array(a.qpos).reshape(-1)[0] for a in b.articulations]),
            cabinet_qvel=np.array([array(a.qvel).reshape(-1)[0] for a in b.articulations]),
            door_world=np.array([array(o.pose.to_transformation_matrix()).reshape(4,4) for o in b.door_links]).reshape(-1,4,4),
            robot_contact_impulses=b.robot_contact_impulses.copy(),object_contact_impulses=b.object_contact_impulses.copy())
        result["tool_tips_world"]=np.array([(array(o.pose.to_transformation_matrix()).reshape(4,4) @
            np.r_[spec.get("tip_local",[0.,0.,0.]),1.])[:3] for o,spec in zip(b.objects,b.scene_spec["objects"])])
        return result


def worker(root,case_id,attempt):
    from .simulation import case_inputs
    from .workbench import export
    from .batch_protocols import protocol
    root=Path(root);experiment,case,scene,task=case_inputs(root,case_id)
    directory=root/"executions"/case_id/attempt
    require(os.environ.get("RESIMPAIR_RESERVATION")==str(directory/"reservation.json"),"supervisor_reservation_required")
    for value in read(directory/"reservation.json")["implementation"].values():checked(value)
    plan=protocol(scene,task)
    from .functional_assets import preflight
    write(directory/"functional_asset_preflight.json",preflight(scene))
    write(directory/"event_protocol.json",plan)
    write(directory/"planning_declaration.json",{"event_protocol":bind(directory/"event_protocol.json"),"case":case,
          "scope":"before_physical_execution","declared_unix":time.time(),"implementation":implementation(),
          "manual_generation_prose":False,"shared_initial_scene":case["scene"]})
    from examples.baselines.hypericl_task_expansion.common import bind_gpu
    gpu=bind_gpu(experiment["gpu"])
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    import gymnasium as gym
    env=rec=None;failure=bundle=None;performance=Accumulator()
    try:
        with span(directory,"simulation_environment_setup"):
            register(scene,performance)
            env=gym.make("ReSimPairNovelMechanisms-v1",sim_backend="physx_cpu",render_backend=gpu["render_backend"],
                         control_mode="pd_joint_pos",obs_mode="rgb",reward_mode="none",render_mode="rgb_array",
                         sensor_configs={"shader_pack":"rt-fast"})
        with span(directory,"simulation_reset_and_recorders"):
            performance.phase="reset";rec=NovelRecorder(env,directory);rec.reset(scene["seed"])
            dynamics=[];collision_shapes=[]
            for actor,item in zip(rec.unwrapped.objects,scene["objects"]):
                shape_count=sum(len(body.collision_shapes) for body in actor._bodies)
                declared=item.get('collision_enabled',True)
                require(bool(shape_count)==declared, 'native_collision_policy_mismatch:'+item['id'])
                collision_shapes.append({'id':item['id'],'kind':item['kind'],
                    'declared_collision_enabled':declared,'native_collision_shape_count':shape_count})
                if item["kind"]!="movable":continue
                body=actor._bodies[0]
                row={"id":item["id"],"native_mass_kg":float(body.mass),"native_inertia_kg_m2":array(body.inertia).tolist(),
                     "native_com_local":body.cmass_local_pose.p.tolist(),"expected_mass_kg":item["mass_kg"],
                     "geometry":item["geometry"]}
                row["native_shape_materials"]=[{"density_kg_m3":float(shape.density),
                    "static_friction":float(shape.physical_material.static_friction),
                    "dynamic_friction":float(shape.physical_material.dynamic_friction),
                    "restitution":float(shape.physical_material.restitution)} for shape in body.collision_shapes]
                require(np.isclose(body.mass,item["mass_kg"],rtol=1e-4),"native_mass_mismatch:"+item["id"])
                if item["shape"]=="sphere":
                    require(np.allclose(body.inertia,.4*item["mass_kg"]*item["radius"]**2,rtol=1e-4,atol=1e-10),
                            "native_sphere_inertia_mismatch:"+item["id"])
                dynamics.append(row)
            write(directory/"asset_runtime_evidence.json",{"assets":dynamics,"source":"native PhysX bodies after reset"})
            write(directory/'collision_policy_evidence.json',{'objects':collision_shapes,
                'source':'native PhysX shape counts after reset; no object state changes'})
        with span(directory,"planning_execution_and_rendering"):
            performance.phase="execution"
            if scene.get("contact_extension"):
                from .contact_extensions import solve
            else:
                from .novel_solvers import solve
            solve(rec,task)
    except Exception as exc:
        import traceback
        failure={"type":type(exc).__name__,"message":str(exc),"trace":[{"file":f.filename,"line":f.lineno,"function":f.name} for f in traceback.extract_tb(exc.__traceback__)]}
        write(directory/"planning_failure.json",failure)
    finally:
        performance.phase="finalize"
        if rec is not None:
            with span(directory,"simulation_video_finalize"):rec.finish()
            with span(directory,"simulation_measurement_export"):bundle=export(rec,case,task,scene,directory,failure)
        if env is not None:env.close()
        performance.persist(directory/"timing_breakdown.json")
    result={"bundle":bundle,"planning_failure":failure,"frames":len(rec.rows) if rec else 0,
            "status":"planned_and_executed" if failure is None else "planning_or_execution_failed"}
    write(directory/"worker_result.json",result)
    return result
