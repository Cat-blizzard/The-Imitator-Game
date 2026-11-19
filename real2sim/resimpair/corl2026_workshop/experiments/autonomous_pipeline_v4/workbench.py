"""Planning-side multi-mechanism scene with task-independent initial assets.

Every task uses the same physical scene and initial state. Mechanism-specific
planning ends at the standard v5 bundle; generation never imports this module.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import os
import time

import numpy as np

from .core import bind, checked, digest, implementation, read, require, write
from . import planning_protocols as p
from .event_protocol import validate
from .simulation import Recorder, array
from .timing import span


def make_scene(scene_id, seed):
    from .library import CAMERAS
    rng = np.random.default_rng(seed)
    scene = {"schema": "resimpair_mechanism_workbench_v1", "id": scene_id, "seed": seed,
             "table_top_z": 0., "robot_roots": [[-.60, -.35, 0.], [-.60, .35, 0.]],
             "cameras": deepcopy(CAMERAS), "objects": [], "supports": [],
             "appearance": {"table_rgb": [.54, .52, .47], "background_rgb": [.73, .79, .82],
                            "ambient_rgb": [.35]*3, "key_light_rgb": [1.5]*3},
             "randomization": {"seed_independent_of_task": True, "source_layout_copied": False}}
    def box(oid, position, half, rgba, description, mass=.04):
        return {"id": oid, "kind": "movable", "shape": "box", "half_size": half,
                "position": position, "yaw": 0., "mass_kg": mass, "rgba": rgba,
                "semantic_description": description, "geometry": {"half_size_m": half, "rgba": rgba}}
    row = [.075 + float(rng.uniform(-.005, .005)), -.085, .0505]
    for i in range(5):
        scene["objects"].append(box(f"domino_{i}", [row[0]+(i-2)*.033, row[1], row[2]],
            [.006, .024, .050], [.89, .72, .29, 1.], f"ochre domino {i+1} of the ordered row", .025))
    ports = [[.055, -.014, .022], [.055, .014, .022]]
    receiver = box("receiver", [-.14, -.31, .0225], [.085, .03, .022], [.25, .30, .36, 1.],
                   "dark gray rectangular receiver with two separate open sockets", .10)
    receiver["shape"] = "compound_boxes"
    receiver["yaw"] = np.pi/2
    receiver["parts"] = []
    for lo, hi in [([-.085,-.03,-.022],[.0465,.03,.022]), ([.0635,-.03,-.022],[.085,.03,.022]),
                   ([.0465,-.03,-.022],[.0635,-.0185,.022]), ([.0465,-.0095,-.022],[.0635,.0095,.022]),
                   ([.0465,.0185,-.022],[.0635,.03,.022]), ([.0465,-.03,-.022],[.0635,.03,-.003])]:
        a, b = np.asarray(lo), np.asarray(hi)
        receiver["parts"].append({"center": ((a+b)/2).tolist(), "half_size": ((b-a)/2).tolist()})
    receiver["geometry"] = {"outer_half_m": [.085,.03,.022], "ports": ports,
        "socket_half_xy_m": [.0085,.0045], "socket_floor_z_m": -.003, "color": "dark gray"}
    plug = box("plug", [-.15,.19,.0425], [.018,.014,.018], [.85,.85,.82,1.],
               "ivory rigid plug with one narrow silver tongue extending below its body", .025)
    plug["shape"] = "compound_boxes"
    plug["parts"] = [{"center": [0.,0.,0.], "half_size": [.018,.014,.018]},
                     {"center": [0.,0.,-.030], "half_size": [.0065,.0025,.012], "rgba": [.55,.58,.62,1.]}]
    plug["geometry"] = {"body_half_m": [.018,.014,.018], "tongue_half_m": [.0065,.0025,.012],
                        "tongue_center_local": [0.,0.,-.030], "color": "ivory and silver"}
    scene["objects"] += [receiver, plug]
    scene["objects"] += [box("green_block", [.26,-.36,.0255], [.025]*3, [.13,.54,.26,1.], "green background cube"),
                         box("blue_block", [.26,-.23,.0255], [.025]*3, [.12,.28,.76,1.], "blue background cube")]
    # A graspable passive lid adds a third mechanism without imposing the real
    # source's furniture layout or claiming exact source-episode equivalence.
    scene["lid"] = {"id": "hinged_lid", "pivot_world": [.21,.36,.085], "axis_world": [0.,1.,0.],
        "panel_center_local": [-.10,0.,0.], "panel_half_size": [.10,.075,.006],
        "handle_center_local": [-.178,0.,.025], "handle_half_size": [.017,.014,.014],
        "rgba": [.58,.25,.13,1.], "handle_rgba": [.77,.79,.81,1.], "mass_kg": .055,
        "joint_limits": [0.,1.55], "joint_friction": .035, "joint_damping": .04}
    # The lid rests on a shallow, genuinely hollow box. No contained object is
    # invented in an occluded region; visible socket and hinge geometry persists.
    scene["supports"] = [{"id": "lid_box", "shape": "tray", "half_size": [.10,.075,.006],
        "position": [.11,.36,0.], "top_z": .079, "wall_height": .079,
        "rgba": [.40,.34,.28,1.], "semantic_description": "shallow brown box under the hinged lid"}]
    return scene


def task_intent(scene, case_id, family, *, port=1, direction=1):
    selected = ([f"domino_{i}" for i in range(5)] if family == "passive_chain" else
                ["receiver", "plug"] if family == "insertion" else ["hinged_lid"])
    skills = {"passive_chain": ["push"], "insertion": ["pick_place", "insert"], "articulation": ["articulate"]}
    require(family in skills, "Unsupported workbench planning mechanism")
    return {"schema": "resimpair_task_intent_v4", "id": case_id, "family": family,
        "required_skills": skills[family], "object_ids": selected,
        "operations": [{"id": "mechanism_action", "primitive": family, "entities": selected,
                        "selected_port": port if family == "insertion" else None,
                        "direction": direction if family == "passive_chain" else None}],
        "required_operation_precedence": [], "equivalence_groups": [],
        "human_hand_policy": "preserve_robot_body_side",
        "non_target_policy": "preserve_identity_and_pose_for_full_execution",
        "planning_parameters": {"port": port, "direction": direction, "arm": 0 if family == "passive_chain" else 1},
        "scope": "declared_composition_of_automatically_observed_source_skills"}


def protocol(scene, task):
    objects = scene["objects"]
    target_ids = set(task["object_ids"])
    entities = [{"id": o["id"], "description": o["semantic_description"], "kind": o["shape"],
        "role": "target" if o["id"] in target_ids else "unchanged", "geometry": o["geometry"],
        "pose_channel": o["id"]+".pose"} for o in objects]
    lid = scene["lid"]
    entities += [{"id": lid["id"], "description": "brown hinged lid with a silver top handle",
        "kind": "articulated_link", "role": "target" if lid["id"] in target_ids else "unchanged",
        "geometry": {k: lid[k] for k in ["panel_center_local","panel_half_size","handle_center_local",
                                       "handle_half_size","axis_world","joint_limits"]},
        "pose_channel": lid["id"]+".pose"}]
    entities += [{"id": s["id"], "description": s["semantic_description"], "kind": s["shape"], "role": "support",
                  "geometry": {"half_size_m": s["half_size"], "height_m": s["wall_height"]},
                  "fixed_position_world": s["position"]} for s in scene["supports"]]
    entities += [{"id":"table", "description":"the tabletop", "kind":"plane", "role":"support", "geometry":{}}]
    plan = p.base(task["id"], task["family"], entities, task["required_skills"], 12 if task["family"]=="insertion" else 8)
    ids = [o["id"] for o in objects]
    poses, grasps = p.rigid_channels(plan, ids)
    joint = p.channel(plan, "hinged_lid.joint", "cabinet_qpos", [0], "joint_position", [lid["id"]])
    p.channel(plan, "hinged_lid.pose", "door_world", [0], "transform", [lid["id"]])
    lid_held = {}
    for arm in range(2):
        name = f"arm{arm}.hinged_lid.grasp"
        lid_held[arm] = p.channel(plan, name, "door_grasp", [arm,0], "grasp", [lid["id"]])
        plan["effectors"][f"arm{arm}"]["grasp_channels"][lid["id"]] = name
    # Initial conditions are scene facts shared by every script. They therefore
    # produce one anchor key when actual initial pixels and roles also agree.
    for oid in ids[:5]:
        plan["initial_conditions"].append(p.condition(oid+".standing", "upright", [oid],
            p.expr("gt", p.expr("upright", poses[oid]), p.const(.9))))
    closed = p.expr("lt", p.expr("abs", joint), p.const(.02))
    plan["initial_conditions"].append(p.condition("lid.closed", "joint_range", [lid["id"]], closed, range_rad=[0.,.02]))
    for oid in [oid for oid in ids if oid not in target_ids]:
        contacts = [p.channel(plan, f"arm{a}.{oid}.contact", "robot_object_force_N", [a,ids.index(oid)],
                             "contact_force", [oid]) for a in range(2)]
        p.unchanged(plan, oid, poses[oid], contact=p.expr("all", *[p.expr("lt", c, p.const(.5)) for c in contacts]))
    if lid["id"] not in target_ids:
        p.invariant(plan, "lid.closed", "joint_range", [lid["id"]], closed, range_rad=[0.,.02])
    cross = p.channel(plan,"robots.contact","cross_robot_force_N",[],"contact_force",[])
    p.invariant(plan,"robot_collision","no_contact",[],p.expr("lt",cross,p.const(.5)),visible=False)
    if task["family"] == "passive_chain":
        selected = ids[:5] if task["planning_parameters"]["direction"]==1 else ids[:5][::-1]
        first = selected[0]
        contact = p.channel(plan,"chain.trigger_contact","robot_contact_impulses",[0,ids.index(first)],"contact_impulse",[first])
        p.event(plan,"chain.trigger","trigger",[first],p.expr("gt",contact,p.const(1e-5)),"arm0")
        previous = "chain.trigger"
        for i,(a,b) in enumerate(zip(selected[:-1],selected[1:])):
            hit=p.channel(plan,f"chain.contact_{i}","object_contact_impulses",[ids.index(a),ids.index(b)],"contact_impulse",[a,b])
            eid=f"chain.propagate_{i}"
            p.event(plan,eid,"passive_contact",[a,b],p.expr("gt",hit,p.const(1e-5)),after=[previous]);previous=eid
        for oid in selected:
            plan["goals"].append(p.condition(oid+".fallen","toppled",[oid],
                p.expr("lt",p.expr("abs",p.expr("upright",poses[oid])),p.const(.45))))
            p.invariant(plan,oid+".never_grasped","no_grasp",[oid],p.expr("not",p.expr("any",grasps[0,oid],grasps[1,oid])))
            for arm in range(2):
                if oid==first and arm==0:continue
                touch=p.channel(plan,f"arm{arm}.{oid}.touch","robot_contact_impulses",[arm,ids.index(oid)],"contact_impulse",[oid])
                p.invariant(plan,f"arm{arm}.{oid}.no_touch","no_contact",[oid],p.expr("le",touch,p.const(1e-5)),effector=f"arm{arm}")
    elif task["family"] == "insertion":
        port=objects[ids.index("receiver")]["geometry"]["ports"][task["planning_parameters"]["port"]]
        plan["terminal_hands"]["arm0"]={"state":"holding","entity":"receiver"}
        relative=p.expr("relative_position",poses["plug"],poses["receiver"])
        lateral=p.expr("norm",p.expr("sub",p.expr("xy",relative),p.const(port[:2])))
        depth=p.expr("sub",p.const(.064),p.component(relative,2))
        aligned=p.expr("all",p.expr("lt",lateral,p.const(.003)),p.expr("ge",depth,p.const(.020)),
            p.expr("le",depth,p.const(.028)),p.expr("lt",p.expr("relative_angle",poses["plug"],poses["receiver"]),p.const(.08)))
        contact=p.channel(plan,"mating.contact","interface_impulses",[],"contact_impulse",["receiver","plug"])
        p.event(plan,"receiver.grasp","grasp",["receiver"],grasps[0,"receiver"],"arm0")
        p.event(plan,"plug.grasp","grasp",["plug"],grasps[1,"plug"],"arm1",["receiver.grasp"])
        p.event(plan,"plug.insert","insert",["plug","receiver"],p.expr("all",aligned,p.expr("gt",contact,p.const(1e-5)),
            grasps[0,"receiver"],grasps[1,"plug"]),"arm1",["plug.grasp"])
        p.event(plan,"plug.release","release",["plug"],p.expr("not",grasps[1,"plug"]),"arm1",["plug.insert"])
        p.invariant(plan,"receiver.continuously_held","continuous_grasp",["receiver"],grasps[0,"receiver"],
                    start="plug.grasp",until="terminal",effector="arm0")
        plan["goals"] += [p.condition("mating_geometry","inserted",["plug","receiver"],aligned,
            selected_port_center_local=port,depth_range_m=[.020,.028],lateral_tolerance_m=.003,
            receiver_face_z_m=.022,tongue_tip_z_m=-.042),
            p.condition("inserted_contact","contact",["plug","receiver"],p.expr("gt",contact,p.const(1e-5)))]
    else:
        held=lid_held[1]
        opened=p.expr("all",p.expr("ge",joint,p.const(1.2)),p.expr("le",joint,p.const(1.4)))
        p.event(plan,"handle.grasp","grasp",[lid["id"]],held,"arm1")
        p.event(plan,"lid.motion","articulate",[lid["id"]],p.expr("all",held,p.expr("gt",joint,p.const(.05))),"arm1",["handle.grasp"])
        p.event(plan,"lid.goal","articulate",[lid["id"]],opened,"arm1",["lid.motion"])
        p.event(plan,"handle.release","release",[lid["id"]],p.expr("not",held),"arm1",["lid.goal"])
        p.invariant(plan,"lid.continuously_held","continuous_grasp",[lid["id"]],held,start="lid.motion",until="lid.goal",effector="arm1")
        velocity=p.channel(plan,"lid.velocity","cabinet_qvel",[0],"joint_velocity",[lid["id"]])
        plan["goals"] += [p.condition("lid.open","joint_range",[lid["id"]],opened,range_rad=[1.2,1.4]),
                          p.condition("lid.stable","stable",[lid["id"]],p.expr("lt",p.expr("abs",velocity),p.const(.02)))]
    for oid in [oid for oid in ids if oid in target_ids]:
        i=ids.index(oid)
        velocity=p.channel(plan,oid+".velocity","object_states",[i,{"slice":[7,10]}],"linear_velocity",[oid])
        angular=p.channel(plan,oid+".angular","object_states",[i,{"slice":[10,13]}],"angular_velocity",[oid])
        plan["goals"].append(p.condition(oid+".stable","stable",[oid],p.expr("all",
            p.expr("lt",p.expr("norm",velocity),p.const(.025)),p.expr("lt",p.expr("norm",angular),p.const(.4)))))
    from .hand_correspondence import declare_from_scene
    declare_from_scene(plan, scene)
    validate(plan)
    return plan


def prepare(root, scene_id="mechanism_workbench_C", seed=2026102317):
    from .library import task_semantic_key
    from .intake import cases
    root=Path(root)
    scene=make_scene(scene_id,seed)
    scene_path=root/"scenes"/(scene_id+".json")
    write(scene_path,scene)
    rows=[]
    for family in ("passive_chain","insertion","articulation"):
        cid=scene_id+"__"+family
        require(cid not in {c["id"] for c in cases(root)},"Task already registered; do not retry")
        task=task_intent(scene,cid,family)
        protocol(scene,task)
        path=root/"tasks"/(cid+".json");write(path,task)
        row={"id":cid,"scene":bind(scene_path),"task":bind(path),"semantic_key":task_semantic_key(task),
             "partition":"development_shared_scene_new_mechanisms","request_human_reference":True,
             "planner_plugin":"workbench_v1","reference_policy":"one_per_distinct_task_semantics"}
        write(root/"registered_cases"/(cid+".json"),row);rows.append(row)
    write(root/"workbench_declaration.json",{"scene":bind(scene_path),"cases":rows,"same_initial_scene_for_all":True,
        "initial_state_and_rgb_equality_must_be_verified":True,"automatic_content_retry":False,
        "downstream_annotations":False,"declared_before_simulation":True})
    return {"scene":scene_id,"cases":[r["id"] for r in rows],"rigid_objects":len(scene["objects"]),"articulated_objects":1}


def build_lid(env):
    import sapien
    from scipy.spatial.transform import Rotation
    spec=env.scene_spec["lid"]
    builder=env.scene.create_articulation_builder()
    root=builder.create_link_builder();root.set_name("lid_fixed_anchor")
    link=builder.create_link_builder(root);link.set_name(spec["id"]);link.set_joint_name("lid_hinge")
    for center,half,color in [(spec["panel_center_local"],spec["panel_half_size"],spec["rgba"]),
                              (spec["handle_center_local"],spec["handle_half_size"],spec["handle_rgba"])]:
        link.add_box_collision(pose=sapien.Pose(center),half_size=half)
        link.add_box_visual(pose=sapien.Pose(center),half_size=half,material=color)
    rotation=Rotation.align_vectors([spec["axis_world"]],[[1.,0.,0.]])[0].as_quat(scalar_first=True)
    link.set_joint_properties("revolute",[spec["joint_limits"]],
        pose_in_parent=sapien.Pose(spec["pivot_world"],rotation),pose_in_child=sapien.Pose(q=rotation),
        friction=spec["joint_friction"],damping=spec["joint_damping"])
    link.set_mass_and_inertia(spec["mass_kg"],sapien.Pose(spec["panel_center_local"]),[.00012,.00019,.00031])
    builder.initial_pose=sapien.Pose()
    env.cabinet=builder.build(fix_root_link=True,name="workbench_lid")
    env.door_links=[next(x for x in env.cabinet.get_links() if x.name==spec["id"])]
    for joint in env.cabinet.active_joints:
        joint.set_drive_properties(stiffness=0.,damping=spec["joint_damping"])
        joint.set_friction(spec["joint_friction"])


def register(scene):
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.types import SimConfig,SceneConfig
    from .simulation import register as register_table
    parent=register_table(scene)
    @register_env("ReSimPairMechanismWorkbench-v1",max_episode_steps=4000)
    class Workbench(parent):
        @property
        def _default_sim_config(self):
            return SimConfig(sim_freq=200,control_freq=20,scene_config=SceneConfig(
                contact_offset=.002,solver_position_iterations=25,solver_velocity_iterations=4))
        def _load_scene(self,options):
            super()._load_scene(options)
            build_lid(self)
            n=len(self.objects)
            self.object_contact_impulses=np.zeros((n,n))
            self.robot_contact_impulses=np.zeros((2,n))
        def _initialize_episode(self,env_idx,options):
            super()._initialize_episode(env_idx,options)
            self.cabinet.set_qpos(np.zeros(1));self.cabinet.set_qvel(np.zeros(1))
        def _before_control_step(self):
            super()._before_control_step()
            self.object_contact_impulses[:]=0.;self.robot_contact_impulses[:]=0.
        def _after_simulation_step(self):
            super()._after_simulation_step()
            for contact in self.scene.get_contacts():
                a,b=[x.entity for x in contact.bodies]
                impulse=sum(float(np.linalg.norm(x.impulse)) for x in contact.points)
                if a in self.object_bodies and b in self.object_bodies:
                    i,j=self.object_bodies[a],self.object_bodies[b]
                    self.object_contact_impulses[i,j]+=impulse;self.object_contact_impulses[j,i]+=impulse
                for first,second in [(a,b),(b,a)]:
                    if first in self.robot_bodies and second in self.object_bodies:
                        self.robot_contact_impulses[self.robot_bodies[first],self.object_bodies[second]]+=impulse
    return Workbench


class WorkbenchRecorder(Recorder):
    def __init__(self,env,directory):
        super().__init__(env,directory)
        self.fixed_arm_commands={}
    def step(self,action):
        action=dict(action)
        for arm,command in self.fixed_arm_commands.items():action[f"panda_wristcam-{arm}"]=command.copy()
        return super().step(action)
    def measure(self):
        result=super().measure();b=self.unwrapped
        # A boolean grasp cannot distinguish one-finger contact, obstruction,
        # and tracking error. Preserve the native vectors for future diagnosis;
        # these are simulation measurements, never inferred human force labels.
        fingers=[[a.finger1_link,a.finger2_link] for a in b.agent.agents]
        result.update(object_contact_impulses=b.object_contact_impulses.copy(),
            robot_contact_impulses=b.robot_contact_impulses.copy(),
            interface_impulses=np.array(b.object_contact_impulses[b.object_ids.index("receiver"),b.object_ids.index("plug")]),
            cabinet_qpos=array(b.cabinet.qpos).reshape(1).copy(),cabinet_qvel=array(b.cabinet.qvel).reshape(1).copy(),
            door_world=np.stack([array(d.pose.to_transformation_matrix()).reshape(4,4) for d in b.door_links]),
            door_grasp=np.array([[bool(array(a.is_grasping(d)).item()) for d in b.door_links] for a in b.agent.agents]),
            finger_world=np.array([[array(f.pose.to_transformation_matrix()).reshape(4,4) for f in fs] for fs in fingers]),
            door_finger_contact_force_N=np.array([[[array(b.scene.get_pairwise_contact_forces(f,d)).reshape(3)
                for f in fs] for d in b.door_links] for fs in fingers]))
        return result


class ObjectView:
    """Planner-only index adapter; the recorder always measures the whole scene."""
    def __init__(self,rec,ids):
        self.rec=rec
        b=rec.unwrapped
        class EnvironmentView:
            def __init__(self):
                self.objects=[b.objects[b.object_ids.index(i)] for i in ids]
            def __getattr__(self,name):return getattr(b,name)
        self.unwrapped=EnvironmentView()
    def __getattr__(self,name):return getattr(self.rec,name)
    def __setattr__(self,name,value):
        if name in {"rec","unwrapped"}:object.__setattr__(self,name,value)
        else:setattr(self.rec,name,value)


def export(rec,case,task,scene,directory,failure):
    if not rec.rows:return None
    directory=Path(directory)
    data={k:np.stack([r[k] for r in rec.rows]) for k in rec.rows[0]}
    data.update(native_commands=np.stack(rec.commands) if rec.commands else np.empty((0,16)),
        timestamps_seconds=np.arange(len(rec.rows),dtype=float)/rec.control_freq,
        initial_physics_state=rec.initial_physics,
        robot_roots_world=np.stack([array(a.robot.pose.to_transformation_matrix()).reshape(4,4) for a in rec.agent.agents]))
    np.savez_compressed(directory/"measurements.npz",**data)
    write(directory/"planned_paths.json",{"paths":rec.paths,"scope":"planner paths, separate from native event predicates"})
    cameras={}
    for name,config in scene["cameras"].items():
        params=rec.unwrapped._sensors[name].get_params()
        cameras[name]={"config":config,"parameters":{k:array(v).reshape((-1,)+tuple(array(v).shape[-2:]))[0].tolist() for k,v in params.items()},
            "video":bind(directory/(name+".mp4")),"initial_image":bind(directory/(name+"_initial.png")),
            "final_image":bind(directory/(name+"_final.png")),"synchronous_with_measurements":True,
            "frame_count":len(rec.rows),"fps":float(rec.control_freq)}
    bundle={"schema":"resimpair_execution_bundle_v5","case":case,"task_intent":task,"scene":scene,
        "event_protocol":bind(directory/"event_protocol.json"),"planning_declaration":bind(directory/"planning_declaration.json"),
        "measurements":bind(directory/"measurements.npz"),"planned_paths":bind(directory/"planned_paths.json"),
        "object_ids":rec.object_ids,"support_ids":rec.support_ids,"cameras":cameras,
        "initial_physics_sha256":rec.initial_physics_sha,"initial_object_state_sha256":digest(data["object_states"][0].tolist()),
        "initial_articulation_state_sha256":digest(data["cabinet_qpos"][0].tolist()),
        "motion_planning_agent_participation":True,"manual_generation_semantics_supplied":False,
        "manual_review_observations_supplied":False,"native_execution":True,"physical_execution_count":1,
        "saved_state_rerender":False,"code":implementation(),"planning_failure":failure,
        "qualification_scope":"native measured event predicates and full-interval non-target checks; not a continuous path proof",
        "source_scene_reconstruction_claimed":False,"training_admitted":False}
    write(directory/"execution_bundle.json",bundle)
    return bind(directory/"execution_bundle.json")


def worker(root,case_id,attempt):
    from .simulation import case_inputs
    root=Path(root)
    experiment,case,scene,task=case_inputs(root,case_id)
    directory=root/"executions"/case_id/attempt
    require(os.environ.get("RESIMPAIR_RESERVATION")==str(directory/"reservation.json"),"Supervisor reservation required")
    for item in read(directory/"reservation.json")["implementation"].values():checked(item)
    plan=protocol(scene,task)
    write(directory/"event_protocol.json",plan)
    write(directory/"planning_declaration.json",{"event_protocol":bind(directory/"event_protocol.json"),"case":case,
        "scope":"before_physical_execution","declared_unix":time.time(),"implementation":implementation(),
        "manual_generation_prose":False,"shared_initial_scene":bind(checked(case["scene"]))})
    from examples.baselines.hypericl_task_expansion.common import bind_gpu
    gpu=bind_gpu(experiment["gpu"])
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    import gymnasium as gym
    env=rec=None;failure=bundle=None
    try:
        with span(directory,"simulation_environment_setup"):
            register(scene)
            env=gym.make("ReSimPairMechanismWorkbench-v1",sim_backend="physx_cpu",render_backend=gpu["render_backend"],
                control_mode="pd_joint_pos",obs_mode="rgb",reward_mode="none",render_mode="rgb_array",sensor_configs={"shader_pack":"rt-fast"})
        with span(directory,"simulation_reset_and_recorders"):
            rec=WorkbenchRecorder(env,directory);rec.reset(scene["seed"])
        with span(directory,"planning_execution_and_rendering"):
            from .workbench_solvers import solve
            solve(rec,task)
    except Exception as exc:
        import traceback
        failure={"type":type(exc).__name__,"message":str(exc),"trace":[{"file":f.filename,"line":f.lineno,"function":f.name} for f in traceback.extract_tb(exc.__traceback__)]}
        write(directory/"planning_failure.json",failure)
    finally:
        if rec is not None:
            with span(directory,"simulation_video_finalize"):
                rec.finish()
            with span(directory,"simulation_measurement_export"):
                bundle=export(rec,case,task,scene,directory,failure)
        if env is not None:env.close()
    result={"bundle":bundle,"planning_failure":failure,"frames":len(rec.rows) if rec else 0,
            "status":"planned_and_executed" if failure is None else "planning_or_execution_failed"}
    write(directory/"worker_result.json",result)
    if failure is None and bundle:
        write(root/"planner_handoffs"/(case_id+".json"),{"schema":"resimpair_planner_handoff_v5","case":case_id,
            "attempt":attempt,"execution_bundle":bundle,"manual_downstream_annotations":False})
    return result
