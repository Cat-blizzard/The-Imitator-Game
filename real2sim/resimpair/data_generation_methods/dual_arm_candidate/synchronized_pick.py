"""Two official mplib planners, one shared native control clock, paired mesh guard."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np

from ..human_to_robot_pipeline.strict_motion import (validate_joint_path, world_fk,
    matrix_to_pose7, pose7_to_matrix, sampled_mesh_points)
from ..human_to_robot_pipeline.planning_contract import bind, save_new

HAND_LINKS = {'panda_hand', 'panda_leftfinger', 'panda_rightfinger'}


def synchronize_paths(paths):
    """Stretch the shorter path to shared duration; never accelerate either path."""
    if len(paths) != 2:
        raise ValueError('Exactly two paths required')
    values = [np.asarray(p, float) for p in paths]
    if any(p.ndim != 2 or p.shape[1] != 7 or not len(p) or not np.isfinite(p).all() for p in values):
        raise ValueError('Finite [time,7] paths required')
    n = max(len(p) for p in values)
    outputs = []
    for p in values:
        old = np.linspace(0., 1., len(p))
        new = np.linspace(0., 1., n)
        outputs.append(np.stack([np.interp(new, old, p[:, j]) for j in range(7)], axis=-1))
    return np.stack(outputs, axis=1)


def paired_dense(path, max_step=.015):
    """Interpolate both robots with the SAME fractional time at every sample."""
    path = np.asarray(path, float)
    if path.ndim != 3 or path.shape[1:] != (2, 9) or not len(path) or not np.isfinite(path).all():
        raise ValueError('Expected finite [time,2,9] paired configurations')
    dense = [path[0]]
    for start, end in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(abs(end-start).max()/max_step)))
        dense.extend(start+(end-start)*(i/n) for i in range(1, n+1))
    return np.asarray(dense)


class PairedPlanning:
    """Independent physical link meshes move together in paired configurations.

    Individual plan candidates use static obstacle clouds. Acceptance additionally
    updates BOTH arms and payloads for every paired sample. No frozen-other-arm
    assumption is used by this acceptance gate.
    """
    def __init__(self, rec, directory):
        import mplib.planner
        from mplib.pymp.collision_detection import fcl
        from mani_skill.utils.geometry.trimesh_utils import get_component_meshes, merge_meshes
        from examples.baselines.hypericl_task_expansion.collect import array
        self.rec, self.directory = rec, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.fcl = fcl
        self.b = rec.unwrapped
        self.agents = self.b.agent.agents
        self.roots = [array(a.robot.pose.to_transformation_matrix()).reshape(4, 4) for a in self.agents]
        self.planners, self.links, self.owned = [], [], []
        self.counter, self.plan_seconds, self.guard_seconds = 0, 0., 0.
        self.payload_to_tcp = [None, None]
        self.world = []
        self.object_geometry = []
        self.meshes = []
        for i, agent in enumerate(self.agents):
            planner = mplib.planner.Planner(urdf=agent.urdf_path,
                srdf=agent.urdf_path.replace('.urdf','.srdf'),
                user_link_names=[x.get_name() for x in agent.robot.get_links()],
                user_joint_names=[x.get_name() for x in agent.robot.get_active_joints()],
                move_group='panda_hand_tcp', joint_vel_limits=np.ones(7)*.8,
                joint_acc_limits=np.ones(7)*.8)
            planner.set_base_pose(np.r_[array(agent.robot.pose.p).reshape(3),array(agent.robot.pose.q).reshape(4)])
            self.planners.append(planner)
            links = []
            for link in agent.robot.get_links():
                mesh = merge_meshes(get_component_meshes(link._objs[0]))
                if mesh is None:
                    continue
                path = self.directory/f'arm{i}_{link.get_name()}_physical_local.obj'
                mesh.export(path)
                geom = fcl.load_mesh_as_BVH(str(path), np.ones(3))
                collision = fcl.CollisionObject(geom)
                self.owned.append((geom, collision))
                links.append((link.get_name(), planner.link_name_2_idx[link.get_name()], collision))
            self.links.append(links)
        actors = [('complete_table', self.b.table_scene.table)]+[(f'container_{i}',a) for i,a in enumerate(self.b.bins)]
        for name, actor in actors:
            mesh = actor.get_first_collision_mesh(to_world_frame=True)
            path = self.directory/(name+'_world.obj');mesh.export(path)
            geom = fcl.load_mesh_as_BVH(str(path),np.ones(3));collision=fcl.CollisionObject(geom)
            self.owned.append((geom,collision));self.world.append((name,collision));self.meshes.append(mesh)
        self.table_z = float(self.meshes[0].bounds[1,2])
        for i, actor in enumerate(self.b.objects):
            mesh=actor.get_first_collision_mesh(to_world_frame=False)
            path=self.directory/f'object_{i}_physical_local.obj';mesh.export(path)
            geom=fcl.load_mesh_as_BVH(str(path),np.ones(3));collision=fcl.CollisionObject(geom)
            self.owned.append((geom,collision));self.object_geometry.append(collision)
        save_new(self.directory/'implementation.json', {
            'planner':bind(mplib.planner.__file__),'candidate':bind(__file__),
            'two_independent_official_planner_instances':True,'legacy_custom_planner_used':False,
            'strict_single_arm_qualification_inherited':False,'paired_dense_step_rad':.015,
            'physical_guard':'simultaneously transformed physical robot link/payload meshes, complete table and containers',
            'allowed_pairs':'own fingers/hand with own object only in grasp/lift; own base/table mount; initial table support under narrow lift rule',
            'continuous_time_collision_proof':False})

    def collision_negative_control(self):
        """Counterfactual overlap of exported physical meshes, never an env write."""
        from examples.baselines.hypericl_task_expansion.collect import array
        current=np.stack([array(a.robot.qpos).reshape(9) for a in self.agents])
        original=self.roots[1].copy()
        try:
            self.roots[1]=self.roots[0].copy();current[1]=current[0]
            tcp=[world_fk(current[i,:7][None],self.roots[i])[0] for i in range(2)]
            failures=self._config_collision(current,'approach',tcp)
        finally:
            self.roots[1]=original
        rejected=any(x['kind']=='inter_arm' for x in failures)
        save_new(self.directory/'inter_arm_overlap_negative_control.json',{
            'expected':'inter_arm collision when both physical robots occupy identical FK world transforms',
            'rejected':rejected,'collision_pairs':failures,'physical_meshes_from_this_execution':True,
            'counterfactual_guard_input_only':True,'executed_in_simulator':False,'environment_state_modified':False})
        if not rejected:raise RuntimeError('Paired mesh guard did not reject coincident-arm negative control')

    def update_clouds(self):
        import trimesh
        from examples.baselines.hypericl_task_expansion.collect import array
        for arm, planner in enumerate(self.planners):
            others=[self.b.objects[1-arm].get_first_collision_mesh(to_world_frame=True),
                    self.agents[1-arm].robot.get_first_collision_mesh(to_world_frame=True)]
            table_points=sampled_mesh_points(self.meshes[0],16000,9907+arm)
            base = self.agents[arm].robot.get_links()[0]
            # Guard below checks complete table except the exact link0 mounting pair.
            from mani_skill.utils.geometry.trimesh_utils import get_component_meshes,merge_meshes
            mesh=merge_meshes(get_component_meshes(base._objs[0]));mesh.apply_transform(self.roots[arm])
            bounds=mesh.bounds
            mount=np.all((table_points[:,:2]>=bounds[0,:2]-.005)&(table_points[:,:2]<=bounds[1,:2]+.005),axis=1)
            rest=trimesh.util.concatenate(self.meshes[1:]+others)
            cloud=np.concatenate([table_points[~mount],sampled_mesh_points(rest,10000,9917+arm)])
            planner.update_point_cloud(cloud,radius=.001)

    def propose(self, arm, goal, stage):
        from examples.baselines.hypericl_task_expansion.collect import array
        current=array(self.agents[arm].robot.qpos).reshape(9).copy()
        planner=self.planners[arm]
        t=time.perf_counter();attempts=[]
        for method in ['plan_screw','plan_qpos_to_pose']:
            result=getattr(planner,method)(np.r_[goal.p,goal.q],current.copy(),
                time_step=self.rec.control_timestep,use_point_cloud=True,use_attach=False,wrt_world=True)
            entry={'method':method,'status':result.get('status')};attempts.append(entry)
            if result.get('status')!='Success':continue
            check=validate_joint_path(result['position'],planner.joint_limits,current)
            entry['joint_check']=check
            if not check['passed']:continue
            end=world_fk(result['position'][-1:],self.roots[arm])[0]
            target=np.asarray(goal.to_transformation_matrix())
            error=float(np.linalg.norm(end[:3,3]-target[:3,3]))
            angle=float(np.arccos(np.clip((np.trace(end[:3,:3].T@target[:3,:3])-1)/2,-1,1)))
            entry.update(endpoint_error_m=error,endpoint_error_rad=angle)
            if error>.002 or angle>.03:continue
            self.plan_seconds+=time.perf_counter()-t
            return result['position'],{'arm':arm,'stage':stage,'attempts':attempts,'goal_world':target.tolist()}
        self.plan_seconds+=time.perf_counter()-t
        raise RuntimeError('Official planner rejected '+stage+' arm '+str(arm)+': '+json.dumps(attempts))

    def attach(self, arm):
        from examples.baselines.hypericl_task_expansion.collect import array,require
        agent=self.agents[arm];obj=self.b.objects[arm]
        require(bool(array(agent.is_grasping(obj)).item()),'Native grasp is false before attachment')
        tcp=array(agent.tcp.pose.to_transformation_matrix()).reshape(4,4)
        pose=array(obj.pose.to_transformation_matrix()).reshape(4,4)
        self.payload_to_tcp[arm]=np.linalg.inv(tcp)@pose

    def _config_collision(self, full, stage, initial_tcp):
        from examples.baselines.hypericl_task_expansion.collect import array
        fcl=self.fcl;links=[];tcp=[]
        for arm,planner in enumerate(self.planners):
            hits=planner.check_for_self_collision(qpos=full[arm])
            if hits:return [{'kind':'self','arm':arm,'pairs':[(str(x.link_name1),str(x.link_name2)) for x in hits]}]
            planner.pinocchio_model.compute_forward_kinematics(full[arm])
            arm_links=[]
            for name,index,collision in self.links[arm]:
                mat=self.roots[arm]@pose7_to_matrix(planner.pinocchio_model.get_link_pose(index))
                collision.set_transformation(matrix_to_pose7(mat));arm_links.append((name,collision))
            links.append(arm_links)
            tcp.append(world_fk(full[arm,:7][None],self.roots[arm])[0])
            obj_pose=(tcp[-1]@self.payload_to_tcp[arm] if self.payload_to_tcp[arm] is not None else
                array(self.b.objects[arm].pose.to_transformation_matrix()).reshape(4,4))
            self.object_geometry[arm].set_transformation(matrix_to_pose7(obj_pose))
        def collides(a,b):return fcl.collide(a,b,fcl.CollisionRequest()).is_collision()
        for n0,c0 in links[0]:
            for n1,c1 in links[1]:
                if collides(c0,c1):return [{'kind':'inter_arm','link0':n0,'link1':n1}]
        if collides(*self.object_geometry):return [{'kind':'object_object'}]
        for arm in range(2):
            for name,collision in links[arm]:
                for world_name,obstacle in self.world:
                    if name=='panda_link0' and world_name=='complete_table':continue
                    if collides(collision,obstacle):return [{'kind':'robot_environment','arm':arm,'link':name,'obstacle':world_name}]
                for obj,obj_col in enumerate(self.object_geometry):
                    allowed=arm==obj and name in HAND_LINKS and (stage in ['grasp_reach','close','lift','hold'] or self.payload_to_tcp[arm] is not None)
                    if not allowed and collides(collision,obj_col):return [{'kind':'robot_object','arm':arm,'link':name,'object':obj}]
            if self.payload_to_tcp[arm] is not None:
                for name,obstacle in self.world:
                    contact=fcl.collide(self.object_geometry[arm],obstacle,fcl.CollisionRequest(num_max_contacts=10000,enable_contact=True))
                    if not contact.is_collision():continue
                    contacts=contact.get_contacts();delta=tcp[arm][:3,3]-initial_tcp[arm][:3,3]
                    allowed=(name=='complete_table' and stage=='lift' and -.0005<=delta[2]<=.002
                        and np.linalg.norm(delta[:2])<=.003 and 0<len(contacts)<10000
                        and all(c.pos[2]<=self.table_z+.005 and c.penetration_depth<=.001 for c in contacts))
                    if not allowed:return [{'kind':'payload_environment','arm':arm,'obstacle':name}]
        return []

    def guard(self, paired, stage, folder):
        from examples.baselines.hypericl_task_expansion.collect import array
        started=time.perf_counter();dense=paired_dense(paired)
        tcp0=[world_fk(dense[0,i,:7][None],self.roots[i])[0] for i in range(2)]
        failure=[]
        for i,full in enumerate(dense):
            failure=self._config_collision(full,stage,tcp0)
            if failure:break
        self.guard_seconds+=time.perf_counter()-started
        report={'passed':not failure,'stage':stage,'dense_samples':len(dense),'checked_samples':i+1,
            'failure_sample':i if failure else None,'collision_pairs':failure,
            'shared_fractional_time_for_both_arms':True,'all_physical_links_updated_together':True,
            'formal_continuous_collision_claimed':False}
        np.savez_compressed(folder/'paired_guard_path.npz',paired_full_qpos=dense)
        save_new(folder/'guard.json',report)
        if failure:raise RuntimeError('Paired physical guard rejected '+stage+': '+json.dumps(failure))
        return report

    def execute(self, paths, grippers, stage, planning_records=None):
        from examples.baselines.hypericl_task_expansion.collect import array,require
        q=synchronize_paths(paths)
        current=[array(a.robot.qpos).reshape(9).copy() for a in self.agents]
        full=np.stack([np.repeat(v[None],len(q),axis=0) for v in current],axis=1)
        full[:,:,:7]=q
        if stage=='close':
            for arm,g in enumerate(grippers):
                if g==-1 and self.payload_to_tcp[arm] is None:
                    full[:,arm,7:]=np.linspace(current[arm][7:],np.zeros(2),len(q))
        folder=self.directory/f'{self.counter:03d}_{stage}';self.counter+=1;folder.mkdir()
        save_new(folder/'plan.json',{'stage':stage,'independent_plan_lengths':[len(p) for p in paths],
            'common_commands':len(q),'planning_records':planning_records,'gripper_commands':grippers,
            'time_stretch_only':True,'single_env_step_per_paired_command':True})
        report=self.guard(full,stage,folder)
        self.rec.stage=stage
        for pair in q:
            action={f'panda_wristcam-{i}':np.r_[pair[i],grippers[i]] for i in range(2)}
            self.rec.step(action)
            require(self.b.control_contact_peaks['inter_arm']<.5,'Actual inter-arm contact exceeded0.5N')
            require(self.b.control_contact_peaks['wrong_object']<.5,'Actual wrong-object contact exceeded0.5N')
            require(self.b.control_contact_peaks['robot_environment']<.5,'Actual robot/environment contact exceeded0.5N')
            require(self.b.control_contact_peaks['fruit_fruit']<.5,'Actual fruit/fruit contact exceeded0.5N')
        return report


def measured_hold_command(finger_positions, compression_m=.0015):
    """Panda normalized target, set from measured aperture after native grasp.

    The fixed 1.5mm compression is a control intervention, not an acceptance
    tolerance. No force or stability claim follows from this target alone.
    """
    positions=np.asarray(finger_positions,float)
    if positions.shape!=(2,) or not np.isfinite(positions).all():
        raise ValueError('Two finite measured finger positions required')
    target=float(np.clip(positions.mean()-compression_m,0.,.04))
    return 2*(target+.01)/.05-1,target


def solve(rec, directory, mode, hold_strategy='full_close'):
    import sapien
    from mani_skill.examples.motionplanning.panda.utils import get_actor_obb,compute_grasp_info_by_obb
    from examples.baselines.hypericl_task_expansion.collect import array,require
    checked=PairedPlanning(rec,directory);checked.collision_negative_control();b=rec.unwrapped
    grasps=[]
    for i,agent in enumerate(b.agent.agents):
        hint=array(agent.tcp.pose.to_transformation_matrix()).reshape(4,4)[:3,1]
        grasp=compute_grasp_info_by_obb(get_actor_obb(b.objects[i]),approaching=[0,0,-1],target_closing=hint,depth=.008)
        require(grasp['extents'][1]<=.077,'Fruit exceeds Panda aperture')
        grasps.append(agent.build_grasp_pose([0,0,-1],grasp['closing'],grasp['center']))
    grips=[1,1];events=[]
    groups=[[0,1]] if mode=='synchronous' else [[0],[1]]
    for active in groups:
        for stage,offset in [('approach',.16),('grasp_reach',0.)]:
            checked.update_clouds();paths=[];records=[]
            for i in range(2):
                if i in active:
                    goal=sapien.Pose(grasps[i].p+[0,0,offset],grasps[i].q)
                    path,record=checked.propose(i,goal,stage)
                else:
                    path=array(b.agent.agents[i].robot.qpos).reshape(9)[:7][None];record={'arm':i,'held_fixed':True}
                paths.append(path);records.append(record)
            checked.execute(paths,grips,stage,records)
            rec.hold(4,tuple(grips))
        for i in active:grips[i]=-1
        fixed=[np.repeat(array(a.robot.qpos).reshape(9)[:7][None],16,axis=0) for a in b.agent.agents]
        checked.execute(fixed,grips,'close')
        for i in active:
            checked.attach(i)
            if hold_strategy=='measured_compression':
                measured=array(b.agent.agents[i].robot.qpos).reshape(9)[7:].copy()
                grips[i],target=measured_hold_command(measured)
                events.append({'arm':i,'event':'measured_hold_target','frame':len(rec.rows),
                    'finger_positions_m':measured.tolist(),'compression_m':.0015,
                    'target_per_finger_m':target,'normalized_command':grips[i],
                    'no_weld_or_pose_write':True})
        checked.update_clouds();paths=[];records=[]
        for i in range(2):
            if i in active:
                goal=sapien.Pose(grasps[i].p+[0,0,.19],grasps[i].q)
                path,record=checked.propose(i,goal,'lift')
            else:
                path=array(b.agent.agents[i].robot.qpos).reshape(9)[:7][None];record={'arm':i,'held_fixed':True}
            paths.append(path);records.append(record)
        checked.execute(paths,grips,'lift',records)
        for i in active:
            require(bool(array(b.agent.agents[i].is_grasping(b.objects[i])).item()),'Fruit grasp lost after lift')
            events.append({'arm':i,'object':i,'native_grasp_after_lift':True,'frame':len(rec.rows)})
    rec.stage='hold';rec.hold(40,tuple(grips))
    return events,{'planning_seconds':checked.plan_seconds,'paired_guard_seconds':checked.guard_seconds,
                   'planner_instances':2,'paired_segments':checked.counter}
