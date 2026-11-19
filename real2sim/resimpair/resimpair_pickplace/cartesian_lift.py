"""Finite Cartesian lift planning and read-only replay of saved failure states.

This planner solves short Cartesian waypoints instead of asking unconstrained
RRT to happen to stay in a lift corridor. It never changes the retained guard.
Offline replay reconstructs the exact saved meshes; it is not a physics run.
"""
from pathlib import Path
import numpy as np
import trimesh
from resimpair_grasp.common import ROOT,read,write,binding
from data_generation_methods.human_to_robot_pipeline.strict_motion import (
    world_fk,matrix_to_pose7,sampled_mesh_points)
from resimpair_grasp.motion import MeshMotion


def cartesian_path(planner, root, current, goal, timestep=.05, waypoint_m=.004):
    current=np.asarray(current,float);goal=np.asarray(goal,float)
    start=world_fk(current[None,:7],root)[0]
    if np.linalg.norm(goal[:2,3]-start[:2,3])>.001 or not .002<=goal[2,3]-start[2,3]<=.30:
        raise ValueError('bounded_vertical_lift_only')
    from scipy.spatial.transform import Rotation
    if Rotation.from_matrix(start[:3,:3].T@goal[:3,:3]).magnitude()>.01:
        raise ValueError('lift_orientation_must_remain_fixed')
    if not .001<=waypoint_m<=.01 or not .01<=timestep<=.10:
        raise ValueError('invalid_cartesian_resolution')
    count=int(np.ceil((goal[2,3]-start[2,3])/waypoint_m))
    values=[current[:7].copy()];q=current.copy();inverse=np.linalg.inv(root)
    for i in range(1,count+1):
        T=start.copy();T[:3,3]=start[:3,3]+(goal[:3,3]-start[:3,3])*i/count
        status,solutions=planner.IK(matrix_to_pose7(inverse@T),q.copy(),
            mask=[False]*7+[True,True],n_init_qpos=2,threshold=.0002)
        if status!='Success':raise RuntimeError(f'cartesian_waypoint_ik_failed:{i}:{status}')
        candidate=min(solutions,key=lambda v:np.linalg.norm(v[:7]-q[:7]))
        if np.max(np.abs(candidate[:7]-q[:7]))>.08:
            raise RuntimeError(f'cartesian_waypoint_branch_jump:{i}')
        q=np.asarray(candidate).copy();values.append(q[:7].copy())
    times,pos,vel,acc,duration=planner.TOPP(np.asarray(values),timestep)
    return {'status':'Success','position':pos,'velocity':vel,'acceleration':acc,
        'time':times,'duration':duration,'cartesian_waypoints':count,
        'requires_retained_full_path_guard':True}


def restore_guard(execution,motion_directory,request):
    """No simulator, render context, GPU, teleport or modified historical file."""
    import mplib
    from mplib.pymp.collision_detection import fcl
    execution,motion_directory=Path(execution),Path(motion_directory)
    bundle=read(execution/'execution_bundle.json');scene=bundle['scene']
    op_id=motion_directory.name.removesuffix('_motion')
    op=next(x for x in bundle['task_intent']['operations'] if x['id']==op_id)
    guard=MeshMotion.__new__(MeshMotion);guard.arm=op['arm']
    guard.root=np.eye(4);guard.root[:3,3]=scene['robot_roots'][guard.arm]
    urdf=ROOT/'PolicyGen/mani_skill/assets/robots/panda/panda_v3.urdf'
    if hasattr(mplib,'set_global_seed'):mplib.set_global_seed(20261027)
    guard.planner=mplib.Planner(urdf=str(urdf),srdf=str(urdf.with_suffix('.srdf')),
        move_group='panda_hand_tcp',joint_vel_limits=np.ones(7)*.8,joint_acc_limits=np.ones(7)*.8)
    guard.planner.set_base_pose(matrix_to_pose7(guard.root))
    guard.objects=[];guard.environment_collision_objects=[];guard.physical_robot_objects=[]
    def collider(path):
        geometry=fcl.load_mesh_as_BVH(str(path),np.ones(3));obj=fcl.CollisionObject(geometry)
        guard.objects.append((geometry,obj));return obj
    world=read(motion_directory/'world.json')
    for obj in world['world']:
        collision=collider(motion_directory/(obj['id']+'.obj'))
        guard.planner.set_normal_object(obj['id'],collision)
        guard.environment_collision_objects.append((obj['id'],collision))
    guard.table_collision=collider(motion_directory/'complete_table.obj')
    for path in sorted(motion_directory.glob('robot_*.obj')):
        name=path.stem.removeprefix('robot_')
        guard.physical_robot_objects.append((name,guard.planner.link_name_2_idx[name],collider(path)))
    table=trimesh.load(motion_directory/'complete_table.obj',force='mesh')
    base=trimesh.load(motion_directory/'robot_panda_link0.obj',force='mesh');base.apply_transform(guard.root)
    cloud=sampled_mesh_points(table,18000,401)
    inside=np.all((cloud[:,:2]>=base.bounds[0,:2]-.005)&(cloud[:,:2]<=base.bounds[1,:2]+.005),axis=1)
    guard.planner.update_point_cloud(cloud[~inside],radius=.001)
    guard.selected_collision=collider(motion_directory/'selected_physical.obj')
    guard.payload_collision=collider(motion_directory/'selected_physical.obj')
    attachments=[(p,read(p)) for p in motion_directory.glob('attachment_*_lift.json')]
    matching=[(p,d) for p,d in attachments if d['frame']==request['frame']]
    if len(matching)!=1:raise ValueError('missing_unambiguous_measured_payload_transform')
    guard.payload_to_tcp=np.array(matching[0][1]['measured_object_in_tcp'])
    guard.use_attach=True;guard.support_floor_z=scene['table_top_z']
    current=np.array(request['current'])
    guard.selected_world_pose=world_fk(current[None,:7],guard.root)[0]@guard.payload_to_tcp
    return guard


def replay(execution,request_path,output):
    import time
    from resimpair_tabletop.sample_report import native_scalars
    execution,request_path,output=map(Path,[execution,request_path,output])
    if output.exists():raise ValueError('Preserve prior offline replay')
    request=read(request_path);motion_directory=request_path.parent.parent
    if request['stage']!='lift' or not request['attached']:raise ValueError('not_an_attached_lift_failure')
    guard=restore_guard(execution,motion_directory,request)
    current=np.array(request['current']);goal=np.array(request['goal'])
    prior_path=request_path.parent/'path.npz'
    with np.load(prior_path,allow_pickle=False) as arrays:
        old,_,_=guard.check({'position':arrays['positions']},current,goal,'lift')
    if old['passed'] or 'outside_local_xy_corridor' not in old.get('geometry',{}).get('failures',{}):
        raise ValueError('historical_corridor_rejection_not_reproduced')
    start=time.monotonic();error=None;candidate=None;report=None
    try:
        candidate=cartesian_path(guard.planner,guard.root,current,goal)
        report,dense,poses=guard.check(candidate,current,goal,'lift')
    except Exception as exc:error=repr(exc)
    result={'schema':'cartesian_lift_offline_replay_v1','request':binding(request_path),
        'original_path':binding(prior_path),'retained_guard':binding(ROOT/'resimpair_grasp/motion.py'),
        'source_meshes':[binding(p) for p in sorted(motion_directory.glob('*.obj'))],
        'implementation':binding(__file__),'original_guard_reproduced':old,
        'candidate_guard':report,'error':error,'planning_seconds':time.monotonic()-start,
        'physical_executions':0,'gpu_executions':0,'api_calls':0,
        'physical_success':'unmeasured','live_controller_promotion':False}
    output.mkdir(parents=True)
    if candidate is not None:
        np.savez_compressed(output/'candidate_path.npz',position=candidate['position'],
            velocity=candidate['velocity'],acceleration=candidate['acceleration'],current=current,goal=goal)
    if report is not None and dense is not None:
        np.savez_compressed(output/'dense.npz',joints=dense,tcp=poses)
    write(output/'report.json',native_scalars(result))
    return {'passed':bool(report and report['passed']),'error':error,'seconds':result['planning_seconds']}
