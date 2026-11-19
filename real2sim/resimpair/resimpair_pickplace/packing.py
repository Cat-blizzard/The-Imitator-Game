"""Candidate finite placement allocation for several payloads in one container.

Input: one scene, ordered movable IDs, one native-qualified container. Output:
role-level pick/place program, derived grasp caches, and allocation evidence.
Failure is explicit no-capacity/no-native-clearance; nothing is teleported and
no new controller or acceptance rule is introduced. Existing placed payloads
are checked as obstacles during allocation and again at their measured poses
by the retained runtime controller.
"""
import copy
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from resimpair_grasp.common import read, resolve, write, binding, digest
from resimpair_grasp.cartesian_clearance import VerticalClearance
from .reachability import pose


def disk_layouts(radii, capacity, gap=.012, margin=.004, limit=64):
    radii=np.asarray(radii,float)
    if not 1<=len(radii)<=6 or not np.isfinite(radii).all() or np.any(radii<=0):
        raise ValueError('one_to_six_positive_payload_radii_required')
    if not np.isfinite(capacity) or capacity<=0 or gap<0 or margin<0 or not 1<=limit<=64:
        raise ValueError('invalid_finite_capacity_request')
    available=capacity-margin
    def valid(points):
        points=np.asarray(points)
        return bool(np.all(np.linalg.norm(points,axis=1)+radii<=available+1e-10) and
            all(np.linalg.norm(points[i]-points[j])>=radii[i]+radii[j]+gap-1e-10
                for i in range(len(radii)) for j in range(i)))
    if len(radii)==1:
        return [np.zeros((1,2))] if valid([[0,0]]) else []
    if len(radii)==2:
        layouts=[]
        for angle in np.linspace(0,2*np.pi,32,endpoint=False):
            axis=np.array([np.cos(angle),np.sin(angle)])
            # Balanced enclosing radius rather than greedily occupying center.
            points=np.array([-axis*(radii[1]+gap/2),axis*(radii[0]+gap/2)])
            if valid(points):layouts.append(points)
        return layouts[:limit]
    # Bounded backtracking for 3..6 roles. Conservative disk capacity may reject
    # an arrangement that would fit with a future exact-polygon allocator.
    step=max(.015,min(radii)*.65)
    grid=np.arange(-available,available+step/2,step)
    points=[np.array([x,y]) for x in grid for y in grid]
    points.sort(key=lambda p:(-float(np.linalg.norm(p)),float(p[0]),float(p[1])))
    choices=[[p for p in points if np.linalg.norm(p)+r<=available] for r in radii]
    nodes=0;layouts=[]
    def visit(placed):
        nonlocal nodes
        if nodes>=20000 or len(layouts)>=limit:return
        i=len(placed)
        if i==len(radii):layouts.append(np.array(placed));return
        for p in choices[i]:
            nodes+=1
            if nodes>=20000:return
            if all(np.linalg.norm(p-q)>=radii[i]+radii[j]+gap for j,q in enumerate(placed)):
                visit(placed+[p])
    visit([])
    return layouts


def compose_packing(scene, movable_ids, container_id, directory, name='collect', ordered=True):
    directory=Path(directory)
    if len(set(movable_ids))!=len(movable_ids):raise ValueError('repeated_object_requires_regrasp_skill')
    objects={o['id']:o for o in scene['objects']}
    supports={s['id']:s for s in scene['supports']}
    support=supports[container_id]
    descriptor=read(resolve(support['container_descriptor']))
    if any(objects[oid]['kind']!='movable' for oid in movable_ids):raise ValueError('movable_role_not_bound')
    radii=[float(np.linalg.norm(objects[oid]['half_size'][:2])) for oid in movable_ids]
    layouts=disk_layouts(radii,descriptor['cavity']['flat_floor_radius_m'])
    trials=[];chosen=None
    container_mesh=trimesh.load(resolve(descriptor['collision']),force='mesh')
    caches={oid:read(resolve(scene['grasp_caches'][oid])) for oid in movable_ids}
    center=np.array(descriptor['cavity']['center_xy_m'])
    for li,offsets in enumerate(layouts):
        placed=[];candidates_by_object={};transforms={};trial={'layout':li,'checks':[]}
        for oid,offset in zip(movable_ids,offsets):
            obj=objects[oid]
            T=pose([*(center+offset),descriptor['cavity']['floor_z_m']+obj['half_size'][2]+.002],obj['yaw']-support['yaw'])
            obstacles=trimesh.util.concatenate([container_mesh,*placed])
            checker=VerticalClearance(obstacles)
            accepted=[]
            for c in caches[oid]['candidates'][:12]:
                ok,gap,reason=checker.check(T@np.asarray(c['tcp_in_asset']),-support['position'][2],.002)
                trial['checks'].append({'object':oid,'candidate':c['id'],'passed':bool(ok),'reason':reason})
                if ok:accepted.append(c)
            if not accepted:break
            candidates_by_object[oid]=accepted;transforms[oid]=T
            d=read(resolve(obj['asset_descriptor']))
            for b in d['collisions']:
                mesh=trimesh.load(resolve(b),force='mesh');mesh.apply_transform(T);placed.append(mesh)
        trials.append(trial)
        if len(candidates_by_object)==len(movable_ids):
            chosen=(li,offsets,candidates_by_object,transforms);break
    evidence={'schema':'native_multi_payload_allocation_v1','movables':movable_ids,
        'container':support['container_descriptor'],'payload_radii_m':radii,
        'floor_radius_m':descriptor['cavity']['flat_floor_radius_m'],
        'inter_payload_gap_m':.012,'floor_boundary_margin_m':.004,
        'geometric_layout_count':len(layouts),'trials':trials,'passed':chosen is not None,
        'physics_status':'unmeasured','runtime_rechecks_actual_placed_objects':True,
        'scope':'conservative floor disks and native world-Z open-hand sweep; full-arm paths checked online'}
    write(directory/'packing_allocation.json',evidence)
    if chosen is None:
        return None,None,'no_disk_capacity' if not layouts else 'no_native_hand_clearance_for_packing'
    li,offsets,candidates_by_object,transforms=chosen
    derived=copy.deepcopy(scene);operations=[]
    for oid,offset in zip(movable_ids,offsets):
        cache=copy.deepcopy(caches[oid])
        cache['signature'].update(parent_cache=scene['grasp_caches'][oid],
            packing_allocation=binding(directory/'packing_allocation.json'),allocation_code=binding(__file__))
        cache['key']=digest(cache['signature']);cache['candidates']=candidates_by_object[oid]
        write(directory/(oid+'_packing_cache.json'),cache)
        derived['grasp_caches'][oid]=binding(directory/(oid+'_packing_cache.json'))
        world_offset=(Rotation.from_euler('z',support['yaw']).as_matrix()@np.r_[center+offset,0])[:2]
        operations.append({'id':'pack_'+oid,'primitive':'pick_place','object':oid,'arm':None,
            'target':{'support':container_id,'offset_xy':world_offset.tolist(),'yaw':objects[oid]['yaw']}})
    task={'schema':'resimpair_task_intent_v4','id':scene['id']+'_'+name,'family':'sort',
        'required_skills':['pick_place'],'object_ids':list(movable_ids),
        'goal':name.replace('_',' '),'roles':{'movables':list(movable_ids),'destination':container_id},
        'operations':operations,'required_operation_precedence':
            [[a['id'],b['id']] for a,b in zip(operations,operations[1:])] if ordered else [],
        'equivalence_groups':[],'incidental_controller_order_is_required':False,
        'human_hand_policy':'preserve_robot_body_side','non_target_policy':'preserve_identity_and_pose',
        'release_precondition_control':True,'source_coverage':'synthetic_only',
        'packing_allocation':binding(directory/'packing_allocation.json'),
        'scope':'repeated existing pick/place with distinct objects; no stacking, handover or regrasp'}
    return derived,task,None
