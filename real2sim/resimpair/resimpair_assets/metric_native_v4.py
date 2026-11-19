"""Canonical native support frame with bounded CPU settling, before grasp solve."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import time
import numpy as np
import trimesh
from resimpair_grasp.common import binding, digest, read, resolve, write
from . import metric_native_v3 as prior

VERSION='metric_native_settled_frame_v4'


def settle(descriptor_ref, output):
    import sapien
    start=time.monotonic(); d=read(resolve(descriptor_ref))
    system=sapien.physx.PhysxCpuSystem();scene=sapien.Scene([system]);scene.set_timestep(1/240)
    material=sapien.physx.PhysxMaterial(.6,.6,0.)
    floor=sapien.Entity();floor.name='support_plane';static=sapien.physx.PhysxRigidStaticComponent()
    static.attach(sapien.physx.PhysxCollisionShapeBox(half_size=[1,1,.05],material=material))
    floor.add_component(static);floor.pose=sapien.Pose([0,0,-.05]);scene.add_entity(floor)
    obj=sapien.Entity();obj.name='payload';body=sapien.physx.PhysxRigidDynamicComponent()
    for ref in d['collisions']:
        shape=sapien.physx.PhysxCollisionShapeConvexMesh(filename=str(resolve(ref)),scale=[1,1,1],material=material)
        shape.density=d['density_kg_m3'];body.attach(shape)
    obj.add_component(body); initial_z=-d['collision_bounds_m'][0][2]+.001
    obj.pose=sapien.Pose([0,0,initial_z]);scene.add_entity(obj)
    records=[];touches=0
    for frame in range(360):
        scene.step()
        pose=obj.pose;R=pose.to_transformation_matrix()[:3,:3]
        v=float(np.linalg.norm(body.linear_velocity));w=float(np.linalg.norm(body.angular_velocity))
        if frame>=300:
            records.append({'frame':frame,'position_m':pose.p.tolist(),'quaternion_wxyz':pose.q.tolist(),
                'linear_speed_m_s':v,'angular_speed_rad_s':w,'asset_up_dot_world_up':float(R[2,2])})
        for contact in scene.get_contacts():
            if {contact.bodies[0].entity.name,contact.bodies[1].entity.name}=={'payload','support_plane'}:
                touches+=1
    last=records[-1];bounds=np.asarray(d['collision_bounds_m']);expected_z=-bounds[0,2]
    criteria={'max_final_window_linear_speed_m_s':.005,'max_final_window_angular_speed_rad_s':.05,
        'minimum_asset_up_dot_world_up':.995,'maximum_xy_drift_m':.003,
        'maximum_support_height_error_m':.003,'contact_observed_required':True}
    checks={'resting':max(r['linear_speed_m_s'] for r in records)<=.005 and max(r['angular_speed_rad_s'] for r in records)<=.05,
        'canonical_up_preserved':min(r['asset_up_dot_world_up'] for r in records)>=.995,
        'xy_preserved':float(np.linalg.norm(last['position_m'][:2]))<=.003,
        'height_preserved':abs(last['position_m'][2]-expected_z)<=.003,
        'native_support_contact_observed':touches>0}
    checks={key:bool(value) for key,value in checks.items()}
    result={'schema':VERSION,'asset':descriptor_ref,'status':'pass' if all(checks.values()) else 'fail',
        'criteria':criteria,'checks':checks,'frames':360,'dt_seconds':1/240,
        'initial_position_m':[0,0,initial_z],'expected_resting_origin_z_m':float(expected_z),
        'final_window':records,'support_contact_frames':touches,
        'mass_kg':float(body.mass),'cmass_local_pose_pq':list(map(float,body.cmass_local_pose.p))+list(map(float,body.cmass_local_pose.q)),
        'principal_inertia_kg_m2':body.inertia.tolist(),'elapsed_seconds':time.monotonic()-start,
        'robot_grasp_or_place_test':False,'real_physical_calibration':False,'GPU_jobs':0,'API_calls':0}
    write(output,result);return result


def canonicalize(descriptor_ref,output):
    """One deterministic most-probable support orientation using native COM."""
    output=Path(output);output.mkdir(parents=True)
    d=read(resolve(descriptor_ref));native=read(resolve(d['native_properties']))
    parts=[trimesh.load(resolve(x),force='mesh',process=True) for x in d['collisions']]
    hull=trimesh.util.concatenate(parts).convex_hull
    transforms,probabilities=trimesh.poses.compute_stable_poses(hull,
        center_mass=np.asarray(native['center_of_mass_m']),sigma=0,n_samples=1,threshold=.01)
    if not len(transforms):raise ValueError('no_native_COM_stable_support_pose')
    T=transforms[int(np.argmax(probabilities))].copy()
    visual=trimesh.load(resolve(d['visual']),force='scene',process=False);visual.apply_transform(T)
    shift=-visual.bounds.mean(0);T[:3,3]+=shift;visual.apply_translation(shift)
    visual.export(output/'visual.glb'); refs=[]
    for i,part in enumerate(parts):
        part.apply_transform(T);path=output/f'collision_{i}.obj';part.export(path);refs.append(binding(path))
    value=copy.deepcopy(d);value.update(visual=binding(output/'visual.glb'),collisions=refs,
        bounds_m=visual.bounds.tolist(),dimensions_m=visual.extents.tolist(),half_size=(visual.extents/2).tolist(),
        collision_bounds_m=np.array([np.min([x.bounds[0] for x in parts],axis=0),np.max([x.bounds[1] for x in parts],axis=0)]).tolist(),
        source_to_asset=(T@np.asarray(d['source_to_asset'])).tolist(),
        canonical_support_transform_from_parent=T.tolist(),
        canonical_frame_semantics='asset_plus_Z_is_support_normal; original semantic up retained separately',
        original_semantic_up_in_canonical=(T[:3,:3]@np.array([0.,0.,1.])).tolist(),
        metric_parent_before_canonicalization=descriptor_ref)
    value['geometry_variant_key']=digest({'parent':descriptor_ref,'canonical_transform':T.tolist()})
    value['id']=value['geometry_variant_key'][:24]
    path=output/'descriptor.json';write(path,value);return binding(path)


def compile_variant(descriptor_path, output):
    output=Path(output);start=time.monotonic()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    first=prior.compile_variant(descriptor_path,output/'initial_native')
    settling=settle(first['asset'],output/'initial_settle.json')
    selected=first;reoriented=False
    if settling['status']!='pass' or first['support_status']!='pass':
        canonical=canonicalize(first['asset'],output/'canonical_geometry')
        selected=prior.compile_variant(resolve(canonical),output/'canonical_native')
        settling=settle(selected['asset'],output/'canonical_settle.json');reoriented=True
    d=read(resolve(selected['asset']));support=read(resolve(d['native_support']))
    support={**support,'geometry_support':d['native_support'],
        'cpu_settle':binding(output/('canonical_settle.json' if reoriented else 'initial_settle.json')),
        'status':'pass' if support['status']=='pass' and settling['status']=='pass' else 'fail',
        'dynamic_settling':settling['status'],'physical_stability':'verified_only_for_bound_table_static_support_test',
        'canonicalized':reoriented,'semantic_up_is_separate_from_stable_support_up':True}
    write(output/'native_support.json',support)
    d.update(native_support=binding(output/'native_support.json'),
        native_compiler={'schema':VERSION,'implementation':binding(__file__),
            'input_metric_descriptor':binding(descriptor_path),'parent_native':selected['asset']})
    d['roles']['rigid_manipulation_candidate'] &= support['status']=='pass'
    d['qualification']['native_support_settle']=settling['status']
    write(output/'descriptor.json',d)
    result={**selected,'schema':VERSION,'asset':binding(output/'descriptor.json'),
        'native_support':binding(output/'native_support.json'),'canonicalized':reoriented,
        'geometry_qualified':d['roles']['rigid_manipulation_candidate'],
        'qualification_stage':'native_cook_geometry_COM_inertia_and_CPU_settle',
        'settle_status':settling['status'],'elapsed_seconds':time.monotonic()-start,
        'dynamic_steps':360*(2 if reoriented else 1)}
    write(output/'result.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--descriptor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(compile_variant(a.descriptor,a.output))
