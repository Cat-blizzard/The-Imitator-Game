"""Additional immutable rules; retained generic_mesh_acceptance_v1 still runs.

The controller does not import this module. New container checks do not change
the old event, contact, motion, mass, camera or non-target thresholds.
"""
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from resimpair_grasp.common import binding,read,resolve,write

VERSION='stationary_container_acceptance_v1'


def bindings():
    from resimpair_grasp import evaluator
    return {**evaluator.bindings(),'container_evaluator.py':binding(__file__)}


def compare_triangle_mesh(expected,actual):
    """All vertices and all triangles, not a bounding-box or color comparison."""
    source=trimesh.Trimesh(vertices=expected.vertices,faces=expected.faces,process=True)
    vertices=np.asarray(actual['vertices'],float);faces=np.asarray(actual['triangles'],int)
    tree=cKDTree(source.vertices);distance,nearest=tree.query(vertices)
    reverse=cKDTree(vertices).query(source.vertices)[0]
    error=float(max(distance.max(),reverse.max()))
    if error>=1e-5:raise ValueError('container_native_vertex_difference')
    a=np.unique(np.sort(nearest[faces],axis=1),axis=0)
    b=np.unique(np.sort(source.faces,axis=1),axis=0)
    if not np.array_equal(a,b):raise ValueError('container_native_triangle_difference')
    if not np.allclose(actual['scale'],[1,1,1]) or not np.allclose(actual['local_pose'],np.eye(4),atol=1e-6):
        raise ValueError('container_native_transform_difference')
    if actual['type']!='PhysxCollisionShapeTriangleMesh':raise ValueError('container_cavity_must_not_use_convex_hull')
    return {'passed':True,'maximum_vertex_difference_m':error,'triangles':len(a)}


def footprint(mesh,object_world,container_world,cavity):
    local=np.linalg.inv(container_world)@object_world
    vertices=mesh.vertices@local[:3,:3].T+local[:3,3]
    radial=np.linalg.norm(vertices[:,:2]-np.asarray(cavity['center_xy_m']),axis=1)
    margin=float(cavity['flat_floor_radius_m']-radial.max())
    return {'passed':bool(margin>=.002),'minimum_floor_disk_margin_m':margin,
            'lowest_vertex_above_declared_floor_m':float(vertices[:,2].min()-cavity['floor_z_m'])}


def evaluate(directory):
    from resimpair_grasp import evaluator as retained
    from resimpair_tabletop.sample_report import native_scalars as json_safe
    from corl2026_workshop.experiments.autonomous_pipeline_v4.core import checked
    directory=Path(directory);output=directory/'container_audit.json'
    if output.exists():return read(output)
    # The retained evaluator can contain numpy.bool_; serialize via the existing
    # report adapter, without changing any acceptance expression.
    old_write=retained.write
    try:
        retained.write=lambda p,v:old_write(p,json_safe(v))
        old=retained.evaluate(directory)
    finally:retained.write=old_write
    result={'version':VERSION,'rule_bindings':bindings(),'retained_audit':binding(directory/'audit.json'),
            'passed':False,'checks':{},'errors':[],'api_calls':0}
    try:
        bundle=read(directory/'execution_bundle.json');scene=bundle['scene']
        rows=read(directory/'native_containers.json')['containers']
        if len(rows)!=len(scene['supports']):raise ValueError('missing_native_container')
        geometry=[];world={}
        for row,item in zip(rows,scene['supports']):
            if row['id']!=item['id'] or row['descriptor']!=item['container_descriptor'] or not row['static']:
                raise ValueError('container_binding_mismatch')
            if len(row['shapes'])!=1:raise ValueError('container_native_shape_count')
            d=read(resolve(item['container_descriptor']))
            m=trimesh.load(resolve(d['collision']),force='mesh',process=False)
            T=np.eye(4);T[:3,:3]=Rotation.from_euler('z',item.get('yaw',0)).as_matrix();T[:3,3]=item['position']
            if not np.allclose(row['world_pose'],T,atol=1e-6):raise ValueError('container_native_pose_difference')
            world[item['id']]=(T,d)
            geometry.append({'id':item['id'],**compare_triangle_mesh(m,row['shapes'][0])})
        result['checks']['native_container_geometry']={'passed':True,'rows':geometry}
        with np.load(checked(bundle['measurements']),allow_pickle=False) as data:
            arrays={k:data[k] for k in data.files}
        task=bundle['task_intent'];checks=[]
        for op in task['operations']:
            if op['target']['support']=='table':continue
            index=[x['id'] for x in scene['objects']].index(op['object'])
            descriptor=read(resolve(scene['objects'][index]['asset_descriptor']))
            mesh=trimesh.load(resolve(descriptor['visual']),force='scene',process=True).to_geometry()
            T,d=world[op['target']['support']]
            # All of the final settled window, not only a convenient screenshot.
            states=arrays['object_states'][-20:,index]
            poses=[]
            for state in states:
                p=np.eye(4);p[:3,3]=state[:3]
                p[:3,:3]=Rotation.from_quat(state[3:7],scalar_first=True).as_matrix()
                poses.append(p)
            samples=[footprint(mesh,p,T,d['cavity']) for p in poses]
            checks.append({'object':op['object'],'support':op['target']['support'],
                           'passed':bool(len(samples)==20 and all(x['passed'] for x in samples)),
                           'samples':samples})
        result['checks']['payload_inside_qualified_floor']={'passed':all(x['passed'] for x in checks),'rows':checks}
        result['passed']=bool(old['passed'] and all(x['passed'] for x in result['checks'].values()))
    except Exception as exc:result['errors'].append(repr(exc))
    write(output,json_safe(result));return result
