"""Original-source mesh -> metric variant -> retained native qualification.

No old native descriptor is a prerequisite. Old inputs are only provenance;
geometry, mass, support frame and any future grasp cache are bound afresh.
"""
import argparse
from pathlib import Path
import time
import traceback

import numpy as np
import trimesh

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from .raw_mesh_io_v1 import load_raw_scene, export_visual
from .raw_size_policy_v1 import decide_raw, metric_policy
from . import metric_variants_v2 as variants
from . import metric_native_v6 as native
from .metric_admission_v1 import inspect


def completed_settle_steps(output):
    return sum(read(p).get('frames',0) for pattern in ('**/initial_settle.json','**/canonical_settle.json')
               for p in Path(output).glob(pattern))


def compile_raw(job_path):
    start = time.monotonic(); job = read(job_path)
    folder = Path(job['output']); row = job['raw_row']
    scene, mesh, raw_receipt = load_raw_scene(row)
    write(folder / 'raw_input.json', raw_receipt)
    hull = mesh.convex_hull
    original_obb = np.asarray(hull.bounding_box_oriented.primitive.extents)
    decision = decide_raw(row, original_obb, metadata_records=job.get('metadata_records',()), seed=job['seed'])
    write(folder / 'size_decision.json', decision)
    if decision['role'] != 'manipulation':
        return {'source_id':row['id'], 'status':'source_size_or_role_requires_other_use',
            'decision':binding(folder/'size_decision.json'),'elapsed_seconds':time.monotonic()-start,
            'recorded_settle_steps':0,'robot_physical_executions':0,'API_calls':0,'GPU_jobs':0}
    scale = decision['source_unit_to_m']
    scaled_hull = hull.copy(); scaled_hull.apply_scale(scale)
    if len(scaled_hull.faces)>400:
        stable = scaled_hull.simplify_quadric_decimation(face_count=400).convex_hull
    else: stable = scaled_hull
    poses, probabilities = trimesh.poses.compute_stable_poses(stable,n_samples=1,sigma=0.,threshold=.005)
    if not len(poses):raise ValueError('raw_metric_no_geometric_support_pose')
    orientation = poses[int(np.argmax(probabilities))]
    transform = orientation @ np.diag([scale,scale,scale,1.])
    visual = scene.copy(); visual.apply_transform(transform)
    translation = -visual.bounds.mean(axis=0)
    transform[:3,3] += translation; visual.apply_translation(translation)
    write(folder/'raw_frame.json',{'source_to_initial_asset':transform.tolist(),
        'source_basis_directions_in_initial_asset':orientation[:3,:3].tolist(),
        'initial_support_pose_selection':'maximum_geometric_stable_pose_probability',
        'semantic_upright_axis':'unknown_not_inferred_from_geometric_support',
        'native_settling_may_apply_a_further_canonical_transform':True})
    actual = visual.to_geometry()
    visual_path = folder/'raw_metric/visual.glb'
    material = export_visual(visual, visual_path, actual)
    write(folder/'raw_metric/material_export.json',material)
    canonical_hull = actual.convex_hull
    # Cheap initial approximation choice; the unchanged native checker decides
    # whether its final surface error is acceptable, not this heuristic.
    points,_ = trimesh.sample.sample_surface(actual,1200,seed=71)
    errors = trimesh.proximity.closest_point(canonical_hull,points)[1]
    collision_path = folder/'raw_metric/initial_hull.obj'
    canonical_hull.export(collision_path,digits=12)
    parent = {'schema':'resimpair_mesh_asset_v1','id':digest([row['id'],raw_receipt,decision])[:24],
        'source_id':row['id'],'name':row.get('name',row['id']),'category':decision['category'],
        'partition':row['partition'],
        'provenance':{'source':row['source'],'original_sha256':row['source_sha'],
            'raw_input':binding(folder/'raw_input.json'),'raw_size_decision':binding(folder/'size_decision.json'),
            'license':job.get('license'),'prior_native':job.get('prior_native'),
            'source_aliases':job.get('source_aliases',[]),'ingestion_group':job['ingestion_group']},
        'visual':binding(visual_path),'collisions':[binding(collision_path)],
        'source_to_asset':transform.tolist(),'dimensions_m':actual.extents.tolist(),
        'bounds_m':actual.bounds.tolist(),'half_size':(actual.extents/2).tolist(),
        'collision_bounds_m':canonical_hull.bounds.tolist(),
        'semantic_role':'manipulation','semantic_category':decision['category'],
        'qualification':{'geometry':'pending','physical_execution':'unmeasured'},
        'roles':{'rigid_manipulation_candidate':False,'static_distractor':False},
        'mass_kg':float(canonical_hull.volume*decision['effective_density_kg_m3']),
        'density_kg_m3':decision['effective_density_kg_m3'],
        'raw_source_uniform_scale':scale,'new_grasp_required':True}
    write(folder/'raw_metric/parent.json',parent)
    parent_ref = binding(folder/'raw_metric/parent.json')
    if float(np.max(errors))>.002:
        parent_ref = native.refine(parent_ref,folder/'metric_initial_decomposition')
    policy = metric_policy(decision,binding(folder/'size_decision.json'))
    write(folder/'metric_policy.json',policy)
    prepared = variants.materialize(resolve(parent_ref),folder/'metric_policy.json',folder/'metric_variant')
    descriptor = resolve(prepared['variants'][0]['descriptor'])
    native_result = native.compile_variant(descriptor,folder/'native')
    state = {'source_id':row['id'],'role':'manipulation','profile':decision['category'],
        'partition':row['partition'],'semantic_admission':'engineered_role_and_scale_candidate',
        'base_longest_m':decision['target_longest_m'],'size_confidence':decision['size_confidence'],
        'policy':binding(folder/'metric_policy.json'),'raw_size_decision':binding(folder/'size_decision.json')}
    write(folder/'source_state.json',state)
    entry = {**native_result,'source_aliases':job.get('source_aliases',[]),
        'ingestion_group':job['ingestion_group'],'role':'manipulation','source_state':binding(folder/'source_state.json')}
    gate = inspect(entry,state) if native_result['geometry_qualified'] else {
        'passed':False,'reasons':['native_geometry_or_support_not_qualified'],
        'current_policy':binding(folder/'metric_policy.json')}
    entry['semantic_scale_admission']=gate
    write(folder/'native_entry.json',entry)
    return {'source_id':row['id'],'status':'native_ready' if native_result['geometry_qualified'] and gate['passed'] else 'native_qualification_failed',
        'entry':binding(folder/'native_entry.json'),'size_decision':binding(folder/'size_decision.json'),
        'state':binding(folder/'source_state.json'),'elapsed_seconds':time.monotonic()-start,
        'recorded_settle_steps':completed_settle_steps(folder),
        'settle_scope':'asset-only CPU dynamics; not robot pick-place execution',
        'robot_physical_executions':0,'API_calls':0,'GPU_jobs':0,'physical_success':'unmeasured'}


def worker(job_path):
    job_path=Path(job_path).resolve();job=read(job_path);folder=Path(job['output']).resolve()
    if folder!=job_path.parent or (folder/'worker_started.json').exists():
        raise ValueError('fresh_bound_worker_output_required')
    reservation=read(folder/'cpu_reservation.json')
    if reservation['job']!=binding(job_path) or reservation['attempt']!=1:
        raise ValueError('registered_single_CPU_attempt_required')
    protocol=read(resolve(reservation['protocol']))
    if str(job_path) not in protocol['job_paths']:
        raise ValueError('worker_outside_frozen_raw_population')
    manifest=read(resolve(protocol['population']))
    expected={row['source_id']:row['job'] for row in manifest['sources']}[job['raw_row']['id']]
    if expected!=reservation['job'] or resolve(expected)!=job_path:
        raise ValueError('CPU_job_differs_from_frozen_population_binding')
    for ref in protocol['implementation_bindings']:resolve(ref)
    write(folder/'worker_started.json',{'job':binding(job_path),'unix':time.time()})
    began=time.monotonic()
    try:result=compile_raw(job_path)
    except Exception as error:
        from scipy.spatial import QhullError
        known=isinstance(error,(ValueError,QhullError))
        status='geometry_or_policy_failed' if known else 'unknown_system_failure'
        if str(error)=='raw_format_requires_explicit_dependency_adapter':status='raw_format_adapter_pending'
        result={'source_id':job['raw_row']['id'],'status':status,
            'reason':type(error).__name__+':'+str(error),'traceback':traceback.format_exc(),
            'elapsed_seconds':time.monotonic()-began,'recorded_settle_steps':completed_settle_steps(folder),
            'robot_physical_executions':0,'API_calls':0,'GPU_jobs':0}
    write(folder/'cpu_result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    a=p.parse_args();print(worker(a.job))
