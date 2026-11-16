"""Isolated fixed-task-v4: both fruit link7 solids use tight25mm coverage.

The unchanged v3 exporter/validator supplies actual PhysX collision vertices,
scale and shape-local XYZ/WXYZ poses. The unchanged six-LP full-solid cell
algorithm uses fixed25mm/maxdepth2, with no q/seed/obstacle-dependent selection.
Both fruit arms use the same rule. Existing v2 payload coverage, every other
robot link, world shape, physical state, limit and collision mask stay exact.
Foreign pre-existing link7 recipes are rejected; a valid v4 snapshot is
idempotent. Other tasks return exactly the previous v2 result.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import difflib
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'infrastructure/development/tig-dual-arm').is_dir())
DEV=ROOT/'infrastructure/development/tig-dual-arm'
for directory in (DEV,ROOT/'infrastructure'):
    if str(directory) not in sys.path:sys.path.insert(0,str(directory))
from mani_skill.examples.motionplanning.dual import curobo_fruit_link7_v3 as v3

v1,v2=v3.v1,v3.v2
LINK,FRUIT=v3.LINK,v3.FRUIT
PITCH_M=.025
POLICY='fixed-task-v4'
RECIPE='fixed-fruit-both-link7-actual-convex-solid-tight25mm-v4'
BASE_RUNNER_SHA256='423168839c332e4a1c18a26741612fd82dee7a954609c8308b7cd2a1ba485cef'
E=v3.E
P=v3.P
geometry_from_initial_scene=v3.geometry_from_initial_scene
export_link7_geometry=v3.export_link7_geometry


def _require(condition: bool,message: str) -> None:
    if not condition:raise ValueError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value,dtype='<f8').tobytes()).hexdigest()


@lru_cache(maxsize=32)
def _cached_cover(key: str) -> tuple[tuple[tuple[float,...],...],str]:
    """Immutable geometry+fixed recipe cache; no scene, state, seed or q enters."""
    data=json.loads(key)
    _require(data.get('recipe')==RECIPE and data.get('pitch_m')==PITCH_M,'Exact fixed v4 geometry policy required')
    rows,proofs=[],[]
    for shape in data['shapes']:
        native=shape['native_local_geometry']
        vertices=np.asarray(native['vertices_shape_local_unscaled_m'],dtype=np.float64)*np.asarray(native['native_scale_xyz'])
        spheres,proof=v1.tight_convex_spheres(vertices,PITCH_M,max_depth=2)
        spheres[:,:3]=v1._transform(spheres[:,:3],v1._pose_matrix(shape['shape_local_pose_raw_xyz_wxyz']))
        proof.update(link=LINK,shape_index=shape['shape_index'],type=shape['type'],
            native_local_geometry_sha256=shape['native_local_geometry_sha256'],
            shape_local_pose_raw_xyz_wxyz=deepcopy(shape['shape_local_pose_raw_xyz_wxyz']),
            source_frame='physical_shape_local',output_frame='physical_link_local',policy=RECIPE,
            task_or_seed_or_pose_or_obstacle_dependent=False,physical_collision_geometry_unchanged=True,
            collision_masks_unchanged=True)
        rows.extend(spheres.tolist());proofs.append(proof)
    array=v1._array(rows,4,'complete v4 link-local sphere cover')
    _require(bool(np.all(array[:,3]>0)),'Positive full-cover sphere radii required')
    return tuple(tuple(float(x) for x in row) for row in array),json.dumps(proofs,sort_keys=True,allow_nan=False)


def _cover(shapes: list[dict[str,Any]]) -> tuple[np.ndarray,list[dict[str,Any]]]:
    rows,proof=_cached_cover(json.dumps({'recipe':RECIPE,'pitch_m':PITCH_M,'shapes':shapes},sort_keys=True,allow_nan=False))
    return np.array(rows,dtype=np.float64,copy=True),json.loads(proof)


def refine_task_snapshot_v4(snapshot: dict[str,Any],geometry_manifest: dict[str,Any]|None=None) -> dict[str,Any]:
    """Fresh baseline/v2 input is normalized to v2, then only link7 changes.

    Idempotence requires the exact v4 recipe/proof/geometry. Any existing v3 or
    foreign link7 recipe is rejected rather than silently converted.
    """
    task,arm=v1._validate_snapshot(snapshot)
    if task!=FRUIT:return v2.refine_task_snapshot_v2(snapshot)
    previous=snapshot['coverage'].get('link7_sphere_refinement')
    if previous is not None:
        _require(isinstance(previous,dict) and previous.get('recipe')==RECIPE,'Foreign link7 recipe cannot be silently promoted to v4')
    shapes=v3._validated_geometry(geometry_manifest,arm)
    base=v2.refine_task_snapshot_v2(snapshot)
    old=[r for r in base['coverage']['active_robot_shapes'] if r['link']==LINK]
    _require(len(old)==len(shapes) and [r['shape_index'] for r in old]==list(range(len(shapes))),
             'Actual complete link7 inventory differs from original export')
    spheres,proofs=_cover(shapes)
    result=deepcopy(base);kin=result['robot_cfg']['kinematics'];coverage=result['coverage']
    original_count=previous['original_sphere_count'] if previous is not None else len(kin['collision_spheres'][LINK])
    kin['collision_spheres'][LINK]=[{'center':row[:3].tolist(),'radius':float(row[3])} for row in spheres]
    coverage['active_robot_shapes']=[r for r in coverage['active_robot_shapes'] if r['link']!=LINK]+deepcopy(proofs)
    coverage['robot_collision_sphere_count']=sum(len(a) for a in kin['collision_spheres'].values())
    coverage['link7_sphere_refinement']={'recipe':RECIPE,'sphere_pitch_m':PITCH_M,'physical_arm_index':arm,
        'original_sphere_count':original_count,'refined_sphere_count':len(spheres),'native_shape_count':len(shapes),
        'actual_geometry_manifest':deepcopy(geometry_manifest),'shape_proofs':deepcopy(proofs),
        'all_actual_link7_solids_covered':True,'physical_geometry_unchanged':True,
        'world_and_nonlink7_geometry_exactly_preserved':True,'collision_permissions_and_limits_unchanged':True,
        'task_or_seed_or_pose_or_obstacle_dependent_sphere_selection':False}
    _require(v3._nonlink7_view(result)==v3._nonlink7_view(base),'A non-link7 v2 field changed')
    if previous is not None:
        _require(previous==coverage['link7_sphere_refinement'] and result==base,'Stale v4 geometry/proof/recipe')
    v1._validate_snapshot(result);json.dumps(result,allow_nan=False)
    return result


def proposed_runner_source(source: str) -> str:
    """Only add explicit v4 and reuse v3 actual geometry export; no writes."""
    _require(hashlib.sha256(source.encode()).hexdigest()==BASE_RUNNER_SHA256,'Reviewed v3 runner base SHA differs')
    edits=(
        ("PAYLOAD_REFINEMENTS = ('baseline', 'fixed-task-v1', 'fixed-task-v2', 'fixed-task-v3')",
         "PAYLOAD_REFINEMENTS = ('baseline', 'fixed-task-v1', 'fixed-task-v2', 'fixed-task-v3', 'fixed-task-v4')"),
        ("        return refine_task_snapshot_v3(snapshot,geometry_manifest)\n    raise ValueError('Unknown payload refinement policy')\n",
         "        return refine_task_snapshot_v3(snapshot,geometry_manifest)\n    if policy == 'fixed-task-v4':\n        from .curobo_fruit_link7_v4 import refine_task_snapshot_v4\n        return refine_task_snapshot_v4(snapshot,geometry_manifest)\n    raise ValueError('Unknown payload refinement policy')\n"),
        ("            if payload_refinement == 'fixed-task-v3' and args.task == TASKS[2]:\n",
         "            if payload_refinement in ('fixed-task-v3','fixed-task-v4') and args.task == TASKS[2]:\n"),
        ("    assert configured_payload_snapshot(apple,'fixed-task-v3') == apple\n",
         "    assert configured_payload_snapshot(apple,'fixed-task-v3') == apple\n    assert configured_payload_snapshot(apple,'fixed-task-v4') == apple\n"))
    result=source
    for before,after in edits:
        _require(result.count(before)==1,'Reviewed v3 policy/export anchor differs')
        result=result.replace(before,after,1)
    before="            if payload_refinement == 'fixed-task-v3' else\n"
    after="            if payload_refinement in ('fixed-task-v3','fixed-task-v4') else\n"
    _require(result.count(before)==2,'Expected both exact existing v3 geometry-scope metadata anchors')
    result=result.replace(before,after)
    ast.parse(result);return result


def interface_patch() -> str:
    source=(E/'candidates/fruit_link7_v3/proposed-validate_curobo_complete.py.txt').read_text()
    name='infrastructure/development/tig-dual-arm/mani_skill/examples/motionplanning/dual/validate_curobo_complete.py'
    return ''.join(difflib.unified_diff(source.splitlines(True),proposed_runner_source(source).splitlines(True),fromfile='a/'+name,tofile='b/'+name))


def self_test() -> dict[str,Any]:
    """Actual complete snapshots/protocols, idempotence, independent cover and negatives."""
    started=time.perf_counter();patch=interface_patch();assert POLICY in patch
    rejected=[];rows=[];_cached_cover.cache_clear()
    def reject(label: str,call: Callable[[],Any]) -> None:
        try:call()
        except (ValueError,TypeError,KeyError):rejected.append(label)
        else:raise AssertionError('Invalid v4 candidate accepted: '+label)
    initial=json.loads((P/'initial-scene.json').read_text())
    geometries=[geometry_from_initial_scene(initial,a) for a in (0,1)]
    scan=json.loads((E/'candidates/fruit_link7_fixed_pitch_scan/pitch25-scan.json').read_text())
    assert scan['both_starts_clear'] and scan['safe_candidates']==8 and scan['mapped_candidate_count']==9
    for arm in (0,1):
        path=P/f'carry_lower-arm-{arm}-snapshot.json';snapshot=json.loads(path.read_text());unchanged=deepcopy(snapshot)
        value=refine_task_snapshot_v4(snapshot,geometries[arm])
        assert snapshot==unchanged and value==refine_task_snapshot_v4(value,geometries[arm])
        assert v3._nonlink7_view(value)==v3._nonlink7_view(v2.refine_task_snapshot_v2(snapshot))
        spheres=np.asarray([[*s['center'],s['radius']] for s in value['robot_cfg']['kinematics']['collision_spheres'][LINK]])
        assert spheres.shape==(64,4) and _array_sha(spheres)==scan['arms'][arm]['sphere_float64_LE_sha256']
        assert np.array_equal(spheres,np.asarray(scan['arms'][arm]['spheres_link_local_xyzr_m']))
        info=_cached_cover.cache_info();again=refine_task_snapshot_v4(snapshot,geometries[arm]);assert again==value
        assert _cached_cover.cache_info().hits==info.hits+1
        # Cache returns fresh mutable arrays/proofs, never aliases its storage.
        a,p=_cover(v3._validated_geometry(geometries[arm],arm));a[0,0]+=1;p[0]['policy']='mutated'
        b,q=_cover(v3._validated_geometry(geometries[arm],arm));assert _array_sha(b)==_array_sha(spheres) and q[0]['policy']==RECIPE
        for profile in ('approach','transport'):
            for operation in ('init_session','update_session'):
                request=v1._request(value,operation,profile)
                v1.validate_task_request(request,FRUIT);v2.legacy_validate_request(v1.validation_view(request,FRUIT))
        rows.append({'arm':arm,'input':str(path),'input_sha256':_sha(path),'sphere_count':len(spheres),
            'sphere_float64_LE_sha256':_array_sha(spheres),'exactly_equal_independent_fullscan_cover':True,
            'all_nonlink7_v2_fields_exact':True,'idempotent':True,'cache_copy_isolated':True,'CPU_protocol_validators':'passed'})
    for task in v1.TASK_IDS[:2]:
        for arm in (0,1):
            raw=json.loads((E/f'task-source-snapshots-dev02/{task}-arm-{arm}.json').read_text())
            assert refine_task_snapshot_v4(raw)==v2.refine_task_snapshot_v2(raw)
    base=json.loads((P/'carry_lower-arm-1-snapshot.json').read_text());geometry=geometries[1]
    reject('missing_actual_geometry',lambda:refine_task_snapshot_v4(base))
    reject('other_arm_geometry',lambda:refine_task_snapshot_v4(base,geometries[0]))
    for label,mutate in (
        ('wrong_geometry_hash',lambda d:d.update(geometry_sha256='0'*64)),
        ('wrong_link',lambda d:d.update(link_name='panda_link6')),
        ('shape_missing',lambda d:d.update(shapes=[])),
        ('visual_geometry',lambda d:d['source'].update(urdf_or_visual_mesh_used=True)),
        ('stepped_source',lambda d:d['source'].update(environment_steps_during_export=1))):
        bad=deepcopy(geometry);mutate(bad);reject(label,lambda bad=bad:refine_task_snapshot_v4(base,bad))
    foreign=deepcopy(base);foreign['coverage']['link7_sphere_refinement']={'recipe':v3.RECIPE}
    reject('old_v3_recipe_rejected',lambda:refine_task_snapshot_v4(foreign,geometry))
    value=refine_task_snapshot_v4(base,geometry);value['coverage']['link7_sphere_refinement']['sphere_pitch_m']=.0125
    reject('stale_v4_metadata',lambda:refine_task_snapshot_v4(value,geometry))
    value=refine_task_snapshot_v4(base,geometry);value['robot_cfg']['kinematics']['collision_spheres'][LINK][0]['radius']*=.5
    reject('stale_v4_spheres',lambda:refine_task_snapshot_v4(value,geometry))
    cube=np.array([[x,y,z] for x in (-.013,.017) for y in (-.011,.019) for z in (-.007,.023)])
    spheres,proof=v1.tight_convex_spheres(cube,PITCH_M,max_depth=2)
    grid=np.stack(np.meshgrid(*[np.linspace(cube[:,i].min(),cube[:,i].max(),13) for i in range(3)],indexing='ij'),axis=-1).reshape(-1,3)
    T=v1._pose_matrix([.1,-.2,.3,np.sqrt(.5),0.,0.,np.sqrt(.5)])
    for points,cover in ((grid,spheres),(v1._transform(grid,T),np.c_[v1._transform(spheres[:,:3],T),spheres[:,3]])):
        assert (np.linalg.norm(points[:,None]-cover[None,:,:3],axis=2)-cover[None,:,3]).min(axis=1).max()<=1e-8
    assert proof['complete_solid_coverage']
    import torch
    assert not torch.cuda.is_initialized()
    return {'status':'passed','fixed_pitch_m':PITCH_M,'max_depth':2,'actual_snapshots':rows,
        'four_nonfruit_arms_exactly_v2':True,'synthetic_solid_grid_points':len(grid),'rigid_transform_checks':2,
        'rejected_inputs':rejected,'cache':_cached_cover.cache_info()._asdict(),
        'actual_live_export_implementation':'unchanged v3 actual PhysX exporter; v4 live integration NOT_RUN',
        'fullscan_reference_sha256':_sha(E/'candidates/fruit_link7_fixed_pitch_scan/pitch25-scan.json'),
        'native_memory_fit':'UNKNOWN','environment_steps':0,'GPU_initialized':False,'native_planning':'NOT_RUN',
        'helper_sha256':_sha(Path(__file__)),'wall_s':time.perf_counter()-started}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--emit-patch',action='store_true')
    parser.add_argument('--output-json',type=Path);args=parser.parse_args()
    if args.emit_patch:
        if args.self_test or args.output_json is not None:parser.error('Patch output is independent of CPU receipt')
        print(interface_patch(),end='');raise SystemExit(0)
    if not args.self_test:parser.error('Provide --self-test or --emit-patch')
    result=self_test();text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output_json is not None:
        with args.output_json.open('x') as f:f.write(text)
    print(text,end='')
