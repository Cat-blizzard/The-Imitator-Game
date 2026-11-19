"""Geometry-only destination census; never assign or change task contracts."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import time

import numpy as np
import trimesh
from resimpair_grasp.common import binding, read, resolve
from resimpair_pickplace.support_surface import check_support

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'validation/paired_assets_20261030/independent/container_affordance_v1'
CAT=ROOT/'runs/metric_assets_20261030_v7/container_union_v6/sampler_catalog.json'
CAP=ROOT/'validation/paired_assets_20261030/independent/metric_capacity_v3_review_v2/FINAL_REVIEW.json'


def save(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def band(height,floor_fraction,depth):
    # Descriptive bins only. These do not grant/withdraw container eligibility.
    if floor_fraction>=.75 and depth<=.03:return 'elevated_shallow_recess_or_support_surface'
    if depth<=.03:return 'shallow_tray_like_cavity'
    if floor_fraction<=.5 and depth/height>=.4:return 'deeper_open_cavity_geometry_candidate'
    return 'intermediate_cavity_geometry_unknown_semantics'


def quantiles(values):
    return dict(zip(['min','p25','median','p75','max'],np.quantile(values,[0,.25,.5,.75,1]).tolist()))


def main():
    OUT.mkdir(exist_ok=False)
    catalog=read(CAT);entries=catalog['assets']
    protocol={'schema':'independent_geometry_affordance_census_v1','reviewer':binding(__file__),
        'catalog':binding(CAT),'capacity_audit':binding(CAP),'registered_variants':len(entries),
        'descriptive_bins_not_acceptance_rules':{'elevated_shallow_recess_or_support_surface':'floor >=75% of native body height AND depth<=30mm',
            'shallow_tray_like_cavity':'remaining depth<=30mm',
            'deeper_open_cavity_geometry_candidate':'remaining floor<=50% of height AND cavity depth>=40% of body height',
            'intermediate_cavity_geometry_unknown_semantics':'all other continuous measurements'},
        'point':'Names excluded from geometry classification; full continuous metrics and nominal object-height/rim relationship reported. No contract rewriting.',
        'native_geometry':'Load bound existing collision triangles without rescaling/cooking; repeat retained97-ray upward support test.',
        'physical_steps':0,'GPU_jobs':0,'API_calls':0}
    save(OUT/'PROTOCOL.json',protocol);began=time.monotonic();rows=[]
    for r in entries:
        d=read(resolve(r));mesh=trimesh.load(resolve(d['collision']),force='mesh',process=False)
        bounds=np.asarray(mesh.bounds);height=float(bounds[1,2]-bounds[0,2]);floor=float(d['cavity']['floor_z_m']);rim=float(d['cavity']['rim_z_m'])
        depth=rim-floor;ff=(floor-bounds[0,2])/height;rf=(rim-bounds[0,2])/height
        support=check_support(mesh,d['cavity']);label=band(height,ff,depth)
        rows.append({'descriptor':r,'source_id':d['source_id'],'variant_id':d.get('variant_id',d['id']),
            'name':d['name'],'partition':d['partition'],'native_body_height_m':height,'native_min_z_m':float(bounds[0,2]),
            'native_max_z_m':float(bounds[1,2]),'floor_z_m':floor,'rim_z_m':rim,'cavity_depth_m':depth,
            'floor_relative_body_height':float(ff),'rim_relative_body_height':float(rf),'depth_relative_body_height':depth/height,
            'flat_floor_radius_m':d['cavity']['flat_floor_radius_m'],'depth_to_floor_diameter_ratio':depth/(2*d['cavity']['flat_floor_radius_m']),
            'declared_and_actual_collision_bounds_max_difference_m':float(np.abs(np.asarray(d['bounds_m'])-bounds).max()),
            'retained_oriented_floor_support':support,'descriptive_geometry_bin':label,
            'semantic_affordance':'unknown_without_visual_or_task_specific_confirmation',
            'original_qualification_unchanged':True})
    by_sha={r['descriptor']['sha256']:r for r in rows};cap=read(CAP);taskrows=[]
    for s in cap['arms']['v3']['scenes']:
        scene=read(resolve(s['scene']));objects={o['id']:o for o in scene['objects']};containers={c['id']:c for c in scene['supports']}
        for task in s['tasks']:
            goals=[]
            for g in task['goal_floor_checks']:
                obj=objects[g['object']];d=read(resolve(obj['asset_descriptor']));c=by_sha[containers[g['container']]['container_descriptor']['sha256']]
                verts=np.concatenate([trimesh.load(resolve(x),force='mesh',process=False).vertices for x in d['collisions']])
                nominal_bottom=g['requested_goal_world'][2]+float(verts[:,2].min());nominal_top=g['requested_goal_world'][2]+float(verts[:,2].max())
                object_height=nominal_top-nominal_bottom
                below=float(np.clip((c['rim_z_m']-nominal_bottom)/object_height,0,1))
                suggested={'elevated_shallow_recess_or_support_surface':'on_support_surface_candidate',
                    'shallow_tray_like_cavity':'on_tray_candidate','deeper_open_cavity_geometry_candidate':'into_open_container_candidate',
                    'intermediate_cavity_geometry_unknown_semantics':'unknown_requires_task_specific_geometry_or_visual_confirmation'}[c['descriptive_geometry_bin']]
                goals.append({'object':g['object'],'object_source_id':d['source_id'],'object_name':d['name'],
                    'container':g['container'],'container_source_id':c['source_id'],'container_name':c['name'],
                    'container_descriptor':c['descriptor'],'geometry_bin':c['descriptive_geometry_bin'],
                    'future_contract_candidate_only':suggested,'current_task_contract_unchanged':True,
                    'object_native_height_m':object_height,'container_depth_m':c['cavity_depth_m'],
                    'floor_relative_body_height':c['floor_relative_body_height'],
                    'nominal_object_fraction_below_rim':below,'nominal_object_top_above_rim_m':nominal_top-c['rim_z_m'],
                    'whole_object_below_rim_at_nominal_goal':nominal_top<=c['rim_z_m'],
                    'interpretation':'Relative height only; partial protrusion need not be invalid, and actual stable pose/semantic containment remains untested.'})
            taskrows.append({'task':task['task'],'id':task['id'],'scene_id':s['id'],'partition':s['partition'],'goals':goals})
    goals=[g for t in taskrows for g in t['goals']]
    report={'schema':'independent_container_affordance_census_result_v1','protocol':binding(OUT/'PROTOCOL.json'),
        'variant_count':len(rows),'source_count':len({r['source_id'] for r in rows}),
        'all_existing_bindings_resolved':True,'native_oriented_floor_support_passes':sum(r['retained_oriented_floor_support']['passed'] for r in rows),
        'geometry_bins_variant_counts':dict(Counter(r['descriptive_geometry_bin'] for r in rows)),
        'geometry_bins_distinct_source_counts_nonadditive':{label:len({r['source_id'] for r in rows if r['descriptive_geometry_bin']==label}) for label in sorted({r['descriptive_geometry_bin'] for r in rows})},
        'distributions':{key:quantiles([r[key] for r in rows]) for key in ['native_body_height_m','floor_z_m','rim_z_m','cavity_depth_m','floor_relative_body_height','depth_relative_body_height','flat_floor_radius_m']},
        'rows':rows,'tasks':taskrows,'task_count':len(taskrows),'goal_count':len(goals),
        'task_goals_by_geometry_bin':dict(Counter(g['geometry_bin'] for g in goals)),
        'whole_nominal_object_below_rim_goals':sum(g['whole_object_below_rim_at_nominal_goal'] for g in goals),
        'nominal_partial_protrusion_goals':sum(not g['whole_object_below_rim_at_nominal_goal'] for g in goals),
        'semantic_container_quality':'unknown; measured cavity and title do not constitute semantic ground truth',
        'recommendation':'Keep current experiment unchanged. Future version should distinguish into_open_container/on_tray/on_support_surface with bound measurable geometry and task-specific occupancy; do not inflate deep-container counts from shallow top recesses.',
        'limitations':['Diagnostic30mm/height-ratio bins describe geometry and are not new acceptance thresholds.',
            '97-ray floor test is sampled support, not proof of full opening/swept payload clearance.',
            'No render/physical execution here; depth and nominal vertical protrusion do not prove task success or failure.',
            'The172 frozen consumer variants omit4 retained semantic-title exclusions from the176-variant bundle.'],
        'elapsed_seconds':time.monotonic()-began,'physical_steps':0,'new_cooking':0,'GPU_jobs':0,'API_calls':0,'qualifications_or_contracts_changed':False}
    save(OUT/'FINAL_CENSUS.json',report)
    cols=['source_id','variant_id','name','partition','native_body_height_m','floor_z_m','rim_z_m','cavity_depth_m','floor_relative_body_height','depth_relative_body_height','flat_floor_radius_m','descriptive_geometry_bin']
    with (OUT/'CONTAINER_GEOMETRY.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r[k] for k in cols} for r in rows)
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','tasks']},ensure_ascii=False))


if __name__=='__main__':main()
