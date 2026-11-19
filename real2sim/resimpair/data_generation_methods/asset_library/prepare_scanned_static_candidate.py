"""One GSO candidate using the original in-package OBJ material resolver.

Keeps every original file unchanged. The bound intermediate GLB bakes an explicit
Z-up to Y-up transform before the existing metric preparation code is reused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from ..human_to_robot_pipeline.household_assets import bind, prepare_asset
from ..r2h_pipeline.common import file_sha256, write_json

SOURCE_ID='3M_Vinyl_Tape_Green_1_x_36_yd'
CATALOG=Path('/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1/scanned_objects/objects.jsonl')


def prepare(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    raw=next(json.loads(line) for line in CATALOG.open() if json.loads(line).get('source_id')==SOURCE_ID)
    original=Path(raw['primary_mesh_path']);package=original.parent.parent
    if file_sha256(original)!=raw['sha256']:raise ValueError('Indexed GSO mesh changed')
    if raw['geometry_check'].get('materials_complete') is not True:
        raise ValueError('Source material dependency check is incomplete')
    resources=[p for p in package.rglob('*') if p.is_file() and p.suffix.lower() in ('.mtl','.png','.jpg','.jpeg')]
    by_name={}
    for p in resources:by_name.setdefault(p.name,[]).append(p)
    archive={str(p.relative_to(package)):p.read_bytes() for p in resources}
    for name,paths in by_name.items():
        if len(paths)==1:archive[name]=archive[str(paths[0].relative_to(package))]
    scene=trimesh.load(original,force='scene',process=False,
                       resolver=trimesh.resolvers.ZipResolver(archive=archive))
    textured=sum(getattr(g.visual,'kind',None)=='texture' and
                 getattr(getattr(g.visual,'material',None),'image',None) is not None
                 for g in scene.geometry.values())
    if textured<1:raise ValueError('GSO texture resolution unexpectedly lost original texture')
    transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('x',-90,degrees=True).as_matrix()
    scene.apply_transform(transform)
    normalized=output/'gso_material_resolved_y_up.glb';normalized.write_bytes(scene.export(file_type='glb'))
    provenance={**raw,'uid':SOURCE_ID,'geometry_path':str(normalized),'sha256':file_sha256(normalized),
                'author':'Google Research; original GSO asset retained',
                'original_GSO_record':raw,
                'original_dependency_bindings':[bind(original)]+[bind(p) for p in resources],
                'original_z_up_to_intermediate_y_up_4x4':transform.tolist(),
                'resolver':'Only unambiguous names within original package; no source file mutation',
                'resolved_original_texture_mesh_count':textured}
    a=prepare_asset(provenance,'tape_roll',.075,'flat_shortest_axis',output)
    a['source_asset_id']='google_scanned_objects_original_fuel:'+SOURCE_ID
    a['usage']='static_distractor';a['semantic_categories']=['tape_roll']
    a['qualification']={'static_geometry_preflight':'passed','static_scene_execution':'unknown_not_executed',
                        'active_manipulation':'unsupported','rendered_semantics':'unknown_not_rendered'}
    a['capabilities']={'static_untouched_obstacle':'preflight_only'}
    a['original_dependency_bindings']=provenance['original_dependency_bindings']
    a['material_resolution_evidence']={'original_textured_meshes':textured,
        'intermediate':bind(normalized),'no_original_file_modification':True,
        'note':'Hole is conservatively filled by static collision; no threading/containment use.'}
    write_json(output/'tape_roll/qualification_asset.json',a)
    a['asset_spec']=bind(output/'tape_roll/qualification_asset.json')
    result={'schema':'household_static_distractor_catalog_v1','assets':[a],'failed_candidates':[],
            'simulation_executed':False,'training_admitted':False,'new_downloads':0}
    write_json(output/'catalog.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=prepare(a.output);print(json.dumps({'prepared':len(r['assets'])}))
