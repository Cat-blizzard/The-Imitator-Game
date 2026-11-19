"""One bounded metric-scale collision refinement when old scaled hulls fail.

Keep the independent 3 mm exterior/native-vertex and CPU settling criteria.
The trigger is measured surface error, not a failed grasp or low task score.
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import time
import sys
import numpy as np
import trimesh
from resimpair_grasp.common import ROOT,binding,digest,read,resolve,write
from . import metric_native_v5 as retained

RECIPE={'backend':'vhacdx','maximum_convex_hulls':32,'resolution':1000000,
    'maximum_vertices_per_hull':128,'minimum_volume_percent_error':.2,
    'shrink_wrap':True,'asynchronous':False,'maximum_decomposition_attempts':1}


def refine(parent_ref,output):
    sys.path.insert(0,str(ROOT/'runtime/collision_cpu_deps'))
    import vhacdx
    output=Path(output);output.mkdir(parents=True)
    d=read(resolve(parent_ref));scene=trimesh.load(resolve(d['visual']),force='scene',process=False)
    mesh=scene.to_geometry();mesh=trimesh.Trimesh(vertices=mesh.vertices,faces=mesh.faces,process=True)
    if not np.isfinite(mesh.vertices).all() or not len(mesh.faces):raise ValueError('invalid_metric_visual_mesh')
    start=time.monotonic()
    raw=vhacdx.compute_vhacd(np.asarray(mesh.vertices,dtype=np.float64),
        np.column_stack((np.full(len(mesh.faces),3),mesh.faces)).astype(np.uint32).ravel(),
        maxConvexHulls=32,resolution=1000000,maxNumVerticesPerCH=128,
        asyncACD=False,shrinkWrap=True,minimumVolumePercentErrorAllowed=.2)
    parts=[trimesh.Trimesh(vertices=v,faces=f,process=True) for v,f in raw]
    if not 1<=len(parts)<=32:raise ValueError('metric_decomposition_part_budget')
    refs=[]
    for i,part in enumerate(parts):
        if not part.is_watertight or not part.is_winding_consistent or part.volume<=0:
            raise ValueError('metric_decomposition_non_solid_part')
        path=output/f'collision_{i}.obj';part.export(path);refs.append(binding(path))
    write(output/'decomposition.json',{'recipe':RECIPE,'parent':parent_ref,
        'backend_binary':binding(vhacdx.__file__),
        'elapsed_seconds':time.monotonic()-start,'parts':len(parts),
        'input_faces':len(mesh.faces),'native_cook_and_independent_qualification':'pending',
        'implementation':binding(__file__)})
    value=copy.deepcopy(d);value.update(collisions=refs,collision_policy='metric_scale_vhacd32_128vertices_before_retained_native_qualification',
        parent_collision_descriptor=parent_ref,metric_decomposition=binding(output/'decomposition.json'))
    value['geometry_variant_key']=digest({'parent':parent_ref,'recipe':RECIPE})
    value['variant_id']=value['geometry_variant_key'];value['id']=value['variant_id'][:24]
    write(output/'descriptor.json',value);return binding(output/'descriptor.json')


def compile_variant(descriptor_path,output,prior_result=None):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);start=time.monotonic()
    if prior_result:
        first=read(prior_result);d=read(resolve(first['asset']))
        expected=d.get('native_compiler',{}).get('input_metric_descriptor')
        if expected!=binding(descriptor_path):raise ValueError('previous_native_result_input_binding_mismatch')
        first_ref=binding(prior_result)
    else:
        first=retained.compile_variant(descriptor_path,output/'first_native');first_ref=binding(output/'first_native/result.json')
        d=read(resolve(first['asset']))
    trigger=d['qualification']['geometry']=='fail' and d['qualification']['native_collision_replay']=='pass'
    if trigger:
        descriptor=refine(first['asset'],output/'metric_collision_refinement')
        selected=retained.compile_variant(resolve(descriptor),output/'refined_native')
    else:selected=first
    result={**selected,'schema':'metric_native_conditional_refinement_v6',
        'prior_native_result':first_ref,'metric_decomposition_triggered':trigger,
        'prior_surface':d['surface_comparison'],'surface_rules_changed':False,
        'elapsed_seconds_this_stage':time.monotonic()-start,'maximum_refinements':1,
        'new_GPU_jobs':0,'API_calls':0}
    write(output/'result.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--descriptor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prior-result',type=Path)
    a=p.parse_args();print(compile_variant(a.descriptor,a.output,a.prior_result))
