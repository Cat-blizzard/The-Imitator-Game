"""Recover source/container scale and recast cavity geometry, no per-bin skills."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import copy
from pathlib import Path
import time
import numpy as np
import trimesh
from resimpair_grasp.common import ROOT,binding,digest,read,resolve,write
from resimpair_assets.metric_inventory_v6 import metric_match
from resimpair_pickplace.containers import cavity


def compile_one(ref,metadata_ref,root):
    import sapien
    start=time.monotonic();d=read(resolve(ref));source=d['source_id'].split(':',1)[0]
    metadata=read(resolve(metadata_ref))['matches'];T=np.asarray(d['source_to_asset'])
    scale=float(np.linalg.svd(T[:3,:3],compute_uv=False).mean())
    mesh=trimesh.load(resolve(d['collision']),force='mesh',process=False)
    original=mesh.copy();original.apply_transform(np.linalg.inv(T))
    original_obb=original.bounding_box_oriented.primitive.extents
    match=metric_match(original_obb,metadata.get(d['source_id'].split(':',1)[1],[])) if source=='abo' else None
    if match and match['status']=='matched_uniform_metric_metadata':
        factor=match['selected']['scale']/scale;confidence='ABO_dimensions_agree_with_source_geometry'
    elif source in {'google_scanned_objects_original_fuel','ycb_official','polyhaven'}:
        factor=1/scale;confidence='original_source_metric_convention_engineering_inference'
    else:
        factor=1.;confidence='retained_0p28m_container_engineering_size_no_metric_source_match'
    variant=digest({'parent':ref,'uniform_factor':factor,'semantic_role':'stationary_container'})
    out=Path(root)/'assets'/variant[:24]
    if (out/'result.json').exists():return read(out/'result.json')
    if out.exists():raise ValueError('incomplete_container_compile_needs_recovery')
    out.mkdir(parents=True)
    write(out/'input.json',{'parent':ref,'metadata':metadata_ref,'uniform_factor':factor,
        'scale_confidence':confidence,'variant_id':variant,'API_calls':0})
    visual=trimesh.load(resolve(d['visual']),force='scene',process=False);visual.apply_scale(factor)
    mesh.apply_scale(factor)
    if max(mesh.extents)>2 or min(mesh.extents)<=0 or not np.isfinite(mesh.vertices).all():
        result={'source_id':d['source_id'],'status':'outside_static_container_numeric_bounds','variant_id':variant}
        write(out/'result.json',result);return result
    try:geometry=cavity(mesh)
    except (ValueError,RuntimeError) as exc:
        result={'source_id':d['source_id'],'status':'cavity_failed_after_metric_scaling',
            'reason':str(exc),'variant_id':variant,'confidence':confidence}
        write(out/'result.json',result);return result
    if not geometry:
        result={'source_id':d['source_id'],'status':'no_cavity_at_source_scale','variant_id':variant,'confidence':confidence}
        write(out/'result.json',result);return result
    visual.export(out/'visual.glb');mesh.export(out/'collision.obj')
    native=sapien.physx.PhysxCollisionShapeTriangleMesh(filename=str(out/'collision.obj'),scale=[1,1,1],
        material=sapien.physx.PhysxMaterial(.6,.6,0.))
    from scipy.spatial import cKDTree
    error=max(cKDTree(mesh.vertices).query(native.vertices)[0].max(),cKDTree(native.vertices).query(mesh.vertices)[0].max())
    if error>1e-5:raise ValueError('native_triangle_geometry_changed')
    value=copy.deepcopy(d);value.update(id=variant[:24],variant_id=variant,
        visual=binding(out/'visual.glb'),collision=binding(out/'collision.obj'),
        source_to_asset=(np.diag([factor,factor,factor,1.])@T).tolist(),
        bounds_m=mesh.bounds.tolist(),dimensions_m=mesh.extents.tolist(),cavity=geometry,
        metric_size={'mode':confidence,'factor_from_parent':factor,'parent':ref,'source_evidence':metadata_ref},
        native_triangle_vertex_error_m=float(error),native_policy='static_triangle_mesh_only; cavity_recased_at_metric_scale',
        physical_status='unmeasured',relative_scale=1.0)
    write(out/'descriptor.json',value)
    result={'source_id':d['source_id'],'status':'ready_geometry_only','asset':binding(out/'descriptor.json'),
        'variant_id':variant,'relative_scale':1.,'dimensions_m':mesh.extents.tolist(),'cavity':geometry,
        'scale_confidence':confidence,'elapsed_seconds':time.monotonic()-start,'GPU_jobs':0,'API_calls':0}
    write(out/'result.json',result);return result


def run(catalog,metadata,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);refs=read(catalog)['assets'];metadata_ref=binding(metadata)
    write(output/'protocol.json',{'schema':'metric_container_population_v3','input':binding(catalog),
        'metadata':metadata_ref,'implementation':binding(__file__),'maximum_assets':len(refs),
        'CPU_workers':4,'dynamic_steps':0,'API_calls':0,'GPU_jobs':0,
        'same_generic_cavity_function_for_every_container':binding(ROOT/'resimpair_pickplace/containers.py')})
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        future={pool.submit(compile_one,r,metadata_ref,output):r for r in refs}
        for f in as_completed(future):
            try:rows.append(f.result())
            except Exception as exc:
                d=read(resolve(future[f]));rows.append({'source_id':d['source_id'],
                    'status':'failed_preserved','error':type(exc).__name__+': '+str(exc),'retry':False})
    assets=[x['asset'] for x in rows if x['status']=='ready_geometry_only']
    write(output/'catalog.json',{'schema':'stationary_container_catalog_v1','assets':assets,
        'rows':rows,'parents':[binding(catalog)],'physics_executions':0,'api_calls':0})
    summary={'attempted':len(rows),'usable_static_container_geometry':len(assets),
        'counts':dict(Counter(r['status'] for r in rows))}
    write(output/'summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(run(a.catalog,a.metadata,a.output))
