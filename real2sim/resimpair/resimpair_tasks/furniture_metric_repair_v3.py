"""Offline correction when publisher dimensions prove a uniform glTF unit error."""
from pathlib import Path
import hashlib
import numpy as np
from resimpair_grasp.common import binding, read, resolve, write


def publisher_scale(size_zup, dimensions_mm):
    measured=np.sort(np.asarray(size_zup,float));declared=np.sort(np.asarray(dimensions_mm,float)/1000.)
    if measured.shape!=(3,) or declared.shape!=(3,) or np.any(measured<=0) or np.any(declared<=0):
        raise ValueError('finite_positive_publisher_geometry_dimensions_required')
    ratios=declared/measured;scale=float(np.median(ratios))
    if not np.isfinite(ratios).all() or not np.allclose(ratios,scale,rtol=.01,atol=1e-5):
        raise ValueError('publisher_dimensions_do_not_prove_uniform_scale')
    if not any(np.isclose(scale,v,rtol=.01) for v in [.001,.01,.1,1.,10.,100.,1000.]):
        raise ValueError('not_a_recognized_uniform_unit_conversion')
    return scale


def repair():
    import trimesh
    parent=Path('assets/visual_furniture_v1/repair_ledger_v2');out=parent/'metric_repair_v3'
    if out.exists():raise ValueError('preserve_existing_metric_repair')
    catalog=read(parent/'catalog.json');meta=read('reports/scene_quality_20261029/polyhaven_models_metadata.json')
    row=next(x for x in read(parent/'protocol.json')['selection'] if x['id']=='steel_frame_shelves_01')
    folder=parent/row['id'];receipts=[]
    for name,record in row['files'].items():
        p=folder/name
        if hashlib.md5(p.read_bytes()).hexdigest()!=record['md5']:raise ValueError('original_publisher_file_changed')
        receipts.append({'file':binding(p),'publisher_md5':record['md5'],'url':record['url']})
    mesh=trimesh.load(folder/row['main_file'],force='scene',process=False)
    rotation=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]],float);mesh.apply_transform(rotation)
    raw_size=np.ptp(mesh.bounds,axis=0);scale=publisher_scale(raw_size,meta[row['id']]['dimensions'])
    mesh.apply_scale(scale);b=mesh.bounds;offset=np.r_[-b[:,:2].mean(axis=0),-b[0,2]];mesh.apply_translation(offset)
    out.mkdir(parents=True);model=out/'steel_frame_shelves_01_metric_z_up.glb';model.write_bytes(mesh.export(file_type='glb'))
    loaded=trimesh.load(model,force='scene',process=False)
    if not np.allclose(loaded.bounds,mesh.bounds,atol=1e-5):raise ValueError('repaired_export_bounds_changed')
    row={'id':row['id'],'status':'completed_after_documented_unit_correction','source':'https://polyhaven.com/a/'+row['id'],
        'license':'CC0-1.0','credit':'Powered by Poly Haven','mesh':binding(model),'bounds_m':loaded.bounds.tolist(),
        'size_m':np.ptp(loaded.bounds,axis=0).tolist(),'original_gltf_size_m':raw_size.tolist(),
        'publisher_dimensions_mm':meta[row['id']]['dimensions'],'uniform_scale_applied':scale,
        'dimension_source':binding('reports/scene_quality_20261029/polyhaven_models_metadata.json'),
        'dimension_unit_documentation':'https://raw.githubusercontent.com/Poly-Haven/Public-API/master/swagger.yml',
        'publisher_inputs':receipts,'original_failure':binding(folder/'result.json'),
        'reason':'all_three_publisher_dimension_ratios_prove_0.1_unit_correction; no grasp-scale normalization',
        'triangles':sum(len(g.faces) for g in loaded.geometry.values()),'usage':'render_only_furniture_not_task_object_or_collision_obstacle'}
    write(out/'catalog.json',{'schema':'metric_visual_furniture_v3','parent':binding(parent/'catalog.json'),
        'implementation':binding(__file__),'entries':catalog['entries']+[row],
        'unresolved_failures':[x for x in catalog['failures'] if x['id']!=row['id']],
        'new_download_requests':0,'paid_api_calls':0,'original_failures_unchanged':True})
    print({'entries':len(catalog['entries'])+1,'corrected_scale':scale,'corrected_size':row['size_m'],'new_download_requests':0})


if __name__=='__main__':repair()
