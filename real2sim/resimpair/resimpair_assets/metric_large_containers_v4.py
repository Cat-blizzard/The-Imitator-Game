"""Category-sized unknown-unit trays/basins; preserve known metric containers.

Every asset uses the same retained cavity rays and native triangle replay.
Sizes are declared engineering variants, not recovered real-world dimensions.
"""
from pathlib import Path
import argparse
import copy
import re
import time
import numpy as np
import trimesh
from resimpair_grasp.common import binding,digest,read,resolve,write
from resimpair_pickplace.containers import cavity

PROFILES={
 'tray':{'pattern':r'\btray\b','base_longest_m':.42,'plausible_longest_m':[.30,.60]},
 'basin':{'pattern':r'\b(basin|wash basin)\b','base_longest_m':.36,'plausible_longest_m':[.25,.55]},
 'basket':{'pattern':r'\bbasket\b','base_longest_m':.40,'plausible_longest_m':[.25,.60]},
 'storage_box':{'pattern':r'\b(storage box|wooden box|container)\b','base_longest_m':.45,'plausible_longest_m':[.30,.60]}}


def run(catalog,output):
    import sapien
    from scipy.spatial import cKDTree
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);rows=[];assets=[]
    write(output/'protocol.json',{'schema':'class_sized_large_container_v4','input':binding(catalog),
        'profiles':PROFILES,'implementation':binding(__file__),'per_container_manual_configuration':False,
        'known_source_metric_preserved':True,'CPU_workers':1,'GPU_jobs':0,'API_calls':0})
    for ref in read(catalog)['assets']:
        d=read(resolve(ref));name=re.sub(r'[_/\-]+',' ',d['name'].lower())
        chosen=next(((key,value) for key,value in PROFILES.items() if re.search(value['pattern'],name)),None)
        if not chosen:continue
        if d['metric_size']['mode']!='retained_0p28m_container_engineering_size_no_metric_source_match':
            rows.append({'source_id':d['source_id'],'status':'known_metric_preserved','asset':ref});continue
        start=time.monotonic();category,profile=chosen
        mesh=trimesh.load(resolve(d['collision']),force='mesh',process=False)
        scale=profile['base_longest_m']/max(mesh.extents)
        variant=digest({'parent':ref,'category':category,'uniform_scale':scale,'profile':profile})
        folder=output/'assets'/variant[:24];folder.mkdir(parents=True)
        write(folder/'input.json',{'parent':ref,'profile':profile,'category':category,
            'uniform_factor_from_parent':scale,'confidence':'engineering_class_scale_unknown_source_units'})
        mesh.apply_scale(scale)
        try:geometry=cavity(mesh)
        except ValueError as exc:
            result={'source_id':d['source_id'],'status':'same_cavity_rule_failed','error':str(exc),'category':category}
            write(folder/'result.json',result);rows.append(result);continue
        visual=trimesh.load(resolve(d['visual']),force='scene',process=False);visual.apply_scale(scale)
        mesh.export(folder/'collision.obj');visual.export(folder/'visual.glb')
        native=sapien.physx.PhysxCollisionShapeTriangleMesh(filename=str(folder/'collision.obj'),scale=[1,1,1],
            material=sapien.physx.PhysxMaterial(.6,.6,0.))
        error=max(cKDTree(mesh.vertices).query(native.vertices)[0].max(),cKDTree(native.vertices).query(mesh.vertices)[0].max())
        if error>1e-5:raise ValueError('native_triangle_replay_changed')
        value=copy.deepcopy(d);value.update(id=variant[:24],variant_id=variant,
            visual=binding(folder/'visual.glb'),collision=binding(folder/'collision.obj'),
            source_to_asset=(np.diag([scale,scale,scale,1.])@np.asarray(d['source_to_asset'])).tolist(),
            bounds_m=mesh.bounds.tolist(),dimensions_m=mesh.extents.tolist(),cavity=geometry,
            metric_size={'mode':'engineering_class_scale_unknown_source_units','profile':profile,
                'category':category,'parent':ref,'factor_from_parent':scale},
            native_triangle_vertex_error_m=float(error),physical_status='unmeasured')
        write(folder/'descriptor.json',value);assets.append(binding(folder/'descriptor.json'))
        result={'source_id':d['source_id'],'status':'ready_geometry_only','asset':assets[-1],
            'variant_id':variant,'category':category,'outer_dimensions_m':mesh.extents.tolist(),
            'inner_flat_floor_diameter_m':2*geometry['flat_floor_radius_m'],
            'elapsed_seconds':time.monotonic()-start,'physical_success_inferred':False}
        write(folder/'result.json',result);rows.append(result)
    write(output/'catalog.json',{'schema':'stationary_container_catalog_v1','assets':assets,'rows':rows,
        'parents':[binding(catalog)],'physics_executions':0,'api_calls':0})
    summary={'new_variants':len(assets),'inner_floor_diameter_025_to_040m':sum(
        .25<=r.get('inner_flat_floor_diameter_m',0)<=.40 for r in rows),
        'known_metric_preserved':sum(r['status']=='known_metric_preserved' for r in rows),
        'same_cavity_rule_failures':sum(r['status']=='same_cavity_rule_failed' for r in rows)}
    write(output/'summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(run(a.catalog,a.output))
