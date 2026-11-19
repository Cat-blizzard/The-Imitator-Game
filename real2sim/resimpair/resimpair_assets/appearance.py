"""Version repaired material exports while proving existing geometry unchanged."""
import argparse,json,sqlite3
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from resimpair_grasp.common import binding,read,resolve,write
from . import source_mesh,acceptance


def repair(catalog,index,output):
    if output.exists():raise ValueError('Appearance revision exists')
    output.mkdir(parents=True);entries=[];rows=[]
    with sqlite3.connect(f'file:{index.resolve()}?mode=ro',uri=True) as db:
        for b in read(catalog)['assets']:
            d=read(resolve(b))
            if d['provenance']['source']!='google_scanned_objects_original_fuel':entries.append(b);continue
            raw=json.loads(db.execute('SELECT record FROM assets WHERE id=?',(d['source_id'],)).fetchone()[0])
            scene,evidence=source_mesh.load(raw);scene.apply_transform(d['source_to_asset'])
            from .batch import reduce_textures
            resized=reduce_textures(scene)
            old=trimesh.load(resolve(d['visual']),force='scene',process=False).to_geometry();new=scene.to_geometry()
            error=max(cKDTree(old.vertices).query(new.vertices)[0].max(),cKDTree(new.vertices).query(old.vertices)[0].max())
            if len(old.faces)!=len(new.faces) or error>1e-6:raise ValueError('Appearance repair changes geometry')
            folder=output/d['id'];folder.mkdir();visual=folder/'visual.glb';scene.export(visual)
            updated={**d,'previous_descriptor':b,'visual':binding(visual),'source_appearance':evidence,
                'texture_resizes':resized,'appearance_revision':{'vertex_error_m':float(error),'face_count':len(new.faces),
                    'collision_and_dynamics_unchanged':True,'visual_identity_confirmation':'unknown'}}
            for k in ['collisions','dimensions_m','half_size','source_to_asset','mass_kg','density_kg_m3','roles']:
                assert updated[k]==d[k]
            path=folder/'descriptor.json';write(path,updated)
            if d['collision_policy']=='static_only_conservative_AABB':
                write(folder/'independent_geometry_check.json',acceptance.evaluate(path))
            entries.append(binding(path));rows.append({'source_id':d['source_id'],'old_visual':d['visual'],'new_visual':updated['visual'],
                'vertex_error_m':float(error),'source_appearance':evidence})
    value={'schema':'resimpair_mesh_catalog_v1','assets':entries,'failures':read(catalog).get('failures',[]),
        'previous_catalog':binding(catalog),'index':binding(index),'repairs':rows,
        'api_calls':0,'scope':'material repair with unchanged geometry; visual semantic identity remains unknown'}
    write(output/'catalog.json',value)
    return {'catalog':str(output/'catalog.json'),'assets':len(entries),'repaired':len(rows)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['catalog','index','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(repair(a.catalog,a.index,a.output),indent=2))
