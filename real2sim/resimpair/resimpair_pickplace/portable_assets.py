"""Copy a small runnable sample, preserving original geometry and provenance."""
import argparse
import json
from pathlib import Path
import shutil
from resimpair_grasp.common import read,write,resolve,binding,digest


def pack(index,containers,container_ids,output):
    if output.exists():raise ValueError('Portable sample exists')
    output.mkdir(parents=True);entries=[];container_bindings=[]
    def copy(b,path):
        path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(resolve(b),path)
        value=binding(path)
        if value['sha256']!=b['sha256']:raise ValueError('Geometry bytes changed during relocation')
        return value
    for e in read(index)['entries']:
        d=read(resolve(e['asset']));cache=read(resolve(e['cache']));folder=output/'objects'/d['id']
        visual=copy(d['visual'],folder/'visual.glb')
        collisions=[copy(b,folder/f'collision_{i}.obj') for i,b in enumerate(d['collisions'])]
        descriptor={**d,'visual':visual,'collisions':collisions,
                    'portable_derivative_of':e['asset'],'runtime_geometry_bytes_unchanged':True}
        p=folder/'descriptor.json';write(p,descriptor);db=binding(p)
        updated={**cache,'signature':{**cache['signature'],'descriptor':db,'portable_parent_cache':e['cache']}}
        updated['key']=digest(updated['signature']);cp=folder/'grasp_cache.json';write(cp,updated)
        entries.append({'asset':db,'cache':binding(cp)})
    for b in read(containers)['assets']:
        d=read(resolve(b))
        if d['id'] not in container_ids:continue
        folder=output/'containers'/d['id']
        visual=copy(d['visual'],folder/'visual.glb');collision=copy(d['collision'],folder/'collision.obj')
        p=folder/'descriptor.json';write(p,{**d,'visual':visual,'collision':collision,
            'portable_derivative_of':b,'runtime_geometry_bytes_unchanged':True})
        container_bindings.append(binding(p))
    if len(container_bindings)!=len(set(container_ids)):raise ValueError('Missing requested container')
    write(output/'cache_index.json',{'schema':'resimpair_grasp_cache_index_v1','entries':entries,'failures':[]})
    write(output/'container_catalog.json',{'schema':'stationary_container_catalog_v1','assets':container_bindings,
        'source_catalog':binding(containers),'rows':[],'scope':'portable subset, not another geometry experiment'})
    write(output/'manifest.json',{'source_cache':binding(index),'source_containers':binding(containers),
        'objects':len(entries),'containers':len(container_bindings),'implementation':binding(__file__),
        'source_provenance_references_may_be_unavailable_on_another_host':True,
        'runtime_visual_collision_cache_and_robot_geometry_are_repository_local':True,
        'no_new_physics_or_source_qualification_inferred':True})
    return {'root':str(output),'objects':len(entries),'containers':len(container_bindings)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True)
    p.add_argument('--containers',type=Path,required=True);p.add_argument('--container',action='append',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(pack(a.index,a.containers,a.container,a.output),indent=2))
