"""Calibrate native density before execution; no trajectory or acceptance change."""
import argparse
import json
from pathlib import Path
import numpy as np
from resimpair_grasp.common import read,write,binding,resolve,digest


def native_mass(collisions,density):
    import sapien
    scene=sapien.Scene([sapien.physx.PhysxCpuSystem()])
    entity=sapien.Entity();body=sapien.physx.PhysxRigidDynamicComponent();entity.add_component(body)
    for collider in collisions:
        shape=sapien.physx.PhysxCollisionShapeConvexMesh(filename=str(resolve(collider)),scale=[1,1,1],
            material=sapien.physx.PhysxMaterial(.6,.6,0))
        shape.density=float(np.float32(density));body.attach(shape)
    scene.add_entity(entity)
    mass=float(body.mass)
    scene.remove_entity(entity)
    if not np.isfinite(mass) or mass<=0:raise ValueError('Invalid native component mass')
    return mass


def calibrated_density(collisions,initial_density,requested_mass):
    if not np.isfinite(requested_mass) or requested_mass<=0:raise ValueError('Positive requested mass required')
    density=float(np.float32(initial_density));history=[]
    for _ in range(3):
        measured=native_mass(collisions,density)
        history.append({'density_kg_m3':density,'native_mass_kg':measured})
        if abs(measured-requested_mass)<1e-6:return density,history
        density=float(np.float32(density*requested_mass/measured))
    raise ValueError('Native mass calibration did not reach 1 mg precision')


def calibrate(index,output):
    if output.exists():raise ValueError('Mass preflight output exists')
    output.mkdir(parents=True);entries=[];rows=[]
    for entry in read(index)['entries']:
        d=read(resolve(entry['asset']));cache=read(resolve(entry['cache']))
        if cache['signature']['descriptor']!=entry['asset']:raise ValueError('Input cache binding mismatch')
        density,history=calibrated_density(d['collisions'],d['density_kg_m3'],d['mass_kg'])
        folder=output/d['id'];folder.mkdir()
        descriptor={**d,'id':d['id']+'_mass1','density_kg_m3':density,
                    'previous_mass_descriptor':entry['asset'],
                    'mass_preflight':{'implementation':binding(__file__),'requested_mass_kg':d['mass_kg'],
                                     'native_component_measurements':history,'dynamic_simulation_steps':0}}
        path=folder/'descriptor.json';write(path,descriptor);db=binding(path)
        cache={**cache,'signature':{**cache['signature'],'descriptor':db,'native_mass_preflight':binding(__file__)}}
        cache['key']=digest(cache['signature']);cp=folder/'grasp_cache.json';write(cp,cache)
        entries.append({'asset':db,'cache':binding(cp)})
        rows.append({'source_id':d['source_id'],'requested_mass_kg':d['mass_kg'],'history':history})
    write(output/'cache_index.json',{'schema':'resimpair_grasp_cache_index_v1','entries':entries,'failures':read(index).get('failures',[])})
    write(output/'report.json',{'input':binding(index),'rows':rows,'api_calls':0,'dynamic_simulation_steps':0,
        'geometry_poses_controller_and_acceptance_unchanged':True,'purpose':'Match declared engineering mass before execution'})
    return {'calibrated':len(rows),'cache_index':str(output/'cache_index.json'),'dynamic_simulation_steps':0,'api_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(calibrate(a.index,a.output),indent=2))
