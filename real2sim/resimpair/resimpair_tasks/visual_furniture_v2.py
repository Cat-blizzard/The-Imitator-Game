"""Bounded CC0 furniture import in publisher metres, for render-only scenes.

Powered by Poly Haven. No grasp normalization, physics, GPU or model APIs.
Original glTF dependencies and download failures are retained.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import time
import urllib.request
from urllib.parse import urlsplit

import numpy as np

from resimpair_grasp.common import binding, read, resolve, write
from resimpair_tasks.appearance_catalog_v2 import atomic

PARENT = Path('assets/visual_furniture_v1')

VERSION = 'metric_visual_furniture_v2_mutable_transfer_ledger'
IDS = ('modern_wooden_cabinet', 'wooden_display_shelves_01',
       'steel_frame_shelves_01', 'modern_arm_chair_01')


def safe_dependency(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('unsafe_gltf_dependency')
    return p


def prepare(metadata, output):
    metadata, output = Path(metadata), Path(output)
    if output.exists(): raise ValueError('resume_same_furniture_campaign')
    if shutil.disk_usage(output.parent).free < 2**40: raise RuntimeError('disk_reserve_required')
    selection = []
    for asset in IDS:
        p = metadata / (asset + '_files.json'); entry = read(p)['gltf']['1k']['gltf']
        files = {Path(urlsplit(entry['url']).path).name: {k: v for k, v in entry.items() if k != 'include'}, **entry['include']}
        for name, f in files.items():
            safe_dependency(name)
            if urlsplit(f['url']).hostname != 'dl.polyhaven.org': raise ValueError('unexpected_publisher_host')
        selection.append({'id': asset, 'files_metadata': binding(p), 'files': files,
                          'main_file': Path(urlsplit(entry['url']).path).name})
    write(output/'protocol.json', {'schema': VERSION, 'selection': selection,
        'implementation': binding(__file__), 'parent_failed_campaign': binding(PARENT/'catalog.json'), 'parent_protocol': binding(PARENT/'protocol.json'), 'source_metadata': binding(metadata/'polyhaven_models_metadata.json'),
        'created_unix': time.time(), 'maximum_files': sum(len(x['files']) for x in selection),
        'maximum_transferred_bytes': 128 * 2**20, 'maximum_attempts_per_file': 2,
        'timeout_seconds_per_file': 45, 'wall_limit_seconds': 1200, 'parallelism': 1,
        'disk_reserve_bytes': 2**40, 'paid_api_calls': 0, 'license': 'CC0-1.0',
        'credit': 'Powered by Poly Haven', 'api_terms': 'https://github.com/Poly-Haven/Public-API',
        'coordinate_convention': 'glTF metre units; Y-up converted to Z-up; bottom anchored; no size normalization'})


def run(output):
    import trimesh
    from PIL import Image
    output = Path(output); protocol = read(output/'protocol.json'); resolve(protocol['implementation'])
    if (output/'catalog.json').exists(): raise ValueError('completed_furniture_campaign_not_repeated')
    ledger_path = output/'transfer_ledger.json'; ledger = read(ledger_path) if ledger_path.exists() else {'bytes': sum(p.stat().st_size for p in PARENT.glob('*/*') if p.is_file() and p.suffix in ('.gltf','.bin','.jpg','.png')), 'attempts': [], 'parent_bytes_charged_from_recovered_publisher_verified_files': True}
    for row in protocol['selection']:
        folder = output/row['id']
        if (folder/'result.json').exists(): continue
        try:
            receipts=[]
            for name, f in row['files'].items():
                p=folder/safe_dependency(name); p.parent.mkdir(parents=True, exist_ok=True)
                cached=PARENT/row['id']/safe_dependency(name)
                if not p.exists() and cached.is_file() and hashlib.md5(cached.read_bytes()).hexdigest()==f['md5']:
                    shutil.copy2(cached,p)
                if not p.exists():
                    for attempt in range(protocol['maximum_attempts_per_file']):
                        if time.time()>protocol['created_unix']+protocol['wall_limit_seconds']: raise TimeoutError('furniture_campaign_deadline')
                        if ledger['bytes'] + f['size'] > protocol['maximum_transferred_bytes']: raise RuntimeError('furniture_transfer_budget')
                        entry={'asset':row['id'],'file':name,'attempt':attempt,'started':time.time(),'bytes':0}
                        ledger['attempts'].append(entry); atomic(ledger_path,ledger)
                        part=p.with_name(p.name+'.part')
                        try:
                            req=urllib.request.Request(f['url'],headers={'User-Agent':'ReSimPair-Visual-Furniture/1.0 (Powered by Poly Haven)'})
                            with urllib.request.urlopen(req,timeout=protocol['timeout_seconds_per_file']) as response, part.open('wb') as stream:
                                while chunk:=response.read(1<<18):
                                    entry['bytes']+=len(chunk);ledger['bytes']+=len(chunk)
                                    if entry['bytes']>f['size'] or ledger['bytes']>protocol['maximum_transferred_bytes']:raise RuntimeError('publisher_size_or_budget_exceeded')
                                    stream.write(chunk)
                            data=part.read_bytes()
                            if len(data)!=f['size'] or hashlib.md5(data).hexdigest()!=f['md5']:raise ValueError('publisher_checksum_mismatch')
                            part.replace(p);entry['completed']=True;atomic(ledger_path,ledger);break
                        except Exception as exc:
                            entry['error']=str(exc);atomic(ledger_path,ledger)
                            if part.exists():part.rename(part.with_name(part.name+f'.failed_{len(ledger["attempts"])}'))
                            if attempt+1==protocol['maximum_attempts_per_file']:raise
                if hashlib.md5(p.read_bytes()).hexdigest()!=f['md5']:raise ValueError('existing_download_changed')
                if p.suffix.lower() in ('.jpg','.png'):
                    with Image.open(p) as image:image.load()
                receipts.append({'file':binding(p),'publisher_md5':f['md5'],'url':f['url']})
            main=folder/row['main_file'];gltf=read(main)
            declared={name for name in row['files'] if name!=row['main_file']}
            for item in gltf.get('buffers',[])+gltf.get('images',[]):
                uri=item.get('uri')
                if uri and (str(safe_dependency(uri)) not in declared):raise ValueError('undeclared_gltf_dependency')
            mesh=trimesh.load(main,force='scene',process=False)
            original_bounds=np.asarray(mesh.bounds);transform=np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]],float)
            mesh.apply_transform(transform);bounds=np.asarray(mesh.bounds);size=bounds[1]-bounds[0]
            if not np.isfinite(bounds).all() or not (.45<size[2]<2.8 and .15<size[:2].min() and size[:2].max()<3.):
                raise ValueError('publisher_metric_furniture_size_implausible:'+str(size))
            offset=np.r_[-bounds[:,:2].mean(axis=0),-bounds[0,2]];mesh.apply_translation(offset)
            model=folder/'metric_z_up.glb';model.write_bytes(mesh.export(file_type='glb'))
            loaded=trimesh.load(model,force='scene',process=False)
            if not np.allclose(loaded.bounds,mesh.bounds,atol=1e-5):raise ValueError('metric_export_bounds_changed')
            write(folder/'result.json',{'schema':VERSION,'status':'completed','id':row['id'],
                'source':'https://polyhaven.com/a/'+row['id'],'license':'CC0-1.0','credit':'Powered by Poly Haven',
                'mesh':binding(model),'publisher_inputs':receipts,'original_gltf_bounds':original_bounds.tolist(),
                'bounds_m':np.asarray(loaded.bounds).tolist(),'size_m':size.tolist(),'y_up_to_z_up':transform.tolist(),
                'bottom_center_translation_m':offset.tolist(),'uniform_scale_applied':1.,
                'triangles':sum(len(g.faces) for g in loaded.geometry.values()),'meshes':len(loaded.geometry),
                'usage':'render_only_furniture_not_task_object_or_collision_obstacle','physical_qualification':'not_applicable'})
        except Exception as exc:
            write(folder/'result.json',{'schema':VERSION,'status':'failed','id':row['id'],'error':{'type':type(exc).__name__,'reason':str(exc)}})
        print(read(folder/'result.json')['id'],read(folder/'result.json')['status'],flush=True)
    results=[read(output/x['id']/'result.json') for x in protocol['selection']]
    write(output/'catalog.json',{'schema':VERSION,'protocol':binding(output/'protocol.json'),
        'entries':[x for x in results if x['status']=='completed'],'failures':[x for x in results if x['status']!='completed'],
        'transfer_ledger':binding(ledger_path),'completed_unix':time.time(),'paid_api_calls':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True);p.add_argument('--metadata',type=Path)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.metadata,a.output)
    else:run(a.output)
