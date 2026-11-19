"""Hash-bound appearance choices: local surfaces and bounded official CC0 HDRIs.

No geometry or collision qualification is implied by a usable visual background.
Downloads are opt-in, sequential, bounded in bytes and time, with no retries.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import requests

from ..human_to_robot_pipeline.household_assets import bind
from ..r2h_pipeline.common import file_sha256, write_json

POLY_CATALOG=Path('/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1/environments/polyhaven/asset_catalog.json')
INDOOR_IDS=['small_empty_room_3','small_empty_room_4','poly_haven_studio']
SURFACES=[('brick_wall_04','angli_veneer'),('bamboo_wall_03','brown_leather'),
          ('bamboo_wall_02','fabric_leather_02')]


def local_surface(record):
    channel=next(c for c in record['channels'] if c['name']=='Diffuse')
    metadata=next(f for f in record['files'] if f['path']==channel['path'])
    b=bind(channel['path'])
    if b['sha256'] != metadata['sha256']:
        raise ValueError('Background source file changed')
    with Image.open(channel['path']) as image:
        image.load()
        if min(image.size)<256:raise ValueError('Background too small')
    return {**b,'asset_id':record['asset_id'], 'license':record['license'],
            'source':record['source'], 'attribution':record['attribution'],
            'usage':'visual_only', 'channel':'Diffuse',
            'surface_appearance_does_not_modify_contact_parameters':True}


def download_hdri(asset_id, folder, remaining_bytes):
    start=time.monotonic();folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    session=requests.Session()
    api=f'https://api.polyhaven.com/files/{asset_id}'
    result=session.get(api,timeout=20);result.raise_for_status()
    files=result.json();write_json(folder/'official_files.json',files)
    info_response=session.get(f'https://api.polyhaven.com/info/{asset_id}',timeout=20)
    info_response.raise_for_status();info=info_response.json();write_json(folder/'official_info.json',info)
    remote=files['hdri']['1k']['hdr']
    if not remote['url'].startswith('https://dl.polyhaven.org/'):
        raise ValueError('Unexpected official download host')
    if remote['size']>remaining_bytes:raise ValueError('Cumulative download byte budget exceeded')
    destination=folder/f'{asset_id}_1k.hdr';part=destination.with_suffix('.hdr.part')
    downloaded=0
    with session.get(remote['url'],stream=True,timeout=30) as response:
        response.raise_for_status()
        with part.open('xb') as stream:
            for chunk in response.iter_content(2**16):
                downloaded+=len(chunk)
                if downloaded>min(remaining_bytes,remote['size']):
                    raise ValueError('Download exceeded announced size or cumulative byte limit')
                if time.monotonic()-start>90:raise TimeoutError('Single background download exceeded 90 seconds')
                stream.write(chunk)
    if downloaded!=remote['size']:raise ValueError('Incomplete download')
    md5=hashlib.md5(part.read_bytes()).hexdigest()
    if md5!=remote['md5']:raise ValueError('Publisher MD5 mismatch')
    part.replace(destination)
    decoded=cv2.imread(str(destination),cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.ndim!=3 or not np.isfinite(decoded).all():
        raise ValueError('HDRI decode failed or nonfinite pixels')
    # Preview is tone-mapped for inspection; renderer consumes the original HDR.
    rgb=cv2.cvtColor(decoded,cv2.COLOR_BGR2RGB)
    white=float(np.percentile(rgb,98));preview=np.clip(rgb/max(white,1e-6),0,1)**(1/2.2)
    Image.fromarray((preview*255).astype('uint8')).save(folder/'preview.jpg')
    record={**bind(destination),'asset_id':'polyhaven:'+asset_id,'license':'CC0-1.0',
            'license_source':'https://polyhaven.com/license', 'source':'https://polyhaven.com/a/'+asset_id,
            'authors':info.get('authors'), 'attribution':'Powered by Poly Haven',
            'usage':'visual_only','publisher_md5':remote['md5'],'publisher_hash_verified':True,
            'download_url':remote['url'],'downloaded_bytes':downloaded,
            'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'elapsed_seconds':time.monotonic()-start,'image_shape':list(decoded.shape),
            'official_file_metadata':bind(folder/'official_files.json'),
            'official_info':bind(folder/'official_info.json'),
            'preview':bind(folder/'preview.jpg'),'physics_qualified':False,'training_admitted':False}
    write_json(folder/'receipt.json',record)
    return record


def prepare(output, download=False, byte_budget=200*2**20):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows={r['asset_id']:r for r in json.loads(POLY_CATALOG.read_text())['records']}
    records=[];failures=[];used=0
    for asset_id in INDOOR_IDS:
        if not download:break
        try:
            r=download_hdri(asset_id,output/'downloaded'/asset_id,byte_budget-used)
            used+=r['downloaded_bytes'];records.append(r)
        except Exception as exc:
            # Count partial bytes too, preventing a failed download from bypassing budget.
            partial=sum(p.stat().st_size for p in (output/'downloaded'/asset_id).glob('*.hdr*'))
            used+=partial
            failures.append({'asset_id':asset_id,'error':str(exc),'partial_bytes':partial})
    cases=[]
    for i,(wall,table) in enumerate(SURFACES):
        background={'wall':local_surface(rows['polyhaven:'+wall]),
                    'table':local_surface(rows['polyhaven:'+table])}
        if i<len(records):background['hdri']=records[i]
        cases.append({'id':f'appearance_{i:02d}','background':background,
                      'physics_geometry_changed':False,'runtime_applied':False,
                      'qualification':'decoded visual files; simulator appearance not yet reviewed'})
    all_local=[]
    for r in rows.values():
        if r['kind']=='texture':all_local.append(local_surface(r))
    result={'schema':'candidate_background_catalog_v1','cases':cases,'local_diffuse_surfaces':all_local,
            'new_hdri_records':records,'download_failures':failures,'downloaded_bytes_including_partials':used,
            'download_byte_budget':byte_budget,'max_concurrency':1,'paid_API_calls':0,
            'catalog_binding':bind(POLY_CATALOG),'training_admitted':False,
            'note':'Visual-only backgrounds; source quantities are not trajectory coverage.'}
    write_json(output/'background_cases.json',result)
    gallery=Image.new('RGB',(960,3*210),'white');draw=ImageDraw.Draw(gallery)
    for i,c in enumerate(cases):
        for j,kind in enumerate(['wall','table','hdri']):
            item=c['background'].get(kind)
            if not item:continue
            image=Image.open(item['preview']['path'] if kind=='hdri' else item['path']).convert('RGB')
            image.thumbnail((310,180));gallery.paste(image,(j*320,i*210+25))
            draw.text((j*320+3,i*210+3),kind+': '+item['asset_id'].removeprefix('polyhaven:'),fill='black')
    gallery.save(output/'gallery.jpg')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--download-indoor-cc0',action='store_true')
    args=p.parse_args();r=prepare(args.output,args.download_indoor_cc0)
    print(json.dumps({'cases':len(r['cases']),'new_hdris':len(r['new_hdri_records']),
                      'bytes':r['downloaded_bytes_including_partials'],'failures':r['download_failures']}))
