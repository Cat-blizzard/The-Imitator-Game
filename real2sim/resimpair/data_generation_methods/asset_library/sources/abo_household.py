"""Official ABO product models: bounded, resumable downloads with provenance.

No image render sets, model training, simulator calls, or guessed physical units.
"""
import argparse
import collections
import concurrent.futures
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import threading
import time
from urllib.parse import quote, urlencode
import urllib.request
import xml.etree.ElementTree as ET

BASE = 'https://amazon-berkeley-objects.s3.amazonaws.com/'
ROOT = Path('/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1/abo')
UA = 'PolicyGen-Household-Asset-Library/1.0'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            sha.update(block);md5.update(block)
    return sha.hexdigest(),md5.hexdigest()


def list_bucket(prefix, root):
    rows, token, page = [], None, 0
    while True:
        query = {'list-type':'2','prefix':prefix,'max-keys':'1000'}
        if token:query['continuation-token']=token
        request = urllib.request.Request(BASE+'?'+urlencode(query),headers={'User-Agent':UA})
        path = root/'metadata'/f"bucket_{prefix.replace('/','_')}_{page}.xml"
        if path.exists():
            data=path.read_bytes()
        else:
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(request,timeout=45) as response:data=response.read()
                    ET.fromstring(data)
                    path.parent.mkdir(parents=True,exist_ok=True)
                    temporary=path.with_suffix('.xml.tmp');temporary.write_bytes(data);temporary.replace(path)
                    break
                except Exception:
                    if attempt==3:raise
                    time.sleep(2**attempt)
        ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
        xml=ET.fromstring(data)
        for item in xml.findall('s:Contents',ns):
            rows.append(dict(key=item.find('s:Key',ns).text,size=int(item.find('s:Size',ns).text),
                etag=item.find('s:ETag',ns).text.strip('"')))
        token=xml.findtext('s:NextContinuationToken',namespaces=ns)
        if not token:return rows
        page+=1


class Store:
    def __init__(self,root,limit):
        self.root,self.limit=root,limit
        self.lock=threading.Lock()
        self.used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    def charge(self,n):
        with self.lock:
            if self.used+n>self.limit:raise RuntimeError('ABO_storage_budget_reached')
            if shutil.disk_usage(self.root).free-n<2**40:raise RuntimeError('1TiB_disk_reserve_reached')
            self.used+=n
    def fetch(self,key,target,size=None,etag=None):
        target.parent.mkdir(parents=True,exist_ok=True)
        url=BASE+quote(key,safe='/')
        part=target.with_name(target.name+'.part')
        if not target.exists():
            for attempt in range(4):
                try:
                    offset=part.stat().st_size if part.exists() else 0
                    if size is not None and offset==size:break
                    headers={'User-Agent':UA}
                    if offset:headers['Range']=f'bytes={offset}-'
                    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=45) as response:
                        if offset and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                            raise ValueError('Server did not confirm precise resume offset')
                        with part.open('ab' if offset else 'wb') as f:
                            while block:=response.read(2**20):
                                self.charge(len(block));f.write(block)
                                if size is not None and f.tell()>size:raise ValueError('Exceeded official file size')
                    if size is not None and part.stat().st_size!=size:raise IOError('Incomplete response')
                    break
                except RuntimeError:raise
                except Exception:
                    if attempt==3:raise
                    time.sleep(2**attempt)
            if size is not None and part.stat().st_size!=size:raise ValueError('Incomplete payload')
            actual_sha,actual_md5=hashes(part)
            if etag and '-' not in etag and actual_md5!=etag:raise ValueError('Official S3 ETag MD5 mismatch')
            part.replace(target)
        else:
            if size is not None and target.stat().st_size!=size:raise ValueError('Existing size mismatch')
            actual_sha,actual_md5=hashes(target)
            if etag and '-' not in etag and actual_md5!=etag:raise ValueError('Existing checksum mismatch')
        return dict(local_path=str(target),source_url=url,bytes=target.stat().st_size,sha256=actual_sha,
            official_S3_ETag=etag,publisher_MD5_verified=bool(etag and '-' not in etag))


def glb_structure(path):
    with Path(path).open('rb') as f:
        magic,version,length=struct.unpack('<4sII',f.read(12))
        if magic!=b'glTF' or version!=2 or length!=Path(path).stat().st_size:raise ValueError('Invalid GLB header or size')
        n,kind=struct.unpack('<I4s',f.read(8))
        if kind!=b'JSON' or n>64*2**20:raise ValueError('Invalid GLB JSON chunk')
        document=json.loads(f.read(n))
    if not document.get('meshes'):raise ValueError('GLB has no mesh')
    for item in document.get('buffers',[])+document.get('images',[]):
        uri=item.get('uri','')
        if uri and not uri.startswith('data:'):raise ValueError('External GLB dependency missing from single-file source')
    return dict(header_and_declared_length_valid=True,self_contained=True,mesh_count=len(document['meshes']),
        material_count=len(document.get('materials',[])),image_count=len(document.get('images',[])),
        geometry_CPU_load='pending',physical_qualified=False)


def english_title(row):
    names=row.get('item_name',[])
    if isinstance(names,str):return names
    return next((r['value'] for r in names if r.get('language_tag')=='en_US'),names[0].get('value','') if names else '')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--budget-gib',type=float,default=200)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(root/'download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(root/'process.json',dict(pid=os.getpid(),started_unix=time.time(),
        proc_start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        root=str(root),proxy_configured=bool(os.environ.get('HTTPS_PROXY'))))
    store=Store(root,int(args.budget_gib*2**30));start=time.time()
    save(root/'state.json',dict(status='fetching_official_metadata',started_unix=start))
    for key in ['README.md','3dmodels/README.md','LICENSE-CC-BY-4.0.txt']:
        store.fetch(key,root/'provenance'/key)
    meta=list_bucket('3dmodels/metadata/',root)[0]
    receipt=store.fetch(meta['key'],root/'metadata/3dmodels.csv.gz',meta['size'],meta['etag'])
    with gzip.open(receipt['local_path'],'rt') as f:models={r['3dmodel_id']:r for r in csv.DictReader(f)}
    objects=list_bucket('3dmodels/original/',root)
    objects={r['key']:r for r in objects if r['key'].endswith('.glb')}
    listing_files=list_bucket('listings/metadata/',root)
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        fs=[pool.submit(store.fetch,r['key'],root/'metadata'/Path(r['key']).name,r['size'],r['etag']) for r in listing_files]
        for future in concurrent.futures.as_completed(fs):future.result()
    products=collections.defaultdict(list)
    for p in sorted((root/'metadata').glob('listings_*.json.gz')):
        with gzip.open(p,'rt') as f:
            for line in f:
                row=json.loads(line);model_id=row.get('3dmodel_id')
                if model_id in models:products[model_id].append(row)
    plan=[]
    for model_id,model in models.items():
        key='3dmodels/original/'+model['path']
        if key not in objects:continue
        items=products.get(model_id,[])
        titles=[english_title(p) for p in items]
        kinds=sorted({x.get('value','unknown') for p in items for x in p.get('product_type',[])})
        plan.append(dict(source_id=model_id,name=next((t for t in titles if t),model_id),
            product_names=titles,categories=kinds or ['unclassified_product'],official_geometry_metadata=model,
            source_item_ids=[p['item_id'] for p in items],object=objects[key]))
    # Finish kitchen, furniture and storage products first; shoes remain useful
    # home objects but must not dominate the early batch or be labelled kitchen.
    plan.sort(key=lambda r:(any('SHOE' in c for c in r['categories']),r['object']['size'],r['source_id']))
    save(root/'selection.json',dict(source='ABO',license='CC-BY-4.0',model_count=len(plan),
        official_payload_bytes=sum(r['object']['size'] for r in plan),by_category=dict(collections.Counter(c for r in plan for c in r['categories'])),
        models=plan,duplicates_by_geometry_pending=True))
    completed={}
    journal=root/'catalog.jsonl'
    if journal.exists():
        for line in journal.read_text().splitlines():
            row=json.loads(line);completed[row['source_id']]=row
    def obtain(row):
        item=row['object'];path=root/'raw'/row['official_geometry_metadata']['path']
        data=store.fetch(item['key'],path,item['size'],item['etag'])
        check=glb_structure(path)
        return dict(asset_id='abo:'+row['source_id'],source='ABO',source_id=row['source_id'],
            name=row['name'],categories=row['categories'],source_item_ids=row['source_item_ids'],
            license='CC-BY-4.0',license_url='https://creativecommons.org/licenses/by/4.0/',
            attribution='Amazon.com; ABO dataset authors',geometry_path=str(path),kind='model',**data,
            download_status='complete',geometry_check=check,official_geometry_metadata=row['official_geometry_metadata'],
            raw_original_preserved=True,coordinate_convention='Official ABO: +Y up, +Z natural front; no rescale applied',
            physics_qualified=False,simulation_qualified=False,training_admitted=False)
    failures=[]
    state=dict(status='downloading',selected=len(plan),complete_models=len(completed),failed_this_run=0,
        downloaded_model_bytes=sum(r['bytes'] for r in completed.values()),updated_unix=time.time())
    save(root/'state.json',state)
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        pending={pool.submit(obtain,r):r for r in plan if r['source_id'] not in completed}
        for future in concurrent.futures.as_completed(pending):
            row=pending[future]
            try:
                record=future.result()
                with journal.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                completed[record['source_id']]=record
            except Exception as exc:
                failed=dict(source_id=row['source_id'],error=str(exc),time_unix=time.time())
                failures.append(failed)
                with (root/'failed_events.jsonl').open('a') as f:f.write(json.dumps(failed)+'\n')
                if isinstance(exc,RuntimeError):
                    for task in pending:task.cancel()
                    raise
            state=dict(status='downloading',selected=len(plan),complete_models=len(completed),failed_this_run=len(failures),
                downloaded_model_bytes=sum(r['bytes'] for r in completed.values()),elapsed_seconds=time.time()-start,
                last_source_id=row['source_id'],source='ABO',physics_qualified=0,updated_unix=time.time())
            save(root/'state.json',state)
            print(json.dumps(state),flush=True)
    state.update(status='download_queue_finished',failures=failures)
    save(root/'REPORT.json',state);save(root/'state.json',state)


if __name__=='__main__':
    main()
