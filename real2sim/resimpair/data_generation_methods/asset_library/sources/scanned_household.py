"""Bounded original GSO/Fuel and official YCB acquisition, never simulation admission.

Run with the project venv. The only asset output is under the explicit --output.
Archive members are copied as regular files after path/type/size validation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gc
import hashlib
import html
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import threading
import time
from urllib.error import HTTPError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import Request, urlopen
import zipfile

DEFAULT_ROOT = Path('/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1/scanned_objects')
YCB_LOCAL = Path('/home/xzzhou/.maniskill/data/assets/mani_skill2_ycb/models')
FUEL = 'https://fuel.gazebosim.org/1.0/GoogleResearch/models'
KUBRIC = 'https://storage.googleapis.com/kubric-public/assets/GSO/GSO.json'
YCB_INDEX = 'https://ycb-benchmarks.s3.amazonaws.com/index.html'
GSO_LICENSE_PAGE = 'https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/'
UA = 'PolicyGenHouseholdAssetInventory/1.0 (research; original material preservation)'
GIB = 1024**3
MAX_ARCHIVE = 512 * 1024**2
MAX_UNPACKED = 2 * GIB
MAX_MEMBERS = 20000


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(data)
    os.replace(tmp, path)


class BudgetStop(RuntimeError):
    pass


class Budget:
    def __init__(self, root, limit, reserve):
        self.root, self.limit, self.reserve = root, limit, reserve
        self.used = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def claim(self, n):
        with self.lock:
            if (self.stop.is_set() or self.used + n > self.limit
                    or shutil.disk_usage(self.root).free - n < self.reserve):
                self.stop.set()
                raise BudgetStop('dataset_budget_or_disk_reserve_reached')
            self.used += n

    def refund(self, n):
        with self.lock:
            self.used -= n


def metadata(url, path, budget, timeout=15):
    if path.is_file():
        return path.read_bytes()
    req = Request(url, headers={'User-Agent': UA})
    with urlopen(req, timeout=timeout) as response:
        data = response.read(16 * 1024**2 + 1)
        if len(data) > 16 * 1024**2:
            raise ValueError('metadata_too_large')
    budget.claim(len(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def destination(root, member):
    if not isinstance(member, str) or '\\' in member or '\x00' in member:
        raise ValueError('unsafe_archive_path')
    name = PurePosixPath(member)
    if name.is_absolute() or '..' in name.parts or any(':' in p for p in name.parts):
        raise ValueError('unsafe_archive_path')
    target = (root / str(name)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('archive_path_escape')
    return target


def copy_member(stream, target, size, budget):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    count = 0
    with os.fdopen(fd, 'wb') as output:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            count += len(chunk)
            if count > size:
                raise ValueError('archive_member_larger_than_declared')
            budget.claim(len(chunk))
            output.write(chunk)
    if count != size:
        raise ValueError('archive_member_truncated')


def extract_archive(archive, output, budget):
    if output.exists():
        raise ValueError('extraction_output_must_be_fresh')
    output.mkdir(parents=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            if len(members) > MAX_MEMBERS or sum(x.file_size for x in members) > MAX_UNPACKED:
                raise ValueError('archive_expansion_budget_exceeded')
            # Validate every path/type before copying any payload.
            for m in members:
                destination(output, m.filename)
                mode = m.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                    raise ValueError('archive_link_or_special_file')
            for m in members:
                target = destination(output, m.filename)
                if m.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    with z.open(m) as source:
                        copy_member(source, target, m.file_size, budget)
    else:
        with tarfile.open(archive, 'r:*') as tar:
            members = tar.getmembers()
            if len(members) > MAX_MEMBERS or sum(m.size for m in members) > MAX_UNPACKED:
                raise ValueError('archive_expansion_budget_exceeded')
            for m in members:
                destination(output, m.name)
                if not (m.isdir() or m.isfile()):
                    raise ValueError('archive_link_or_special_file')
            for m in members:
                target = destination(output, m.name)
                if m.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    with tar.extractfile(m) as source:
                        copy_member(source, target, m.size, budget)


def download(url, path, budget):
    receipt = path.with_suffix(path.suffix + '.download.json')
    if path.is_file():
        old = json.loads(receipt.read_text())
        if old['source_url'] != url or old['sha256'] != sha(path):
            raise ValueError('completed_download_binding_changed')
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.part')
    size = part.stat().st_size if part.exists() else 0
    headers = {'User-Agent': UA}
    if size:
        headers['Range'] = f'bytes={size}-'
    start = time.monotonic()
    with urlopen(Request(url, headers=headers), timeout=45) as response:
        if urlsplit(response.url).scheme != 'https':
            raise ValueError('non_https_download_redirect')
        if size and response.status != 206:
            budget.refund(size)
            size = 0
        elif size and not response.headers.get('Content-Range', '').startswith(f'bytes {size}-'):
            raise ValueError('invalid_resume_range')
        declared = response.headers.get('Content-Length')
        if declared and int(declared) + size > MAX_ARCHIVE:
            raise ValueError('archive_exceeds_per_object_limit')
        written = 0
        with part.open('ab' if size else 'wb') as target:
            for data in iter(lambda: response.read(1 << 20), b''):
                if size + written + len(data) > MAX_ARCHIVE:
                    raise ValueError('archive_exceeds_per_object_limit')
                budget.claim(len(data))
                target.write(data)
                written += len(data)
        if declared and written != int(declared):
            raise ValueError('download_content_length_mismatch')
        result = dict(source_url=url, resolved_url=response.url, sha256=sha(part),
                      bytes=part.stat().st_size, downloaded_this_call_bytes=written,
                      elapsed_seconds=time.monotonic()-start, etag=response.headers.get('ETag'),
                      last_modified=response.headers.get('Last-Modified'), retrieved_utc=now())
    os.replace(part, path)
    write(receipt, result)
    return result


def geometry(root):
    import numpy as np
    import trimesh
    from PIL import Image
    meshfiles = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.obj', '.ply', '.glb', '.stl', '.dae')]
    if not meshfiles:
        return None, {'status': 'no_supported_visual_mesh', 'readable': False}
    def priority(p):
        n = p.name.lower()
        return ('collision' in n, 'textured' not in n and 'model.obj' != n, p.suffix != '.obj', str(p))
    meshfiles.sort(key=priority)
    primary = meshfiles[0]
    try:
        # Original Gazebo packages may keep model.mtl under meshes/ while its
        # texture.png lives under materials/textures/. Resolve only unambiguous
        # in-package basenames without modifying source OBJ/MTL/PNG files.
        material_files=[p for p in root.rglob('*') if p.is_file()
                        and p.suffix.lower() in ('.mtl','.png','.jpg','.jpeg')]
        by_name={}
        for p in material_files:by_name.setdefault(p.name,[]).append(p)
        resource_map={str(p.relative_to(root)):p.read_bytes() for p in material_files}
        for name,paths in by_name.items():
            if len(paths)==1:resource_map[name]=resource_map[str(paths[0].relative_to(root))]
        resolver=trimesh.resolvers.ZipResolver(archive=resource_map)
        scene = trimesh.load(primary, force='scene', process=False, resolver=resolver)
        meshes = [m for m in scene.geometry.values() if isinstance(m, trimesh.Trimesh)]
        if not meshes:
            raise ValueError('scene_has_no_triangle_geometry')
        verts = sum(len(m.vertices) for m in meshes)
        faces = sum(len(m.faces) for m in meshes)
        bounds = np.asarray(scene.bounds)
        finite = all(np.isfinite(m.vertices).all() for m in meshes) and np.isfinite(bounds).all()
        nonzero = bool(bounds.shape == (2, 3) and (bounds[1]-bounds[0]).max() > 0)
        missing, indirect = [], []
        for obj in root.rglob('*.obj'):
            for line in obj.read_text(errors='replace').splitlines():
                if line.startswith('mtllib '):
                    name = line[7:].strip()
                    q = (obj.parent/name).resolve()
                    if not q.is_relative_to(root.resolve()) or not q.is_file():
                        missing.append(str(obj.relative_to(root)) + ':mtllib:' + name)
        for mtl in root.rglob('*.mtl'):
            for line in mtl.read_text(errors='replace').splitlines():
                if line.strip().startswith(('map_Kd ', 'map_Ka ', 'map_Bump ', 'bump ')):
                    name = line.split(maxsplit=1)[1].strip()
                    q = (mtl.parent/name).resolve()
                    if not q.is_relative_to(root.resolve()) or not q.is_file():
                        alternatives=by_name.get(Path(name).name,[]) if not Path(name).is_absolute() else []
                        if len(alternatives)==1:
                            indirect.append({'mtl':str(mtl.relative_to(root)),'reference':name,
                                             'resolved_to':str(alternatives[0].relative_to(root)),
                                             'resolution':'unique_basename_within_original_asset_only'})
                        else:missing.append(str(mtl.relative_to(root)) + ':texture:' + name)
        textures = []
        for image in root.rglob('*'):
            if image.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                try:
                    with Image.open(image) as im:
                        shape = list(im.size); im.verify()
                    textures.append({'path':str(image),'resolution':shape,'decode_verified':True})
                except Exception as exc:
                    textures.append({'path':str(image),'decode_verified':False,'error':type(exc).__name__})
        result = {'status':'readable_geometry_candidate' if finite and nonzero and faces else 'invalid_geometry',
                  'readable':bool(finite and nonzero and faces),'vertices':verts,'faces':faces,
                  'bounds_source_units':bounds.tolist(),'source_units_independently_verified':False,
                  'geometries':len(meshes),'watertight_all':all(bool(m.is_watertight) for m in meshes),
                  'mesh_files':len(meshfiles),'texture_files':textures,
                  'missing_material_references':missing,'materials_complete':not missing and all(t['decode_verified'] for t in textures),
                  'indirect_original_material_bindings':indirect,
                  'direct_obj_material_paths_complete':not missing and not indirect,
                  'original_asset_material_resolver_required':bool(indirect),
                  'loaded_texture_visual_meshes':sum(m.visual.kind=='texture' and getattr(getattr(m.visual,'material',None),'image',None) is not None for m in meshes),
                  'geometry_check_version':2,
                  'collision_or_simulation_qualified':False}
        del scene, meshes
        gc.collect()
        return str(primary), result
    except Exception as exc:
        return str(primary), {'status':'geometry_parse_failed','readable':False,'error':type(exc).__name__}


def asset_key(name):
    clean = re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('.')[:170] or 'object'
    return clean + '_' + hashlib.sha256(name.encode()).hexdigest()[:8]


def inventory_and_plan(root, budget):
    docs = root/'metadata'
    kubric_path = root/'gso_kubric_catalog.json'
    kubric = json.loads(metadata(KUBRIC, kubric_path, budget))
    # Source license evidence is retained independently of derivative metadata.
    try:
        metadata(GSO_LICENSE_PAGE, docs/'google_scanned_objects_official_blog.html', budget, timeout=8)
    except Exception as exc:
        # Original per-asset Fuel license fields below remain mandatory. A
        # slow marketing/blog host must not block the actual asset queue.
        write(docs/'google_blog_fetch_status.json',{'url':GSO_LICENSE_PAGE,'error':type(exc).__name__,
              'license_basis':'mandatory_original_official_Fuel_per_asset_license_metadata'})
    ycb_html = metadata(YCB_INDEX, docs/'ycb_official_index.html', budget).decode()
    models = {}
    def fetch_page(page):
        p=root/'gso_fuel_page1.json' if page==1 else docs/f'gso_fuel_page{page:02d}.json'
        for attempt in range(3):
            try:return json.loads(metadata(FUEL+f'?page={page}&per_page=100',p,budget))
            except HTTPError as exc:
                if exc.code==404:return []
                if attempt==2:raise
            except Exception:
                if attempt==2:raise
            time.sleep(1+attempt)
    # The official derivative catalog bounds the expected original collection.
    # Two extra pages establish the actual original pagination terminator.
    last_page=math.ceil(len(kubric['assets'])/100)+2
    print(json.dumps({'phase':'official_metadata_pages','bounded_pages':last_page}),flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        pages=list(pool.map(fetch_page,range(1,last_page+1)))
    terminated=False
    for page,content in enumerate(pages,1):
        if not isinstance(content, list):
            raise ValueError('invalid_official_fuel_catalog')
        for row in content:
            if row.get('owner') != 'GoogleResearch' or row.get('private') is not False:
                raise ValueError('unexpected_nonpublic_fuel_owner')
            models[row['name']] = row
        if len(content) < 100:
            terminated=True
            break
    if not terminated:raise ValueError('fuel_catalog_page_limit')
    jobs = []
    for name, row in sorted(models.items()):
        meta = kubric['assets'].get(name, {})
        category = meta.get('metadata', {}).get('category')
        if category in (None, '', 'None'):
            category = None
        license_url = row.get('license_url', '')
        if 'creativecommons.org/licenses/by/4.0' not in license_url:
            raise ValueError('original_fuel_license_not_verified:' + name)
        jobs.append({'source':'google_scanned_objects_original_fuel','source_id':name,'name':name,
                     'category':category,'category_provenance':'official_google_research_kubric_GSO_metadata_not_independently_verified',
                     'license':'CC-BY-4.0','license_url':license_url,'raw_source_metadata':row,
                     'derivative_metadata_license':meta.get('license'),'derivative_metadata':meta,
                     'source_url':FUEL+'/'+quote(name,safe='')+'/1/'+quote(name,safe='')+'.zip',
                     'kind':'zip','local_existing_path':None})
    # Kitchen/containers first, then complete the full original object collection.
    def rank(job):
        text=(job['name']+' '+str(job['category'])).lower()
        return (not any(w in text for w in ('bowl','cup','mug','bottle','plate','food','kitchen','can ','box','ramekin','utensil')),
                'shoe' in text, job['source_id'])
    jobs.sort(key=rank)
    ycb_rows=[]
    for row in re.findall(r'<tr[^>]*>(.*?)</tr>',ycb_html,re.S|re.I):
        names=re.findall(r'(\d{3}[-_][A-Za-z0-9_-]+)',re.sub('<[^>]+>',' ',row))
        if not names:
            continue
        name=names[0]
        links=[urljoin(YCB_INDEX,html.unescape(s)) for s in re.findall(r'href=[\"\']([^\"\']+)[\"\']',row,re.I)]
        candidates=[u for u in links if u.endswith('_google_16k.tgz')]
        if not candidates:
            candidates=[u for u in links if u.endswith('_berkeley_meshes.tgz')]
        local=YCB_LOCAL/name
        base={'source':'ycb_official','source_id':name,'name':name,'category':None,
              'category_provenance':'official_object_name_only_no_category_annotation',
              'license':'CC-BY-4.0','license_url':'https://creativecommons.org/licenses/by/4.0/',
              'source_url':candidates[0] if candidates else None,'kind':'tgz',
              'local_existing_path':str(local) if local.is_dir() else None}
        ycb_rows.append({**base,'official_mesh_release_available':bool(candidates)})
        if local.is_dir() or candidates:
            jobs.append(base)
    write(root/'source_catalog.json',{'schema':'scanned_household_source_catalog_v1','created_utc':now(),
          'original_GSO_objects':len(models),'GSO_kubric_metadata_assets':len(kubric['assets']),
          'official_YCB_objects':len(ycb_rows),'existing_YCB_references':sum(bool(r['local_existing_path']) for r in ycb_rows),
          'YCB_no_available_mesh': [r for r in ycb_rows if not r['source_url'] and not r['local_existing_path']],
          'YCB_inventory':ycb_rows,'jobs':jobs,'physical_qualified':False})
    return jobs


def process(job, root, budget):
    budget.claim(0)
    folder=root/'objects'/job['source']/asset_key(job['source_id'])
    record_path=folder/'record.json'
    if record_path.is_file():
        old=json.loads(record_path.read_text())
        if (old.get('status') in ('downloaded_checked','existing_local_checked')
                and old.get('geometry_check',{}).get('geometry_check_version')==2):
            primary=old.get('primary_mesh_path')
            if primary and Path(primary).is_file() and sha(primary)==old.get('sha256'):
                return old
            raise ValueError('previous_asset_primary_changed')
    folder.mkdir(parents=True,exist_ok=True)
    existing=job['local_existing_path']
    archive=None
    if existing:
        payload=Path(existing)
        download_receipt={'bytes':0,'sha256':None,'reused_existing_local_asset':True}
    else:
        archive=folder/('original.zip' if job['kind']=='zip' else 'original.tgz')
        error=None
        for attempt in range(3):
            try:
                download_receipt=download(job['source_url'],archive,budget)
                error=None;break
            except BudgetStop:
                raise
            except Exception as exc:
                error=exc
                if attempt<2:time.sleep(1+attempt)
        if error:raise error
        payload=folder/'original'
        ready=folder/'EXTRACTED.json'
        if not ready.is_file():
            if payload.exists():
                # An interrupted partial extraction is local derived state only.
                budget.refund(sum(p.stat().st_size for p in payload.rglob('*') if p.is_file()))
                shutil.rmtree(payload)
            extract_archive(archive,payload,budget)
            write(ready,{'archive_sha256':download_receipt['sha256'],'utc':now()})
        elif json.loads(ready.read_text())['archive_sha256'] != download_receipt['sha256']:
            raise ValueError('extraction_archive_binding_changed')
    primary, check=geometry(payload)
    files=[{'path':str(p),'relative_path':str(p.relative_to(payload)),'bytes':p.stat().st_size,'sha256':sha(p)}
           for p in sorted(payload.rglob('*')) if p.is_file()]
    manifest=folder/'files.json'
    write(manifest,{'files':files,'all_original_material_files_preserved':True,'source_payload_reencoded':False})
    record={**job,'schema':'scanned_household_object_v1','status':'existing_local_checked' if existing else 'downloaded_checked',
            'checked_utc':now(),'primary_mesh_path':primary,'local_primarymesh_path':primary,
            'sha256':sha(primary) if primary else None,'geometry_check':check,
            'download_bytes':download_receipt['bytes'],'archive_sha256':download_receipt['sha256'],
            'archive_path':str(archive) if archive else None,'download_receipt':download_receipt,
            'payload_bytes':sum(f['bytes'] for f in files),'payload_file_count':len(files),
            'file_manifest':str(manifest),'file_manifest_sha256':sha(manifest),
            'material_sources_preserved':True,'physical_qualified':False,'collision_qualified':False,
            'simulation_qualified':False,'training_admitted':False,
            'license_scope':'Original Fuel/YCB asset license; derivative Kubric metadata license remains separately recorded.'}
    write(record_path,record)
    return record


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--budget-gib',type=float,default=100)
    parser.add_argument('--reserve-gib',type=float,default=1024)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--max-assets',type=int)
    args=parser.parse_args(argv)
    if not 1<=args.workers<=3:raise ValueError('workers_must_be_1_to_3')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    # One owning queue. Lock auto-releases on process exit, enabling safe resume.
    import fcntl
    lock=(root/'queue.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    budget=Budget(root,int(args.budget_gib*GIB),int(args.reserve_gib*GIB))
    write(root/'OWNER.json',{'pid':os.getpid(),'started_utc':now(),'source_code':str(Path(__file__).resolve()),
                           'source_sha256':sha(__file__),'output':str(root),'workers':args.workers,
                           'budget_bytes':budget.limit,'disk_reserve_bytes':budget.reserve,
                           'simulation_started':False,'training_started':False})
    jobs=inventory_and_plan(root,budget)
    if args.max_assets:jobs=jobs[:args.max_assets]
    counts=collections_counter();done=[];errors=[]
    with (root/'objects.jsonl').open('a',buffering=1) as ledger, ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(process,j,root,budget):j for j in jobs}
        for future in as_completed(pending):
            job=pending[future]
            try:
                record=future.result();done.append(record)
                counts['existing_local' if record['local_existing_path'] else 'downloaded']+=1
                counts['geometry_readable']+=int(record['geometry_check'].get('readable',False))
                ledger.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                print(json.dumps({'source':record['source'],'id':record['source_id'],'status':record['status'],
                                  'downloaded':counts['downloaded'],'existing':counts['existing_local'],
                                  'readable':counts['geometry_readable'],'used_gib':round(budget.used/GIB,3)}),flush=True)
            except BudgetStop as exc:
                counts['budget_stopped']+=1
                errors.append({'source':job['source'],'source_id':job['source_id'],'error':str(exc)})
            except Exception as exc:
                counts['failed']+=1
                error={'source':job['source'],'source_id':job['source_id'],'error':type(exc).__name__+':'+str(exc)[:300]}
                errors.append(error);print(json.dumps(error),flush=True)
            write(root/'state.json',{'status':'budget_stopping' if budget.stop.is_set() else 'running',
                  'updated_utc':now(),'counts':dict(counts),'planned_objects':len(jobs),
                  'dataset_bytes_budget_accounted':budget.used,'free_bytes':shutil.disk_usage(root).free})
    unique={(r['source'],r['source_id']):r for r in done}
    write(root/'SUMMARY.json',{'status':'budget_stopped' if budget.stop.is_set() else 'terminal',
          'updated_utc':now(),'counts':dict(counts),'unique_object_identities_completed':len(unique),
          'archives_downloaded':sum(not r['local_existing_path'] for r in unique.values()),
          'downloaded_archive_bytes':sum(r['download_bytes'] for r in unique.values()),
          'objects_by_source':dict(count_by(r['source'] for r in unique.values())),
          'errors':errors,'dataset_bytes_budget_accounted':budget.used,
          'physical_qualified_objects':0,'simulation_started':False,'training_started':False})
    write(root/'state.json',{'status':'budget_stopped' if budget.stop.is_set() else 'terminal',
                           'updated_utc':now(),'counts':dict(counts),'planned_objects':len(jobs)})


def collections_counter():
    from collections import Counter
    return Counter()


def count_by(values):
    from collections import Counter
    return Counter(values)


if __name__=='__main__':
    main()
