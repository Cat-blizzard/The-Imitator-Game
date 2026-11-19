"""Bounded, resumable household Objaverse acquisition; raw GLBs never modified.

Run from the workspace using PolicyGen/.venv/bin/python. All data/receipts and
mutable download journals stay under the --root. No simulator/GPU/model import.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .objaverse_household_taxonomy import GROUPS, name_groups

OLD = Path('/mnt/sdf/datasets/open_assets/task_factory_20261016_v1')
OPEN_ASSETS = OLD.parent
DEFAULT_ROOT = Path('/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1/objaverse')
WORKSPACE = Path(__file__).resolve().parents[3]
REPO = Path('/mnt/sdf/dataset/zhouxunzhe/zhiqi/PolicyGen')
PYTHON = REPO/'.venv/bin/python'
REVISION = '556637099bf4fa79ea7b239d0d9c328b8a2e9ac8'
BASE = f'https://huggingface.co/datasets/allenai/objaverse/resolve/{REVISION}'
SOURCES = {
    'dataset': 'https://huggingface.co/datasets/allenai/objaverse',
    'official_code': 'https://github.com/allenai/objaverse-xl/tree/main/objaverse',
    'official_API_source': 'https://raw.githubusercontent.com/allenai/objaverse-xl/main/objaverse/__init__.py',
    'license_note': 'Dataset collection is ODC-By; download eligibility uses each individual object license, not repository code license.',
}
LICENSES = {'cc0': ('CC0-1.0', 'https://creativecommons.org/publicdomain/zero/1.0/'),
            'by': ('CC-BY-4.0', 'https://creativecommons.org/licenses/by/4.0/'),
            'by-sa': ('CC-BY-SA-4.0', 'https://creativecommons.org/licenses/by-sa/4.0/')}
UID = re.compile(r'^[0-9a-f]{32}$')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def json_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def save(path, document):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def append(path, document):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(document, ensure_ascii=False, allow_nan=False)+'\n')
        handle.flush()


def iter_rows(path):
    if Path(path).exists():
        with Path(path).open(encoding='utf-8') as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)


class Budget:
    def __init__(self, root, maximum, reserve):
        self.root, self.maximum, self.reserve = root, maximum, reserve
        self.lock, self.pending = threading.Lock(), 0
        self.committed = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())

    def acquire(self, amount):
        with self.lock:
            if self.committed + self.pending + amount > self.maximum:
                raise RuntimeError('asset_source_quota_reached')
            if shutil.disk_usage(self.root).free - self.pending - amount < self.reserve:
                raise RuntimeError('global_disk_reserve_reached')
            self.pending += amount

    def finish(self, reserved, actual=0):
        with self.lock:
            self.pending -= reserved
            self.committed += actual

    def refresh(self):
        with self.lock:
            self.committed = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())


def fetch(url, path, maximum, budget, expected_size=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(path.suffix+'.source.json')
    if path.exists():
        if not receipt.exists():
            raise ValueError('existing_file_without_download_receipt')
        old = load(receipt)
        if old['url'] != url or old['sha256'] != sha(path):
            raise ValueError('existing_source_hash_conflict')
        return old
    reservation = maximum
    budget.acquire(reservation)
    part = path.with_suffix(path.suffix+'.part')
    actual = 0
    try:
        for attempt in range(3):
            try:
                started = time.monotonic()
                # curl's system TLS transport succeeds on this server's configured
                # proxy where Python's urllib TLS handshake returned unexpected EOF.
                # Stream through this process to retain hard byte/disk guards.
                command = ['curl', '--fail', '--silent', '--show-error', '--location',
                           '--connect-timeout', '12', '--max-time', '180', '--max-filesize', str(maximum),
                           '--user-agent', 'PolicyGen-HouseholdAssetIntake/1.0', url]
                child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    with part.open('wb') as target:
                        actual = 0
                        while True:
                            block = child.stdout.read(1 << 20)
                            if not block:
                                break
                            actual += len(block)
                            if actual > maximum:
                                raise ValueError('stream_exceeds_per_asset_limit')
                            if shutil.disk_usage(path.parent).free < budget.reserve + len(block):
                                raise RuntimeError('global_1TiB_reserve_reached')
                            target.write(block)
                    code = child.wait(timeout=15)
                    error = child.stderr.read().decode(errors='replace')
                    if code:
                        raise RuntimeError('curl_HTTP_transfer_failed_'+str(code)+':'+error[-300:])
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:child.kill(); child.wait(timeout=5)
                    child.stdout.close(); child.stderr.close()
                if actual == 0:
                    raise ValueError('empty_HTTP_object')
                digest = sha(part); part.replace(path)
                result = {'url': url, 'transport': 'curl_official_https', 'sha256': digest, 'bytes': actual,
                          'retrieved_unix': time.time(), 'elapsed_seconds': time.monotonic()-started,
                          'metadata_size_bytes': expected_size,
                          'metadata_size_matches': expected_size is None or expected_size == actual}
                save(receipt, result)
                return result
            except Exception as exc:
                part.unlink(missing_ok=True)
                actual = 0
                if isinstance(exc, ValueError) or attempt == 2:
                    raise
                time.sleep(2 ** (attempt + 1))
    finally:
        budget.finish(reservation, actual if path.exists() else 0)


def check_glb(path):
    path = Path(path)
    with path.open('rb') as source:
        header = source.read(12)
        if len(header) != 12:
            raise ValueError('truncated_GLB_header')
        magic, version, size = struct.unpack('<4sII', header)
        if magic != b'glTF' or version != 2 or size != path.stat().st_size:
            raise ValueError('invalid_GLB_magic_version_or_length')
        first = source.read(8)
        if len(first) != 8:
            raise ValueError('truncated_GLB_JSON_header')
        length, kind = struct.unpack('<I4s', first)
        if kind != b'JSON' or not 0 < length <= min(size - 20, 32 << 20):
            raise ValueError('invalid_GLB_JSON_chunk')
        data = json.loads(source.read(length))
        while source.tell() < size:
            chunk = source.read(8)
            if len(chunk) != 8:
                raise ValueError('truncated_GLB_binary_chunk')
            count, _ = struct.unpack('<I4s', chunk)
            if count > size - source.tell():
                raise ValueError('GLB_chunk_exceeds_file')
            source.seek(count, 1)
        if source.tell() != size:
            raise ValueError('GLB_chunk_alignment_error')
    external = []
    for kind in ('buffers', 'images'):
        for i, record in enumerate(data.get(kind, [])):
            uri = record.get('uri')
            if uri and not uri.startswith('data:'):
                external.append({'kind': kind, 'index': i, 'uri': uri})
    if not data.get('meshes'):
        raise ValueError('GLB_has_no_meshes')
    return {'status': 'structurally_valid', 'GLB_version': version,
            'mesh_count': len(data.get('meshes', [])), 'material_count': len(data.get('materials', [])),
            'image_count': len(data.get('images', [])), 'texture_count': len(data.get('textures', [])),
            'animation_count': len(data.get('animations', [])), 'external_resources': external,
            'self_contained': not external, 'required_extensions': data.get('extensionsRequired', []),
            'original_geometry_and_material_bytes_preserved': True,
            'visual_semantics_approved': False, 'physics_qualified': False}


def existing_uids():
    index = defaultdict(list)
    for directory, subdirs, names in os.walk(OPEN_ASSETS):
        # New source-specific runs have their own journals; root handles cross-source aliases.
        subdirs[:] = [d for d in subdirs if d != 'household_scaleup_20261020_v1']
        parent = Path(directory)
        for name in names:
            if not name.lower().endswith(('.glb', '.gltf')):
                continue
            path = parent/name
            uid = path.stem if UID.fullmatch(path.stem) else parent.name if UID.fullmatch(parent.name) else None
            if uid:
                index[uid].append(str(path))
    return dict(index)


def inspect_chunk(path, paths, membership, existing):
    rows, quarantined, rejection = [], [], Counter()
    content = json_gz(path); digest = sha(path)
    for uid, a in content.items():
        if uid not in paths:
            continue
        groups, text_evidence = name_groups(a)
        if text_evidence.startswith('explicit_'):
            rejection[text_evidence] += 1; continue
        lvis = membership.get(uid, {})
        groups = sorted(set(groups) | set(lvis))
        if not groups:
            continue
        if uid in existing:
            rejection['already_present_UID'] += 1; continue
        archive = a.get('archives', {}).get('glb', {})
        token = str(a.get('license', '')).lower()
        base = {'source': 'objaverse', 'source_id': uid, 'uid': uid, 'name': a.get('name', ''),
                'categories': groups, 'LVIS_categories': sorted({c for cs in lvis.values() for c in cs}),
                'category_evidence': 'LVIS_and_'+text_evidence if lvis else 'source_'+text_evidence,
                'license_token': token, 'source_url': a.get('viewerUrl') or 'https://sketchfab.com/3d-models/'+uid,
                'author': {k: a.get('user', {}).get(k) for k in ('uid', 'username', 'displayName', 'profileUrl')},
                'repository_revision': REVISION, 'object_path': paths[uid],
                'download_url': BASE+'/'+paths[uid], 'archive_metadata': archive,
                'metadata_chunk': {'path': str(path), 'sha256': digest},
                'physics_qualified': False, 'visual_semantics_approved': False}
        if token not in LICENSES:
            quarantined.append({**base, 'license': token or 'unknown', 'download_status': 'metadata_only_license_quarantine',
                                'reason': 'unknown_license' if not token else 'outside_CC0_BY_BY_SA_subset'})
            continue
        size = archive.get('size')
        if not isinstance(size, int) or size <= 0 or size > 100 << 20:
            rejection['missing_size_or_over_100MiB'] += 1; continue
        if not isinstance(archive.get('faceCount'), int) or archive['faceCount'] < 12:
            rejection['missing_or_negligible_surface_geometry'] += 1; continue
        base.update(license=LICENSES[token][0], license_url=LICENSES[token][1],
                    retrieval_priority=(0 if lvis and text_evidence == 'name_supported' else 1 if lvis else 2 if text_evidence == 'name_supported' else 3))
        rows.append(base)
    return rows, quarantined, rejection


def discover(root, paths, lvis, metadata_files, existing):
    membership = defaultdict(lambda: defaultdict(list))
    found_categories = {}
    for group, labels in GROUPS.items():
        found_categories[group] = [label for label in labels if label in lvis]
        for label in found_categories[group]:
            for uid in lvis[label]:
                membership[uid][group].append(label)
    rows, unknown, rejected = {}, {}, Counter()
    for path in metadata_files:
        selected, quarantine, counts = inspect_chunk(path, paths, membership, existing)
        rows.update((r['uid'], r) for r in selected)
        unknown.update((r['uid'], r) for r in quarantine)
        rejected.update(counts)
    ordered = balanced(list(rows.values()))
    with (root/'candidates.jsonl').open('w', encoding='utf-8') as stream:
        for row in ordered:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    with (root/'license_quarantine_metadata.jsonl').open('w', encoding='utf-8') as stream:
        for row in unknown.values():
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    report = {'schema': 'objaverse_household_discovery_v1', 'candidate_count': len(ordered),
              'category_counts': dict(Counter(g for r in ordered for g in r['categories'])),
              'license_counts': dict(Counter(r['license'] for r in ordered)),
              'source_category_counts': {g: len(v) for g, v in found_categories.items()},
              'exact_LVIS_household_categories': sum(map(len, found_categories.values())),
              'metadata_chunks': len(metadata_files), 'eligible_declared_GLB_bytes': sum(r['archive_metadata']['size'] for r in ordered),
              'metadata_only_license_quarantine': len(unknown), 'rejections': dict(rejected),
              'existing_UIDs_not_downloaded': len(existing), 'group_semantics_not_render_verified': True,
              'source_revision': REVISION, 'official_sources': SOURCES, 'updated_unix': time.time()}
    save(root/'discovery_summary.json', report)
    print(json.dumps({'stage': 'discovery', **{k: report[k] for k in ('candidate_count', 'metadata_chunks', 'eligible_declared_GLB_bytes', 'exact_LVIS_household_categories')}}), flush=True)
    return ordered, report


def balanced(rows):
    # Round-robin primary family, preferring LVIS+title support and manageable raw size.
    counts, bins = Counter(), defaultdict(deque)
    for row in sorted(rows, key=lambda r: (r['retrieval_priority'], r['archive_metadata']['size'] > 12 << 20,
                                          abs(math.log2(max(1024, r['archive_metadata']['size'])) - 20), r['uid'])):
        primary = min(row['categories'], key=lambda g: (counts[g], g))
        counts[primary] += 1; bins[primary].append(row)
    result = []
    while any(bins.values()):
        for family in sorted(bins):
            if bins[family]: result.append(bins[family].popleft())
    return result


def download_rows(root, rows, budget, workers, limit, stage):
    complete = {r['uid'] for r in iter_rows(root/'catalog.jsonl')}
    # Finite failed attempts are preserved; --retry-failed explicitly opts in later.
    failed = {r['uid'] for r in iter_rows(root/'failed_events.jsonl') if r.get('permanent', True)}
    todo = [r for r in rows if r['uid'] not in complete | failed][:max(0, limit-len(complete))]
    begun, success, failures, total_bytes = time.time(), 0, 0, 0
    def one(row):
        path = root/'raw'/row['uid'][:2]/(row['uid']+'.glb')
        receipt = fetch(row['download_url'], path, 100 << 20, budget, row['archive_metadata']['size'])
        geometry = check_glb(path)
        if not geometry['self_contained']:
            raise ValueError('GLB_references_external_resources_not_archived')
        return {**row, 'local_path': str(path), 'geometry_path': str(path), 'sha256': receipt['sha256'],
                'bytes': receipt['bytes'], 'download_status': 'downloaded', 'geometry_check': geometry,
                'retrieved_unix': receipt['retrieved_unix'], 'metadata_size_matches': receipt['metadata_size_matches'],
                'download_receipt': str(path.with_suffix(path.suffix+'.source.json')),
                'physics_qualified': False, 'training_admitted': False, 'retrieval_stage': stage}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        iterator = iter(todo); active = {}
        def submit():
            try:r = next(iterator)
            except StopIteration:return False
            active[executor.submit(one, r)] = r; return True
        for _ in range(workers):submit()
        stop = None
        while active:
            future = next(as_completed(active)); row = active.pop(future)
            try:
                record = future.result(); append(root/'catalog.jsonl', record)
                append(root/'events.jsonl', {'event': 'download_completed', 'uid': row['uid'], 'bytes': record['bytes'], 'time_unix': time.time()})
                success += 1; total_bytes += record['bytes']
            except Exception as exc:
                message = str(exc)
                permanent = not any(x in message for x in ('quota_reached', 'reserve_reached'))
                append(root/'failed_events.jsonl', {'uid': row['uid'], 'name': row['name'], 'source_url': row['source_url'],
                      'categories': row['categories'], 'download_status': 'not_catalogued', 'error': repr(exc),
                      'permanent': permanent, 'stage': stage, 'time_unix': time.time()})
                failures += 1
                if not permanent:stop = message
            state = {'status': 'downloading', 'stage': stage, 'successful_this_stage': success,
                     'failed_this_stage': failures, 'bytes_this_stage': total_bytes,
                     'successful_previous_stages': len(complete), 'selected_this_stage': len(todo),
                     'elapsed_stage_seconds': time.time()-begun, 'updated_unix': time.time(),
                     'source_storage_limit_bytes': budget.maximum, 'global_reserve_bytes': budget.reserve,
                     'workers': workers, 'stop_reason': stop, 'physics_qualified': 0}
            save(root/'state.json', state)
            if (success+failures) % 10 == 0 or success < 3:
                print(json.dumps(state), flush=True)
            if not stop:submit()
    budget.refresh()
    return {'stage': stage, 'successes': success, 'failed': failures, 'bytes': total_bytes, 'stop': stop}


def sample_qa(root, maximum=24):
    rows = list(iter_rows(root/'catalog.jsonl'))
    previous = {r['uid'] for r in iter_rows(root/'sample_geometry_checks.jsonl')}
    chosen, covered = [], Counter()
    for r in rows:
        if r['uid'] in previous:continue
        if all(covered[g] >= 2 for g in r['categories']):continue
        chosen.append(r); covered.update(r['categories'])
        if len(chosen) >= maximum:break
    for row in chosen:
        try:
            process = subprocess.run([str(PYTHON), '-B', '-m', 'data_generation_methods.asset_library.sources.objaverse_household_scaleup', '--check-glb', row['local_path']],
                                     capture_output=True, text=True, timeout=90,
                                     cwd=WORKSPACE,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            if process.returncode:raise ValueError(process.stderr[-1000:] or 'trimesh_child_failed')
            result = json.loads(process.stdout)
            append(root/'sample_geometry_checks.jsonl', {'uid': row['uid'], 'name': row['name'], 'categories': row['categories'],
                   'geometry_path': row['local_path'], 'source_sha256': row['sha256'], **result, 'physics_qualified': False})
        except Exception as exc:
            append(root/'sample_geometry_checks.jsonl', {'uid': row['uid'], 'status': 'trimesh_open_failed', 'error': repr(exc), 'physics_qualified': False})
    return len(chosen)


def run(args):
    root = args.root.resolve()
    if root != DEFAULT_ROOT:
        raise ValueError('Output is confined to the assigned Objaverse source directory')
    if not 1 <= args.workers <= 4 or not 0 < args.max_gib <= 550 or args.reserve_gib < 1024:
        raise ValueError('Requested resources exceed assigned budget')
    root.mkdir(parents=True, exist_ok=True)
    guard = (root/'worker.lock').open('a'); fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(root/'process.json', {'pid': os.getpid(), 'started_unix': time.time(), 'python': sys.executable, 'argv': sys.argv,
                            'workers': args.workers, 'output': str(root), 'no_simulation_or_training': True})
    budget = Budget(root, int(args.max_gib * 2**30), int(args.reserve_gib * 2**30))
    discovery = OLD/'discovery'
    repository = load(discovery/'objaverse_repository.json')
    if repository['sha'] != REVISION:raise ValueError('Existing object index revision differs')
    paths = json_gz(discovery/'object-paths.json.gz'); lvis = json_gz(discovery/'objaverse_lvis.json.gz')
    existing = existing_uids()
    save(root/'existing_UID_inventory.json', {'uids': existing, 'count': len(existing), 'source_tree': str(OPEN_ASSETS),
                                             'new_household_run_excluded_from_old_UID_census': True})
    save(root/'source_index.json', {'official_sources': SOURCES, 'revision': REVISION,
         'existing_indexes': {str(p): sha(p) for p in [discovery/'objaverse_repository.json', discovery/'object-paths.json.gz', discovery/'objaverse_lvis.json.gz']},
         'category_retrieval_not_visual_approval': True,
         'script_sha256': sha(__file__), 'taxonomy_sha256': sha(Path(__file__).with_name('objaverse_household_taxonomy.py')),
         'budget_bytes': budget.maximum, 'reserve_bytes': budget.reserve, 'per_asset_limit_bytes': 100 << 20})
    old_files = sorted((discovery/'objaverse_metadata').glob('*.json.gz'))
    new_meta = root/'metadata'; new_meta.mkdir(exist_ok=True)
    files = {p.name:p for p in old_files}
    files.update({p.name:p for p in new_meta.glob('*.json.gz')})
    candidates, initial = discover(root, paths, lvis, list(files.values()), existing)
    stages = [download_rows(root, candidates, budget, args.workers, min(args.max_models, 24), 'cached_metadata_first24')]
    sample_qa(root, 12)
    needed = sorted({p.split('/')[1]+'.json.gz' for p in paths.values()} - set(files))
    metadata_errors = []
    def metadata(name):
        path = new_meta/name
        fetch(BASE+'/metadata/'+name, path, 64 << 20, budget)
        json_gz(path)
        return path
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(metadata, name): name for name in needed}
        for future in as_completed(futures):
            name = futures[future]
            try:files[name] = future.result()
            except Exception as exc:
                metadata_errors.append({'chunk': name, 'error': repr(exc)})
                append(root/'metadata_errors.jsonl', metadata_errors[-1])
            save(root/'state.json', {'status': 'acquiring_missing_official_metadata', 'cached_initial_chunks': len(old_files),
                 'available_chunks': len(files), 'target_chunks': len(needed)+len(old_files),
                 'metadata_failures': len(metadata_errors), 'first_model_stage': stages[0], 'updated_unix': time.time()})
            if (len(files)+len(metadata_errors)) % 10 == 0:print(json.dumps(load(root/'state.json')), flush=True)
    budget.refresh()
    candidates, report = discover(root, paths, lvis, [files[k] for k in sorted(files)], existing)
    stages.append(download_rows(root, candidates, budget, args.workers, args.max_models, 'full_metadata_household'))
    sample_qa(root, 36)
    rows = list(iter_rows(root/'catalog.jsonl'))
    summary = {'status': 'bounded_household_acquisition_complete', 'completed_unix': time.time(), 'stages': stages,
         'catalogued_models': len(rows), 'actual_raw_model_bytes': sum(r['bytes'] for r in rows),
         'category_counts': dict(Counter(g for r in rows for g in r['categories'])),
         'candidate_count': len(candidates), 'metadata_chunks_available': len(files), 'metadata_errors': metadata_errors,
         'catalogue': str(root/'catalog.jsonl'), 'sample_geometry_checks': str(root/'sample_geometry_checks.jsonl'),
         'physics_qualified': 0, 'training_admitted': 0, 'no_source_mesh_or_material_rewriting': True}
    save(root/'summary.json', summary); save(root/'state.json', summary)
    print(json.dumps(summary), flush=True)


def inspect_surface_geometry(path):
    """Check triangle surfaces separately from valid non-surface glTF geometry."""
    import numpy as np
    import trimesh
    scene = trimesh.load(path, force='scene', process=False)
    if not scene.geometry:raise ValueError('No geometry parsed')
    meshes, non_surface = {}, []
    for name, geometry in scene.geometry.items():
        if isinstance(geometry, trimesh.Trimesh):
            meshes[name] = geometry
        else:
            coordinates = np.asarray(getattr(geometry, 'vertices', []))
            non_surface.append({'name': name, 'geometry_type': type(geometry).__name__,
                                'vertices': len(coordinates),
                                'finite_coordinates': bool(np.isfinite(coordinates).all())})
    result = {'mesh_count': len(meshes), 'non_surface_geometry': non_surface,
              'non_surface_geometry_count': len(non_surface), 'original_scale_is_not_metric_GT': True,
              'qa_checker_source_sha256': sha(__file__)}
    if not meshes:
        return {**result, 'status': 'trimesh_no_surface_mesh', 'surface_geometry_usable': False,
                'reason': 'File parses but contains no triangular surface mesh; unsuitable for surface-based asset intake.'}
    for name, mesh in meshes.items():
        if not np.isfinite(mesh.vertices).all():raise ValueError('Nonfinite surface vertices: '+name)
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:raise ValueError('Empty surface mesh: '+name)
        if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
            raise ValueError('Nontriangular surface faces: '+name)
        if (mesh.faces < 0).any() or (mesh.faces >= len(mesh.vertices)).any():
            raise ValueError('Invalid surface face indices: '+name)
    surfaces = trimesh.Scene()
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        if geometry_name in meshes:
            surfaces.add_geometry(meshes[geometry_name], node_name=node, transform=transform)
    bounds = np.asarray(surfaces.bounds)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or not (np.ptp(bounds, axis=0) > 1e-10).any():
        raise ValueError('Invalid surface scene bounds')
    return {**result, 'status': 'trimesh_open_passed', 'surface_geometry_usable': True,
            'vertices': sum(len(m.vertices) for m in meshes.values()),
            'faces': sum(len(m.faces) for m in meshes.values()), 'bounds_source_units': bounds.tolist()}


def main():
    if '--check-glb' in sys.argv:
        print(json.dumps(inspect_surface_geometry(sys.argv[sys.argv.index('--check-glb')+1])));return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-gib', type=float, default=550)
    parser.add_argument('--reserve-gib', type=float, default=1024)
    parser.add_argument('--max-models', type=int, default=100000)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':main()
