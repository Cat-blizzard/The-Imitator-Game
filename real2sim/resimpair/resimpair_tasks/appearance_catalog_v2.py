"""Bounded, resumable public appearance catalog; no model or simulator calls.

Uses Poly Haven's metadata, rather than a fixed asset-name whitelist. Candidate
selection is frozen before transfers. Every downloaded file is publisher-hash
checked and decoded; failures remain terminal and are never replaced by a seed.
Powered by Poly Haven. Assets: CC0; API: https://github.com/Poly-Haven/Public-API.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
import urllib.request
from urllib.parse import urlsplit

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from data_generation_methods.asset_library.sources import environment_common as transfer

VERSION = 'resimpair_appearance_catalog_v2'
UA = 'ReSimPair-Appearance-Catalog/2.0 (research; Powered by Poly Haven)'
FAMILIES = ['kitchen', 'dining', 'office', 'workshop', 'storage']
DIMENSION_SPEC = 'https://raw.githubusercontent.com/Poly-Haven/Public-API/master/swagger.yml'


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def family_group(asset_id):
    # Numbered variants of one photographed source stay in one appearance split.
    return re.sub(r'(?:_\d+)+$', '', asset_id)


def classify(asset_id, info, kind):
    text = ' '.join([asset_id, info.get('name', ''), *info.get('categories', []),
                     *info.get('tags', [])]).lower().replace('_', ' ')
    cats = set(info.get('categories', []))
    if kind == 'hdri':
        if 'indoor' not in cats:
            return None
        if any(w in text for w in ('abandoned', 'derelict', 'ruin', 'roofless', 'demolish', 'construction site')):
            return None
        rules = [('kitchen', ('kitchen', 'canteen', 'cafeteria')),
                 ('dining', ('dining', 'restaurant', 'cafe', 'living room', 'lounge')),
                 ('office', ('office', 'library', 'classroom', 'study', 'meeting', 'computer lab')),
                 ('workshop', ('workshop', 'garage', 'laboratory', 'industrial', 'factory')),
                 ('storage', ('storage', 'warehouse', 'store room', 'storeroom', 'utility'))]
        families = [name for name, words in rules if any(w in text for w in words)]
        group = families[0] if families else 'studio' if 'studio' in text else 'other_indoor'
        return {'roles': ['environment'], 'categories': families or ['generic_indoor'],
                'scene_group': group, 'source_material_family': None}
    forbidden = {'terrain', 'aerial', 'asphalt', 'road', 'roofing', 'bark', 'sand', 'rock', 'snow', 'ice'}
    if cats & forbidden or any(w in text for w in ('mossy', 'muddy', 'rubble', 'tree bark', 'gravel')):
        return None
    dirty = any(w in text for w in ('damaged', 'rust', 'broken', 'dirty', 'rotting', 'peeling'))
    if 'wood' in cats or 'veneer' in text:
        family = 'wood'; roles = ['floor', 'wall', 'cabinet']
        if not dirty: roles.append('tabletop')
        families = ['workshop', 'storage'] if dirty else FAMILIES
    elif 'marble' in text or 'granite' in text or 'terrazzo' in text:
        family = 'stone'; roles = ['tabletop', 'floor', 'wall']; families = ['kitchen', 'dining', 'office']
    elif 'tile' in text:
        family = 'tile'; roles = ['floor', 'wall']; families = ['kitchen', 'dining', 'storage']
    elif 'metal' in cats or 'metal' in text:
        family = 'metal'; roles = ['metal', 'cabinet', 'wall']
        if not dirty and not any(w in text for w in ('corrugated', 'sheet', 'mesh', 'grate')): roles.append('tabletop')
        families = ['workshop', 'storage'] if dirty else ['kitchen', 'workshop', 'storage']
    elif 'brick' in cats or 'brick' in text:
        family = 'brick'; roles = ['wall']; families = ['office', 'workshop', 'storage']
    elif 'plaster' in cats or 'paint' in text or 'plaster' in text:
        family = 'plaster'; roles = ['wall']; families = ['workshop', 'storage'] if dirty else FAMILIES
    elif 'concrete' in cats or 'concrete' in text:
        family = 'concrete'; roles = ['floor', 'wall']; families = ['workshop', 'storage']
    else:
        return None
    return {'roles': roles, 'categories': list(families), 'scene_group': None,
            'source_material_family': family}


def select(metadata, kind, limit, seed):
    if type(limit) is not int or limit < 1:
        raise ValueError('positive_finite_candidate_limit_required')
    groups = defaultdict(list); excluded = Counter()
    for asset_id, info in metadata.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', asset_id):
            excluded['unsafe_identifier'] += 1; continue
        tags = classify(asset_id, info, kind)
        if tags is None:
            excluded['incompatible_metadata'] += 1; continue
        group = tags['scene_group'] if kind == 'hdri' else tags['source_material_family']
        source_group = kind + ':' + family_group(asset_id)
        row = {'id': 'polyhaven:' + asset_id, 'source_asset_id': asset_id, 'kind': kind, **tags,
               'split_group': source_group,
               'asset_partition': 'heldout_appearance' if int(hashlib.sha256(source_group.encode()).hexdigest()[:8], 16) % 5 == 0 else 'development',
               'source_metadata': info, 'semantic_tag_basis': 'deterministic_publisher_metadata_heuristic_not_verified_room_affordance'}
        groups[group].append(row)
    for rows in groups.values():
        rows.sort(key=lambda x: hashlib.sha256(f'{seed}:{x["id"]}'.encode()).hexdigest())
    chosen = []; used = set(); used_families = set()
    # Balance semantic/material groups. Prefer new source families over variants.
    for unique_only in (True, False):
        while len(chosen) < limit:
            progress = False
            for group in sorted(groups):
                row = next((r for r in groups[group] if r['id'] not in used and
                            (not unique_only or r['split_group'] not in used_families)), None)
                if row is None: continue
                chosen.append(row); used.add(row['id']); used_families.add(row['split_group']); progress = True
                if len(chosen) == limit: break
            if not progress: break
    if len(chosen) != limit:
        raise ValueError(f'insufficient_metadata_candidates:{kind}:{len(chosen)}/{limit}')
    return chosen, {'eligible': sum(map(len, groups.values())), 'excluded': dict(excluded),
                    'selected_groups': dict(Counter(r['scene_group'] or r['source_material_family'] for r in chosen))}


def prepare(root, hdri_metadata, texture_metadata, *, hdris=120, materials=240, seed=20261029, budget_gib=4.):
    root = Path(root).resolve()
    if root.exists(): raise ValueError('catalog_campaign_exists_resume_it_instead')
    if not 0 < budget_gib <= 16: raise ValueError('bounded_storage_budget_required')
    if shutil.disk_usage(root.parent).free < 2**40 + budget_gib * 2**30:
        raise RuntimeError('insufficient_disk_reserve')
    h, hsummary = select(read(hdri_metadata), 'hdri', hdris, seed)
    m, msummary = select(read(texture_metadata), 'material', materials, seed)
    root.mkdir(parents=True)
    for name, source in [('hdris', hdri_metadata), ('textures', texture_metadata)]:
        shutil.copy2(source, root / (name + '_metadata.json'))
    protocol = {'schema': VERSION, 'implementation': binding(__file__),
        'transfer_implementation': binding(transfer.__file__), 'created_unix': time.time(),
        'seed': seed, 'selection': h + m, 'selection_summary': {'hdri': hsummary, 'material': msummary},
        'metadata': {'hdris': binding(root / 'hdris_metadata.json'), 'textures': binding(root / 'textures_metadata.json')},
        'asset_limit': len(h) + len(m), 'byte_limit': int(budget_gib * 2**30),
        'disk_reserve_bytes': 2**40, 'workers': 2, 'metadata_min_interval_seconds': .65,
        'metadata_attempts_per_URL': 3, 'metadata_timeout_seconds': 30,
        'asset_download_attempts_per_file': 4, 'asset_download_timeout_seconds': 60,
        'wall_limit_seconds': 7200, 'required_resolution': '1k', 'paid_model_calls': 0,
        'license': 'CC0-1.0', 'credit': 'Powered by Poly Haven',
        'api_terms': 'https://github.com/Poly-Haven/Public-API/blob/master/ToS.md',
        'dimensions_documentation': DIMENSION_SPEC, 'retry_policy': 'bounded_public_GET_transport_only; terminal_asset_failures_not_replaced',
        'partition_scope': 'appearance_source_families; does_not_override_physical_source_split'}
    write(root / 'protocol.json', protocol)
    return {'root': str(root), 'selected': len(h) + len(m), 'groups': protocol['selection_summary']}


class MetadataFetcher:
    def __init__(self, protocol):
        self.protocol = protocol; self.lock = threading.Lock(); self.last_start = 0.

    def fetch(self, url, path):
        path = Path(path)
        if path.exists(): return read(path)
        for attempt in range(self.protocol['metadata_attempts_per_URL']):
            try:
                with self.lock:
                    delay = self.protocol['metadata_min_interval_seconds'] - (time.monotonic() - self.last_start)
                    if delay > 0: time.sleep(delay)
                    self.last_start = time.monotonic()
                with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}),
                                            timeout=self.protocol['metadata_timeout_seconds']) as response:
                    data = response.read(8 << 20)
                value = json.loads(data)
                write(path, value)
                write(path.with_suffix('.source.json'), {'url': url, 'retrieved_unix': time.time(),
                                                       'content': binding(path), 'attempts': attempt + 1})
                return value
            except Exception:
                if attempt + 1 == self.protocol['metadata_attempts_per_URL']: raise
                time.sleep(2**attempt)


def choose_resources(files, kind):
    channels = {'environment': 'hdri'} if kind == 'hdri' else {
        'base_color': 'Diffuse', 'roughness': 'Rough', 'normal': 'nor_gl'}
    result = {}
    for name, key in channels.items():
        formats = files.get(key, {}).get('1k', {})
        selected = next((formats[f] for f in (['hdr'] if kind == 'hdri' else ['jpg', 'png']) if f in formats), None)
        if selected is None: raise ValueError('missing_required_1k_channel:' + key)
        if not selected.get('md5') or not 0 < selected.get('size', 0) < 64 << 20:
            raise ValueError('publisher_hash_and_bounded_file_size_required')
        url = urlsplit(selected['url'])
        if url.scheme != 'https' or url.hostname not in {'dl.polyhaven.org', 'dl.polyhaven.com'}:
            raise ValueError('unexpected_asset_host')
        result[name] = selected
    return result


def validate_image(path, kind):
    if kind == 'hdri':
        import cv2
        import numpy as np
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[1] != image.shape[0] * 2 or not np.isfinite(image).all():
            raise ValueError('nonfinite_or_non_equirectangular_HDRI')
        return {'shape': list(image.shape), 'finite': True, 'maximum': float(image.max()), 'minimum': float(image.min())}
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        if min(image.size) < 256: raise ValueError('material_resolution_too_small')
        return {'shape': [image.height, image.width], 'full_decode': True}


def one(root, row, guard, fetcher, deadline):
    folder = root / 'assets' / row['source_asset_id']; folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'result.json'
    if result_path.exists(): return read(result_path)
    start = time.monotonic()
    try:
        if time.time() >= deadline: raise RuntimeError('campaign_deadline_reached')
        files = fetcher.fetch('https://api.polyhaven.com/files/' + row['source_asset_id'], folder / 'files.json')
        resources = choose_resources(files, row['kind']); refs = {}; receipts = {}; decoded = {}
        for name, spec in resources.items():
            if time.time() >= deadline: raise RuntimeError('campaign_deadline_reached')
            dest = folder / (name + Path(urlsplit(spec['url']).path).suffix)
            receipts[name] = transfer.download(spec, dest, guard)
            decoded[name] = validate_image(dest, row['kind']); refs[name] = binding(dest)
        dims = row['source_metadata'].get('dimensions')
        size = ([float(v) / 1000 for v in dims[:2]] if isinstance(dims, list) and len(dims) >= 2 and
                all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in dims[:2]) else None)
        entry = {k: v for k, v in row.items() if k != 'source_metadata'}
        info = row['source_metadata']
        entry.update(name=info.get('name'), source='https://polyhaven.com/a/' + row['source_asset_id'],
            source_categories=info.get('categories', []), source_tags=info.get('tags', []), authors=info.get('authors'),
            resources=refs, license='CC0-1.0', attribution='Powered by Poly Haven', texture_size_m=size,
            texture_size_source='publisher_dimensions_mm' if size else None,
            usage='visual_only', physical_qualification='not_applicable_to_texture_or_panorama',
            dimension_documentation=DIMENSION_SPEC, file_manifest=binding(folder / 'files.json'),
            default_metallic=1.0 if row['source_material_family'] == 'metal' else 0.0)
        receipt_path = folder / 'receipt.json'
        write(receipt_path, {'schema': VERSION, 'source_metadata': info, 'files': receipts,
            'decode_checks': decoded, 'license': 'CC0-1.0', 'elapsed_seconds': time.monotonic() - start})
        entry['receipt'] = binding(receipt_path)
        result = {'status': 'completed', 'entry': entry, 'elapsed_seconds': time.monotonic() - start}
    except Exception as error:
        result = {'status': 'failed', 'id': row['id'], 'kind': row['kind'],
                  'error': type(error).__name__ + ':' + str(error), 'elapsed_seconds': time.monotonic() - start,
                  'terminal_failure_not_retried_or_replaced': True}
    write(result_path, result)
    return result


def summarize(rows, protocol):
    entries = []; failures = []; duplicates = []; seen = {}
    for result in sorted(rows, key=lambda r: r.get('id', r.get('entry', {}).get('id', ''))):
        if result['status'] != 'completed': failures.append(result); continue
        e = result['entry']; fingerprint = tuple(sorted((k, v['sha256']) for k, v in e['resources'].items()))
        if fingerprint in seen:
            duplicates.append({'id': e['id'], 'same_bytes_as': seen[fingerprint], 'excluded_from_active_catalog': True}); continue
        seen[fingerprint] = e['id']; entries.append(e)
    return {'schema': VERSION, 'entries': entries, 'protocol': protocol,
            'license': 'CC0-1.0', 'attribution': 'Powered by Poly Haven',
            'counts': {'completed': len(rows) - len(failures), 'failed': len(failures),
                       'unique_active_assets': len(entries), 'duplicate_content': len(duplicates),
                       'by_kind': dict(Counter(e['kind'] for e in entries)),
                       'by_role': dict(Counter(role for e in entries for role in e['roles'])),
                       'by_scene_category': dict(Counter(c for e in entries for c in e['categories'])),
                       'by_partition': dict(Counter(e['asset_partition'] for e in entries))},
            'failures': failures, 'duplicates': duplicates,
            'actual_render_validation': 'separate_frozen_screen_and_video_experiment_required',
            'new_paid_model_calls': 0, 'no_physical_trajectory_claim': True}


def run(root):
    root = Path(root).resolve(); p = read(root / 'protocol.json')
    for key in ('implementation', 'transfer_implementation'): resolve(p[key])
    for ref in p['metadata'].values(): resolve(ref)
    with (root / '.download.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'catalog.json').exists():
            return {'status': 'already_completed', 'catalog': binding(root / 'catalog.json')}
        clock = root / 'clock.json'
        if not clock.exists(): write(clock, {'start_unix': time.time(), 'deadline_unix': time.time() + p['wall_limit_seconds']})
        deadline = read(clock)['deadline_unix']
        guard = transfer.Guard(root, p['byte_limit'], reserve_bytes=p['disk_reserve_bytes'])
        fetcher = MetadataFetcher(p)
        write(root / 'sessions' / (str(time.time_ns()) + '.json'), {'pid': os.getpid(), 'started_unix': time.time(),
              'protocol': binding(root / 'protocol.json'), 'resumes_same_selection': True})
        rows = []
        with ThreadPoolExecutor(max_workers=p['workers']) as pool:
            jobs = [pool.submit(one, root, row, guard, fetcher, deadline) for row in p['selection']]
            for future in as_completed(jobs):
                rows.append(future.result())
                value = summarize(rows, binding(root / 'protocol.json'))
                value['transfer_bytes_charged'] = guard.used
                value['updated_unix'] = time.time()
                atomic(root / 'catalog_progress.json', value)
                if len(rows) % 20 == 0: print(json.dumps({'finished': len(rows), **value['counts']}), flush=True)
        result = summarize(rows, binding(root / 'protocol.json'))
        result['transfer_bytes_charged'] = guard.used
        result['elapsed_seconds'] = time.time() - read(clock)['start_unix']
        write(root / 'catalog.json', result)
        write(root / 'completion.json', {'catalog': binding(root / 'catalog.json'), 'counts': result['counts'],
              'finished_unix': time.time(), 'new_paid_model_calls': 0})
        return result['counts']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--root', type=Path, required=True)
    a.add_argument('--hdri-metadata', type=Path, required=True); a.add_argument('--texture-metadata', type=Path, required=True)
    a.add_argument('--hdris', type=int, default=120); a.add_argument('--materials', type=int, default=240)
    a.add_argument('--seed', type=int, default=20261029); a.add_argument('--budget-gib', type=float, default=4.)
    a = sub.add_parser('run'); a.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.root, args.hdri_metadata, args.texture_metadata, hdris=args.hdris, materials=args.materials,
                  seed=args.seed, budget_gib=args.budget_gib) if args.command == 'prepare' else run(args.root))
