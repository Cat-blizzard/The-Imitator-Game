"""Finite CPU conversion of local full-size furniture, never grasp assets.

ABO glTF Y-up / +Z-front is transformed to Z-up and bottom-centred. Source
scale is exactly one. Product dimensions are checked and disagreements are
retained, not repaired with anisotropic scaling. Embedded PBR images are capped
for rendering, preserving originals. Every export must reload before admission.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, sha, write

VERSION = 'local_fullsize_visual_furniture_bulk_v1'
RAW = Path('/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1')
DEFAULT_OUT = ROOT / 'assets/visual_furniture_bulk_20261030_v1'
SOURCE_PARTITIONS = ROOT / 'assets/expanded_library/raw_roles_v4.sqlite'
CATEGORIES = {
    'CABINET': 'cabinet', 'DRESSER': 'dresser', 'SHELF': 'shelf',
    'CHAIR': 'chair', 'SOFA': 'sofa', 'TABLE': 'table', 'DESK': 'desk',
    'STOOL_SEATING': 'stool', 'OTTOMAN': 'ottoman', 'BENCH': 'bench',
    'CLOTHES_RACK': 'clothes_rack', 'STORAGE_RACK': 'storage_rack',
    'LAMP': 'lamp', 'STORAGE_BOX': 'storage_box',
}
LENGTH = {'inches': .0254, 'inch': .0254, 'centimeters': .01,
          'millimeters': .001, 'meters': 1., 'feet': .3048}
MASS = {'pounds': .45359237, 'ounces': .028349523125, 'kilograms': 1., 'grams': .001}
Y_TO_Z = np.array([[1., 0., 0., 0.], [0., 0., -1., 0.],
                   [0., 1., 0., 0.], [0., 0., 0., 1.]])
TEXTURE_FIELDS = ('baseColorTexture', 'normalTexture', 'metallicRoughnessTexture',
                  'occlusionTexture', 'emissiveTexture', 'image')


def unit_value(item, mapping):
    value = item.get('normalized_value', item)
    number = value.get('value')
    if value.get('unit') not in mapping or type(number) not in (int, float):
        return None
    result = number * mapping[value['unit']]
    return float(result) if np.isfinite(result) and result > 0 else None


def listings(metadata, wanted):
    matches = defaultdict(list); refs = []
    for path in sorted(Path(metadata).glob('listings_*.json.gz')):
        ref = binding(path); refs.append(ref)
        with gzip.open(path, 'rt') as stream:
            for lineno, line in enumerate(stream, 1):
                record = json.loads(line); key = record.get('3dmodel_id')
                if key not in wanted:
                    continue
                raw_dimensions = record.get('item_dimensions', {})
                matches[key].append({'metadata_file': ref, 'line_number': lineno,
                    'item_id': record.get('item_id'), 'model_id': key,
                    'dimensions_raw': raw_dimensions,
                    'dimensions_m': {axis: unit_value(value, LENGTH) for axis, value in raw_dimensions.items()},
                    'weights': [{'raw': value, 'kg': unit_value(value, MASS)} for value in record.get('item_weight', [])],
                    'item_name': record.get('item_name'), 'product_type': record.get('product_type'),
                    'material': record.get('material')})
    return dict(matches), refs


def dimension_evidence(native_size_y_up, records, tolerance=.10):
    """Compare without altering scale; multiple conflicting listings stay visible."""
    size = np.sort(np.asarray(native_size_y_up, float)); rows = []
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('finite_positive_native_dimensions_required')
    for record in records:
        values = [record['dimensions_m'].get(axis) for axis in ('height', 'length', 'width')]
        if not all(v is not None and np.isfinite(v) and v > 0 for v in values):
            rows.append({'item_id': record['item_id'], 'complete': False, 'passed': None})
            continue
        expected = np.sort(values); errors = np.abs(size - expected) / expected
        rows.append({'item_id': record['item_id'], 'complete': True, 'passed': bool(np.all(errors <= tolerance)),
            'mesh_sorted_native_dimensions': size.tolist(), 'metadata_sorted_dimensions_m': expected.tolist(),
            'relative_errors': errors.tolist(), 'tolerance': tolerance})
    complete = [r for r in rows if r['complete']]
    status = ('source_metric_record_consistent' if complete and all(r['passed'] for r in complete) else
              'source_dimensions_conflict' if complete else 'source_gltf_units_only_not_independently_confirmed')
    return {'status': status, 'comparisons': rows, 'uniform_scale_applied': 1.,
        'missing_dimensions_claimed_measured': False,
        'comparison_basis': 'sorted full native AABB versus listing item dimensions, no best-record selection'}


def inventory(raw=RAW):
    candidates = []; counts = Counter(); all_counts = Counter(); filtered = Counter()
    for line in (Path(raw) / 'abo/catalog.jsonl').open():
        row = json.loads(line); all_counts['ABO_records'] += 1
        p = Path(row['geometry_path'])
        if p.is_file(): all_counts['ABO_local_files'] += 1
        match = sorted(set(row.get('categories', [])) & set(CATEGORIES))
        if not match: continue
        category = CATEGORIES[match[0]]; counts[category] += 1
        meta = row['official_geometry_metadata']; dimensions = [float(meta['extent_' + a]) for a in 'xyz']
        reason = None
        if not p.is_file(): reason = 'local_file_missing'
        elif row.get('license') != 'CC-BY-4.0': reason = 'unrecognized_source_license'
        elif row['bytes'] > 96 * 2**20: reason = 'input_exceeds_96MiB'
        elif int(meta['faces']) > 250000: reason = 'input_exceeds_250k_triangles'
        elif not np.isfinite(dimensions).all() or min(dimensions) < .04 or max(dimensions) > 4.:
            reason = 'outside_declared_background_native_bounds'
        if reason:
            filtered[reason] += 1; continue
        candidates.append({**row, 'background_category': category, 'raw_dimensions_y_up': dimensions})
    for line in (Path(raw) / 'objaverse/catalog.jsonl').open():
        row = json.loads(line); all_counts['Objaverse_records'] += 1
        if Path(row['geometry_path']).is_file(): all_counts['Objaverse_local_files'] += 1
    scanned = read(Path(raw) / 'scanned_objects/source_catalog.json')
    all_counts['GSO_original_catalog_objects'] = scanned['original_GSO_objects']
    return candidates, {'source_counts': dict(all_counts), 'ABO_furniture_category_counts': dict(counts),
        'ABO_local_bounded_candidate_counts': dict(Counter(r['background_category'] for r in candidates)),
        'ABO_light_processing_candidates': len(candidates), 'filtered_counts': dict(filtered),
        'Objaverse_used_in_first_batch': False,
        'reason': 'ABO has enough local full-size furniture with documented axes, licenses and product dimensions'}


def balanced(candidates, count, seed):
    groups = defaultdict(list)
    for row in candidates:
        groups[row['background_category']].append(row)
    for values in groups.values():
        values.sort(key=lambda r: digest([seed, r['asset_id']]))
    selected = []; seen_source_hashes = set()
    while len(selected) < count and any(groups.values()):
        for category in sorted(groups):
            while groups[category]:
                row = groups[category].pop(0)
                if row['sha256'] not in seen_source_hashes:
                    seen_source_hashes.add(row['sha256']); selected.append(row); break
            if len(selected) == count: break
    return selected


def prepare(output=DEFAULT_OUT, *, raw=RAW, count=400, seed=2026103009):
    output = Path(output).resolve()
    if output.exists(): raise ValueError('preserve_existing_background_bulk_directory')
    if type(count) is not int or not 1 <= count <= 400: raise ValueError('one_to_400_source_limit')
    candidates, report = inventory(raw); selected = balanced(candidates, count, seed)
    records, metadata_refs = listings(Path(raw) / 'abo/metadata', {r['source_id'] for r in selected})
    partition_ref = binding(SOURCE_PARTITIONS)
    with sqlite3.connect(f'file:{SOURCE_PARTITIONS}?mode=ro', uri=True) as db:
        source_partitions = {r['asset_id']: db.execute('SELECT partition FROM assets WHERE id=?', (r['asset_id'],)).fetchone()
                             for r in selected}
    jobs = []
    for i, row in enumerate(selected):
        match = source_partitions[row['asset_id']]
        job = {'schema': VERSION, 'index': i, 'source': row,
            'source_partition': match[0] if match else 'unknown_not_admitted', 'source_partition_evidence': partition_ref,
            'source_metric_records': records.get(row['source_id'], []),
            'output': str(output / 'models' / row['source_id']), 'texture_max_pixels': 1024,
            'source_scale': 1., 'purpose': 'render_only_background', 'physics_or_grasp_assets_created': False}
        path = output / 'jobs' / row['source_id'] / 'job.json'; write(path, job)
        jobs.append(binding(path))
    protocol = {'schema': VERSION, 'implementation': binding(__file__), 'created_unix': time.time(),
        'raw_catalog': binding(Path(raw) / 'abo/catalog.jsonl'), 'metadata': metadata_refs,
        'source_axis_documentation': binding(Path(raw) / 'abo/provenance/3dmodels/README.md'),
        'source_license_documentation': binding(Path(raw) / 'abo/provenance/README.md'),
        'seed': seed, 'jobs': jobs, 'maximum_source_attempts': len(jobs), 'parallel_CPU_workers': 2,
        'single_job_timeout_seconds': 120, 'total_timeout_seconds': 3600,
        'total_timeout_definition': 'cumulative measured CPU worker wall seconds across pilot and remaining; stricter than serial elapsed wall budget',
        'maximum_output_bytes': 12 * 2**30, 'minimum_free_bytes': 20 * 2**30,
        'maximum_asset_output_bytes': 96 * 2**20,
        'texture_max_pixels': 1024, 'pilot_count': min(14, len(jobs)),
        'automatic_retries': 0, 'source_scale_policy': 'preserve_native_source_scale_exactly_one',
        'network_downloads': 0, 'GPU_jobs': 0, 'new_physical_executions': 0, 'API_calls': 0}
    write(output / 'protocol.json', protocol)
    write(output / 'inventory.json', {**report, 'selected': len(jobs),
        'selected_by_category': dict(Counter(r['background_category'] for r in selected)),
        'selected_with_listing_metadata': sum(bool(records.get(r['source_id'])) for r in selected)})
    (output / 'ATTRIBUTION.txt').write_text(
        'Amazon Berkeley Objects (ABO), CC-BY-4.0. Source data and images: Amazon.com.\n'
        'Dataset: Matthieu Guillaumin, Thomas Dideriksen, Kenan Deng, Himanshu Arora, '
        'Jasmine Collins, Jitendra Malik.\nhttps://amazon-berkeley-objects.s3.amazonaws.com/index.html\n'
        'https://creativecommons.org/licenses/by/4.0/\n'
        'Changes: Y-up to Z-up rigid transform, XY centering and grounding; '
        'image resolution limited to 1024 pixels, GLB repackaging. No geometric rescaling.\n'
        'Per-item source URLs, hashes, metadata and conflicts are retained in catalog.json.\n')
    return {'selected': len(jobs), 'inventory': report, 'protocol': binding(output / 'protocol.json')}


def texture_report(scene, resize=None):
    from PIL import Image
    rows = []; seen = set()
    for mesh in scene.geometry.values():
        material = getattr(mesh.visual, 'material', None)
        if material is None or id(material) in seen: continue
        seen.add(id(material))
        for field in TEXTURE_FIELDS:
            image = getattr(material, field, None)
            if image is None: continue
            if not isinstance(image, Image.Image): raise ValueError('unsupported_embedded_texture_type')
            image.load(); before = list(image.size)
            if resize and max(image.size) > resize:
                image = image.copy(); image.thumbnail((resize, resize), Image.Resampling.LANCZOS)
                setattr(material, field, image)
            rows.append({'material': getattr(material, 'name', None), 'channel': field,
                'input_size_px': before, 'output_size_px': list(image.size), 'mode': image.mode})
    return rows


def geometry_signature(scene):
    """Exact canonical-coordinate geometry key, excluding material color variants."""
    geometry = scene.to_geometry(); v = np.round(np.asarray(geometry.vertices), 6)
    v[v == 0.] = 0.  # Canonicalize negative zero after axis conversion.
    triangles = v[np.asarray(geometry.faces)]
    # Vertex order and face order do not distinguish color-only SKU derivatives.
    order = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1)
    triangles = np.take_along_axis(triangles, order[:, :, None], axis=1).reshape(-1, 9)
    triangles = triangles[np.lexsort(tuple(triangles[:, i] for i in reversed(range(9))))]
    return hashlib.sha256(np.ascontiguousarray(triangles, dtype=np.float64).tobytes()).hexdigest()


def convert(job_path):
    import trimesh
    job_path = Path(job_path); job = read(job_path); row = job['source']; out = Path(job['output'])
    if out.exists(): raise ValueError('no_asset_conversion_retry_or_overwrite')
    source = Path(row['geometry_path'])
    if sha(source) != row['sha256']: raise ValueError('original_source_hash_changed')
    started = time.monotonic(); out.mkdir(parents=True)
    scene = trimesh.load(source, force='scene', process=False)
    if not scene.geometry: raise ValueError('source_contains_no_meshes')
    if any(not isinstance(g, trimesh.Trimesh) or not len(g.faces) or not np.isfinite(g.vertices).all()
           for g in scene.geometry.values()): raise ValueError('finite_triangle_geometry_required')
    before_bounds = np.asarray(scene.bounds); before_size = np.ptp(before_bounds, axis=0)
    recorded_size = np.asarray(row['raw_dimensions_y_up'])
    if not np.allclose(before_size, recorded_size, rtol=.01, atol=1e-4):
        raise ValueError('native_mesh_extent_differs_from_publisher_geometry_record')
    before_textures = texture_report(scene, resize=job['texture_max_pixels'])
    scene.apply_transform(Y_TO_Z)
    bounds = np.asarray(scene.bounds); shift = np.r_[-bounds[:, :2].mean(axis=0), -bounds[0, 2]]
    scene.apply_translation(shift)
    transformed = np.asarray(scene.bounds); geometry_hash = geometry_signature(scene)
    if not np.allclose(np.sort(np.ptp(transformed, axis=0)), np.sort(before_size), rtol=1e-6, atol=1e-6):
        raise ValueError('rigid_coordinate_transform_changed_native_size')
    mesh_path = out / 'background_z_up.glb'; payload = scene.export(file_type='glb')
    if len(payload) > 96 * 2**20: raise ValueError('converted_asset_exceeds_96MiB_output_limit')
    mesh_path.write_bytes(payload)
    loaded = trimesh.load(mesh_path, force='scene', process=False)
    if not np.allclose(loaded.bounds, transformed, rtol=0, atol=1e-5): raise ValueError('export_reload_bounds_changed')
    texture_after = texture_report(loaded)
    if Counter(t['channel'] for t in before_textures) != Counter(t['channel'] for t in texture_after):
        raise ValueError('texture_channels_lost_during_export')
    if any(max(t['output_size_px']) > 1024 for t in texture_after): raise ValueError('texture_size_limit_not_respected')
    triangles = sum(len(g.faces) for g in loaded.geometry.values())
    if triangles != sum(len(g.faces) for g in scene.geometry.values()): raise ValueError('export_changed_triangle_count')
    size = np.ptp(loaded.bounds, axis=0)
    evidence = dimension_evidence(before_size, job['source_metric_records'])
    placement = 'floor'
    if row['background_category'] == 'lamp' and size[2] < .95:
        placement = 'furniture_top'
    floor_min = float(loaded.bounds[0, 2])
    if abs(floor_min) > 1e-6: raise ValueError('output_not_grounded')
    value = {'id': row['asset_id'], 'source_id': row['asset_id'], 'name': row['name'],
        'category': row['background_category'], 'source_categories': row['categories'],
        'source': row['source_url'], 'source_dataset': 'Amazon Berkeley Objects',
        'license': row['license'], 'license_url': row['license_url'], 'attribution': row['attribution'],
        'source_mesh': binding(source), 'source_record': row, 'source_metric_records': job['source_metric_records'],
        'mesh': binding(mesh_path), 'bounds_m': np.asarray(loaded.bounds).tolist(), 'size_m': size.tolist(),
        'source_bounds_y_up': before_bounds.tolist(), 'native_source_scale': 1.,
        'scale_confidence': evidence['status'], 'dimension_evidence': evidence,
        'axis_transform': Y_TO_Z.tolist(), 'grounding_translation_m': shift.tolist(),
        'source_axis_convention': '+Y up / +Z natural front, publisher documented',
        'output_axis_convention': '+Z up / -Y natural front, XY centered / min Z = 0',
        'texture_conversion': before_textures, 'textures_after_reload': texture_after,
        'textured': bool(before_textures), 'triangles': triangles,
        'geometry_signature': geometry_hash, 'grounding_error_m': abs(floor_min),
        'split_group': 'geometry:' + geometry_hash,
        'source_partition': job.get('source_partition', 'unknown_not_admitted'),
        'source_partition_evidence': job.get('source_partition_evidence'),
        'asset_partition': job.get('source_partition', 'unknown_not_admitted'),
        'source_and_same_geometry_derivatives_share_partition': True,
        'mass_or_contact_parameters_assigned': False,
        'usage': 'render_only_furniture_not_task_object_or_collision_obstacle',
        'allowed_roles': ['background'], 'pick_place_eligible': False,
        'placement_support': placement, 'mesh_reload_passed': True,
        'GPU_render_validated': False, 'visual_semantics': 'publisher_category_not_independent_image_review',
        'converted_bytes': mesh_path.stat().st_size, 'conversion_seconds': time.monotonic() - started,
        'implementation': binding(__file__)}
    write(out / 'asset.json', value)
    return value


def run_one(job_ref, timeout):
    job_path = resolve(job_ref); directory = job_path.parent
    if (directory / 'terminal.json').exists(): return read(directory / 'terminal.json')
    if (directory / 'reservation.json').exists(): raise ValueError('unresolved_CPU_conversion_no_resubmit')
    write(directory / 'reservation.json', {'job': job_ref, 'started_unix': time.time(), 'CPU_workers': 1,
        'new_physical_executions': 0, 'API_calls': 0, 'GPU_jobs': 0})
    started = time.monotonic()
    with (directory / 'worker.log').open('x') as log:
        try:
            p = subprocess.run([sys.executable, '-m', 'resimpair_tasks.visual_furniture_bulk_v1',
                'worker', '--job', str(job_path)], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
                stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            code = p.returncode; timed_out = False
        except subprocess.TimeoutExpired:
            code = 'timeout'; timed_out = True
    result = {'job': job_ref, 'returncode': code, 'timeout': timed_out,
        'elapsed_seconds': time.monotonic() - started, 'finished_unix': time.time(),
        'new_physical_executions': 0, 'API_calls': 0, 'GPU_jobs': 0}
    write(directory / 'terminal.json', result)
    return result


def summarize(output):
    output = Path(output); protocol = read(output / 'protocol.json'); entries = []; failures = []; pending = []
    geometry_owner = {}; duplicates = []; geometry_partitions = defaultdict(set)
    for ref in protocol['jobs']:
        path = resolve(ref); job = read(path); terminal = path.parent / 'terminal.json'
        if not terminal.exists(): pending.append(job['source']['asset_id']); continue
        result = read(terminal); asset_path = Path(job['output']) / 'asset.json'
        if result['returncode'] != 0 or not asset_path.exists():
            failures.append({'id': job['source']['asset_id'], 'terminal': binding(terminal),
                'log': binding(path.parent / 'worker.log'), 'returncode': result['returncode']}); continue
        row = read(asset_path); resolve(row['mesh'])
        row['preparation_record'] = binding(asset_path)
        key = row['geometry_signature']
        geometry_partitions[key].add(row['source_partition'])
        if key in geometry_owner:
            duplicates.append({'id': row['id'], 'same_geometry_as': geometry_owner[key], 'asset': binding(asset_path)})
            continue
        geometry_owner[key] = row['id']; entries.append(row)
    for row in entries:
        partitions = sorted(geometry_partitions[row['geometry_signature']])
        row['all_same_geometry_source_partitions'] = partitions
        row['asset_partition'] = partitions[0] if len(partitions) == 1 else 'partition_conflict_not_admitted'
    return {'schema': VERSION, 'protocol': binding(output / 'protocol.json'), 'entries': entries,
        'source_attempts': len(protocol['jobs']) - len(pending), 'unique_geometry_assets': len(entries),
        'by_category': dict(Counter(r['category'] for r in entries)),
        'by_scale_confidence': dict(Counter(r['scale_confidence'] for r in entries)),
        'by_placement_support': dict(Counter(r['placement_support'] for r in entries)),
        'textured_assets': sum(r['textured'] for r in entries),
        'failed': failures, 'pending': pending, 'same_geometry_derivatives': duplicates,
        'source_geometry_duplicate_definition': 'canonical-coordinate 1 micron rounded triangle set, excluding materials',
        'converted_bytes': sum(read(Path(read(resolve(r))['output']) / 'asset.json')['converted_bytes']
            for r in protocol['jobs'] if (Path(read(resolve(r))['output']) / 'asset.json').exists()),
        'all_models_background_only': True, 'pick_place_assets_added': 0,
        'actual_GPU_rendered_assets': 0, 'visual_quality_scope': 'CPU export/reload, dimensions and texture channels only',
        'new_physical_executions': 0, 'API_calls': 0, 'GPU_jobs': 0}


def run(output=DEFAULT_OUT, *, phase):
    output = Path(output).resolve(); protocol = read(output / 'protocol.json'); resolve(protocol['implementation'])
    if phase not in ('pilot', 'remaining'): raise ValueError('pilot_or_remaining_phase_required')
    with (output / '.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if phase == 'remaining':
            gate = read(output / 'pilot_gate.json')
            if gate['decision'] != 'continue_finite_batch' or gate['protocol'] != binding(output / 'protocol.json'):
                raise ValueError('real_cross_category_pilot_required')
        selected = protocol['jobs'][:protocol['pilot_count']] if phase == 'pilot' else protocol['jobs'][protocol['pilot_count']:]
        pending = [r for r in selected if not (resolve(r).parent / 'terminal.json').exists()]
        if any((resolve(r).parent / 'reservation.json').exists() for r in pending):
            raise ValueError('unresolved_attempt_requires_recovery')
        prior_seconds = sum(read(p)['elapsed_seconds'] for p in (output / 'jobs').glob('*/terminal.json'))
        # CPU worker seconds are stricter than wall time at parallelism two.
        remaining_seconds = protocol['total_timeout_seconds'] - prior_seconds
        start = time.monotonic(); results = []
        for i in range(0, len(pending), protocol['parallel_CPU_workers']):
            remaining_seconds = protocol['total_timeout_seconds'] - prior_seconds - sum(r['elapsed_seconds'] for r in results)
            if remaining_seconds <= 2: break
            if shutil.disk_usage(output).free < protocol['minimum_free_bytes']: break
            size = sum(p.stat().st_size for p in (output / 'models').rglob('*') if p.is_file()) if (output / 'models').exists() else 0
            # Reserve worst-case output for both simultaneous jobs before launch.
            if size + protocol['parallel_CPU_workers'] * (protocol['maximum_asset_output_bytes'] + 2**20) > protocol['maximum_output_bytes']: break
            with ThreadPoolExecutor(max_workers=protocol['parallel_CPU_workers']) as pool:
                futures = [pool.submit(run_one, ref, min(protocol['single_job_timeout_seconds'], max(1, (remaining_seconds - 2) / protocol['parallel_CPU_workers'])))
                           for ref in pending[i:i+protocol['parallel_CPU_workers']]]
                for future in as_completed(futures):
                    result = future.result(); results.append(result)
                    print({'completed': len(results), 'returncode': result['returncode'],
                           'job': result['job']['path'], 'seconds': round(result['elapsed_seconds'], 2)}, flush=True)
        result = summarize(output)
        if phase == 'pilot':
            pilot_refs = protocol['jobs'][:protocol['pilot_count']]
            pilot_assets = [read(Path(read(resolve(r))['output']) / 'asset.json') for r in pilot_refs
                            if (Path(read(resolve(r))['output']) / 'asset.json').exists()]
            ready = len(pilot_assets) >= 10 and len({r['category'] for r in pilot_assets}) >= 10
            write(output / 'pilot_gate.json', {'schema': VERSION, 'protocol': binding(output / 'protocol.json'),
                'decision': 'continue_finite_batch' if ready else 'stop_for_pilot_diagnosis',
                'actual_assets': len(pilot_assets), 'categories': sorted({r['category'] for r in pilot_assets}),
                'failures': result['failed'], 'same_validation_for_remaining': True,
                'all_successful_exports_reloaded_and_native_scale_preserved': True})
            write(output / 'pilot_summary.json', result)
        else:
            write(output / 'catalog.json', result)
        return {k: result[k] for k in ('source_attempts', 'unique_geometry_assets', 'by_category', 'by_scale_confidence', 'textured_assets', 'converted_bytes')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'pilot', 'remaining', 'worker'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare': print(prepare(args.output))
    elif args.command == 'worker':
        value = convert(args.job); print({'id': value['id'], 'size_m': value['size_m'], 'seconds': value['conversion_seconds']})
    else: print(run(args.output, phase='pilot' if args.command == 'pilot' else 'remaining'))
