"""Incremental full-size ABO background conversion with a finite, resumable ledger.

The v1 converter is retained byte-for-byte. Old converted source records are
reused; every previously unprocessed local source is attempted once. Failed
assets are isolated. A catalog is a CPU qualification, not visual approval.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import shutil
import sqlite3
import time
from pathlib import Path

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import visual_furniture_bulk_v1 as old

VERSION = 'incremental_fullsize_visual_furniture_bulk_v2'
DEFAULT_OUT = ROOT / 'assets/visual_furniture_bulk_20261030_v2'


def prepare(output=DEFAULT_OUT, *, prior=old.DEFAULT_OUT, seed=2026103020):
    output = Path(output).resolve(); prior = Path(prior).resolve()
    if output.exists(): raise ValueError('new_output_directory_required')
    prior_protocol = read(prior / 'protocol.json')
    old_ids = set(); reused = []
    for ref in prior_protocol['jobs']:
        job_path = resolve(ref); job = read(job_path); old_ids.add(job['source']['asset_id'])
        terminal = job_path.parent / 'terminal.json'
        if not terminal.exists(): raise ValueError('prior_conversion_unresolved')
        if read(terminal)['returncode'] == 0:
            reused.append(binding(Path(job['output']) / 'asset.json'))
    candidates, report = old.inventory()
    groups = defaultdict(list)
    for row in candidates:
        if row['asset_id'] not in old_ids: groups[row['background_category']].append(row)
    for rows in groups.values(): rows.sort(key=lambda row: digest([seed, row['asset_id']]))
    selected = []
    while any(groups.values()):
        for category in sorted(groups):
            if groups[category]: selected.append(groups[category].pop(0))
    if len(selected) > 4500: raise ValueError('incremental_source_budget_exceeded')
    records, metadata_refs = old.listings(old.RAW / 'abo/metadata', {r['source_id'] for r in selected})
    partition_ref = binding(old.SOURCE_PARTITIONS)
    jobs = []
    with sqlite3.connect(f'file:{old.SOURCE_PARTITIONS}?mode=ro', uri=True) as db:
        for i, row in enumerate(selected):
            partition = db.execute('SELECT partition FROM assets WHERE id=?', (row['asset_id'],)).fetchone()
            job = {'schema': VERSION, 'index': i, 'source': row,
                'source_partition': partition[0] if partition else 'unknown_not_admitted',
                'source_partition_evidence': partition_ref,
                'source_metric_records': records.get(row['source_id'], []),
                'output': str(output / 'models' / row['source_id']), 'texture_max_pixels': 1024,
                'source_scale': 1., 'purpose': 'render_only_background', 'physics_or_grasp_assets_created': False}
            path = output / 'jobs' / row['source_id'] / 'job.json'; write(path, job); jobs.append(binding(path))
    protocol = {'schema': VERSION, 'implementation': binding(__file__), 'converter': binding(old.__file__),
        'created_unix': time.time(), 'seed': seed, 'prior_protocol': binding(prior / 'protocol.json'),
        'prior_sources_reused': reused, 'prior_sources_never_reexecuted': sorted(old_ids),
        'raw_catalog': binding(old.RAW / 'abo/catalog.jsonl'), 'metadata': metadata_refs,
        'source_axis_documentation': binding(old.RAW / 'abo/provenance/3dmodels/README.md'),
        'source_license_documentation': binding(old.RAW / 'abo/provenance/README.md'),
        'jobs': jobs, 'maximum_source_attempts': 4500, 'registered_source_jobs': len(jobs),
        'parallel_CPU_workers': 4, 'single_job_timeout_seconds': 120,
        'total_wall_timeout_seconds': 10800,
        'wall_clock_origin': 'first reservation for this allocation, including pauses between phases',
        'maximum_output_bytes': 30 * 2**30, 'minimum_free_bytes': 40 * 2**30,
        'maximum_asset_output_bytes': 96 * 2**20, 'pilot_count': min(14, len(jobs)),
        'automatic_retries': 0, 'source_scale_policy': 'native scale one, never fit furniture into a grasp slot',
        'network_downloads': 0, 'GPU_jobs': 0, 'new_physical_executions': 0, 'API_calls': 0}
    write(output / 'protocol.json', protocol)
    write(output / 'inventory.json', {**report, 'old_source_records_reused': len(reused),
        'new_source_jobs': len(jobs), 'new_jobs_by_category': dict(Counter(r['background_category'] for r in selected))})
    shutil.copyfile(prior / 'ATTRIBUTION.txt', output / 'ATTRIBUTION.txt')
    return {'reused': len(reused), 'new_jobs': len(jobs), 'protocol': binding(output / 'protocol.json')}


def summarize(output):
    output = Path(output); protocol = read(output / 'protocol.json')
    refs = list(protocol['prior_sources_reused']); failures = []; pending = []
    elapsed = 0.; attempted = 0
    for ref in protocol['jobs']:
        path = resolve(ref); job = read(path); terminal = path.parent / 'terminal.json'
        if not terminal.exists(): pending.append(job['source']['asset_id']); continue
        result = read(terminal); elapsed += result['elapsed_seconds']; attempted += 1
        asset_path = Path(job['output']) / 'asset.json'
        if result['returncode'] == 0 and asset_path.exists(): refs.append(binding(asset_path))
        else: failures.append({'id': job['source']['asset_id'], 'terminal': binding(terminal),
            'log': binding(path.parent / 'worker.log'), 'returncode': result['returncode']})
    geometry = defaultdict(list)
    for ref in refs:
        row = read(resolve(ref)); row['preparation_record'] = ref
        geometry[row['geometry_signature']].append(row)
    entries = []; duplicates = []
    for key, rows in sorted(geometry.items()):
        # Prefer a representative with supported dimensions; aliases retain every source record.
        rows.sort(key=lambda r: (r['scale_confidence'] != 'source_metric_record_consistent', r['id']))
        owner = rows[0]; partitions = sorted({r['source_partition'] for r in rows})
        owner['all_same_geometry_source_partitions'] = partitions
        owner['asset_partition'] = partitions[0] if len(partitions) == 1 else 'partition_conflict_not_admitted'
        owner['geometry_aliases'] = [{'id': r['id'], 'preparation_record': r['preparation_record'],
            'partition': r['source_partition'], 'scale_confidence': r['scale_confidence']} for r in rows]
        entries.append(owner)
        duplicates.extend({'id': r['id'], 'same_geometry_as': owner['id'], 'asset': r['preparation_record']} for r in rows[1:])
    return {'schema': VERSION, 'protocol': binding(output / 'protocol.json'), 'entries': entries,
        'prior_sources_reused': len(protocol['prior_sources_reused']), 'new_attempts': attempted,
        'successful_source_records': len(refs), 'unique_geometry_assets': len(entries),
        'new_CPU_worker_wall_seconds_sum': elapsed,
        'by_category': dict(Counter(r['category'] for r in entries)),
        'by_scale_confidence': dict(Counter(r['scale_confidence'] for r in entries)),
        'by_partition': dict(Counter(r['asset_partition'] for r in entries)),
        'textured_assets': sum(r['textured'] for r in entries),
        'failed': failures, 'pending': pending, 'same_geometry_derivatives': duplicates,
        'source_records': refs, 'geometry_deduplication': 'canonical native-scale triangles rounded to one micrometre; materials ignored',
        'new_converted_bytes': sum(read(resolve(r))['converted_bytes'] for r in refs[len(protocol['prior_sources_reused']):]),
        'all_models_background_only': True, 'pick_place_assets_added': 0,
        'actual_GPU_rendered_assets': 0, 'visual_quality_scope': 'CPU export/reload; consumer qualification recorded separately',
        'new_physical_executions': 0, 'API_calls': 0, 'GPU_jobs': 0}


def run(output=DEFAULT_OUT, *, phase):
    output = Path(output).resolve(); protocol = read(output / 'protocol.json')
    resolve(protocol['implementation']); resolve(protocol['converter'])
    with (output / '.supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if phase == 'remaining':
            gate = read(output / 'pilot_gate.json')
            if gate['decision'] != 'continue_finite_batch' or gate['protocol'] != binding(output / 'protocol.json'):
                raise ValueError('cross_category_pilot_required')
        selected = protocol['jobs'][:protocol['pilot_count']] if phase == 'pilot' else protocol['jobs'][protocol['pilot_count']:]
        pending = deque(r for r in selected if not (resolve(r).parent / 'terminal.json').exists())
        if any((resolve(r).parent / 'reservation.json').exists() for r in pending):
            raise ValueError('unresolved_attempt_requires_recovery_before_resubmission')
        origin_path = output / 'wall_clock_origin.json'
        if not origin_path.exists(): write(origin_path, {'started_unix': time.time(), 'protocol': binding(output / 'protocol.json')})
        deadline = read(origin_path)['started_unix'] + protocol['total_wall_timeout_seconds']
        current_bytes = sum(p.stat().st_size for p in (output / 'models').rglob('*') if p.is_file())
        count = 0; stop = None
        with ThreadPoolExecutor(max_workers=protocol['parallel_CPU_workers']) as pool:
            active = {}
            while pending or active:
                while pending and len(active) < protocol['parallel_CPU_workers'] and stop is None:
                    remaining = deadline - time.time()
                    if remaining < 2: stop = 'wall_clock_budget_exhausted'; break
                    if shutil.disk_usage(output).free < protocol['minimum_free_bytes']: stop = 'minimum_free_space'; break
                    reserve = (len(active) + 1) * (protocol['maximum_asset_output_bytes'] + 2**20)
                    if current_bytes + reserve > protocol['maximum_output_bytes']: stop = 'output_byte_budget'; break
                    ref = pending.popleft()
                    active[pool.submit(old.run_one, ref, min(120, remaining - 1))] = ref
                if not active: break
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    ref = active.pop(future); result = future.result(); count += 1
                    out = Path(read(resolve(ref))['output'])
                    current_bytes += sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) if out.exists() else 0
                    if count % 20 == 0 or phase == 'pilot' or result['returncode'] != 0:
                        print({'phase_completed': count, 'remaining': len(pending), 'active': len(active),
                            'returncode': result['returncode'], 'job': ref['path'],
                            'output_GiB': round(current_bytes / 2**30, 3), 'wall_seconds_left': round(deadline-time.time())}, flush=True)
        result = summarize(output)
        if phase == 'pilot':
            assets = [read(Path(read(resolve(r))['output']) / 'asset.json') for r in selected
                if (Path(read(resolve(r))['output']) / 'asset.json').exists()]
            ready = len(assets) >= 10 and len({r['category'] for r in assets}) >= 10
            write(output / 'pilot_gate.json', {'schema': VERSION, 'protocol': binding(output / 'protocol.json'),
                'decision': 'continue_finite_batch' if ready else 'stop_for_pilot_diagnosis',
                'new_successful_assets': len(assets), 'categories': sorted({r['category'] for r in assets}),
                'failures': result['failed'], 'same_converter_for_remaining': True})
            write(output / 'pilot_catalog.json', result)
        else: write(output / 'catalog.json', result)
        write(output / (phase + '_terminal.json'), {'completed_unix': time.time(), 'stop_reason': stop,
            'new_attempts': result['new_attempts'], 'pending': len(result['pending']), 'protocol': binding(output / 'protocol.json')})
        return {k: result[k] for k in ('prior_sources_reused', 'new_attempts', 'unique_geometry_assets', 'by_scale_confidence', 'new_converted_bytes')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'pilot', 'remaining', 'summarize'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--prior', type=Path, default=old.DEFAULT_OUT)
    args = parser.parse_args()
    if args.command == 'prepare': print(prepare(args.output, prior=args.prior))
    elif args.command == 'summarize':
        result = summarize(args.output); write(args.output / 'catalog.json', result)
        print({k: result[k] for k in ('new_attempts', 'unique_geometry_assets', 'by_scale_confidence')})
    else: print(run(args.output, phase=args.command))
