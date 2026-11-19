"""Fast manipulable-asset intake: relaxed geometry, source-metadata labels, cheap grasp checks, one measured test.

The earlier intake stacked strict gates (3 mm surface fidelity, exact closing-width agreement, VLM labels) and kept
212 of 22,344 sources. This one keeps only checks that predict a failed grasp, and lets a real pick-carry-place run
decide; 70-80 % grasp success is the target, not certainty.

Candidates are rollout sources whose collision geometry was built (native-ready, or failed only the surface gate).
Per asset, one CPU job (`prepare`):
  geometry   catalog.admit: surface p95 <= 10 mm, support and settle pass
  size       longest side within LONGEST_RANGE_M; source profile not in REJECT_PROFILES
  label      the source's own name, cleaned (no VLM); color and luminance from the textured surface
  grasps     cuDex proposals unless the open hand collides, the opening is out of range, a finger finds no contact
             on closing, or the approach tilts past MAX_TILT_DEG; plus top-down antipodal proposals; ordered by lever
`bench` runs the BENCH_K lowest-lever distinct grasps of each candidate through the real skill (grasp_bench scenes,
CPU executor) in chunks, pruning each finished run to its result; `admit` keeps assets with >= MIN_SUCCESSES
successes and writes catalog rows whose caches lead with the grasps that succeeded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write

VERSION = 'pp2_asset_intake_v2'
ROLLOUT = ROOT / 'runs/raw_asset_rollout_20261031_v1'
INVENTORY = ROOT / 'runs/metric_assets_20261030_v7/inventory_r1/source_state_index.json'
LONGEST_RANGE_M = (.03, .25)
REJECT_PROFILES = {'furniture', 'lamp_background', 'large_appliance', 'deformable_background', 'art_background',
                   'large_scene', 'luggage'}
OPENING_RANGE_M = (.012, .077)
MAX_TILT_DEG = 60.
CACHE_SIZE = 16
BENCH_K, MIN_SUCCESSES = 4, 3
KEEP_AFTER_BENCH = {'worker_result.json', 'terminal.json', 'reservation.json', 'effective_task.json', 'process.json',
                    'measurements.npz'}
NAME_NOISE = {'v', 'obj', 'fbx', 'glb', 'gltf', 'low', 'poly', 'lowpoly', 'model', 'free', 'game', 'ready', 'pbr',
              'scan', 'scanned', 'photogrammetry', 'asset', 'prop', 'hd', 'k', 'textured', 'realistic', 'simple', 'the',
              'copy', 'of', 'untitled', 'new', 'my', 'final', 'test', 'project', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat',
              'sun', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'}


def clean_name(name, profile):
    """Source name as a short noun phrase: lower case, no ids, versions or modelling jargon."""
    text = re.split(r'\s(?:from|by)\s', str(name or ''), maxsplit=1, flags=re.I)[0]  # "X from Poly by Google"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'[^A-Za-z]+', ' ', text).lower()
    words = [w for w in text.split() if w not in NAME_NOISE and len(w) > 1]
    return ' '.join(words[-3:]) or (profile or 'object').replace('_', ' ')


def lever(tcp, com):
    tcp = np.asarray(tcp, float); offset = np.asarray(com, float) - tcp[:3, 3]; axis = tcp[:3, 1]
    return float(np.linalg.norm(offset - axis * (offset @ axis)))


def cudex_candidates(job):
    """Relaxed cuDex proposals: keep unless a check predicts a miss (not mere disagreement with cuDex's width)."""
    gpu = ROLLOUT / 'gpu' / job
    if not (gpu / 'native_qualification/decisions.json').exists():
        return []
    rows = read(gpu / 'native_qualification/decisions.json')['rows']; poses = np.load(gpu / 'raw.npz')['poses']
    out = []
    for row in rows:
        if row.get('rejected') in ('duplicate', 'hand_inside_object', 'gripper_table', 'open_hand_or_world_z_approach_collision'):
            continue
        if not (row.get('open_hand') or {}).get('passed'):
            continue
        fingers = (row.get('closing_sweep') or {}).get('fingers') or []
        contact = [f.get('first_contact_width_m') for f in fingers]
        if len(contact) != 2 or any(c is None for c in contact):
            continue
        width = float(np.mean(contact))
        if not OPENING_RANGE_M[0] <= width <= OPENING_RANGE_M[1]:
            continue
        pose = poses[row['rank']]
        if np.degrees(np.arccos(np.clip(-pose[2, 2], -1, 1))) > MAX_TILT_DEG:
            continue
        out.append({'id': f"c{row['rank']:03d}", 'tcp_in_asset': pose.tolist(), 'width_m': width, 'source': 'cudex',
                    'cudex_check': row.get('rejected') or 'passed', 'physical_status': 'unmeasured'})
    return out


def prepare_one(job):
    source_id, folder, meta, output = job
    from resimpair_grasp import grasps
    from . import catalog
    start = time.monotonic(); out = Path(output) / 'assets' / digest(source_id)[:24]
    try:
        if meta.get('semantic_profile') in REJECT_PROFILES:
            raise ValueError('profile_' + meta['semantic_profile'])
        descriptor_binding, admission = catalog.admit(source_id, Path(folder), out)
        descriptor = read(resolve(descriptor_binding))
        dims = descriptor['dimensions_m']
        if not LONGEST_RANGE_M[0] <= max(dims) <= LONGEST_RANGE_M[1]:
            raise ValueError('longest_side_outside_range')
        rgb = catalog.surface_samples(resolve(descriptor['visual']))
        color, evidence = catalog.color_name(rgb)
        luminance = float((rgb @ np.array([.2126, .7152, .0722])).mean())
        candidates = cudex_candidates(Path(folder).name)
        try:
            anti_binding, _, _ = catalog.native_grasps(descriptor_binding, out / 'antipodal')
            candidates += [{**c, 'source': 'antipodal'} for c in read(resolve(anti_binding))['candidates']]
        except Exception as error:  # cuDex proposals may still carry the asset
            admission['antipodal_error'] = str(error)[:120]
        com = descriptor.get('native_COM_m') or [0., 0., 0.]
        seen, kept = set(), []
        for c in sorted(candidates, key=lambda c: lever(c['tcp_in_asset'], com)):
            t = np.asarray(c['tcp_in_asset']); yaw = float(np.arctan2(t[1, 1], t[0, 1]))
            bucket = (*np.round(t[:3, 3] / .01).astype(int).tolist(), int(np.round(yaw / .52)))
            if bucket not in seen:
                seen.add(bucket); kept.append({**c, 'lever_m': lever(t, com)})
        kept = kept[:CACHE_SIZE]
        if len(kept) < BENCH_K:
            raise ValueError(f'only_{len(kept)}_distinct_grasps')
        signature = {'descriptor': descriptor_binding, 'gripper': grasps.GRIPPER, 'intake': VERSION}
        write(out / 'grasp_cache.json', {'schema': 'resimpair_mesh_grasp_cache_v1', 'signature': signature,
              'key': digest(signature), 'candidates': kept, 'physical_qualification': 'bench_pending'})
        category = clean_name(meta.get('name'), meta.get('semantic_profile'))
        return {'source_id': source_id, 'role': 'manipulable', 'category': category, 'color': color,
                'color_evidence': evidence, 'luminance': luminance, 'label_source': 'source_name',
                'source_name': meta.get('name'), 'profile': meta.get('semantic_profile'),
                'partition': meta.get('partition') or descriptor.get('partition'), 'descriptor': descriptor_binding,
                'cache': binding(out / 'grasp_cache.json'), 'grasps': len(kept), 'dimensions_m': dims,
                'mass_kg': descriptor.get('mass_kg'), 'admission': admission,
                'grasp_sources': dict(Counter(c['source'] for c in kept)), 'seconds': time.monotonic() - start}
    except Exception as error:
        return {'source_id': source_id, 'rejected': str(error)[:120], 'seconds': time.monotonic() - start}


def sources(sample=None, seed=0, exclude=()):
    """Rollout sources with built geometry, with their inventory metadata."""
    meta = {e['source_id']: e for e in read(INVENTORY)['entries']}
    rows = []
    for r in read(ROLLOUT / 'population.json')['sources']:
        folder = ROOT / Path(r['job']['path']).parent
        if r['source_id'] in exclude or not (folder / 'native_entry.json').exists():
            continue
        rows.append((r['source_id'], str(folder), meta.get(r['source_id'], {})))
    if sample:
        rng = np.random.default_rng(seed); rows = [rows[i] for i in sorted(rng.choice(len(rows), sample, replace=False))]
    return rows


def prepare(output, workers=64, sample=None, seed=0, exclude=()):
    output = Path(output).resolve()
    if (output / 'candidates.json').exists():
        raise ValueError('preserve_existing_intake')
    started = time.monotonic()
    jobs = [(sid, folder, m, str(output)) for sid, folder, m in sources(sample, seed, exclude)]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(prepare_one, jobs, chunksize=4))
    kept = [r for r in rows if 'rejected' not in r]
    write(output / 'candidates.json', {'schema': VERSION, 'sources': len(rows), 'candidates': kept,
          'rejections': dict(Counter(re.sub(r'\d+', 'N', r['rejected']) for r in rows if 'rejected' in r)),
          'rules': {'longest_range_m': LONGEST_RANGE_M, 'reject_profiles': sorted(REJECT_PROFILES), 'opening_range_m': OPENING_RANGE_M,
                    'max_tilt_deg': MAX_TILT_DEG, 'cache_size': CACHE_SIZE, 'surface_limit_m': .010},
          'seconds': time.monotonic() - started, 'implementation': binding(Path(__file__))})
    return {'sources': len(rows), 'candidates': len(kept), 'seconds': round(time.monotonic() - started)}


def bench_factory(output, workers=64):
    from . import grasp_bench
    output = Path(output).resolve(); bench = output / 'bench'
    if (bench / 'factory.json').exists():
        raise ValueError('preserve_existing_bench_factory')
    jobs = []
    for asset in read(output / 'candidates.json')['candidates']:
        for index, c in enumerate(read(resolve(asset['cache']))['candidates'][:BENCH_K]):
            jobs.append((asset, {'candidate': c, 'lever_m': c['lever_m']}, 'intake', index, bench))
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(grasp_bench.build_case, jobs, chunksize=8))
    write(bench / 'factory.json', {'schema': VERSION + '_bench', 'candidates': binding(output / 'candidates.json'),
          'scenes': rows, 'bench_k': BENCH_K, 'physical_executions': 0, 'API_calls': 0})
    return {'cases': len(rows)}


def prune(cohort):
    """Finished bench runs keep only their result, terminal record and measurements."""
    freed = 0
    for d in (Path(cohort) / 'executions').iterdir():
        if not (d / 'terminal.json').exists() or (d / '.pruned').exists():
            continue
        for p in d.iterdir():
            if p.name not in KEEP_AFTER_BENCH:
                freed += sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.is_dir() else p.stat().st_size
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        (d / '.pruned').write_text('kept ' + ' '.join(sorted(KEEP_AFTER_BENCH)))
    return freed


def bench(output, workers=160, chunk=4000):
    from . import execute
    output = Path(output).resolve(); cohort = output / 'bench/cohort'
    if not (cohort / 'cohort.json').exists():
        cases = len(read(output / 'bench/factory.json')['scenes'])
        execute.prepare(output / 'bench', cohort, 'production', cases, worker_module='resimpair_pp2.worker')
    rounds = []
    while True:
        result = execute.run(cohort, workers, limit=chunk)
        freed = prune(cohort); rounds.append({**result, 'freed_bytes': freed}); print(json.dumps(rounds[-1]), flush=True)
        if not result['new_attempts']:
            break
    return rounds


def bench_index(case):
    return int(case['id'].split('__')[0].rsplit('_', 1)[1])


def case_success(cohort, case_id):
    from . import grasp_bench
    d = Path(cohort) / 'executions' / case_id
    if (read(d / 'worker_result.json') if (d / 'worker_result.json').exists() else {}).get('failure'):
        return False
    if not (d / 'measurements.npz').exists():
        return False
    return bool(np.linalg.norm(np.load(d / 'measurements.npz')['object_states'][-1, 0, :2] - np.asarray(grasp_bench.GOAL_XY)) <= .03)


def bench_adaptive(output, workers=128):
    """Same admission rule with fewer runs: grasps 0-2 first; grasp 3 only where exactly two of them succeeded
    (3/3 is already admitted, <= 1/3 can no longer reach MIN_SUCCESSES of BENCH_K)."""
    from concurrent.futures import ThreadPoolExecutor
    from . import execute
    output = Path(output).resolve(); cohort = output / 'bench/cohort'
    limits = read(cohort / 'budget/limits.json'); cases = read(cohort / 'cohort.json')['cases']
    for d in (cohort / 'executions').iterdir() if (cohort / 'executions').exists() else []:
        if (d / 'reservation.json').exists() and not (d / 'terminal.json').exists():
            shutil.rmtree(d)  # a bench run killed mid-flight left no result; it is simply run again
    finished = lambda c: (cohort / 'executions' / c['id'] / 'terminal.json').exists()

    def launch(batch):
        started = time.monotonic()
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(lambda c: execute._launch(cohort, c, limits), batch))
        return {'runs': len(batch), 'wall_seconds': time.monotonic() - started, 'freed_bytes': prune(cohort)}
    rounds = [launch([c for c in cases if bench_index(c) < BENCH_K - 1 and not finished(c)])]
    print(json.dumps(rounds[-1]), flush=True)
    wins = Counter(c['scene_id'].rsplit('_', 1)[0] for c in cases if bench_index(c) < BENCH_K - 1 and case_success(cohort, c['id']))
    need = [c for c in cases if bench_index(c) == BENCH_K - 1 and wins[c['scene_id'].rsplit('_', 1)[0]] == MIN_SUCCESSES - 1
            and not finished(c)]
    rounds.append(launch(need)); print(json.dumps(rounds[-1]), flush=True)
    return rounds


def admit(output):
    """Assets whose bench grasps succeeded often enough; caches reordered to lead with the successful grasps."""
    from . import grasp_bench
    output = Path(output).resolve()
    scored = grasp_bench.score(output / 'bench/cohort', output / 'bench')
    by_asset = {}
    for row in scored:
        by_asset.setdefault(row['source_id'], []).append(row)
    ran = lambda r: (output / 'bench/cohort/executions' / (r['scene_id'] + '__task_0') / 'terminal.json').exists()
    admitted, outcomes = [], Counter(r['outcome'] for r in scored if ran(r))
    for asset in read(output / 'candidates.json')['candidates']:
        runs = [r for r in by_asset.get(asset['source_id'], []) if (output / 'bench/cohort/executions' / (r['scene_id'] + '__task_0') / 'terminal.json').exists()]
        wins = {r['candidate'] for r in runs if r['outcome'] == 'success'}
        asset = {**asset, 'bench': {'tested': len(runs), 'successes': len(wins)}}
        if len(wins) < MIN_SUCCESSES:
            continue
        cache = read(resolve(asset['cache']))
        tested = {r['candidate'] for r in runs}
        ordered = ([c for c in cache['candidates'] if c['id'] in wins] + [c for c in cache['candidates'] if c['id'] not in tested])
        for c in ordered:
            c['physical_status'] = 'bench_success' if c['id'] in wins else 'untested'
        path = Path(resolve(asset['cache'])).parent / 'grasp_cache_admitted.json'
        write(path, {**cache, 'candidates': ordered, 'physical_qualification': f'bench {len(wins)}/{len(runs)} pick-carry-place'})
        admitted.append({**asset, 'cache': binding(path), 'grasps': len(ordered)})
    write(output / 'admitted.json', {'schema': VERSION + '_admitted', 'rule': f'>= {MIN_SUCCESSES} of {BENCH_K} bench grasps succeed',
          'assets': admitted, 'bench_outcomes': dict(outcomes), 'candidates': binding(output / 'candidates.json')})
    return {'candidates': len(read(output / 'candidates.json')['candidates']), 'admitted': len(admitted),
            'bench_runs': sum(outcomes.values()), 'bench_success_rate': round(outcomes['success'] / max(1, sum(outcomes.values())), 3),
            'by_source': dict(Counter(a['source_id'].split(':')[0] for a in admitted))}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--output', type=Path, required=True)
    a.add_argument('--workers', type=int, default=64); a.add_argument('--sample', type=int); a.add_argument('--seed', type=int, default=0)
    b = sub.add_parser('bench'); b.add_argument('--output', type=Path, required=True)
    b.add_argument('--workers', type=int, default=160); b.add_argument('--chunk', type=int, default=4000)
    c = sub.add_parser('admit'); c.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.output, args.workers, args.sample, args.seed)))
    elif args.command == 'bench':
        print(json.dumps(bench_factory(args.output))); print(json.dumps(bench(args.output, args.workers, args.chunk)))
    else:
        print(json.dumps(admit(args.output), indent=1))
