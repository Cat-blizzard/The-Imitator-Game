"""Read-only method funnels, matched dynamics and non-overlapping timings."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import statistics

import numpy as np

from resimpair_grasp.common import binding, read, write, sha
from .campaign import BASE, LEDGER, METHODS, proposal_folder


def stats(values):
    values = [float(v) for v in values if v is not None]
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'p95': float(np.quantile(values, .95)) if values else None,
            'total': sum(values)}


def summarize():
    entries = read(BASE / 'selection.json')['entries']
    novelty_file = BASE / 'source_novelty_audit.json'
    novelty = {r['source_id']: r['novelty'] for r in read(novelty_file)['rows']} if novelty_file.exists() else {}
    rows = []; by_method = {}; physics = []; paired = defaultdict(dict)
    for split in ['development', 'heldout_asset']:
        root = BASE / ('physics_' + split)
        if not (root / 'cohort.json').exists(): continue
        for case in read(root / 'cohort.json')['cases']:
            directory = root / 'executions' / case['id']
            term = read(directory / 'terminal.json') if (directory / 'terminal.json').exists() else {}
            audit = read(directory / 'container_audit.json') if (directory / 'container_audit.json').exists() else {}
            failure = read(directory / 'planning_failure.json') if (directory / 'planning_failure.json').exists() else {}
            row = {'case': case['id'], 'source_id': case['source_id'], 'method': case['method'],
                'split': split, 'eligible': case['eligible'], 'physical_attempts': term.get('physical_attempts', 0),
                'novelty': novelty.get(case['source_id'], 'unknown'),
                'terminal_present': bool(term), 'passed': audit.get('passed'),
                'failure': failure.get('message') or case.get('preflight_reason'),
                'wall_seconds': term.get('elapsed_seconds'), 'endpoint_seconds': case.get('endpoint_seconds'),
                'audit': binding(directory / 'container_audit.json') if audit else None}
            if (directory / 'measurements.npz').exists():
                data = np.load(directory / 'measurements.npz', allow_pickle=False)
                row['two_finger_grasp_observed'] = bool(data['grasp'][:, :, 0].any())
                row['max_target_lift_m'] = float(data['object_states'][:, 0, 2].max() - data['object_states'][0, 0, 2])
                row['state_frames'] = len(data['states'])
                row['command_digest'] = __import__('hashlib').sha256(data['native_commands'].tobytes()).hexdigest()
            if (directory / 'execution_bundle.json').exists():
                b = read(directory / 'execution_bundle.json')
                row['initial_physics_sha256'] = b['initial_physics_sha256']
                paired[case['source_id']][case['method']] = b['initial_physics_sha256']
                row['videos'] = {n: binding(directory / (n + '.mp4')) for n in ['robot_side', 'opposite_high']
                                 if (directory / (n + '.mp4')).exists()}
            timing = directory / 'timing_breakdown.json'
            if timing.exists():
                value = read(timing)
                if not value.get('exclusive_times_may_be_summed'): raise ValueError('Timing scopes may overlap')
                sums = Counter()
                for item in value['rows']: sums[item['operation']] += item['exclusive_seconds']
                row['exclusive_seconds'] = dict(sums)
                row['unattributed_wall_seconds'] = row['wall_seconds'] - sum(sums.values()) if row['wall_seconds'] else None
            physics.append(row)
    for method in METHODS:
        method_rows = []
        for entry in entries:
            folder = proposal_folder(method, entry['id'])
            result = read(folder / 'result.json') if (folder / 'result.json').exists() else {}
            terminal = read(folder / 'terminal.json') if (folder / 'terminal.json').exists() else {}
            qpath = BASE / 'qualified_visual' / method / entry['id'] / 'decisions.json'
            q = read(qpath) if qpath.exists() else {}
            native_path = BASE / 'qualified_native' / method / entry['id'] / 'decisions.json'
            native = read(native_path) if native_path.exists() else {}
            row = {'id': entry['id'], 'source_id': entry['source_id'], 'source': entry['source'],
                'method': method, 'split': entry['split'], 'physical_subset': entry['physical_subset'],
                'novelty': novelty.get(entry['source_id'], 'unknown'),
                'generation_completed': bool(result), 'terminal_present': bool(terminal),
                'worker_seconds': result.get('worker_seconds'),
                'canonical_batch_wall_share_seconds': terminal.get('batch_wall_seconds', 0) / terminal.get('batch_jobs', 1) if 'batch_wall_seconds' in terminal else None,
                'generation_core_seconds': result.get('inference_seconds', result.get('solve_seconds')),
                'strict_optimizer_success': result.get('strict_optimizer_success'),
                'visual_qualification_completed': bool(q), 'visual_candidates': q.get('retained', 0),
                'qualification_seconds': q.get('elapsed_seconds'), 'visual_rejections': q.get('rejections', {}),
                'native_candidates': native.get('retained'),
                'result': binding(folder / 'result.json') if result else None,
                'raw': result.get('raw'), 'visual_qualification': binding(qpath) if q else None}
            method_rows.append(row); rows.append(row)
        prows = [r for r in physics if r['method'] == method]
        by_method[method] = {'source_attempts': len(method_rows),
            'generation_completed': sum(r['generation_completed'] for r in method_rows),
            'with_visual_candidates': sum(r['visual_candidates'] > 0 for r in method_rows),
            'retained_visual_pose_count': sum(r['visual_candidates'] for r in method_rows),
            'generation_core_seconds': stats(r['generation_core_seconds'] for r in method_rows),
            'worker_seconds': stats(r['worker_seconds'] for r in method_rows),
            'canonical_generation_wall_seconds_per_asset': stats(r['canonical_batch_wall_share_seconds'] for r in method_rows),
            'qualification_seconds': stats(r['qualification_seconds'] for r in method_rows),
            'physics_registered': len(prows), 'physics_eligible': sum(r['eligible'] for r in prows),
            'physics_attempted': sum(r['physical_attempts'] for r in prows),
            'physics_passed': sum(r['passed'] is True for r in prows),
            'grasp_observed': sum(r.get('two_finger_grasp_observed', False) for r in prows),
            'physical_wall_seconds': stats(r['wall_seconds'] for r in prows if r['physical_attempts']),
            'successful_physical_wall_seconds': stats(r['wall_seconds'] for r in prows if r['passed'] is True),
            'failure_counts': dict(Counter(r['failure'] or 'audit_failed_or_unresolved' for r in prows if r['passed'] is not True)),
            'splits': {split: {'source_attempts': sum(r['split'] == split for r in method_rows),
                'with_visual_candidates': sum(r['split'] == split and r['visual_candidates'] > 0 for r in method_rows),
                'physics_registered': sum(r['split'] == split for r in prows),
                'physics_attempted': sum(r['physical_attempts'] for r in prows if r['split'] == split),
                'physics_passed': sum(r['passed'] is True for r in prows if r['split'] == split)}
                for split in ['development', 'heldout_asset']}}
        batches = [read(p) for p in (BASE / 'batches').glob('*' + method + '*/terminal.json')
                   if read(p).get('returncode') == 0]
        by_method[method]['batch_wall_seconds_including_startup'] = stats(b['elapsed_seconds'] for b in batches)
        by_method[method]['reserved_jobs_in_batches'] = sum(b['reserved'] for b in batches)
        by_method[method]['unseen_sources_only'] = {
            'source_attempts': sum(r['novelty'] == 'unseen_source' for r in method_rows),
            'with_visual_candidates': sum(r['novelty'] == 'unseen_source' and r['visual_candidates'] > 0 for r in method_rows),
            'physics_registered': sum(r['novelty'] == 'unseen_source' for r in prows),
            'physics_attempted': sum(r['physical_attempts'] for r in prows if r['novelty'] == 'unseen_source'),
            'physics_passed': sum(r['passed'] is True for r in prows if r['novelty'] == 'unseen_source')}
    timings = defaultdict(list)
    learned_times = {r['id']: r['canonical_batch_wall_share_seconds'] for r in rows if r['method'] == 'graspgenx'}
    by_method['hybrid']['cold_generation_including_learned_seed_seconds'] = stats(
        r['canonical_batch_wall_share_seconds'] + learned_times[r['id']]
        for r in rows if r['method'] == 'hybrid' and r['canonical_batch_wall_share_seconds'] is not None
        and learned_times.get(r['id']) is not None)
    for row in physics:
        for operation, value in row.get('exclusive_seconds', {}).items(): timings[operation].append(value)
    conversions = read(BASE / 'native_conversion/report.json')['rows']
    return {'schema': 'three_way_grasp_result_v1', 'protocol': binding(BASE / 'protocol.json'),
        'methods': by_method, 'pose_rows': rows, 'physics_rows': physics,
        'matched_initial_states': [{'source_id': source, 'methods': sorted(v),
            'equal': len(set(v.values())) == 1, 'hashes': v} for source, v in paired.items()],
        'exclusive_physics_timing': {k: stats(v) for k, v in timings.items()},
        'physical_wall_seconds': stats(r['wall_seconds'] for r in physics if r['physical_attempts']),
        'successful_physical_wall_seconds': stats(r['wall_seconds'] for r in physics if r['passed'] is True),
        'endpoint_selection_seconds': stats(r['endpoint_seconds'] for r in physics if r['endpoint_seconds'] is not None),
        'native_conversion_seconds': stats(r['elapsed_seconds'] for r in conversions),
        'gpu_reserved': len(list((LEDGER / 'jobs').glob('*.json'))),
        'physical_reserved': len(list((LEDGER / 'reservations').glob('*.json'))),
        'prior_physical_reserved': 44, 'api_calls': 0, 'api_cost_usd': 0,
        'new_human_videos': 0, 'formal_real_source_pairs': 0,
        'scope': 'stratified already-tabletop-feasible assets at fixed scale/support; not whole-corpus success rate'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path); a = p.parse_args()
    result = summarize()
    if a.output: write(a.output, result)
    print({m: {k: v for k, v in s.items() if k in ['source_attempts', 'generation_completed',
        'with_visual_candidates', 'physics_registered', 'physics_attempted', 'physics_passed']}
        for m, s in result['methods'].items()})
