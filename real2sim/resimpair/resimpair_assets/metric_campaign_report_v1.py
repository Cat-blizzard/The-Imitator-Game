"""Read-only coverage, failure and measured timing report for metric campaigns."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import binding, read, resolve, write


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'count': 0}
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('invalid_observed_duration')
    return {'count': len(values), 'mean_seconds': float(values.mean()),
        'median_seconds': float(np.median(values)), 'p90_seconds': float(np.quantile(values, .9)),
        'p99_seconds': float(np.quantile(values, .99)), 'sum_worker_seconds': float(values.sum())}


def summarize(cpu_root, stream_root, output):
    cpu_root, stream_root, output = map(Path, [cpu_root, stream_root, output])
    if output.exists():
        raise FileExistsError(output)
    observed = time.time()
    cpu = read(cpu_root / 'source_state_index.json')
    cpu_status = read(cpu_root / 'status.json')
    gpu_status = read(stream_root / 'status.json')
    gpu_status.pop('rows', None)
    cpu_protocol = read(cpu_root / 'protocol.json')
    source_manifest = read(resolve(cpu_protocol['manifest']))
    states = {r['source_id']: r for r in cpu['entries']}
    roles = defaultdict(Counter)
    qualification = Counter()
    native_failures = Counter()
    refined = Counter()
    input_failures = Counter()
    cpu_times = defaultdict(list)
    source_rows = []
    for job in source_manifest['jobs']:
        folder = cpu_root / 'recoveries_v6' / job['key']
        if not (folder / 'job.json').exists():
            folder = cpu_root / 'jobs' / job['key']
        state = states[job['source_id']]
        terminal = read(folder / 'terminal.json') if (folder / 'terminal.json').exists() else None
        outcome = terminal['status'] if terminal else state.get('batch_status', 'pending')
        role = job['decision']['role']
        roles[role][outcome] += 1
        if (folder / 'failure.json').exists():
            failed = read(folder / 'failure.json')
            input_failures[failed.get('error', failed.get('reason', 'unknown')).split(':')[0]] += 1
        duration = None
        if terminal and (folder / 'process.json').exists():
            started = read(folder / 'process.json')['unix']
            duration = (folder / 'terminal.json').stat().st_mtime - started
            cpu_times['all_terminal'].append(duration)
            cpu_times[role].append(duration)
        source_rows.append({'source_id': job['source_id'], 'role': role, 'outcome': outcome,
            'terminal': binding(folder / 'terminal.json') if terminal else None,
            'observed_subprocess_seconds': duration})
        for entry in state.get('variants', []):
            if entry.get('role') != 'manipulation':
                continue
            good = entry.get('geometry_qualified', False)
            qualification['native_qualified' if good else 'native_not_qualified'] += 1
            if entry.get('metric_decomposition_triggered'):
                refined['qualified' if good else 'still_failed'] += 1
            if not good:
                surface = entry.get('surface', {})
                if surface.get('visual_to_collision_p95_m', 0) > .003:
                    native_failures['surface_forward_above_3mm'] += 1
                if surface.get('union_external_p95_m', 0) > .003:
                    native_failures['surface_reverse_above_3mm'] += 1
                if entry.get('settle_status') != 'pass':
                    native_failures['CPU_settle_not_pass'] += 1
                if entry.get('support_status') != 'pass':
                    native_failures['COM_support_not_pass'] += 1
    gpu_outcomes = Counter()
    rejections = Counter()
    stages = defaultdict(list)
    gpu_rows = []
    for path in (stream_root / 'jobs').glob('*/terminal.json'):
        terminal = read(path)
        gpu_outcomes[terminal['status']] += 1
        folder = path.parent
        if not (folder / 'result.json').exists():
            continue
        value = read(folder / 'result.json')
        count = value.get('retained', 0)
        gpu_outcomes['nonempty' if count else 'empty'] += 1
        rejections.update(value.get('rejections', {}))
        for key in ['setup_seconds', 'solve_seconds', 'worker_seconds', 'export_seconds']:
            if key in value:
                stages[key].append(value[key])
        gpu_rows.append({'result': binding(folder / 'result.json'), 'retained': count,
            'reserved_optimizer': (folder / 'reservation.json').exists()})
    scale = read(cpu_root / 'worker_scale_16_receipt.json')
    terminal_count = sum(v for k, v in cpu_status['counts'].items() if k not in {'pending', 'running'})
    elapsed = cpu_status['updated_unix'] - scale['created_unix']
    value = {'schema': 'metric_population_measured_report_v1', 'observed_unix': observed,
        'implementation': binding(__file__), 'source_plan': source_manifest['source_state_index'],
        'CPU_protocol': binding(cpu_root / 'protocol.json'),
        'CPU_16worker_receipt': binding(cpu_root / 'worker_scale_16_receipt.json'),
        'GPU_protocol': binding(stream_root / 'protocol.json'),
        'producer_snapshots': {'CPU': cpu_status, 'GPU': gpu_status},
        'role_outcomes': {k: dict(v) for k, v in roles.items()},
        'native_qualification': dict(qualification),
        'native_failure_dimensions_not_exclusive': dict(native_failures),
        'conditional_collision_refinement': dict(refined),
        'source_input_failure_reasons': dict(input_failures),
        'CPU_observed_durations': {k: distribution(v) for k, v in cpu_times.items()},
        'CPU_duration_method': 'terminal_first_write_file_mtime_minus_recorded_subprocess_start_unix_includes_import_and_output_cost; historical_recovered_jobs_may_include_recovery_delay',
        'CPU_16worker_observed_throughput': {
            'new_terminal_sources': terminal_count - scale['prior_terminal_source_count'],
            'elapsed_seconds_to_status_snapshot': elapsed,
            'terminal_sources_per_second': (terminal_count - scale['prior_terminal_source_count']) / elapsed,
            'mix_includes_static_dynamic_reuse_and_known_failures': True,
            'not_a_cold_raw_preprocessing_cost_estimate': True},
        'GPU_terminal_outcomes': dict(gpu_outcomes),
        'GPU_native_qualifier_rejections_not_physics_failures': dict(rejections),
        'GPU_internal_stage_durations': {k: distribution(v) for k, v in stages.items()},
        'source_rows': source_rows, 'GPU_rows': gpu_rows,
        'raw_pending_is_not_unusable': True, 'physical_pick_place_success_inferred': False,
        'API_calls': 0, 'no_jobs_launched_or_stopped': True}
    write(output, value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-root', type=Path, required=True)
    parser.add_argument('--stream-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.cpu_root, args.stream_root, args.output)
    print({key: result[key] for key in ['native_qualification', 'CPU_16worker_observed_throughput', 'GPU_terminal_outcomes']})
