"""Read-only campaign funnel and exclusive timing summary, including failures.

No evaluator is changed or rerun, no provider is queried, and no job is
submitted. Missing terminals/timing remain explicitly unknown. This tool
reports only the campaigns supplied, never extrapolated corpus throughput.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import statistics
from resimpair_grasp.common import read, write, binding


def summarize(roots):
    roots = [Path(p).resolve() for p in roots]
    if len(set(roots)) != len(roots): raise ValueError('duplicate_campaign_root')
    records = []; timings = defaultdict(list); sources = set(); mechanisms = set()
    counts = Counter(); failure_types = Counter(); inputs = []
    for root in roots:
        cohort = root/'cohort.json'; inputs.append(binding(cohort))
        cases = read(cohort)['cases']
        if len({c['id'] for c in cases}) != len(cases): raise ValueError('duplicate_case_in_campaign')
        for case in cases:
            counts['registered_cases'] += 1
            directory = root/'executions'/case['id']
            terminal_path = directory/'terminal.json'
            terminal = read(terminal_path) if terminal_path.exists() else None
            row = {'campaign': str(root), 'case': case['id'], 'eligible': bool(case['eligible']),
                   'terminal': binding(terminal_path) if terminal else None}
            sources.update([case['source_id']] if case.get('source_id') else case.get('active_source_ids', []))
            if not case['eligible']:
                counts['preflight_rejected'] += 1
                row['status'] = 'preflight_rejected'
            elif terminal is None:
                reserved = (directory/'reservation.json').exists()
                row['status'] = 'reserved_without_terminal' if reserved else 'not_started'
                counts[row['status']] += 1
            else:
                attempts = terminal.get('physical_attempts')
                if attempts is None:
                    counts['unknown_physical_attempt_count'] += 1
                else:
                    counts['physical_attempts'] += int(attempts)
                row.update(status='terminal', physical_attempts=attempts,
                           elapsed_seconds=terminal.get('elapsed_seconds'))
                audit = directory/'container_audit.json'
                if audit.exists():
                    result = read(audit); row['audit'] = binding(audit)
                    row['passed'] = result['passed']
                    counts['passed' if result['passed'] else 'failed'] += 1
                else:
                    counts['missing_independent_audit'] += 1
                failure = directory/'planning_failure.json'
                if failure.exists():
                    f = read(failure); label = f.get('message', f.get('type','unknown')).split(':')[0]
                    failure_types[label] += 1; row['failure_type'] = label
                bundle = directory/'execution_bundle.json'
                if bundle.exists():
                    b = read(bundle); mechanisms.update(b.get('task_intent',{}).get('required_skills', []))
            path = directory/'timing_breakdown.json'
            if path.exists() and read(path).get('exclusive_times_may_be_summed') is True:
                sums = Counter()
                for t in read(path)['rows']: sums[t['operation']] += t['exclusive_seconds']
                for op, value in sums.items(): timings[op].append(value)
                row['timing'] = binding(path); row['exclusive_seconds'] = dict(sums)
                if terminal and terminal.get('elapsed_seconds') is not None:
                    row['unattributed_wall_seconds'] = terminal['elapsed_seconds']-sum(sums.values())
                    # Significant negative residual means incompatible scopes.
                    if row['unattributed_wall_seconds'] < -.01:
                        counts['inconsistent_timing_scope'] += 1
                counts['executions_with_timings'] += 1
            elif terminal and terminal.get('physical_attempts',0):
                counts['executions_without_exclusive_timings'] += 1
            records.append(row)
    stats = {op:{'observed_executions':len(v),'mean_seconds':statistics.mean(v),
                 'median_seconds':statistics.median(v),'total_seconds':sum(v)} for op,v in sorted(timings.items())}
    elapsed = [r['elapsed_seconds'] for r in records if r.get('physical_attempts') and r.get('elapsed_seconds') is not None]
    return {'schema':'pickplace_batch_diagnostics_v1','inputs':inputs,'counts':dict(counts),
            'unique_source_ids':sorted(sources),'required_skill_labels':sorted(mechanisms),
            'independent_trajectory_count':'unknown; executions are not deduplicated action trajectories',
            'timing':stats,'observed_execution_wall_seconds':{'n':len(elapsed),
                'mean':statistics.mean(elapsed) if elapsed else None,
                'median':statistics.median(elapsed) if elapsed else None},
            'failure_types':dict(failure_types),'records':records,'api_calls':0,'new_physical_executions':0,
            'scope':'provided historical campaigns only; timings include failed executions where available; no human-generation timing or full-pipeline rate inferred'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(); result=summarize(a.campaign);write(a.output,result)
    print({'counts':result['counts'],'wall_seconds':result['observed_execution_wall_seconds']})
