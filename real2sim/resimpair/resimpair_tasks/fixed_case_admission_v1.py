"""Execute only the named cases in a reviewed finite repair manifest.

The retained driver still owns simulation, shared locks, budgets and rules.
This entry prevents a resumed ``run(limit=1)`` from silently selecting the next
case in a historical multi-case cohort. Existing reservations are never retried.
"""
import argparse
import fcntl
from pathlib import Path
import time
from unittest.mock import patch

from resimpair_grasp.common import binding, read, resolve, write
from . import production_integration_v2 as retained


def guarded_driver(root, case):
    """Enforce the exact case again at the driver's first budget mutation."""
    root = Path(root).resolve()
    folder = root / 'executions' / case['id']
    if (folder / 'terminal.json').exists():
        if not (folder / 'reservation.json').exists():
            raise ValueError('terminal_without_original_reservation')
        return {'state': 'already_terminal_no_submission', 'terminal': binding(folder / 'terminal.json')}
    if (folder / 'reservation.json').exists():
        raise ValueError('existing_reserved_case_requires_recovery_never_retry')
    protocol = read(root / 'protocol.json')
    campaign = Path(protocol['campaign_root'])
    original_write = retained.write

    def exact_case_write(path, value):
        path = Path(path)
        budget = path.parent == campaign / 'budget' / 'reservations'
        execution = path.name == 'reservation.json' and path.parent.parent == root / 'executions'
        if budget or execution:
            if (value.get('case') != case or value.get('campaign') != str(root)
                    or value.get('cohort') != root.name):
                raise ValueError('driver_attempted_unapproved_case_before_budget_reservation')
            if execution and path != folder / 'reservation.json':
                raise ValueError('driver_attempted_wrong_execution_directory')
        return original_write(path, value)

    with patch.object(retained, 'write', exact_case_write):
        result = retained.run(root, limit=1)
    if result.get('new_attempts') != 1 or not (folder / 'terminal.json').exists():
        raise ValueError('requested_case_has_no_verified_terminal_after_driver_return')
    return {'state': 'executed_once', 'driver_return': result,
            'reservation': binding(folder / 'reservation.json'), 'terminal': binding(folder / 'terminal.json')}


def run(protocol_path, gate_path, output):
    protocol_path, output = Path(protocol_path).resolve(), Path(output).resolve()
    protocol = read(protocol_path); gate = read(gate_path)
    expected = protocol['allowed_submissions']
    if (gate.get('passed') is not True or gate.get('protocol') != binding(protocol_path)
            or gate.get('runner') != binding(__file__) or gate.get('allowed_submissions') != expected
            or gate.get('maximum_new_actual_executions') != protocol['maximum_new_actual_executions']):
        raise ValueError('independently_reviewed_exact_case_manifest_required')
    resolve(gate['independent_review'])
    if len(expected) != protocol['maximum_new_actual_executions']:
        raise ValueError('finite_exact_case_count_required')
    excluded = protocol.get('explicitly_not_submitted')
    if excluded and any(x['root'] == excluded['root'] and x['case'] == excluded['case'] for x in expected):
        raise ValueError('excluded_case_is_not_authorized')
    cases = []
    for target in expected:
        root = Path(target['root']).resolve()
        matches = [c for c in read(root / 'cohort.json')['cases'] if c['id'] == target['case']]
        if len(matches) != 1 or target['maximum_runner_limit'] != 1:
            raise ValueError('one_exact_case_per_fixed_submission_required')
        for ref in read(root / 'frozen_inputs.json').values():
            resolve(ref)
        if read(root / 'rules.json') != read(resolve(protocol['unchanged_rules'])):
            raise ValueError('original_physical_rules_changed')
        cases.append((root, matches[0]))
    output.mkdir(parents=True, exist_ok=True)
    config = {'protocol': binding(protocol_path), 'gate': binding(gate_path), 'implementation': binding(__file__)}
    with (output / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(output / 'REGISTERED.json', config)
        session = output / 'sessions' / str(time.time_ns())
        write(session / 'LAUNCH.json', {**config, 'unix': time.time()})
        rows = []
        for root, case in cases:
            try:
                result = guarded_driver(root, case)
            except Exception as error:
                write(session / 'STOPPED.json', {'root': str(root), 'case': case['id'],
                    'error': repr(error), 'earlier_cases': rows, 'new_retry': False})
                raise
            rows.append({'root': str(root), 'case': case['id'], **result})
            write(session / f'{len(rows):02d}.json', rows[-1])
        result = {'rows': rows, 'new_attempts': sum(r['state'] == 'executed_once' for r in rows),
                  'all_named_cases_terminal': True, 'API_calls': 0, 'GPU_jobs': 0,
                  'scope': 'submission status only; independent physical acceptance remains required'}
        write(session / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'gate', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(run(args.protocol, args.gate, args.output))
