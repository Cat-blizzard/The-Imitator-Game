"""Independent budget dispatch tests using local functions and no transport calls."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Event
import argparse

from data_generation_methods.r2h_pipeline import unified_anchor_candidate as candidate
from data_generation_methods.r2h_pipeline.batch_pipeline import binding
from data_generation_methods.r2h_pipeline.common import file_sha256, read_json, write_json


def audit(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(candidate.__file__)
    source_before = file_sha256(source)
    calls, cases = [], []

    def budget(name, *, image=1, video=0):
        p = output/(name+'.json')
        write_json(p, {'schema': 'r2h_unified_budget_v1',
                      'campaign_id': 'independent_local_fixture_' + output.name + '_' + name,
                      'ledger_path': str(output/(name+'_ledger')),
                      'cumulative_call_limits': {'image': image, 'video': video, 'review': 0},
                      'max_concurrent': 1, 'timeout_seconds': 30})
        return p

    def local_only(unused_transport):
        calls.append('local_only_no_transport')
        return {'fixture_only': True, 'network_called': False}

    def expect_blocked(name, operation):
        try:
            operation()
        except (ValueError, RuntimeError, BlockingIOError) as exc:
            cases.append({'case': name, 'blocked': True, 'exception_type': type(exc).__name__,
                          'reason': str(exc)})
        else:
            cases.append({'case': name, 'blocked': False, 'reason': 'Unexpected dispatch'})

    original = budget('copy')
    copied = output/'byte_identical_copy.json'
    copied.write_bytes(original.read_bytes())
    first = candidate._campaign_budgeted_call(original, 'image', {'fixture_attempt': 1}, local_only)
    count = len(calls)
    recovered = candidate._campaign_budgeted_call(copied, 'image', {'fixture_attempt': 1}, local_only)
    cases.append({'case': 'copied_config_recovers_same_completed_stage',
                  'passed': recovered == first and len(calls) == count})
    expect_blocked('copied_config_does_not_reset_budget', lambda:
                   candidate._campaign_budgeted_call(copied, 'image', {'fixture_attempt': 2}, local_only))
    for field, replacement in (('ledger_path', str(output/'other_ledger')),
                               ('cumulative_call_limits', {'image': 2, 'video': 0, 'review': 0})):
        changed = read_json(original)
        changed[field] = replacement
        p = output/('changed_'+field+'.json')
        write_json(p, changed)
        expect_blocked('same_campaign_changed_'+field, lambda p=p:
                       candidate._campaign_budgeted_call(p, 'image', {'fixture_attempt': 2}, local_only))

    unknown = budget('unknown', image=2)
    def unresolved(unused_transport):
        calls.append('local_fixture_raised_before_any_transport')
        raise RuntimeError('Local fixture unresolved result; no network request sent')
    try:
        candidate._campaign_budgeted_call(unknown, 'image', {'fixture_attempt': 1}, unresolved)
    except RuntimeError:
        pass
    unknown_copy = output/'unknown_copy.json'
    unknown_copy.write_bytes(unknown.read_bytes())
    expect_blocked('unresolved_stage_blocks_copied_config', lambda:
                   candidate._campaign_budgeted_call(unknown_copy, 'image', {'fixture_attempt': 2}, local_only))

    concurrent = budget('concurrent', image=2)
    entered, release = Event(), Event()
    def held(unused_transport):
        calls.append('held_local_fixture_no_transport')
        entered.set()
        if not release.wait(10):
            raise RuntimeError('Fixture synchronization timed out')
        return {'fixture_only': True, 'network_called': False}
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(candidate._campaign_budgeted_call, concurrent, 'image',
                                 {'fixture_attempt': 1}, held)
        if not entered.wait(10):
            raise RuntimeError('Fixture did not enter first operation')
        try:
            expect_blocked('concurrent_second_dispatch_rejected', lambda:
                           candidate._campaign_budgeted_call(concurrent, 'image', {'fixture_attempt': 2}, local_only))
        finally:
            release.set()
        future.result(timeout=10)

    zero = budget('zero', image=0)
    expect_blocked('zero_budget_dispatch_rejected', lambda:
                   candidate._campaign_budgeted_call(zero, 'image', {'fixture_attempt': 1}, local_only))
    cases.append({'case': 'zero_budget_creates_no_started_marker',
                  'passed': not list(Path(read_json(zero)['ledger_path']).glob('stages/*/call_started.json'))})
    video = budget('video', image=1, video=1)
    def fake_accepted_video(unused_transport):
        calls.append('local_video_fixture_no_transport')
        return {'task_id': 'not_a_real_provider_task', 'fixture_only': True}
    candidate._campaign_budgeted_call(video, 'video', {'fixture_attempt': 1}, fake_accepted_video)
    video_copy = output/'video_copy.json'
    video_copy.write_bytes(video.read_bytes())
    expect_blocked('async_video_still_occupies_campaign_after_copy', lambda:
                   candidate._campaign_budgeted_call(video_copy, 'image', {'fixture_attempt': 2}, local_only))
    source_after = file_sha256(source)
    result = {'schema': 'independent_unified_budget_repair_audit_v1',
              'source': binding(source), 'source_unchanged': source_before == source_after,
              'audit_source': binding(__file__), 'cases': cases,
              'all_passed': all(c.get('passed', c.get('blocked', False)) for c in cases),
              'local_function_invocations': calls, 'local_only_fixture': True,
              'actual_api_calls': 0, 'actual_cost': 0,
              'scope': 'Ledger and dispatch invariants; no network, generated image, or paid provider task.'}
    write_json(output/'audit.json', result)
    if not result['all_passed'] or not result['source_unchanged']:
        raise ValueError('Independent budget audit failed')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output)
    print({'cases': len(result['cases']), 'all_passed': result['all_passed'], 'actual_api_calls': 0})
