"""Finite 500-request execution with a frozen 20-request expansion gate.

Every retry resumes the same output and shared reservation ledger. No failed
task is replaced, and qualification executions are part of the 500 requests.
"""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from resimpair_grasp.common import ROOT, binding, read, resolve, write


def qualification_scene_ids(scenes):
    groups = defaultdict(list)
    for scene in scenes: groups[scene['partition']].append(scene['scene_id'])
    selected = []
    for partition in sorted(groups): selected.extend(sorted(groups[partition])[:2])
    if len(selected) > 4: raise ValueError('More than two partitions require a new qualification protocol')
    for scene in sorted(s['scene_id'] for s in scenes):
        if len(selected) == 4: break
        if scene not in selected: selected.append(scene)
    if len(selected) != 4: raise ValueError('Four shared scenes are required')
    return selected


def validate_chunks(rows, chunks, scenes):
    """Prove that executable requests are exactly the independently audited set."""
    expected = {}; requests = {}; metadata = {}; partitions = defaultdict(set)
    for row in rows:
        if row['id'] in expected or row['id'] != row['scene_id'] + '_' + row['requested_task_id']:
            raise ValueError('Duplicate or inconsistent manifest task identity')
        expected[row['id']] = row
        partitions[row['scene_id']].add(row['partition'])
        source = read(resolve(row['source_request']))
        if source['id'] != row['scene_id']: raise ValueError('Manifest scene identity changed')
        if row['scene_id'] in requests and requests[row['scene_id']] != source:
            raise ValueError('One scene has conflicting bound requests')
        requests[row['scene_id']] = source
    for scene in scenes:
        if scene['scene_id'] in metadata: raise ValueError('Duplicate scene metadata')
        metadata[scene['scene_id']] = scene
        if partitions[scene['scene_id']] != {scene['partition']}:
            raise ValueError('Scene metadata partition differs from the audited task partition')
        if read(resolve(scene['request'])) != requests.get(scene['scene_id']):
            raise ValueError('Scene metadata and manifest request differ')
    if set(metadata) != set(requests): raise ValueError('Scene metadata coverage differs')
    seen_scenes = set(); seen_cases = set(); seen_chunks = set()
    for chunk in chunks:
        if type(chunk['chunk']) is not int or chunk['chunk'] < 0 or chunk['chunk'] in seen_chunks:
            raise ValueError('Invalid or duplicate executable chunk')
        seen_chunks.add(chunk['chunk'])
        value = read(resolve(chunk['requests']))
        if value.get('schema') != 'cached_pickplace_requests_v1': raise ValueError('Chunk request schema changed')
        actual_ids = [s['id'] for s in value['scenes']]
        if actual_ids != chunk['scene_ids']: raise ValueError('Declared chunk scene IDs differ')
        count = 0
        for i, scene in enumerate(value['scenes']):
            sid = scene['id']
            if sid in seen_scenes or requests.get(sid) != scene:
                raise ValueError('Missing, repeated or changed executable scene')
            seen_scenes.add(sid)
            if chunk['seed_start'] + i != metadata[sid]['seed']:
                raise ValueError('Executable scene seed differs from frozen clutter seed')
            for task in scene['tasks']:
                identity = sid + '_' + task['id']; count += 1
                if identity not in expected or identity in seen_cases:
                    raise ValueError('Unexpected or duplicate executable task')
                seen_cases.add(identity)
        if count != chunk['task_count']: raise ValueError('Declared chunk task count differs')
    if seen_cases != set(expected) or seen_scenes != set(requests):
        raise ValueError('Executable chunks do not cover the full audited manifest')
    return expected


def validate_cohort(campaign, chunk, expected):
    requested = read(resolve(chunk['requests']))['scenes']
    wanted = {s['id'] + '_' + t['id'] for s in requested for t in s['tasks']}
    cases = read(campaign / 'cohort.json')['cases']; seen = set()
    for case in cases:
        identity = case['id']
        if identity in seen or identity not in wanted or case['execution_id'] != identity:
            raise ValueError('Registered cohort differs from frozen executable tasks')
        seen.add(identity); original = expected[identity]
        if (case['requested_task_id'] != original['requested_task_id'] or
                read(resolve(case['source_request'])) != read(resolve(original['source_request']))):
            raise ValueError('Registered task or source request changed')
    if seen != wanted: raise ValueError('Registered cohort is incomplete')


def initialize(root, factory, library, budget, admission, motion, contact):
    from .coverage_audit_v1 import audit
    root, factory, library = map(lambda p: Path(p).resolve(), (root, factory, library))
    if root.exists(): raise ValueError('Production exists; resume run')
    manifest = factory / 'coverage_manifest.json'
    rows = read(manifest)['rows']
    coverage = audit(rows, expected_count=500,
        known_source_ids=read(Path(budget).parent / 'known_source_ids.json')['source_ids'])
    if not coverage['passed']: raise ValueError('Frozen coverage audit failed')
    if coverage['counts']['scene_ids'] != 100 or set(coverage['scene_task_counts'].values()) != {5}:
        raise ValueError('Production requires exactly 100 shared scenes with five tasks each')
    scenes = read(factory / 'scenes.json')['scenes']
    validate_chunks(rows, read(factory / 'chunks.json')['chunks'], scenes)
    ids = qualification_scene_ids(scenes)
    wanted = [r['id'] for r in rows if r['scene_id'] in ids]
    if len(wanted) != 20: raise ValueError('Qualification must contain all five tasks of each scene')
    write(root / 'coverage_audit.json', {**coverage, 'manifest': binding(manifest)})
    write(root / 'protocol.json', {'schema': 'diverse500_execution_v1',
        'implementation': binding(__file__), 'factory': str(factory),
        'manifest': binding(manifest), 'chunks': binding(factory / 'chunks.json'), 'scenes': binding(factory / 'scenes.json'),
        'library': str(library), 'library_manifest': binding(library / 'manifest.json'),
        'budget': str(Path(budget).resolve()), 'admission': binding(admission),
        'qualification_scenes': ids, 'qualification_cases': wanted,
        'motion_policy': motion, 'contact_policy': contact, 'release_policy': 'legacy_v1',
        'gpu': 2, 'api_calls': 0, 'physical_retries': 0, 'default_promoted': False})
    return {'qualification_tasks': len(wanted), 'production_tasks': len(rows), 'coverage': coverage['counts']}


def _campaign(root, chunk):
    return root / (root.name + '_chunk_' + str(chunk['chunk']).zfill(3))


def require_recorded_execution(root, case, execution):
    """Run both before skipping a terminal and immediately after a worker exits."""
    if case['eligible'] and (execution / 'terminal.json').exists() and not (execution / 'execution_bundle.json').exists():
        write(root / 'infrastructure_stop.json', {'case': case['id'],
            'reason': 'worker_started_but_no_execution_bundle',
            'planning_failure': binding(execution / 'planning_failure.json') if (execution / 'planning_failure.json').exists() else None,
            'automatic_retry': False, 'physical_budget_still_charged': True})
        raise RuntimeError('Infrastructure failure before recorded execution; stop the batch and retain this attempt')


def _ensure_campaign(root, protocol, chunk):
    from resimpair_pickplace.cache_runtime import initialize as register
    campaign = _campaign(root, chunk)
    if (campaign / 'cohort.json').exists(): return campaign
    if campaign.exists(): raise ValueError('Unfinished registration: inspect existing state before resuming ' + str(campaign))
    started = time.monotonic()
    args = SimpleNamespace(root=campaign, budget=Path(protocol['budget']),
        requests=resolve(chunk['requests']), library=Path(protocol['library']),
        method='cudex', gpu=2, assets=1, layouts=1, physical_limit=1,
        seed=chunk['seed_start'], release_policy='legacy_v1',
        motion_policy=protocol['motion_policy'], contact_policy=protocol['contact_policy'],
        contact_diagnostic=False)
    result = register(args)
    write(campaign / 'registration_timing.json', {'seconds': time.monotonic() - started, **result})
    print({'registered': campaign.name, **result}, flush=True)
    return campaign


def status(root):
    root = Path(root).resolve(); protocol = read(root / 'protocol.json')
    rows = []
    for campaign in sorted(root.glob(root.name + '_chunk_*')):
        if not (campaign / 'cohort.json').exists(): continue
        for case in read(campaign / 'cohort.json')['cases']:
            directory = campaign / 'executions' / case['id']
            terminal = directory / 'terminal.json'; audit = directory / 'requested_goal_audit.json'
            if terminal.exists():
                t = read(terminal); a = read(audit) if audit.exists() else {}
                state = 'accepted' if a.get('passed') else ('rejected' if audit.exists() or not t['physical_attempts'] else 'audit_pending')
                if case['eligible'] and not (directory / 'execution_bundle.json').exists(): state = 'infrastructure_failure'
                rows.append({'case': case['id'], 'campaign': str(campaign), 'state': state,
                    'passed': a.get('passed', False), 'physical_attempts': t['physical_attempts'],
                    'seconds': t['elapsed_seconds'], 'errors': a.get('errors', [t.get('reason', 'audit_pending')]),
                    'audit': binding(audit) if audit.exists() else None})
            elif (directory / 'reservation.json').exists():
                rows.append({'case': case['id'], 'campaign': str(campaign), 'state': 'reserved_unresolved', 'passed': False})
    value = {'schema': 'diverse500_progress_v1', 'updated_unix': time.time(),
        'frozen_tasks': 500, 'rows': rows, 'states': dict(Counter(r['state'] for r in rows)),
        'physical_terminal': sum(r.get('physical_attempts', 0) for r in rows),
        'accepted': sum(r['passed'] for r in rows), 'api_calls': 0,
        'qualification': read(root / 'qualification_decision.json') if (root / 'qualification_decision.json').exists() else 'pending'}
    temporary = root / ('progress.' + str(os.getpid()) + '.tmp'); temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, root / 'progress.json')
    return value


def run(root):
    from resimpair_pickplace.stability_runner import run as execute
    from resimpair_pickplace.requested_goal_audit import audit
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import resources
    root = Path(root).resolve()
    with (root / '.production.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'infrastructure_stop.json').exists():
            raise RuntimeError('This candidate stopped on infrastructure failure; retain it and use an explicitly repaired candidate')
        p = read(root / 'protocol.json')
        for key in ['implementation', 'manifest', 'chunks', 'scenes', 'library_manifest', 'admission']: resolve(p[key])
        if (root / 'completion.json').exists():
            return read(resolve(read(root / 'completion.json')['progress']))
        chunks = read(resolve(p['chunks']))['chunks']
        expected = validate_chunks(read(resolve(p['manifest']))['rows'], chunks, read(resolve(p['scenes']))['scenes'])
        wanted = set(p['qualification_cases'])
        for phase in ['qualification', 'production']:
            if phase == 'production' and not read(root / 'qualification_decision.json')['expand']:
                print({'state': 'qualification_failed_expansion_stopped'}, flush=True)
                return status(root)
            for chunk in chunks:
                if phase == 'qualification' and not set(chunk['scene_ids']) & set(p['qualification_scenes']): continue
                campaign = _ensure_campaign(root, p, chunk)
                validate_cohort(campaign, chunk, expected)
                for case in read(campaign / 'cohort.json')['cases']:
                    if phase == 'qualification' and case['id'] not in wanted: continue
                    execution = campaign / 'executions' / case['id']
                    require_recorded_execution(root, case, execution)
                    if (execution / 'terminal.json').exists(): continue
                    for attempt in range(16):
                        snapshot = resources.gpu_snapshot(2)
                        try: resources.check_idle(snapshot); break
                        except RuntimeError:
                            if snapshot['compute_owners'] or attempt == 15: raise
                            time.sleep(1)
                    execute(SimpleNamespace(root=campaign, case=[case['id']], limit=1))
                    require_recorded_execution(root, case, execution)
                    result = audit(SimpleNamespace(root=campaign))
                    print({'case': case['id'], 'campaign_accepted': result['passed']}, flush=True)
                    status(root)
                audit(SimpleNamespace(root=campaign))
            progress = status(root)
            if phase == 'qualification':
                rows = [r for r in progress['rows'] if r['case'] in wanted]
                if len(rows) != 20 or any(r['state'] not in {'accepted', 'rejected'} for r in rows):
                    raise ValueError('Qualification has unresolved results')
                by_scene = Counter()
                manifest = {r['id']: r for r in read(resolve(p['manifest']))['rows']}
                for row in rows:
                    if row['passed']: by_scene[manifest[row['case']]['scene_id']] += 1
                passed = sum(r['passed'] for r in rows); complete = sum(v == 5 for v in by_scene.values())
                gate = read(resolve(p['admission']))
                decision = {'tasks': 20, 'passed': passed, 'complete_scenes': complete,
                    'expand': passed >= gate['threshold_passed'] and complete >= gate['minimum_complete_scenes'],
                    'rows': rows, 'rules_changed': False, 'failed_tasks_replaced': False}
                write(root / 'qualification_decision.json', decision)
        if ({r['case'] for r in progress['rows']} != set(expected) or len(progress['rows']) != 500
                or any(r['state'] not in {'accepted', 'rejected'} for r in progress['rows'])):
            raise ValueError('Cannot complete: the 500 unique requests do not all have final audited states')
        write(root / 'final_report.json', progress)
        write(root / 'completion.json', {'state': 'all_500_requests_resolved',
            'progress': binding(root / 'final_report.json'), 'accepted': progress['accepted']})
        return progress


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['initialize', 'run', 'status'])
    parser.add_argument('--root', type=Path, required=True)
    for name in ['factory', 'library', 'budget', 'admission']: parser.add_argument('--' + name, type=Path)
    parser.add_argument('--motion', choices=['legacy_v1', 'cartesian_transfer_v1', 'transport_fallback_v2'], default='cartesian_transfer_v1')
    parser.add_argument('--contact', choices=['native_default', 'small_tabletop_v1'], default='small_tabletop_v1')
    args = parser.parse_args()
    if args.command == 'initialize': result = initialize(args.root, args.factory, args.library, args.budget, args.admission, args.motion, args.contact)
    elif args.command == 'run': result = run(args.root)
    else: result = status(args.root)
    print(json.dumps({k: v for k, v in result.items() if k not in {'rows', 'qualification'}}, ensure_ascii=False))
