"""Compare the original three heights using saved source IK, without simulation.

The control and alternatives share the exact saved grasp qpos and a common
bank of native random initializations. Only existing development slot scenes,
both existing arms and the first twelve cached grasps are in scope. An IK
failure is bounded-search evidence, not a proof of global unreachability.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_pickplace import above_rim_release_v1 as release
from resimpair_pickplace import whole_path_motion_v1 as whole
from . import source_first_reachable_layout_v3 as endpoints

VERSION = 'source_first_height_profile_audit_v1'
PROFILES = ((.20, .18), (.12, .12), (.26, .24))


@contextmanager
def common_initializations(planner, bank):
    """Keep upstream IK, collisions and limits; replay only its random starts."""
    original = planner.pinocchio_model
    class ReplayModel:
        def __init__(self):
            self.index = 0
        def __getattr__(self, name):
            return getattr(original, name)
        def get_random_configuration(self):
            if self.index >= len(bank):
                raise ValueError('original_eight_initialization_budget_exceeded')
            value = np.asarray(bank[self.index], float).copy()
            self.index += 1
            return value
    planner.pinocchio_model = ReplayModel()
    try:
        yield
    finally:
        planner.pinocchio_model = original


def run_steps(selector, arm, stages, seed_qpos, bank):
    q = np.asarray(seed_qpos, float).copy()
    checks = []
    for stage, target in stages:
        with common_initializations(selector.planners[arm], bank):
            next_q, status = selector.ik(arm, target, q)
        checks.append({'stage': stage, 'world_tcp': target.tolist(), 'seed_qpos': q.tolist(),
            'status': status, 'qpos': None if next_q is None else next_q.tolist()})
        if next_q is None:
            return {'passed': False, 'checks': checks}
        q = next_q
    return {'passed': True, 'checks': checks}


def operation_key(operation):
    return digest([operation['object'], operation['target'], operation.get('release')])


def compare_operation(scene, operation, prior, selector, reused=None):
    item = next(o for o in scene['objects'] if o['id'] == operation['object'])
    source = np.asarray(item.get('measured_world_transform', endpoints.pose(item['position'], item['yaw'])))
    cache = read(resolve(scene['grasp_caches'][item['id']]))
    candidates = {c['id']: c for c in cache['candidates'][:12]}
    rows = []; calls = 0
    for old in prior['decisions']:
        arm, cid = old['arm'], old['candidate']
        if arm not in (0, 1) or cid not in candidates:
            raise ValueError('saved_decision_outside_two_arms_first_twelve_original_candidates')
        row = {'arm': arm, 'candidate': cid, 'legacy_decision': old, 'profiles': [],
               'height_only_false_negative': False, 'profile_chain_passed': False}
        prefix = old['checks'][:2]
        if ([c['stage'] for c in prefix] != ['pregrasp', 'grasp']
                or any(c['qpos'] is None for c in prefix)):
            row['classification'] = 'unchanged_source_or_release_geometry_failure'
            rows.append(row)
            continue
        local = np.asarray(candidates[cid]['tcp_in_asset'])
        desired, geometry = release.release_geometry(scene, operation, local)
        grasp, target = source @ local, desired @ local
        above_control = target.copy(); above_control[2, 3] += .18
        np.testing.assert_allclose(grasp, prefix[-1]['world_tcp'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(above_control, old['checks'][2]['world_tcp'], rtol=0, atol=1e-12)
        if not old.get('container_hand_clearance', {'passed': True})['passed']:
            raise ValueError('saved_source_prefix_requires_retained_release_clearance')
        saved = ([r for r in reused['rows'] if r['arm'] == arm and r['candidate'] == cid]
                 if reused is not None else [])
        bank = (saved[0]['common_random_initializations'] if saved else
                [np.asarray(selector.planners[arm].pinocchio_model.get_random_configuration()).tolist()
                 for _ in range(8)])
        row.update(saved_source_prefix=prefix, release_geometry=geometry, common_random_initializations=bank)
        for heights in PROFILES:
            above = target.copy(); above[2, 3] += heights[1]
            lift = grasp.copy(); lift[2, 3] += heights[0]
            previous = next((r for r in saved if tuple(r['height_profile_m']) == heights), None)
            if previous is None:
                tail = run_steps(selector, arm, [('above_goal', above), ('release', target)], prefix[-1]['qpos'], bank)
                calls += len(tail['checks'])
            else:
                tail = {'passed': previous['accepted'], 'checks': previous['checks'],
                        'reused_identical_saved_input_probe': True}
            branch = {'height_profile_m': list(heights), 'controlled_legacy_tail': tail,
                      'complete_profile_endpoint_chain': None}
            # A full same-arm profile also needs its lift endpoint. It is still
            # only an endpoint proposal: never bypass preview_route or check.
            if tail['passed']:
                chain = run_steps(selector, arm, [('lift', lift), ('above_goal', above), ('release', target)],
                                  prefix[-1]['qpos'], bank)
                calls += len(chain['checks'])
                branch['complete_profile_endpoint_chain'] = chain
            row['profiles'].append(branch)
        control = row['profiles'][0]['controlled_legacy_tail']['passed']
        alternatives = [p for p in row['profiles'][1:] if p['controlled_legacy_tail']['passed']
                        and p['complete_profile_endpoint_chain']['passed']]
        row['profile_chain_passed'] = any(p['complete_profile_endpoint_chain'] is not None
            and p['complete_profile_endpoint_chain']['passed'] for p in row['profiles'])
        failed_above = not old['accepted'] and old['checks'][-1]['stage'] == 'above_goal'
        row['height_only_false_negative'] = bool(failed_above and not control and alternatives)
        if row['height_only_false_negative']:
            row['classification'] = 'controlled_fixed_height_false_negative_with_lift_endpoint'
        elif not old['accepted'] and control:
            row['classification'] = 'legacy_rejection_but_control_replay_passes_not_attributed_to_height'
        elif row['profile_chain_passed']:
            row['classification'] = 'profile_endpoint_candidate'
        else:
            row['classification'] = 'no_solution_in_this_bounded_original_profile_check'
        rows.append(row)
    considered = {(r['arm'], r['candidate']) for r in rows}
    missing = [{'arm': a, 'candidate': cid} for a in (0, 1) for cid in candidates if (a, cid) not in considered]
    return {'schema': VERSION, 'object': item['id'], 'operation': operation,
        'legacy_passed': bool(prior['passed']), 'profile_chain_passed': any(r['profile_chain_passed'] for r in rows),
        'controlled_height_false_negative': any(r['height_only_false_negative'] for r in rows),
        'rows': rows, 'not_in_saved_legacy_report': missing,
        'missing_reason': 'legacy stopped after three accepted candidates on that arm; replay does not add source draws',
        'new_endpoint_IK_calls': calls, 'original_cache': scene['grasp_caches'][item['id']],
        'physical_success': 'unmeasured', 'physics_steps': 0,
        'scope': 'saved source endpoint plus lift/above-goal/release IK; no full-path or physical success'}, calls


def audit(source, output, *, reuse_object4=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_height_audit')
    saved = read(source)
    proposals = saved['proposals']
    if saved['schema'] != 'source_first_slot_layout_v1' or not 1 <= len(proposals) <= 8:
        raise ValueError('at_most_eight_existing_slot_scenes_required')
    if whole.HEIGHTS != PROFILES:
        raise ValueError('original_whole_path_profiles_changed')
    frozen = [binding(p) for p in (release.__file__, whole.__file__, endpoints.__file__,
        'resimpair_pickplace/cartesian_reachability.py', 'resimpair_grasp/motion.py',
        'resimpair_pickplace/stability_motion.py', '.venv/lib/python3.12/site-packages/mplib/planner.py')]
    protocol = {'schema': VERSION, 'implementation': binding(__file__), 'source': binding(source),
        'frozen_implementations': frozen, 'profiles': PROFILES, 'maximum_scenes': 8,
        'maximum_arms': 2, 'maximum_candidates_per_cache': 12, 'native_IK_initializations': 8,
        'native_IK_threshold': .001, 'controller_workspace': endpoints.runtime_workspace(),
        'common_prefix': 'exact source qpos retained in existing endpoint reports',
        'random_control': 'all profiles share the same native random initialization bank; no changed seed draws',
        'new_source_pose_samples': 0, 'new_goal_samples': 0, 'new_grasps': 0,
        'physics_steps': 0, 'GPU_jobs': 0, 'API_calls': 0, 'default_promoted': False}
    if reuse_object4 is not None:
        protocol['reuse_object4_probe'] = binding(Path(reuse_object4) / 'RESULT.json')
    write(output / 'PROTOCOL.json', protocol)
    rows = []; calls = 0; classifications = Counter(); began = time.monotonic()
    for proposal in proposals:
        number = proposal['proposal']; cohort = read(resolve(proposal['cohort']))
        cases = cohort['cases']
        if (not cases or any(c['partition'] != 'development' for c in cases)
                or len({digest(c['scene']) for c in cases}) != 1):
            raise ValueError('one_fixed_development_scene_per_slot_required')
        scene = read(resolve(cases[0]['scene']))
        if len(scene['robot_roots']) != 2:
            raise ValueError('exactly_two_original_arms_required')
        tasks = [read(resolve(c['task'])) for c in cases]
        legacy_result = read(resolve(proposal['endpoint_result']))
        selector = endpoints.WorkspaceReleaseSelector(scene['robot_roots'], scene['seed'])
        operation_rows = []; by_key = {}
        for op_number, (task_id, operation) in enumerate(endpoints.unique_operations(tasks)):
            prior_row = legacy_result['checked'][op_number]
            if prior_row['object'] != operation['object']:
                raise ValueError('saved_endpoint_operation_order_mismatch')
            prior = read(resolve(prior_row['proof']))
            reused = None
            if reuse_object4 is not None and operation['object'] == 'object_4':
                reused = read(Path(reuse_object4) / f'{number:02d}.json')
                if reused['old_endpoint_proof'] != prior_row['proof']:
                    raise ValueError('reused_probe_requires_exact_legacy_proof')
            report, count = compare_operation(scene, operation, prior, selector, reused)
            calls += count; report['old_endpoint_proof'] = prior_row['proof']
            report['scene'] = cases[0]['scene']; report['workspace_passed'] = prior_row['workspace_passed']
            path = output / f'{number:02d}' / f'{op_number:02d}_{operation["object"]}.json'
            write(path, report)
            summary = {k: report[k] for k in ('object', 'legacy_passed', 'profile_chain_passed', 'controlled_height_false_negative')}
            summary.update(task=task_id, proof=binding(path), new_endpoint_IK_calls=count)
            operation_rows.append(summary); by_key[operation_key(operation)] = summary
            classifications.update(r['classification'] for r in report['rows'])
        task_rows = [{'task': t['requested_task_id'],
            'legacy_passed': all(by_key[operation_key(op)]['legacy_passed'] for op in t['operations']),
            'profile_endpoint_passed': all(by_key[operation_key(op)]['profile_chain_passed'] for op in t['operations'])}
            for t in tasks]
        row = {'proposal': number, 'scene': cases[0]['scene'], 'cohort': proposal['cohort'],
            'operations': operation_rows, 'tasks': task_rows, 'physical_success': 'unmeasured'}
        write(output / f'{number:02d}' / 'RESULT.json', row); rows.append(row)
        print({'proposal': number, 'old_passed': [r['object'] for r in operation_rows if r['legacy_passed']],
               'profile_endpoints_passed': [r['object'] for r in operation_rows if r['profile_chain_passed']],
               'controlled_height_false_negatives': [r['object'] for r in operation_rows if r['controlled_height_false_negative']]}, flush=True)
    for ref in frozen:
        resolve(ref)
    result = {'schema': VERSION, 'rows': rows, 'classifications': dict(classifications),
        'new_endpoint_IK_calls': calls, 'seconds': time.monotonic() - began,
        'frozen_implementations_unchanged': True, 'default_promoted': False,
        'physical_success': 'unmeasured', 'physics_steps': 0, 'GPU_jobs': 0, 'API_calls': 0}
    write(output / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-object4', type=Path)
    args = parser.parse_args()
    audit(args.source, args.output, reuse_object4=args.reuse_object4)
