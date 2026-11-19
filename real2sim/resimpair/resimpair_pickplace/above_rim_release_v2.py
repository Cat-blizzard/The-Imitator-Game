"""Opt-in profile-aligned endpoint selection for the retained low-drop worker.

Only the three original whole-path profiles are candidates. The original
release geometry, workspace, IK, complete mesh guard and physical acceptance
remain authoritative. A profile is committed together with its arm and grasp;
the existing controller consumes that same profile before physical movement.
No historical worker or evaluator is edited or made to opt in automatically.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_tasks.source_first_reachable_layout_v3 import runtime_workspace, within_workspace
from . import above_rim_release_v1 as release
from . import whole_path_motion_v1 as whole
from .cartesian_reachability import INITIAL, pose

VERSION = 'profile_aligned_above_rim_release_v2'
PROFILES = ((.20, .18), (.12, .12), (.26, .24))


def profile(value):
    try:
        array = np.asarray(value, float)
    except (TypeError, ValueError) as error:
        raise ValueError('one_original_whole_path_height_profile_required') from error
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError('one_original_whole_path_height_profile_required')
    result = tuple(map(float, array))
    if whole.HEIGHTS != PROFILES or result not in PROFILES:
        raise ValueError('one_original_whole_path_height_profile_required')
    return result


def retained_bindings():
    from resimpair_grasp import motion
    return {'release_geometry': binding(release.__file__), 'whole_path': binding(whole.__file__),
            'native_mesh_guard': binding(motion.__file__),
            'endpoint_ik': binding(Path(__file__).with_name('cartesian_reachability.py')),
            'workspace_adapter': binding(Path(__file__).parents[1] / 'resimpair_tasks/source_first_reachable_layout_v3.py')}


class ReleaseSelector(release.ReleaseSelector):
    def ik(self, arm, target, seed_qpos):
        if not within_workspace(target):
            return None, 'outside_retained_complete_path_workspace'
        return super().ik(arm, target, seed_qpos)

    def _steps(self, arm, stages, seed_qpos):
        q = seed_qpos.copy(); checks = []; cost = 0.
        for name, target in stages:
            next_q, status = self.ik(arm, target, q)
            checks.append({'stage': name, 'world_tcp': target.tolist(), 'status': status,
                           'qpos': None if next_q is None else next_q.tolist()})
            if next_q is None:
                return checks, None, cost
            cost += float(np.linalg.norm(next_q[:7] - q[:7])); q = next_q
        return checks, q, cost

    def select(self, scene, operation, container=None):
        from resimpair_grasp.cartesian_clearance import VerticalClearance
        import trimesh
        if len(self.planners) != 2:
            raise ValueError('two_retained_arms_required')
        for heights in PROFILES:
            profile(heights)
        item = next(o for o in scene['objects'] if o['id'] == operation['object'])
        source = np.asarray(item.get('measured_world_transform', pose(item['position'], item['yaw'])), float)
        cache = read(resolve(scene['grasp_caches'][item['id']]))
        candidates = cache['candidates'][:12]
        if len({c['id'] for c in candidates}) != len(candidates):
            raise ValueError('unique_original_cache_candidate_ids_required')
        obstacle = None
        if container is not None:
            descriptor = read(resolve(container['container_descriptor']))
            obstacle = VerticalClearance(trimesh.load(resolve(descriptor['collision']), force='mesh'))
            container_T = pose(container['position'], container.get('yaw', 0.))
        decisions = []; actual_ik_calls = 0
        for arm in (0, 1):
            for candidate in candidates:
                base = {'arm': arm, 'candidate': candidate['id'], 'accepted': False, 'checks': []}
                local = np.asarray(candidate['tcp_in_asset'], float)
                try:
                    desired, geometry = release.release_geometry(scene, operation, local)
                except ValueError as error:
                    for heights in PROFILES:
                        decisions.append({**deepcopy(base), 'height_profile_m': list(heights), 'reason': str(error)})
                    continue
                base['release_geometry'] = geometry
                grasp, target = source @ local, desired @ local
                if obstacle is not None:
                    ok, clearance, reason = obstacle.check(np.linalg.inv(container_T) @ target,
                                                           -container['position'][2], .002)
                    base['container_hand_clearance'] = {'passed': bool(ok), 'reason': reason,
                                                        'table_clearance_m': clearance}
                    if not ok:
                        for heights in PROFILES:
                            decisions.append({**deepcopy(base), 'height_profile_m': list(heights)})
                        continue
                approach = grasp.copy(); approach[2, 3] += .17
                prefix, grasp_q, source_cost = self._steps(arm, [('pregrasp', approach), ('grasp', grasp)], INITIAL)
                actual_ik_calls += len(prefix)
                # Source IK is shared. Every profile restarts from exactly that
                # same successful grasp, never a different arm/candidate/branch.
                for heights in PROFILES:
                    row = {**deepcopy(base), 'height_profile_m': list(heights), 'checks': deepcopy(prefix),
                           'source_prefix_shared_across_profiles': True}
                    if grasp_q is not None:
                        lift = grasp.copy(); lift[2, 3] += heights[0]
                        above = target.copy(); above[2, 3] += heights[1]
                        tail, end_q, cost = self._steps(arm, [('lift', lift), ('above_goal', above), ('release', target)], grasp_q)
                        actual_ik_calls += len(tail); row['checks'].extend(tail)
                        if end_q is not None:
                            row.update(accepted=True, joint_travel_proxy=source_cost + cost)
                    decisions.append(row)
        return {'schema': VERSION, 'passed': any(r['accepted'] for r in decisions), 'decisions': decisions,
            'height_profiles_m': [list(h) for h in PROFILES], 'robot_bindings': self.bindings,
            'workspace': runtime_workspace(), 'original_candidate_count': len(candidates),
            'actual_endpoint_IK_calls': actual_ik_calls, 'maximum_candidates_per_cache': 12,
            'source_pose': 'measured_full_SE3' if 'measured_world_transform' in item else 'declared_scene_SE3',
            'physical_success': 'unmeasured',
            'scope': 'same arm/grasp/profile endpoint IK only; original complete path guard and physical acceptance still required'}


def ordered_choices(report):
    """Keep up to three distinct grasps per arm, with their eligible profiles."""
    groups = {}
    for row in report['decisions']:
        if not row['accepted']:
            continue
        heights = profile(row['height_profile_m']); arm = row['arm']
        if arm not in (0, 1):
            raise ValueError('original_two_arm_choice_required')
        group = groups.setdefault((arm, row['candidate']), {'arm': arm, 'candidate': row['candidate'], 'profiles': []})
        if any(tuple(r['height_profile_m']) == heights for r in group['profiles']):
            raise ValueError('duplicate_arm_candidate_profile')
        group['profiles'].append(row)
    per_arm = {}
    for arm in (0, 1):
        values = [g for g in groups.values() if g['arm'] == arm]
        for group in values:
            group['joint_travel_proxy'] = min(r['joint_travel_proxy'] for r in group['profiles'])
            group['profiles'].sort(key=lambda r: PROFILES.index(tuple(r['height_profile_m'])))
        per_arm[arm] = sorted(values, key=lambda g: g['joint_travel_proxy'])[:whole.MAX_GRASPS_PER_ARM]
    arm_order = sorted(per_arm, key=lambda a: per_arm[a][0]['joint_travel_proxy'] if per_arm[a] else float('inf'))
    return [per_arm[a][i] for i in range(whole.MAX_GRASPS_PER_ARM) for a in arm_order if i < len(per_arm[a])]


def prepare(rec, task):
    """Select a guarded route; no endpoint-only proposal can bind execution."""
    scene, result = deepcopy(rec.scene_spec), deepcopy(task)
    selector = ReleaseSelector(scene['robot_roots'], scene['seed'])
    choices = {}
    for operation in result['operations']:
        index = rec.object_ids.index(operation['object'])
        source = whole.array(rec.objects[index].pose.to_transformation_matrix()).reshape(4, 4)
        item = next(o for o in scene['objects'] if o['id'] == operation['object'])
        item['measured_world_transform'] = source.tolist()
        support = next((s for s in scene['supports'] if s['id'] == operation['target']['support']), None)
        report = selector.select(scene, operation, support if support and 'container_descriptor' in support else None)
        cache_ref = scene['grasp_caches'][operation['object']]; cache = read(resolve(cache_ref))
        by_id = {c['id']: c for c in cache['candidates'][:12]}
        folder = rec.directory / 'whole_path_selection' / operation['id']
        write(folder / 'endpoint_proposals.json', report)
        guards = {}; decisions = []; selected = None; began = time.monotonic()
        for group in ordered_choices(report):
            arm, cid = group['arm'], group['candidate']; candidate = by_id[cid]
            if arm not in guards:
                guard = whole.make_solver(rec, arm, folder / f'arm_{arm}_motion')
                # Dynamic lookup preserves existing certified-prefix adapters.
                guard.__class__ = release.DropMotion; guard.install(index); guards[arm] = guard
            guard = guards[arm]
            for proposal in group['profiles']:
                if time.monotonic() - began >= whole.MAX_OPERATION_SECONDS:
                    break
                heights = profile(proposal['height_profile_m'])
                try:
                    desired, geometry = release.release_geometry(scene, operation, np.asarray(candidate['tcp_in_asset']))
                    proof = whole.preview_route(guard, source @ np.asarray(candidate['tcp_in_asset']), desired, candidate, heights)
                    if proof.get('passed') is not True or profile(proof['height_profile_m']) != heights:
                        raise ValueError('complete_preview_must_certify_the_selected_profile')
                    proof['release_geometry'] = geometry
                    selected = (arm, candidate, heights)
                    decisions.append({'arm': arm, 'candidate': cid, **proof}); break
                except (RuntimeError, ValueError, TimeoutError) as error:
                    decisions.append({'arm': arm, 'candidate': cid, 'height_profile_m': list(heights),
                                      'passed': False, 'reason': str(error)})
            if selected is not None or time.monotonic() - began >= whole.MAX_OPERATION_SECONDS:
                break
        write(folder / 'selection.json', {'schema': VERSION, 'decisions': decisions,
            'endpoint_proposals': binding(folder / 'endpoint_proposals.json'),
            'selected': None if selected is None else {'arm': selected[0], 'candidate': selected[1]['id'],
                                                      'height_profile_m': list(selected[2])},
            'elapsed_seconds': time.monotonic() - began, 'no_physical_retry': True,
            'scope': 'initial-state geometric preview; measured execution must replan and pass the original guard'})
        if selected is None:
            raise RuntimeError('no_complete_pick_place_proposal:' + operation['id'])
        arm, candidate, heights = selected
        operation['arm'] = arm; choices[operation['id']] = heights
        signature = {**cache['signature'], 'whole_path_selector': binding(__file__), 'parent_cache': cache_ref,
            'operation': operation, 'height_profile_m': list(heights), 'profile_selection_schema': VERSION,
            'selection': binding(folder / 'selection.json'), 'retained_implementations': retained_bindings()}
        selected_cache = {**cache, 'signature': signature, 'key': digest(signature), 'candidates': [candidate]}
        path = folder / 'selected_cache.json'; write(path, selected_cache)
        scene['grasp_caches'][operation['object']] = binding(path)
    rec.scene_spec['grasp_caches'] = scene['grasp_caches']
    rec.whole_path_choices = choices
    return result


def execution_choice(rec, task, arm, directory):
    """Fail closed before constructing a motion solver if its triple changed."""
    op_id = Path(directory).name.removesuffix('_motion')
    operation = next((o for o in task['operations'] if o['id'] == op_id), None)
    if operation is None:
        raise ValueError('declared_operation_required_for_profile_binding')
    choices = getattr(rec, 'whole_path_choices', {})
    if op_id not in choices:
        raise ValueError('guarded_profile_choice_required_before_execution')
    heights = profile(choices[op_id])
    cache_ref = rec.scene_spec['grasp_caches'][operation['object']]
    cache = read(resolve(cache_ref)); signature = cache['signature']
    expected_operation = {**deepcopy(operation), 'arm': arm}
    if (signature.get('profile_selection_schema') != VERSION
            or signature.get('whole_path_selector') != binding(__file__)
            or signature.get('operation') != expected_operation
            or profile(signature.get('height_profile_m', ())) != heights
            or cache.get('key') != digest(signature) or len(cache['candidates']) != 1):
        raise ValueError('execution_arm_grasp_profile_must_match_guarded_selection')
    for ref in signature['retained_implementations'].values():
        resolve(ref)
    if signature['retained_implementations'] != retained_bindings():
        raise ValueError('retained_release_and_native_guards_must_be_unchanged')
    candidate = cache['candidates'][0]
    parent = read(resolve(signature['parent_cache']))
    if not any(c == candidate for c in parent['candidates'][:12]):
        raise ValueError('execution_grasp_must_be_in_original_first_twelve')
    selection = read(resolve(signature['selection']))
    expected = {'arm': arm, 'candidate': candidate['id'], 'height_profile_m': list(heights)}
    if selection.get('schema') != VERSION or selection.get('selected') != expected:
        raise ValueError('execution_requires_bound_complete_path_selection')
    return {'schema': VERSION, 'operation': op_id, **expected, 'selected_cache': cache_ref,
            'complete_path_selection': signature['selection'], 'physical_success': 'unmeasured'}


@contextmanager
def _worker_hooks(task):
    """Change dispatch only; preserve the retained driver's adapter order."""
    from . import stability_motion, transport_fallback_v2
    operations = task.get('operations', [])
    if not operations or len({o['id'] for o in operations}) != len(operations):
        raise ValueError('nonempty_unique_declared_operations_required')
    original_prepare, original_solver = release.prepare, stability_motion.make_solver
    def bound_solver(rec, arm, directory):
        motion_class = transport_fallback_v2.TransportFallbackMotion
        if not issubclass(motion_class, release.DropMotion):
            raise ValueError('profile_binding_requires_retained_whole_path_worker_context')
        receipt = execution_choice(rec, task, arm, directory)
        guard = original_solver(rec, arm, directory)
        guard.height_profile = profile(receipt['height_profile_m'])
        write(Path(directory) / 'height_profile_execution.json', receipt)
        return guard
    # v1.install resolves its module global prepare at entry, so even an older
    # production driver's nested install uses this adapter, not the fixed gate.
    release.prepare, stability_motion.make_solver = prepare, bound_solver
    try:
        yield
    finally:
        release.prepare, stability_motion.make_solver = original_prepare, original_solver


@contextmanager
def install(task):
    """Replace v1.install at the same point in a single rolling worker.

    The original worker must still enter whole_path_motion_v1.install_for_worker.
    This process-local adapter never calls an actuator or allocates an attempt.
    Use install_for_worker when wrapping a driver that enters v1.install itself.
    """
    with _worker_hooks(task), release.install(task):
        yield


@contextmanager
def install_for_worker(root, task):
    """Wrap an old production driver while retaining its v1 installation order.

    Its original v1.install installs the measured view, release controller and
    motion class in the same position as before. That install sees the new
    prepare dispatch. Installing the drop class early would change the order
    of other retained adapters, such as payload_descent_v2.
    """
    from .runner import verify
    from .rolling_pickplace_v3 import VERSION as ROLLING
    spec = verify(Path(root))
    if (spec.get('candidate_release_selector') != VERSION
            or spec.get('candidate_motion_override') != whole.VERSION
            or spec.get('execution_selection_policy') != ROLLING
            or spec.get('motion_policy') != 'transport_fallback_v2'):
        raise ValueError('explicit_profile_selector_and_retained_rolling_worker_required')
    if spec['code'].get('resimpair_pickplace/above_rim_release_v2.py') != binding(__file__):
        raise ValueError('profile_selector_source_not_frozen')
    with _worker_hooks(task):
        yield
