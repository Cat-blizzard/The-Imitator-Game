"""Bounded measured-state selection for multi-object pick/place programs.

Reuse the frozen whole-path proposal, exact collision guard and physical skill.
No body state is set. Each actual operation is committed before its first step.
Only unrequired ordering may change; failed geometry previews consume no rollout.
This version runs under the worker's read-only native articulation mesh access.
The exact before/after state guard remains unchanged.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import binding, read, resolve, write
from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array, targets
from . import whole_path_motion_v1 as retained

VERSION = 'rolling_state_v3'


def state_sha(rec):
    state = array(rec.unwrapped.get_state())
    return hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()


def ready_operations(task, completed):
    operations = task['operations']; ids = [o['id'] for o in operations]
    if not 1 <= len(ids) <= 4 or len(set(ids)) != len(ids):
        raise ValueError('one_to_four_unique_operations_required')
    if len({o['object'] for o in operations}) != len(ids):
        raise ValueError('regrasp_not_implemented')
    if not set(completed) <= set(ids):
        raise ValueError('unknown_completed_operation')
    edges = task.get('required_operation_precedence', [])
    if any(len(e) != 2 or e[0] not in ids or e[1] not in ids or e[0] == e[1] for e in edges):
        raise ValueError('invalid_required_precedence')
    remaining = [o for o in operations if o['id'] not in completed]
    ready = [o for o in remaining if all(a in completed for a, b in edges if b == o['id'])]
    if remaining and not ready:
        raise ValueError('cyclic_or_unsatisfied_precedence')
    return ready


class PlanningView:
    """Local metadata view; all measured bodies and robot state remain live."""
    def __init__(self, rec, directory, caches):
        self._rec = rec; self.directory = Path(directory)
        self.scene_spec = copy.deepcopy(rec.scene_spec)
        self.scene_spec['grasp_caches'] = copy.deepcopy(caches)
        for oid, actor in zip(rec.object_ids, rec.objects):
            obj = next(o for o in self.scene_spec['objects'] if o['id'] == oid)
            T = array(actor.pose.to_transformation_matrix()).reshape(4, 4)
            obj['position'] = T[:3, 3].tolist()
            obj['yaw'] = float(Rotation.from_matrix(T[:3, :3]).as_euler('xyz')[2])

    def __getattr__(self, name):
        return getattr(self._rec, name)


def operation_task(task, operation):
    """A one-operation protocol declares exactly its selected target.

    Other objects become non-targets only in this local fragment. The final
    protocol is rebuilt from the full task, never by concatenating fragments.
    """
    if not any(o['id'] == operation['id'] and o['object'] == operation['object'] for o in task['operations']):
        raise ValueError('unknown_operation_for_fragment')
    value = copy.deepcopy(task)
    value['operations'] = [copy.deepcopy(operation)]
    value['object_ids'] = [operation['object']]
    value['required_operation_precedence'] = []
    return value


def select_operation(rec, task, operation, directory, original_caches):
    """The retained geometry search on current state, with no physical retry."""
    before = state_sha(rec); frames = len(rec.rows)
    view = PlanningView(rec, directory, original_caches)
    one = operation_task(task, operation)
    try:
        selected = retained.prepare(view, one)
    finally:
        if state_sha(rec) != before or len(rec.rows) != frames:
            raise RuntimeError('geometry_preview_changed_physical_state')
    chosen = selected['operations'][0]
    report = Path(directory) / 'whole_path_selection' / operation['id'] / 'selection.json'
    return chosen, view.scene_spec['grasp_caches'][operation['object']], view.whole_path_choices, binding(report)


def target_occupancy(rec, operation, blocker, directory):
    """Exact native mesh overlap at the requested destination, not a rollout."""
    from mplib.pymp.collision_detection import fcl
    from data_generation_methods.human_to_robot_pipeline.strict_motion import matrix_to_pose7
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    before = state_sha(rec)
    oi = rec.object_ids.index(operation['object']); bi = rec.object_ids.index(blocker)
    movable = rec.objects[oi].get_first_collision_mesh(to_world_frame=False)
    obstacle = rec.objects[bi].get_first_collision_mesh(to_world_frame=True)
    if movable is None or obstacle is None:
        raise ValueError('native_mesh_required_for_occupancy_probe')
    a = directory / 'payload_local.obj'; b = directory / 'blocker_world.obj'
    movable.export(a); obstacle.export(b)
    ga = fcl.load_mesh_as_BVH(str(a), np.ones(3)); gb = fcl.load_mesh_as_BVH(str(b), np.ones(3))
    ca, cb = fcl.CollisionObject(ga), fcl.CollisionObject(gb)
    T = np.eye(4); T[:3, :3] = Rotation.from_euler('z', operation['target']['yaw']).as_matrix()
    T[:3, 3] = targets(rec.scene_spec, {'operations': [operation]})[operation['object']]
    ca.set_transformation(matrix_to_pose7(T))
    result = fcl.collide(ca, cb, fcl.CollisionRequest(num_max_contacts=1000, enable_contact=True))
    row = {'object': operation['object'], 'blocker': blocker, 'native_collision': bool(result.is_collision()),
           'frame': len(rec.rows), 'state_sha256': before, 'target_transform': T.tolist(),
           'payload_native_mesh': binding(a), 'blocker_native_mesh': binding(b),
           'scope': 'native surface collision necessary endpoint check; no claim of complete initial path feasibility'}
    if state_sha(rec) != before:
        raise RuntimeError('occupancy_probe_changed_physics')
    write(directory / 'evidence.json', row)
    return {**row, 'evidence': binding(directory / 'evidence.json')}


def execute(rec, task, *, release_policy, motion_policy):
    from .stability_motion import solve
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import planning_protocols
    original = copy.deepcopy(task); caches = copy.deepcopy(rec.scene_spec['grasp_caches'])
    completed = set(); executed = []; decisions = []
    # Full-task intent/goal/precedence exists before any action; actual effectors
    # are independently committed below before their own physical operation.
    write(rec.directory / 'initial_task_intent.json', original)
    write(rec.directory / 'rolling_execution_start.json', {'schema': VERSION, 'implementation': binding(__file__),
        'state_sha256': state_sha(rec), 'frame': len(rec.rows), 'created_unix': time.time(),
        'initial_task_intent': binding(rec.directory / 'initial_task_intent.json'), 'actor_pose_updates': 0})
    rec.whole_path_choices = {}
    clear = task.get('requested_task_id') == 'clear_then_place'
    occupancy = None
    if clear:
        if len(task['operations']) != 2:
            raise ValueError('two_operation_vacated_destination_required')
        blocker = task['operations'][0]['object']; dependent = task['operations'][1]
        occupancy = {'schema': 'measured_vacated_destination_v1', 'object': dependent['object'], 'blocker': blocker,
                     'before': target_occupancy(rec, dependent, blocker, rec.directory / 'occupancy_before')}
    try:
        while len(completed) < len(task['operations']):
            ready = ready_operations(task, completed); selected = None
            step = len(completed)
            for operation in ready:
                folder = rec.directory / 'rolling_selection' / f'{step:02d}_{operation["id"]}'
                began = time.monotonic()
                try:
                    chosen, cache, heights, evidence = select_operation(rec, task, operation, folder, caches)
                    selected = chosen, cache, heights, evidence
                    decisions.append({'step': step, 'operation': operation['id'], 'passed': True,
                                      'seconds': time.monotonic() - began, 'selection': evidence})
                    break
                except (RuntimeError, ValueError, TimeoutError) as error:
                    if str(error) == 'geometry_preview_changed_physical_state': raise
                    decisions.append({'step': step, 'operation': operation['id'], 'passed': False,
                                      'seconds': time.monotonic() - began, 'reason': str(error)})
            write(rec.directory / 'rolling_decision_snapshots' / f'{step:02d}.json',
                  {'rows': decisions, 'no_physical_retry': True})
            if selected is None:
                raise RuntimeError('no_ready_operation_has_complete_path')
            chosen, cache, heights, evidence = selected
            task['operations'] = [chosen if o['id'] == chosen['id'] else o for o in task['operations']]
            rec.scene_spec['grasp_caches'][chosen['object']] = cache
            rec.whole_path_choices.update(heights)
            one = operation_task(task, chosen)
            ids = rec.object_ids; supports = ids + [s['id'] for s in rec.scene_spec['supports']] + ['table']
            protocol_path = rec.directory / 'operation_commitments' / f'{step:02d}_events.json'
            write(protocol_path, planning_protocols.pick_place(rec.scene_spec, one, ids, supports, targets(rec.scene_spec, one)))
            commitment = {'schema': VERSION, 'operation': chosen, 'step_index': step,
                'before_frame': len(rec.rows), 'actual_state_sha256': state_sha(rec), 'created_unix': time.time(),
                'single_operation_event_protocol': binding(protocol_path), 'selection': evidence,
                'initial_task_intent': binding(rec.directory / 'initial_task_intent.json'),
                'remaining_operation_ids': [o['id'] for o in task['operations'] if o['id'] not in completed],
                'geometry_only_preview': True, 'physical_retries': 0,
                'protocol_scope': 'operation_fragment_not_global_non_target_invariants'}
            cp = rec.directory / 'operation_commitments' / f'{step:02d}.json'; write(cp, commitment)
            if occupancy is not None and chosen['id'] == dependent['id']:
                occupancy['after'] = target_occupancy(rec, chosen, blocker, rec.directory / 'occupancy_after')
                occupancy['after_complete_path_selection'] = evidence
            row = {'operation_id': chosen['id'], 'commitment': binding(cp), 'first_frame': len(rec.rows),
                   'last_frame': None, 'completed': False}
            executed.append(row)
            hold = rec.hold
            # Retained solve ends with 40 global settle steps. Keep that once,
            # just as the baseline multi-operation call, not once per operation.
            def operation_hold(steps, *args, **kwargs):
                if rec.stage == 'final_settle' and steps == 40 and step + 1 < len(task['operations']):
                    return None
                return hold(steps, *args, **kwargs)
            rec.hold = operation_hold
            try:
                solve(rec, one, 'cached_antipodal', release_policy=release_policy, motion_policy=motion_policy)
                row['completed'] = True; completed.add(chosen['id'])
            finally:
                rec.hold = hold; row['last_frame'] = len(rec.rows) - 1
                row['end_state_sha256'] = state_sha(rec)
                write(rec.directory / 'operation_outcomes' / f'{step:02d}.json', row)
    finally:
        write(rec.directory / 'executed_operations.json', {'operations': executed})
        write(rec.directory / 'rolling_decisions.json', {'rows': decisions, 'no_physical_retry': True})
        if occupancy is not None:
            write(rec.directory / 'goal_occupancy_probe.json', occupancy)
        write(rec.directory / 'rolling_execution_summary.json', {'schema': VERSION,
            'completed_operation_ids': sorted(completed), 'requested_operations': len(task['operations']),
            'geometry_preview_attempts': len(decisions), 'actor_pose_updates': 0,
            'reset_count': 1, 'physical_retries': 0, 'API_calls': 0})
    return task
