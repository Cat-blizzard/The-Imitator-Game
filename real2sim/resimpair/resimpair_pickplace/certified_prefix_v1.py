"""Reuse the checked, pre-contact prefix of a successful whole-route preview.

Only approach/grasp_reach are reusable. The physical scene must still be the
same, and the original dense path guard runs again before returning a path.
Once contact or any physical state changes, transport uses measured attachment
and ordinary planning. No physical state, planner threshold or evaluator changes.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write
from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array
from . import whole_path_motion_v1 as whole
from . import above_rim_release_v1 as drop

VERSION = 'same_state_certified_precontact_prefix_v1'
STAGES = ('approach', 'grasp_reach')


def owner(rec):
    while '_rec' in vars(rec):
        rec = vars(rec)['_rec']
    return rec


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def scene_dependencies(rec):
    scene = rec.scene_spec
    objects = [{k: o[k] for k in ('id', 'kind', 'shape', 'asset_descriptor', 'half_size') if k in o}
               for o in scene['objects']]
    supports = [{k: o[k] for k in ('id', 'shape', 'container_descriptor', 'position', 'yaw', 'half_size') if k in o}
                for o in scene['supports']]
    return {'objects': objects, 'supports': supports, 'robot_roots': scene['robot_roots'],
            'table_top_z': scene['table_top_z'],
            'control_timestep': float(rec.control_timestep),
            'control_modes': [a.control_mode for a in rec.agent.agents]}


def planner_settings(guard):
    return {name: json_value(getattr(guard.planner, name)) for name in
            ('joint_limits', 'joint_vel_limits', 'joint_acc_limits') if hasattr(guard.planner, name)}


def validate_prefix(record, *, state_hash, dependencies, arm, object_id, candidate,
                    descriptor, gripper, parent_cache, current, goal, stage):
    """Fail closed on stale paths; numerical identity tolerance is NOT a guard."""
    if record['schema'] != VERSION or stage not in STAGES:
        raise ValueError('unsupported_prefix_version_or_stage')
    if record['initial_state_sha256'] != state_hash:
        raise ValueError('prefix_physical_state_changed')
    for name, value in [('dependencies', dependencies), ('arm', arm), ('object_id', object_id),
                        ('candidate', candidate), ('descriptor', descriptor), ('gripper', gripper),
                        ('source_cache', parent_cache)]:
        if digest(record[name]) != digest(value):
            raise ValueError('prefix_binding_changed:' + name)
    piece = record['prefixes'][stage]
    q, target = np.asarray(current, float), np.asarray(goal, float)
    if (q.shape != (9,) or target.shape != (4, 4)
            or not np.isfinite(q).all() or not np.isfinite(target).all()):
        raise ValueError('finite_prefix_current_and_goal_required')
    # SAPIEN's float32 pose construction can differ by one final rounding.
    # The retained guard below still enforces its original path/goal tolerances.
    if not np.allclose(q, piece['current'], atol=1e-6, rtol=0):
        raise ValueError('prefix_initial_joint_configuration_changed')
    if not np.allclose(target, piece['goal'], atol=1e-6, rtol=0):
        raise ValueError('prefix_goal_changed')
    return piece


def capture_preview(original, guard, grasp, desired, candidate, heights):
    root_rec = owner(guard.rec)
    initial_state = whole._state_hash(root_rec)
    captured = {}
    original_plan = guard.plan
    prior_instance = vars(guard).get('plan')
    def capture(waypoint, stage, current=None):
        q = array(guard.agent.robot.qpos).reshape(9).copy() if current is None else current.copy()
        target = np.asarray(waypoint.to_transformation_matrix()).copy()
        path = original_plan(waypoint, stage, current)
        if stage in STAGES:
            captured[stage] = {'current': q.tolist(), 'goal': target.tolist(),
                'path': json_value(deepcopy(path)), 'original_guard': binding(path['guard_evidence'])}
        return path
    guard.plan = capture
    try:
        proof = original(guard, grasp, desired, candidate, heights)
    finally:
        if prior_instance is None:
            del guard.plan
        else:
            guard.plan = prior_instance
    if not proof.get('passed') or set(captured) != set(STAGES):
        raise RuntimeError('complete_checked_route_with_two_prefixes_required')
    if whole._state_hash(root_rec) != initial_state:
        raise RuntimeError('prefix_capture_changed_physics')
    oid = guard.rec.object_ids[guard.object_index]
    item = next(o for o in guard.rec.scene_spec['objects'] if o['id'] == oid)
    cache_ref = guard.rec.scene_spec['grasp_caches'][oid]
    cache = read(resolve(cache_ref))
    record = {'schema': VERSION, 'implementation': binding(__file__),
        'initial_state_sha256': initial_state, 'initial_frame': len(root_rec.rows),
        'dependencies': scene_dependencies(root_rec), 'planner_settings': planner_settings(guard),
        'arm': guard.arm, 'object_id': oid,
        'candidate': candidate, 'descriptor': item['asset_descriptor'],
        'gripper': cache['signature']['gripper'], 'source_cache': cache_ref,
        'prefixes': captured, 'route_proof': proof, 'predicted_attachment_reused': False}
    folder = root_rec.directory / 'certified_prefixes' / digest(record)[:20]
    path = folder / 'prefix.json'
    write(path, record)
    if not hasattr(root_rec, 'certified_precontact_prefixes'):
        root_rec.certified_precontact_prefixes = {}
    root_rec.certified_precontact_prefixes[(oid, guard.arm, candidate['id'])] = binding(path)
    return {**proof, 'certified_precontact_prefix': binding(path)}


@contextmanager
def install(task):
    """Install inside drop.install(task), in one isolated physics worker."""
    if not task.get('operations'):
        raise ValueError('nonempty_declared_task_required')
    old_preview, old_drop, old_whole = whole.preview_route, drop.DropMotion, whole.WholePathMotion
    if old_whole is not old_drop:
        raise ValueError('full_SE3_drop_adapter_context_required')

    class CertifiedMotion(old_drop):
        def plan(self, waypoint, stage, current=None):
            if getattr(self, 'preview', False) or self.use_attach or stage not in STAGES:
                return super().plan(waypoint, stage, current)
            root_rec = owner(self.rec)
            oid = self.rec.object_ids[self.object_index]
            cache = read(resolve(self.rec.scene_spec['grasp_caches'][oid]))
            candidates = cache.get('candidates', [])
            records = getattr(root_rec, 'certified_precontact_prefixes', {})
            ref = records.get((oid, self.arm, candidates[0]['id'])) if len(candidates) == 1 else None
            if ref is None:
                return super().plan(waypoint, stage, current)
            directory = self.directory / f'{self.counter:03d}_{stage}_certified_prefix'
            self.counter += 1
            q = array(self.agent.robot.qpos).reshape(9).copy() if current is None else current.copy()
            goal = np.asarray(waypoint.to_transformation_matrix()).copy()
            record = read(resolve(ref))
            item = next(o for o in self.rec.scene_spec['objects'] if o['id'] == oid)
            before = whole._state_hash(root_rec)
            try:
                resolve(record['implementation']); resolve(record['descriptor']); resolve(record['source_cache'])
                if digest(record['planner_settings']) != digest(planner_settings(self)):
                    raise ValueError('prefix_planner_limits_changed')
                for geometry in record['gripper'].get('geometry_bindings', []):
                    resolve(geometry)
                piece = validate_prefix(record, state_hash=before, dependencies=scene_dependencies(root_rec),
                    arm=self.arm, object_id=oid, candidate=candidates[0], descriptor=item['asset_descriptor'],
                    gripper=cache['signature']['gripper'], parent_cache=cache['signature'].get('parent_cache'),
                    current=q, goal=goal, stage=stage)
                prior = read(resolve(piece['original_guard']))
                if prior.get('accepted') is not True or prior.get('report', {}).get('passed') is not True:
                    raise ValueError('prefix_original_guard_not_accepted')
            except (ValueError, OSError, KeyError) as error:
                write(directory / 'reuse_rejection.json', {'reason': str(error), 'prefix': ref,
                    'state_sha256': before, 'fallback': 'ordinary_planning_and_unchanged_full_guard'})
                return super().plan(waypoint, stage, current)
            path = deepcopy(piece['path'])
            for key in ('position', 'velocity', 'acceleration', 'time'):
                if key in path:
                    path[key] = np.asarray(path[key], dtype=float)
            with root_rec.performance.section('certified_prefix_dense_guard'):
                report, dense, poses = self.check(path, q, goal, stage)
            if whole._state_hash(root_rec) != before:
                raise RuntimeError('certified_prefix_guard_changed_physics')
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / 'path.npz', positions=path['position'], current=q, goal=goal)
            if dense is not None:
                np.savez_compressed(directory / 'dense.npz', joints=dense, tcp=poses)
            decision = {'accepted': bool(report['passed']), 'report': report,
                'planner_status': path['status'], 'reused_prefix': ref,
                'original_guard': piece['original_guard'], 'actual_state_sha256': before,
                'stage': stage, 'no_new_planner_call': True, 'complete_preview_passed': True,
                'full_guard_repeated_without_threshold_change': True}
            write(directory / 'decision.json', decision)
            if not report['passed']:
                raise RuntimeError('certified_prefix_current_guard_rejected:' + stage)
            path['guard_evidence'] = str(directory / 'decision.json')
            return path

    whole.preview_route = lambda *args, **kwargs: capture_preview(old_preview, *args, **kwargs)
    drop.DropMotion = whole.WholePathMotion = CertifiedMotion
    try:
        yield
    finally:
        whole.preview_route, drop.DropMotion, whole.WholePathMotion = old_preview, old_drop, old_whole
