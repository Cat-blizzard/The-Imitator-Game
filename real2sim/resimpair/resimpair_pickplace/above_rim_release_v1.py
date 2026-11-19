"""Explicit low-drop placement, retaining final goals and all native guards.

This is a versioned, process-local adapter for the frozen pick/place worker.
No physics actor is moved by this module. A release waypoint is not a final
goal: only the pre-opening controller target changes, never the evaluator.
"""
from contextlib import contextmanager
from functools import lru_cache
import copy
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import binding, digest, read, resolve, write
from . import whole_path_motion_v1 as retained
from .cartesian_reachability import ArmSelector as OriginalSelector, INITIAL, pose

VERSION = 'explicit_above_rim_release_v1'
MODE = 'drop_from_above'


def policy(operation):
    cfg = operation.get('release', {})
    if not cfg or cfg.get('mode') == 'supported_place':
        return None
    if cfg.get('mode') != MODE or cfg.get('allow_drop') is not True:
        raise ValueError('explicit_drop_authorization_required_in_task')
    if cfg.get('fragile') is not False:
        raise ValueError('drop_requires_explicit_nonfragile_assumption')
    values = [cfg.get('bottom_above_rim_m', .03), cfg.get('hand_above_rim_m', .01),
              cfg.get('maximum_fall_m', .15)]
    if not np.isfinite(values).all() or not (.005 <= values[0] <= .06 and .002 <= values[1] <= .04
                                          and values[0] <= values[2] <= .20):
        raise ValueError('bounded_low_drop_policy_required')
    if operation['target']['support'] == 'table':
        raise ValueError('this_drop_policy_requires_a_container')
    return dict(zip(('bottom_above_rim_m', 'hand_above_rim_m', 'maximum_fall_m'), map(float, values)))


@lru_cache(maxsize=128)
def mesh_vertices(path, sha256):
    import trimesh
    mesh = trimesh.load(resolve({'path': path, 'sha256': sha256}), force='mesh', process=False)
    value = np.asarray(mesh.vertices, float)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value) or not np.isfinite(value).all():
        raise ValueError('finite_native_mesh_required')
    return value


@lru_cache(maxsize=1)
def hand_vertices():
    from resimpair_tabletop.geometry import panda_meshes
    return np.vstack([m.vertices for _, _, m in panda_meshes()[0]])


def raised_pose(final_object_T, tcp_in_object, object_vertices, hand_points, rim_z, cfg):
    """Raise enough for BOTH the payload bottom and the open hand, in metres."""
    final, local = np.asarray(final_object_T, float), np.asarray(tcp_in_object, float)
    vertices, hand = np.asarray(object_vertices, float), np.asarray(hand_points, float)
    if final.shape != (4, 4) or local.shape != (4, 4) or not all(np.isfinite(x).all() for x in (final, local, vertices, hand)):
        raise ValueError('finite_release_geometry_required')
    tcp = final @ local
    payload_bottom = float((vertices @ final[:3, :3].T + final[:3, 3])[:, 2].min())
    hand_bottom = float((hand @ tcp[:3, :3].T + tcp[:3, 3])[:, 2].min())
    lift = max(0., rim_z + cfg['bottom_above_rim_m'] - payload_bottom,
               rim_z + cfg['hand_above_rim_m'] - hand_bottom)
    if not np.isfinite(lift) or lift > cfg['maximum_fall_m']:
        raise ValueError('required_drop_exceeds_declared_maximum')
    desired = final.copy(); desired[2, 3] += lift
    return desired, {'release_height_over_final_pose_m': lift, 'container_rim_world_z_m': float(rim_z),
        'payload_bottom_above_rim_m': payload_bottom + lift - rim_z,
        'open_hand_bottom_above_rim_m': hand_bottom + lift - rim_z,
        'ballistic_fall_time_upper_proxy_s': float(np.sqrt(2 * lift / 9.81)),
        'final_target_unchanged': True}


def release_geometry(scene, operation, tcp_in_object):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import targets
    final = pose(np.asarray(targets(scene, {'operations': [operation]})[operation['object']]) + [0, 0, .002],
                 operation['target']['yaw'])
    cfg = policy(operation)
    if cfg is None:
        return final, {'mode': 'supported_place', 'final_target_unchanged': True}
    item = next(o for o in scene['objects'] if o['id'] == operation['object'])
    asset = read(resolve(item['asset_descriptor']))
    vertices = np.vstack([mesh_vertices(**ref) for ref in asset['collisions']])
    support = next(s for s in scene['supports'] if s['id'] == operation['target']['support'])
    descriptor = read(resolve(support['container_descriptor']))
    ref = descriptor['collision']; container = mesh_vertices(**ref)
    container_T = pose(support['position'], support.get('yaw', 0.))
    rim = float((container @ container_T[:3, :3].T + container_T[:3, 3])[:, 2].max())
    desired, proof = raised_pose(final, tcp_in_object, vertices, hand_vertices(), rim, cfg)
    height_over_goal = float(desired[2, 3] - (final[2, 3] - .002))
    if height_over_goal > cfg['maximum_fall_m']:
        raise ValueError('release_waypoint_exceeds_contract_height')
    proof['release_height_over_goal_m'] = height_over_goal
    return desired, {'mode': MODE, 'policy': cfg, **proof}


class ReleaseSelector(OriginalSelector):
    def select(self, scene, operation, container=None):
        from resimpair_grasp.cartesian_clearance import VerticalClearance
        import trimesh
        item = next(x for x in scene['objects'] if x['id'] == operation['object'])
        start = np.asarray(item.get('measured_world_transform', pose(item['position'], item['yaw'])), float)
        cache = read(resolve(scene['grasp_caches'][item['id']]))
        obstacle = None
        if container is not None:
            descriptor = read(resolve(container['container_descriptor']))
            obstacle = VerticalClearance(trimesh.load(resolve(descriptor['collision']), force='mesh'))
            container_T = pose(container['position'], container.get('yaw', 0.))
        decisions = []
        for arm in range(len(self.planners)):
            accepted = 0
            for candidate in cache['candidates'][:12]:
                row = {'arm': arm, 'candidate': candidate['id'], 'accepted': False, 'checks': []}
                local = np.asarray(candidate['tcp_in_asset'])
                try:
                    desired, proof = release_geometry(scene, operation, local)
                except ValueError as error:
                    row['reason'] = str(error); decisions.append(row); continue
                row['release_geometry'] = proof
                grasp, release = start @ local, desired @ local
                if obstacle is not None:
                    ok, clearance, reason = obstacle.check(np.linalg.inv(container_T) @ release, -container['position'][2], .002)
                    row['container_hand_clearance'] = {'passed': bool(ok), 'reason': reason, 'table_clearance_m': clearance}
                    if not ok:
                        decisions.append(row); continue
                stages = [('pregrasp', grasp.copy()), ('grasp', grasp.copy()), ('above_goal', release.copy()), ('release', release.copy())]
                stages[0][1][2, 3] += .17; stages[2][1][2, 3] += .18
                q = INITIAL.copy(); cost = 0.
                for name, T in stages:
                    next_q, status = self.ik(arm, T, q)
                    row['checks'].append({'stage': name, 'world_tcp': T.tolist(), 'status': status,
                                          'qpos': None if next_q is None else next_q.tolist()})
                    if next_q is None: break
                    cost += float(np.linalg.norm(next_q[:7] - q[:7])); q = next_q
                else:
                    row.update(accepted=True, joint_travel_proxy=cost); accepted += 1
                decisions.append(row)
                if accepted >= 3: break
        return {'passed': any(d['accepted'] for d in decisions), 'decisions': decisions,
                'scope': 'release waypoint endpoint IK only; complete native path guard remains mandatory',
                'source_pose': 'measured_full_SE3', 'robot_bindings': self.bindings}


class DropMotion(retained.WholePathMotion):
    def plan(self, waypoint, stage, current=None):
        if not getattr(self, 'preview', False) and stage in {'above_bowl', 'release_above_rim'}:
            from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array
            import sapien
            op_id = self.directory.name.removesuffix('_motion')
            operation = self.rec.drop_operations[op_id]
            if policy(operation) is not None:
                actual = array(self.selected_actor.pose.to_transformation_matrix()).reshape(4, 4)
                tcp = array(self.agent.tcp.pose.to_transformation_matrix()).reshape(4, 4)
                local = np.linalg.inv(actual) @ tcp
                desired, evidence = release_geometry(self.rec.scene_spec, operation, local)
                T = desired @ local
                if stage == 'above_bowl': T[2, 3] += .18
                self.rec.drop_release_targets[op_id] = (desired, evidence)
                write(self.directory / f'drop_waypoint_{self.counter:03d}_{stage}.json',
                      {'operation': op_id, 'desired_object': desired.tolist(), 'world_tcp': T.tolist(), **evidence})
                waypoint = sapien.Pose(T)
        return super().plan(waypoint, stage, current)


@contextmanager
def install(task):
    """Single isolated worker only; never used in a shared Python thread pool."""
    from . import rolling_pickplace_v3 as rolling
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import release_control
    old_prepare, old_class, old_view = retained.prepare, retained.WholePathMotion, rolling.PlanningView
    old_readiness = release_control.readiness
    active = {}
    class MeasuredView(old_view):
        def __init__(self, rec, directory, caches):
            super().__init__(rec, directory, caches)
            from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array
            active['rec'] = rec
            rec.drop_operations = {o['id']: o for o in task['operations']}
            if not hasattr(rec, 'drop_release_targets'): rec.drop_release_targets = {}
            for oid, actor in zip(rec.object_ids, rec.objects):
                obj = next(o for o in self.scene_spec['objects'] if o['id'] == oid)
                obj['measured_world_transform'] = array(actor.pose.to_transformation_matrix()).reshape(4, 4).tolist()
    def readiness(actual, destination, held):
        rec = active.get('rec'); op_id = rec.stage.split(':')[0] if rec is not None else ''
        operation = rec.drop_operations.get(op_id) if rec is not None else None
        if operation is None or policy(operation) is None:
            return old_readiness(actual, destination, held)
        desired, evidence = rec.drop_release_targets[op_id]
        result = old_readiness(actual, desired[:3, 3], held)
        result.update(release_mode=MODE, final_destination=list(destination),
                      release_destination=desired[:3, 3].tolist(), geometry=evidence,
                      final_goal_acceptance_unchanged=True)
        return result
    retained.prepare, retained.WholePathMotion = prepare, DropMotion
    rolling.PlanningView, release_control.readiness = MeasuredView, readiness
    try: yield
    finally:
        retained.prepare, retained.WholePathMotion, rolling.PlanningView = old_prepare, old_class, old_view
        release_control.readiness = old_readiness


def prepare(rec, task):
    """Bind actual arm/grasp before declaring the physical event protocol."""
    scene = copy.deepcopy(rec.scene_spec); result = copy.deepcopy(task)
    selector = ReleaseSelector(scene['robot_roots'], scene['seed'])
    choices = {}; destination = retained.targets(scene, result)
    for op in result['operations']:
        index = rec.object_ids.index(op['object']); item = scene['objects'][index]
        source = retained.array(rec.objects[index].pose.to_transformation_matrix()).reshape(4, 4)
        desired = np.eye(4); desired[:3, :3] = Rotation.from_euler('z', op['target']['yaw']).as_matrix()
        desired[:3, 3] = np.array(destination[op['object']]) + [0, 0, .002]
        support = next((s for s in scene['supports'] if s['id'] == op['target']['support']), None)
        report = selector.select(scene, op, support if support and 'container_descriptor' in support else None)
        cache_ref = scene['grasp_caches'][op['object']]; cache = read(resolve(cache_ref))
        by_id = {c['id']: c for c in cache['candidates'][:12]}
        candidates = sorted((d for d in report['decisions'] if d['accepted']), key=lambda d: d['joint_travel_proxy'])
        folder = rec.directory / 'whole_path_selection' / op['id']
        write(folder / 'endpoint_proposals.json', report)
        guards = {}; decisions = []; selected = None; began = time.monotonic()
        # Alternate arm candidates before spending the entire budget on one arm.
        per_arm = {a: [d for d in candidates if d['arm'] == a][:retained.MAX_GRASPS_PER_ARM] for a in range(2)}
        ordered = [per_arm[a][i] for i in range(retained.MAX_GRASPS_PER_ARM) for a in sorted(per_arm, key=lambda a: min(
            [d['joint_travel_proxy'] for d in per_arm[a]] or [float('inf')])) if i < len(per_arm[a])]
        for row in ordered:
            arm = row['arm']; candidate = by_id[row['candidate']]
            if arm not in guards:
                guard = retained.make_solver(rec, arm, folder / f'arm_{arm}_motion')
                guard.__class__ = DropMotion; guard.install(index); guards[arm] = guard
            guard = guards[arm]
            for heights in retained.HEIGHTS:
                if time.monotonic() - began >= retained.MAX_OPERATION_SECONDS:
                    break
                try:
                    release_desired, release_proof = release_geometry(scene, op, np.asarray(candidate['tcp_in_asset']))
                    proof = retained.preview_route(guard, source @ np.asarray(candidate['tcp_in_asset']), release_desired, candidate, heights)
                    proof['release_geometry'] = release_proof
                    selected = (arm, candidate, heights)
                    decisions.append({'arm': arm, 'candidate': candidate['id'], **proof}); break
                except (RuntimeError, ValueError, TimeoutError) as error:
                    decisions.append({'arm': arm, 'candidate': candidate['id'], 'height_profile_m': list(heights),
                                      'passed': False, 'reason': str(error)})
            if selected is not None or time.monotonic() - began >= retained.MAX_OPERATION_SECONDS:
                break
        write(folder / 'selection.json', {'schema': VERSION, 'decisions': decisions,
              'selected': None if selected is None else {'arm': selected[0], 'candidate': selected[1]['id'],
                                                        'height_profile_m': list(selected[2])},
              'elapsed_seconds': time.monotonic() - began, 'no_physical_retry': True,
              'scope': 'initial-state geometric preview; measured execution must replan and pass the original guard'})
        if selected is None:
            raise RuntimeError('no_complete_pick_place_proposal:' + op['id'])
        arm, candidate, heights = selected; op['arm'] = arm; choices[op['id']] = heights
        signature = {**cache['signature'], 'whole_path_selector': binding(__file__), 'parent_cache': cache_ref,
                     'operation': op, 'height_profile_m': list(heights)}
        selected_cache = {**cache, 'signature': signature, 'key': digest(signature), 'candidates': [candidate]}
        path = folder / 'selected_cache.json'; write(path, selected_cache)
        scene['grasp_caches'][op['object']] = binding(path)
    rec.scene_spec['grasp_caches'] = scene['grasp_caches']
    rec.whole_path_choices = choices
    return result

