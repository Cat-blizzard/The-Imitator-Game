"""Opt-in complete geometric pick/place proposals before physical movement.

Search existing cached grasps, both arms and three bounded height profiles.
All paths use the retained native mesh guard. Preview attachment is geometry
only: no simulator state, contact, mass, friction or acceptance rule changes.
Actual execution re-plans from measured attachment with the retained controller.
"""
from contextlib import contextmanager
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import ROOT, binding, read, resolve, write, digest
from data_generation_methods.human_to_robot_pipeline.strict_motion import make_solver, world_fk
from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array, targets
from .continuation_motion_v3 import ContinuationMotion
from .shape_preserving_toppra_v2 import PlannerAdapter
from .transport_fallback_v2 import se3_transport_path
from .stability import opening_width

VERSION = 'whole_path_joint_selection_v1'
HEIGHTS = ((.20, .18), (.12, .12), (.26, .24))
MAX_GRASPS_PER_ARM = 3
MAX_OPERATION_SECONDS = 90


def geometric_transport(planner, root, current, goal, timestep=.05):
    """Bounded Cartesian geometry followed by unchanged command/path checks."""
    current, goal = np.asarray(current, float), np.asarray(goal, float)
    if current.shape != (9,) or goal.shape != (4, 4) or not np.isfinite(current).all() or not np.isfinite(goal).all():
        raise ValueError('finite_native_state_and_transform_required')
    start = world_fk(current[None, :7], root)[0]
    delta = goal[:3, 3] - start[:3, 3]
    rotation = Rotation.from_matrix(start[:3, :3].T @ goal[:3, :3]).as_rotvec()
    count = max(1, int(np.ceil(np.linalg.norm(delta) / .005)), int(np.ceil(np.linalg.norm(rotation) / .025)))
    if count > 320 or np.linalg.norm(delta) > 1.2:
        raise ValueError('bounded_transport_scope_exceeded')
    limits = np.asarray(planner.joint_limits, float)[:7]
    q = current[:7].copy(); values = [q.copy()]; traces = []; began = time.monotonic()
    for i in range(1, count + 1):
        target = start.copy(); target[:3, 3] += delta * i / count
        target[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(rotation * i / count).as_matrix()
        previous = q.copy(); attempts = []
        def residual(joints):
            actual = world_fk(joints[None], root)[0]
            return np.r_[actual[:3, 3] - target[:3, 3],
                         .15 * Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec(),
                         .00001 * (joints - previous)]
        for radius in (.16, .32):
            if time.monotonic() - began > 12:
                raise TimeoutError('bounded_geometric_transport_timeout')
            lower = np.maximum(limits[:, 0] + 1e-7, previous - radius)
            upper = np.minimum(limits[:, 1] - 1e-7, previous + radius)
            if np.any(lower >= upper):
                raise ValueError('empty_transport_trust_region')
            result = least_squares(residual, np.clip(previous, lower, upper), bounds=(lower, upper),
                                   max_nfev=50, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            actual = world_fk(result.x[None], root)[0]
            pe = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
            re = float(Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).magnitude())
            attempts.append({'radius_rad': radius, 'position_error_m': pe, 'rotation_error_rad': re})
            if pe <= .0002 and re <= .002:
                q = result.x.copy(); break
        else:
            raise RuntimeError(f'geometric_transport_waypoint_failed:{i}:{pe}:{re}')
        values.append(q.copy()); traces.append({'waypoint': i, 'attempts': attempts})
    times, positions, velocities, accelerations, duration = PlannerAdapter(planner).TOPP(np.asarray(values), timestep)
    return {'status': 'Success', 'position': positions, 'velocity': velocities, 'acceleration': accelerations,
            'time': times, 'duration': duration, 'geometry_trace': traces,
            'requires_retained_full_path_guard': True, 'policy': VERSION}


class WholePathMotion(ContinuationMotion):
    def refresh_attachment(self, stage):
        if not getattr(self, 'preview', False):
            return super().refresh_attachment(stage)

    def independent_table_collisions(self, full, tcp):
        if getattr(self, 'preview', False) and not self.use_attach:
            self.selected_world_pose = self.preview_object_pose.copy()
        return super().independent_table_collisions(full, tcp)

    def plan(self, pose, stage, current=None):
        import sapien
        if not getattr(self, 'preview', False):
            heights = getattr(self, 'height_profile', HEIGHTS[0])
            if stage in {'lift', 'above_bowl'}:
                modified = np.array(pose.to_transformation_matrix(), dtype=float)
                modified[2, 3] += heights[0] - .20 if stage == 'lift' else heights[1] - .18
                pose = sapien.Pose(modified)
        current = array(self.agent.robot.qpos).reshape(9).copy() if current is None else current.copy()
        try:
            return super().plan(pose, stage, current)
        except RuntimeError as original_error:
            if stage not in {'above_bowl', 'release_above_rim'} or not self.use_attach:
                raise
            goal = np.array(pose.to_transformation_matrix(), dtype=float)
            # A solver error is also a valid trigger for an alternative proposal.
            # Every proposal must still pass the unchanged full guard.
            for mode, propose in [('se3_bounded', se3_transport_path), ('se3_geometric', geometric_transport)]:
                directory = self.directory / f'{self.counter:03d}_{stage}_{mode}_wholepath'
                self.counter += 1; directory.mkdir()
                write(directory / 'request.json', {'stage': stage, 'current': current.tolist(),
                      'goal': goal.tolist(), 'attached': self.use_attach, 'preview': getattr(self, 'preview', False),
                      'trigger': str(original_error), 'frame': len(self.rec.rows)})
                try:
                    with self.rec.performance.section('motion_plan_wholepath_' + mode):
                        path = propose(self.planner, self.root, current, goal, self.rec.control_timestep)
                    with self.rec.performance.section('dense_physical_collision_guard'):
                        report, dense, poses = self.check(path, current, goal, stage)
                    np.savez_compressed(directory / 'path.npz', positions=path['position'], current=current, goal=goal)
                    if dense is not None:
                        np.savez_compressed(directory / 'dense.npz', joints=dense, tcp=poses)
                    write(directory / 'decision.json', {'accepted': report['passed'], 'report': report, 'policy': VERSION})
                    if report['passed']:
                        path['guard_evidence'] = str(directory / 'decision.json'); return path
                except Exception as error:
                    if not (directory / 'decision.json').exists():
                        write(directory / 'decision.json', {'accepted': False, 'error': repr(error), 'policy': VERSION})
            raise original_error


def _state_hash(rec):
    return hashlib.sha256(np.ascontiguousarray(array(rec.unwrapped.get_state())).tobytes()).hexdigest()


def preview_route(guard, grasp, desired, candidate, heights):
    """Complete planning sequence, no action commands or simulator changes."""
    import sapien
    from mplib.pymp.collision_detection import fcl
    source = array(guard.selected_actor.pose.to_transformation_matrix()).reshape(4, 4).copy()
    current = array(guard.agent.robot.qpos).reshape(9).copy()
    initial = world_fk(current[None, :7], guard.root)[0]
    before = _state_hash(guard.rec); guard.preview = True; guard.preview_object_pose = source.copy()
    guard.use_attach = False; evidence = []
    def move(T, stage):
        path = guard.plan(sapien.Pose(T), stage, current)
        current[:7] = path['position'][-1]
        evidence.append({'stage': stage, 'guard': binding(path['guard_evidence']),
                         'duration_seconds': float(path.get('duration', len(path['position']) * guard.rec.control_timestep))})
    try:
        approach = grasp.copy(); approach[2, 3] += .17
        move(approach, 'approach'); move(grasp, 'grasp_reach')
        # Predict the closed hand only for path feasibility. Actual grasping is
        # tested later by the native physical controller, without retries.
        current[7:] = float(candidate['width_m']) / 2
        achieved_grasp = world_fk(current[None, :7], guard.root)[0]
        guard.payload_to_tcp = np.linalg.inv(achieved_grasp) @ source
        geometry = fcl.load_mesh_as_BVH(str(guard.selected_mesh_path), np.ones(3))
        guard.payload_collision = fcl.CollisionObject(geometry)
        guard.objects.append((geometry, guard.payload_collision)); guard.use_attach = True
        lift = achieved_grasp.copy(); lift[2, 3] += heights[0]; move(lift, 'lift')
        release = desired @ np.linalg.inv(guard.payload_to_tcp)
        above = release.copy(); above[2, 3] += heights[1]; move(above, 'above_bowl')
        move(release, 'release_above_rim')
        guard.use_attach = False; guard.preview_object_pose = desired.copy(); guard.check_stage = 'retreat'
        actual_tcp = world_fk(current[None, :7], guard.root)[0]
        width = opening_width(float(current[7:].sum())); opening = []
        for aperture in np.linspace(current[7:].sum(), width, 17):
            full = current.copy(); full[7:] = aperture / 2
            collisions = guard.independent_table_collisions(full, actual_tcp)
            opening.append({'width_m': float(aperture), 'collisions': collisions})
            if collisions:
                raise RuntimeError('preview_opening_sweep_collision')
        current[7:] = width / 2
        retreat = actual_tcp.copy(); retreat[2, 3] += .18; move(retreat, 'retreat')
        current[7:] = .04
        if guard.independent_table_collisions(current, world_fk(current[None, :7], guard.root)[0]):
            raise RuntimeError('preview_full_open_collision')
        move(initial, 'return')
        return {'passed': True, 'paths': evidence, 'opening_sweep': opening,
                'height_profile_m': list(heights), 'predicted_width_m': float(candidate['width_m']),
                'physical_grasp_status': 'unmeasured', 'state_hash_before': before}
    finally:
        guard.preview = False; guard.use_attach = False; guard.payload_to_tcp = None; guard.payload_collision = None
        guard.planner.robot.set_qpos(array(guard.agent.robot.qpos).reshape(9).copy(), True)
        if _state_hash(guard.rec) != before:
            raise RuntimeError('preview_changed_physical_state')


def prepare(rec, task):
    """Bind actual arm/grasp before declaring the physical event protocol."""
    from .cartesian_reachability import ArmSelector
    scene = copy.deepcopy(rec.scene_spec); result = copy.deepcopy(task)
    selector = ArmSelector(scene['robot_roots'], scene['seed'])
    choices = {}; destination = targets(scene, result)
    for op in result['operations']:
        index = rec.object_ids.index(op['object']); item = scene['objects'][index]
        source = array(rec.objects[index].pose.to_transformation_matrix()).reshape(4, 4)
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
        per_arm = {a: [d for d in candidates if d['arm'] == a][:MAX_GRASPS_PER_ARM] for a in range(2)}
        ordered = [per_arm[a][i] for i in range(MAX_GRASPS_PER_ARM) for a in sorted(per_arm, key=lambda a: min(
            [d['joint_travel_proxy'] for d in per_arm[a]] or [float('inf')])) if i < len(per_arm[a])]
        for row in ordered:
            arm = row['arm']; candidate = by_id[row['candidate']]
            if arm not in guards:
                guard = make_solver(rec, arm, folder / f'arm_{arm}_motion')
                guard.__class__ = WholePathMotion; guard.install(index); guards[arm] = guard
            guard = guards[arm]
            for heights in HEIGHTS:
                if time.monotonic() - began >= MAX_OPERATION_SECONDS:
                    break
                try:
                    proof = preview_route(guard, source @ np.asarray(candidate['tcp_in_asset']), desired, candidate, heights)
                    selected = (arm, candidate, heights)
                    decisions.append({'arm': arm, 'candidate': candidate['id'], **proof}); break
                except (RuntimeError, ValueError, TimeoutError) as error:
                    decisions.append({'arm': arm, 'candidate': candidate['id'], 'height_profile_m': list(heights),
                                      'passed': False, 'reason': str(error)})
            if selected is not None or time.monotonic() - began >= MAX_OPERATION_SECONDS:
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


@contextmanager
def install_for_worker(root):
    from .runner import verify
    from . import transport_fallback_v2, stability_motion
    spec = verify(Path(root))
    if spec.get('candidate_motion_override') != VERSION or spec.get('motion_policy') != 'transport_fallback_v2':
        raise ValueError('explicit_whole_path_candidate_required')
    if spec['code'].get('resimpair_pickplace/whole_path_motion_v1.py') != binding(__file__):
        raise ValueError('whole_path_candidate_source_not_frozen')
    prior_class = transport_fallback_v2.TransportFallbackMotion; prior_solver = stability_motion.make_solver
    def configured_solver(rec, arm, directory):
        guard = prior_solver(rec, arm, directory)
        op_id = Path(directory).name.removesuffix('_motion')
        guard.height_profile = rec.whole_path_choices[op_id]
        return guard
    transport_fallback_v2.TransportFallbackMotion = WholePathMotion
    stability_motion.make_solver = configured_solver
    try:
        yield
    finally:
        transport_fallback_v2.TransportFallbackMotion = prior_class
        stability_motion.make_solver = prior_solver
