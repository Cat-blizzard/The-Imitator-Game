"""Explicit new compiler geometry branch; original compiler files stay frozen.

Local function bindings reuse the original compile/factory bytecode. Only the
floor predicate and task-goal pair predicate use verified native footprints.
Initial layouts, exterior circles, table limits, height, cache/source/split
rules and physical evaluators remain the retained implementations. Every new
cohort starts on review hold, including all semantic holds supplied as data.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_pickplace import requested_tasks as compiler
from . import metric_task_factory_v3 as factory
from . import floor_polygon_capacity_v1 as geometry
from . import source_first_task_selection_v2 as release

VERSION = 'verified_floor_polygon_factory_v1'


def locally_bound(function, **overrides):
    """Reuse exact code with a private globals mapping, never patch a module."""
    result = FunctionType(function.__code__, {**function.__globals__, **overrides},
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    return result


def vertices_key(vertices):
    value = np.ascontiguousarray(np.asarray(vertices, dtype=np.float64))
    return (value.shape, hashlib.sha256(value.tobytes()).hexdigest())


def point_key(xy, radius):
    xy = compiler.finite(xy, (2,))
    return (*map(float, xy), float(radius))


def admission(scene_id, holds):
    return {'status': 'review_hold', 'usable_candidate': False, 'physical_qualified': False,
            'geometry_policy': VERSION, 'scene': scene_id,
            'holds': [{'reason': 'pending_independent_polygon_geometry_review'}, *deepcopy(holds)],
            'semantic_status': 'hold' if holds else 'not_reassessed', 'physical_success': 'unmeasured'}


class GeometryBranch:
    def __init__(self, spec, entries, container_entries, directory, native_geometry_refs):
        self.spec = deepcopy(spec); self.directory = Path(directory)
        self.floor_checks = []; self.pair_checks = []
        self.floors = {}; self.receivers = {}; self.footprints = {}; self.by_vertices = {}
        self.goals = {}; self.goals_by_id = {}
        partitions = {read(resolve(e['asset']))['partition'] for e in entries}
        if len(partitions) != 1:
            raise ValueError('one_original_source_partition_required')
        refs = factory.select_container_variants(container_entries, spec.get('containers', []),
            spec.get('container_variant_choices', {}), next(iter(partitions)))
        for ref in refs:
            descriptor = read(resolve(ref)); floor = geometry.load_floor(ref)
            path = self.directory / 'floors' / (ref['sha256'] + '.json'); write(path, floor)
            key = digest(descriptor['cavity'])
            if key in self.floors and self.floors[key]['receiver'] != ref:
                raise ValueError('ambiguous_equal_cavity_summaries_require_separate_compiles')
            floor['evidence'] = binding(path); self.floors[key] = floor
            self.receivers[descriptor['source_id']] = descriptor
        by_source = {e['source_id']: e for e in entries}
        for obj in spec['objects']:
            entry = by_source[obj['source_id']]
            footprint = geometry.load_footprint(entry, native_geometry_refs[entry['asset']['sha256']])
            path = self.directory / 'footprints' / (entry['asset']['sha256'] + '.json')
            write(path, footprint['receipt']); footprint['evidence'] = binding(path)
            self.footprints[obj['id']] = footprint
            self.by_vertices.setdefault(vertices_key(footprint['visual']), []).append(footprint)
        supports = {c['id']: c for c in spec.get('containers', [])}
        objects = {o['id']: o for o in spec['objects']}
        release_checker = release.ReleaseCompatibility()
        for task in spec['tasks']:
            for placement in task['placements']:
                goal = placement['target']; oid = placement['object']
                if goal['kind'] != 'container':
                    raise ValueError('explicit_container_goals_required_for_polygon_branch')
                support = supports[goal['container']]; descriptor = self.receivers[support['source_id']]
                entry = by_source[objects[oid]['source_id']]
                release_checker.check((entry, read(resolve(entry['asset']))),
                    spec['container_variant_choices'][support['source_id']], descriptor)
                yaw = float(goal.get('yaw', objects[oid].get('yaw', 0.)))
                local = np.asarray(descriptor['cavity']['center_xy_m']) + compiler.finite(goal.get('offset_xy_m', [0., 0.]), (2,))
                world_xy = np.asarray(support['xy']) + compiler.rotation(support.get('yaw', 0.)) @ local
                footprint = self.footprints[oid]
                points = footprint['envelope'][:, :2] @ compiler.rotation(yaw).T + world_xy
                bounds = np.array([points.min(0), points.max(0)])
                radius = float(np.linalg.norm(footprint['receipt']['half_size_m'][:2]))
                record = {'object': oid, 'task': task['id'], 'bounds': bounds, 'xy': world_xy,
                          'radius': radius, 'footprint': footprint['evidence'], 'support': support['id']}
                self.goals.setdefault(point_key(world_xy, radius), []).append(record)
                self.goals_by_id[(oid, point_key(world_xy, radius))] = record
        self.release_preflight = release_checker.receipt()
        write(self.directory / 'RELEASE_PREFLIGHT.json', self.release_preflight)
        if not all(row['passed'] for row in self.release_preflight['checks']):
            raise ValueError('original_first_twelve_cache_release_height_preflight_failed')

    def floor_fit(self, vertices, cavity, object_yaw, container_yaw, offset, margin=.008):
        floor = self.floors.get(digest(cavity)); footprints = self.by_vertices.get(vertices_key(vertices))
        if floor is None or not footprints:
            raise ValueError('floor_and_every_payload_must_bind_original_geometry')
        if not np.isfinite([object_yaw, container_yaw, margin]).all():
            raise ValueError('finite_original_yaws_and_margin_required')
        required = max(geometry.FLOOR_MARGIN_M, float(margin))
        transform = compiler.rotation(object_yaw - container_yaw)
        origin = compiler.finite(offset, (2,)) + np.asarray(cavity['center_xy_m'])
        visual = np.asarray(vertices)[:, :2] @ transform.T + origin
        visual_margin = geometry.floor_margin(floor, visual)
        native_margins = [geometry.floor_margin(floor, f['native'][:, :2] @ transform.T + origin) for f in footprints]
        remaining = min(visual_margin, *native_margins)
        result = {'passed': remaining >= required, 'floor_margin_m': remaining,
            'required_margin_m': required, 'geometry_policy': VERSION, 'floor': floor['evidence'],
            'native_footprints': [f['evidence'] for f in footprints], 'visual_floor_margin_m': visual_margin,
            'native_floor_margins_m': native_margins, 'legacy_requested_margin_m': margin,
            'legacy_circle_diagnostic': compiler.floor_fit(vertices, cavity, object_yaw, container_yaw, offset, margin),
            'circle_diagnostic_is_not_the_polygon_acceptance_rule': True}
        self.floor_checks.append(result)
        return result

    def separation(self, xy, radius, occupied, exclude=(), margin=.025):
        # The retained compiler uses .015 only for goal-to-goal destinations.
        # Its initial objects, source/goal checks, containers and clutter retain
        # their original .025 circle rules through the untouched function.
        if margin != .015:
            return compiler.check_separation(xy, radius, occupied, exclude, margin)
        if exclude:
            raise ValueError('unexpected_exclusion_in_goal_pair_predicate')
        matches = self.goals.get(point_key(xy, radius))
        if not matches:
            raise ValueError('task_goal_pair_must_bind_original_native_footprint')
        # Reused coordinates in different tasks get a conservative union. No
        # ambiguous lookup can accidentally check a smaller payload instead.
        bounds = np.array([np.min([r['bounds'][0] for r in matches], axis=0),
                           np.max([r['bounds'][1] for r in matches], axis=0)])
        for oid, point, other_radius in occupied:
            other = self.goals_by_id.get((oid, point_key(point, other_radius)))
            if other is None:
                raise ValueError('unbound_previous_goal_in_polygon_pair_check')
            distance = geometry.aabb_distance(bounds, other['bounds'])
            passed = distance >= geometry.PAYLOAD_GAP_M
            row = {'passed': passed, 'objects_at_current_goal': sorted({r['object'] for r in matches}),
                'other_object': oid, 'gap_m': distance, 'required_gap_m': geometry.PAYLOAD_GAP_M,
                'legacy_requested_margin_m': margin, 'current_bounds_world_xy_m': bounds.tolist(),
                'other_bounds_world_xy_m': other['bounds'].tolist(),
                'native_footprints': [r['footprint'] for r in matches] + [other['footprint']],
                'geometry_policy': VERSION, 'scope': 'native-and-visual enclosing AABBs at the exact requested yaw'}
            self.pair_checks.append(row)
            if not passed:
                raise compiler.Rejected('native_goal_footprints_too_close', **row)

    def compile(self, spec, entries, container_entries, seed):
        bound = locally_bound(compiler.compile_scene, floor_fit=self.floor_fit, check_separation=self.separation)
        try:
            scene, tasks, checks = bound(spec, entries, container_entries, seed)
            checks['floor_polygon_geometry'] = {'schema': VERSION,
                'floors': [f['evidence'] for f in self.floors.values()],
                'footprints': {oid: f['evidence'] for oid, f in self.footprints.items()},
                'goal_pair_checks': self.pair_checks, 'new_floor_margin_m': geometry.FLOOR_MARGIN_M,
                'new_payload_gap_m': geometry.PAYLOAD_GAP_M,
                'initial_table_and_exterior_circle_rules_unchanged': True,
                'legacy_circle_checks_are_separate_diagnostics': True, 'physical_success': 'unmeasured'}
            checks['floor_polygon_geometry']['release_preflight'] = binding(self.directory / 'RELEASE_PREFLIGHT.json')
            return scene, tasks, checks
        finally:
            write(self.directory / 'COMPILER_CHECKS.json', {'schema': VERSION, 'floor_checks': self.floor_checks,
                'goal_pair_checks': self.pair_checks, 'physics_steps': 0, 'physical_success': 'unmeasured'})


def compile_requests(bundle, requests, output, seed, *, native_geometry_refs, source_holds=()):
    """Real v3 task compilation through explicit, private geometry callbacks."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_polygon_factory_outputs')
    requested = factory.request_schema(read(requests))
    holds_by_scene = {}
    for scene in requested['scenes']:
        sources = {o['source_id'] for o in scene['objects']}
        holds = [deepcopy(h) for h in source_holds if h['source_id'] in sources]
        for hold in holds:
            resolve(hold['evidence'])
        holds_by_scene[scene['id']] = holds
    original_hashes = [binding(p) for p in (compiler.__file__, factory.__file__, factory.previous.__file__)]
    def compile_exact(spec, entries, containers, current_seed):
        branch = GeometryBranch(spec, entries, containers, output / 'floor_polygon_geometry' / spec['id'],
                                native_geometry_refs)
        namespace = SimpleNamespace(**{**vars(compiler), 'compile_scene': branch.compile})
        exact = locally_bound(factory.compile_exact_scene, retained=namespace)
        return exact(spec, entries, containers, current_seed)
    def annotated_write(path, value):
        relative = Path(path).resolve().relative_to(output)
        value = deepcopy(value)
        if relative.parts == ('protocol.json',):
            value.update(geometry_policy=VERSION, geometry_adapter=binding(__file__),
                floor_extraction=binding(geometry.__file__),
                new_floor_margin_m=geometry.FLOOR_MARGIN_M, new_payload_gap_m=geometry.PAYLOAD_GAP_M,
                saved_native_geometry=deepcopy(native_geometry_refs),
                original_release_preflight=binding(release.__file__),
                unchanged_rules='table, initial/source/exterior circles, source/cache/partition/height, physical evaluator',
                original_circle_geometry_is_separate_control=True, automatic_admission=False)
        elif relative.name == 'scene.json' and relative.parts[0] == 'scenes':
            value['geometry_admission'] = admission(relative.parts[1], holds_by_scene[relative.parts[1]])
        elif relative.parts == ('cohort.json',):
            for case in value['cases']:
                scene = read(resolve(case['scene'])); state = scene['geometry_admission']
                case.update(eligible=False, geometric_compile_passed=True, admission_status='review_hold',
                    geometry_policy=VERSION, geometry_admission=deepcopy(state), physical_status='not_executed')
            value.update(geometry_policy=VERSION, default_promoted=False, usable_candidates=0)
        elif relative.parts == ('summary.json',):
            value.update(geometry_policy=VERSION, admission_status='review_hold', usable_candidates=0,
                         physical_success='unmeasured', physics_steps=0, GPU_jobs=0)
        write(path, value)
    run = locally_bound(factory.compile_requests, compile_exact_scene=compile_exact, write=annotated_write)
    run(bundle, requests, output, seed)
    for ref in original_hashes:
        resolve(ref)
    return read(output / 'summary.json')
