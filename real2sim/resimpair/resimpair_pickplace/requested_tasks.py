"""Compile metric pick/place requests using existing assets and cached grasps.

Object identity is data, never a controller branch. Source poses are on the
table in the cache's fixed support orientation. All translations are meters,
all yaw angles are radians. Container offsets are in its LOCAL XY frame.
"""
import re

import numpy as np
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write

SCHEMA = 'cached_pickplace_requests_v1'
TABLE_XY = np.array([[-.46, .46], [-.90, .90]])
MAX_DISTRACTOR_EXCLUSIONS = 4096


class Rejected(ValueError):
    def __init__(self, code, **details):
        self.code, self.details = code, details
        super().__init__(code)


def fields(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise Rejected('unsupported_or_missing_request_fields', required=list(required),
                       received=list(value) if isinstance(value, dict) else None)


def identity(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,80}', value):
        raise Rejected('invalid_request_identifier')
    return value


def finite(value, shape):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all(): raise Rejected('invalid_metric_coordinate')
    return a


def rotation(yaw):
    return Rotation.from_euler('z', float(finite(yaw, ()))).as_matrix()[:2, :2]


def floor_fit(vertices, cavity, object_yaw, container_yaw, offset, margin=.008):
    """Actual payload vertices, not the container's outside bounding box."""
    xy = np.asarray(vertices)[:, :2] @ rotation(object_yaw - container_yaw).T + finite(offset, (2,))
    remaining = float(cavity['flat_floor_radius_m'] - np.linalg.norm(xy, axis=1).max())
    return {'passed': remaining >= margin, 'floor_margin_m': remaining, 'required_margin_m': margin}


def table_fit(xy, radius):
    xy = finite(xy, (2,))
    if np.any(xy - radius < TABLE_XY[:, 0]) or np.any(xy + radius > TABLE_XY[:, 1]):
        raise Rejected('footprint_outside_table', xy=xy.tolist(), radius_m=radius)


def check_separation(xy, radius, occupied, exclude=(), margin=.025):
    for name, point, other_radius in occupied:
        if name not in exclude and np.linalg.norm(np.asarray(xy) - point) <= radius + other_radius + margin:
            raise Rejected('footprint_overlap', other=name)


def request_index(value):
    fields(value, ['schema', 'scenes'])
    if value['schema'] != SCHEMA or not 1 <= len(value['scenes']) <= 24:
        raise Rejected('unsupported_request_schema_or_count')
    scene_ids, task_ids = set(), set()
    for scene in value['scenes']:
        fields(scene, ['id', 'objects', 'tasks'], ['containers', 'distractors'])
        sid = identity(scene['id'])
        if sid in scene_ids: raise Rejected('duplicate_scene_id')
        scene_ids.add(sid)
        if not 1 <= len(scene['objects']) <= 4 or not 1 <= len(scene['tasks']) <= 8:
            raise Rejected('bounded_object_and_task_counts_required')
        for task in scene['tasks']:
            fields(task, ['id', 'placements'])
            key = sid + '_' + identity(task['id'])
            if key in task_ids: raise Rejected('duplicate_task_id')
            task_ids.add(key)
    if len(task_ids) > 48: raise Rejected('bounded_total_tasks_required')
    return value


def add_distractors(scene, settings, occupied, seed):
    from resimpair_grasp.assets import select
    from resimpair_grasp.scenes import item
    fields(settings, [], ['count', 'max_dimension_m', 'exclude_source_ids', 'catalog', 'source_role_index'])
    count = settings.get('count', 4)
    maximum = float(settings.get('max_dimension_m', .16))
    if type(count) is not int or not 3 <= count <= 8 or not .02 <= maximum <= .18:
        raise Rejected('invalid_distractor_limits')
    exclusions = settings.get('exclude_source_ids', [])
    if (type(exclusions) is not list or len(exclusions) > MAX_DISTRACTOR_EXCLUSIONS
            or any(type(source) is not str or not 1 <= len(source) <= 512
                   or source != source.strip() or any(ord(char) < 32 for char in source)
                   for source in exclusions)
            or len(set(exclusions)) != len(exclusions)):
        raise Rejected('invalid_distractor_exclude_source_ids', maximum_count=MAX_DISTRACTOR_EXCLUSIONS,
                       required='unique nonempty source-ID strings, each at most 512 characters')
    rng = np.random.default_rng(seed)
    source_ids = {read(resolve(o['asset_descriptor']))['source_id'] for o in scene['objects']}
    source_ids.update(exclusions)
    if ('catalog' in settings) != ('source_role_index' in settings):
        raise Rejected('expanded_catalog_requires_bound_source_role_index')
    if 'catalog' in settings:
        from .source_role_catalog_v1 import select as select_indexed
        pool = select_indexed(settings['catalog'], settings['source_role_index'],
                              scene['partition'], maximum, source_ids)
    else:
        pool = [(b, d) for b, d in select(ROOT / 'assets/expanded_library/scene_catalog_v3.json', max_dimension=maximum)
                if d['source_id'] not in source_ids and d['partition'] in {scene['partition'], 'shared_distractor'}]
    rng.shuffle(pool)
    distinct, repeated, categories = [], [], set()
    for pair in pool:
        (repeated if pair[1]['category'] in categories else distinct).append(pair)
        categories.add(pair[1]['category'])
    selected = []
    for b, d in distinct + repeated:
        radius = float(np.linalg.norm(d['half_size'][:2]))
        for _ in range(100):
            xy = ([rng.uniform(.28, .38), rng.uniform(-.73, .73)] if rng.random() < .65 else
                  [rng.uniform(-.36, .28), rng.choice([-1, 1]) * rng.uniform(.65, .79)])
            try:
                table_fit(xy, radius); check_separation(xy, radius, occupied)
            except Rejected: continue
            oid = 'clutter_' + str(len(selected))
            scene['objects'].append(item(b, d, oid, xy, rng.uniform(-np.pi, np.pi)))
            occupied.append((oid, np.asarray(xy), radius)); selected.append(d['source_id']); break
        if len(selected) == count: break
    if len(selected) != count: raise Rejected('requested_distractor_count_not_placeable', placed=len(selected))
    return selected


def compile_scene(spec, entries, container_entries, seed):
    """No physical rollout, solver regeneration, rescaling, or task-specific code."""
    import trimesh
    from resimpair_grasp.scenes import item
    from .support_surface import check_support
    from corl2026_workshop.experiments.autonomous_pipeline_v4.library import make_scene
    by_source = {read(resolve(e['asset']))['source_id']: e for e in entries}
    objects, object_data, vertices, caches, occupied = [], {}, {}, {}, []
    geometry_checks = {'objects': [], 'containers': [], 'goals': []}
    for request in spec['objects']:
        fields(request, ['id', 'source_id', 'xy'], ['yaw', 'max_dimension_m'])
        oid = identity(request['id'])
        if oid == 'table' or oid.startswith('clutter_') or oid in object_data: raise Rejected('duplicate_or_reserved_object_id')
        if request['source_id'] not in by_source: raise Rejected('asset_not_in_native_pose_bundle', source_id=request['source_id'])
        entry = by_source[request['source_id']]; descriptor = read(resolve(entry['asset']))
        candidates = read(resolve(entry['cache']))['candidates']
        if not candidates: raise Rejected('no_cached_grasp', source_id=request['source_id'])
        dimensions = 2 * np.asarray(descriptor['half_size'])
        if 'max_dimension_m' in request:
            maximum = float(finite(request['max_dimension_m'], ()))
            if maximum <= 0 or dimensions.max() > maximum:
                raise Rejected('asset_exceeds_requested_size', dimensions_m=dimensions.tolist(), limit_m=maximum)
        xy = finite(request['xy'], (2,)); yaw = float(finite(request.get('yaw', 0.), ()))
        radius = float(np.linalg.norm(descriptor['half_size'][:2]))
        table_fit(xy, radius); check_separation(xy, radius, occupied)
        occupied.append((oid, xy, radius))
        objects.append(item(entry['asset'], descriptor, oid, xy, yaw, True))
        object_data[oid] = descriptor; caches[oid] = entry['cache']
        mesh = trimesh.load(resolve(descriptor['visual']), force='scene', process=False).to_geometry()
        vertices[oid] = np.asarray(mesh.vertices)
        geometry_checks['objects'].append({'object': oid, 'dimensions_m': dimensions.tolist(),
            'cached_poses': len(candidates), 'grasp_width_range_m': [min(c['width_m'] for c in candidates), max(c['width_m'] for c in candidates)],
            'rescaled': False, 'source_id': descriptor['source_id']})
    partitions = {d['partition'] for d in object_data.values()}
    if len(partitions) != 1: raise Rejected('cross_partition_scene')
    partition = next(iter(partitions)); source_ids = sorted(d['source_id'] for d in object_data.values())
    containers = {identity(c['id']): c for c in spec.get('containers', [])}
    if len(containers) != len(spec.get('containers', [])) or set(containers) & (set(object_data) | {'table'}):
        raise Rejected('duplicate_container_id')
    objects_by_id = {o['id']: o for o in objects}; goals_by_container = {k: [] for k in containers}
    for task in spec['tasks']:
        selected = set()
        if not 1 <= len(task['placements']) <= len(objects): raise Rejected('invalid_placement_count')
        for op in task['placements']:
            fields(op, ['object', 'target'])
            oid = op['object']; goal = op['target']
            if oid not in object_data or oid in selected: raise Rejected('unknown_or_repeated_object_in_task')
            selected.add(oid)
            if goal.get('kind') == 'container':
                fields(goal, ['kind', 'container'], ['offset_xy_m', 'yaw'])
                if goal['container'] not in containers: raise Rejected('unknown_target_container')
                goals_by_container[goal['container']].append((oid, goal))
            elif goal.get('kind') == 'table': fields(goal, ['kind', 'xy'], ['yaw'])
            else: raise Rejected('unsupported_place_target')
    support_data, supports = {}, []
    for cid, request in containers.items():
        fields(request, ['id', 'xy'], ['source_id', 'yaw', 'max_dimension_m'])
        xy = finite(request['xy'], (2,)); yaw = float(finite(request.get('yaw', 0.), ()))
        choices = []; rejected = []
        for b in container_entries:
            d = read(resolve(b))
            if d['partition'] != partition: continue
            if request.get('source_id') not in {None, d['source_id']}: continue
            if re.search(r'\b(soup|filled|water|cereal|fruit|food)\b', d['name'], re.I): continue
            if 'max_dimension_m' in request and max(d['dimensions_m']) > float(finite(request['max_dimension_m'], ())): continue
            fits = [floor_fit(vertices[oid], d['cavity'], goal.get('yaw', objects_by_id[oid]['yaw']), yaw,
                              goal.get('offset_xy_m', [0., 0.])) for oid, goal in goals_by_container[cid]]
            if not all(f['passed'] for f in fits):
                rejected.append({'source_id': d['source_id'], 'reason': 'payload_exceeds_inner_floor', 'checks': fits}); continue
            check = check_support(trimesh.load(resolve(d['collision']), force='mesh', process=False), d['cavity'])
            if not check['passed']:
                rejected.append({'source_id': d['source_id'], 'reason': check['reason']}); continue
            choices.append((max(d['dimensions_m']), d['id'], b, d, fits))
        if not choices: raise Rejected('no_container_with_requested_capacity', container=cid, candidates_rejected=rejected)
        choices.sort(key=lambda row: row[:2]); _, _, b, d, fits = choices[0]
        radius = float(np.linalg.norm(np.asarray(d['dimensions_m'])[:2] / 2))
        table_fit(xy, radius); check_separation(xy, radius, occupied)
        occupied.append((cid, xy, radius)); support_data[cid] = d
        supports.append({'id': cid, 'shape': 'stationary_mesh_container', 'container_descriptor': b,
            'position': [*xy.tolist(), 0.], 'yaw': yaw, 'half_size': (np.asarray(d['dimensions_m']) / 2).tolist(),
            'top_z': d['cavity']['floor_z_m'], 'semantic_description': 'the ' + d['name'],
            'color_name': 'original material', 'rgba': [.5, .5, .5, 1.]})
        geometry_checks['containers'].append({'container': cid, 'source_id': d['source_id'], 'size_m': d['dimensions_m'],
            'floor_radius_m': d['cavity']['flat_floor_radius_m'], 'payload_checks': fits,
            'selection': 'smallest fitting native container; no physical outcomes used', 'rejected': rejected})
    scene = make_scene(spec['id'], seed, nuisance=True, layout=False)
    scene.update(schema='requested_cached_pickplace_scene_v1', scene_instance_id=spec['id'],
                 split_group=source_ids, partition=partition, source_coverage='synthetic_only',
                 objects=objects, supports=supports, grasp_caches=caches)
    tasks = []; goal_reservations = []
    for request in spec['tasks']:
        ops, destinations = [], []
        for index, placement in enumerate(request['placements']):
            oid = placement['object']; goal = placement['target']; obj = objects_by_id[oid]
            radius = float(np.linalg.norm(obj['half_size'][:2]))
            yaw = float(finite(goal.get('yaw', obj['yaw']), ()))
            if goal['kind'] == 'table':
                xy = finite(goal['xy'], (2,)); support = 'table'; offset = xy; z = scene['table_top_z'] + obj['half_size'][2]
            else:
                support = goal['container']; c = next(s for s in supports if s['id'] == support)
                local = np.asarray(support_data[support]['cavity']['center_xy_m']) + finite(goal.get('offset_xy_m', [0., 0.]), (2,))
                offset = rotation(c['yaw']) @ local; xy = np.asarray(c['position'][:2]) + offset
                z = c['top_z'] + obj['half_size'][2]
            table_fit(xy, radius); check_separation(xy, radius, occupied, exclude=[oid, support])
            check_separation(xy, radius, destinations, margin=.015)
            destinations.append((oid, xy, radius)); goal_reservations.append((request['id'] + '_' + oid, xy, radius))
            ops.append({'id': 'transfer_' + str(index), 'primitive': 'pick_place', 'object': oid, 'arm': None,
                        'target': {'support': support, 'offset_xy': offset.tolist(), 'yaw': yaw}})
            geometry_checks['goals'].append({'task': request['id'], 'object': oid, 'world_position_m': [*xy.tolist(), z],
                'yaw_radians': yaw, 'explicit_orientation_required': 'yaw' in goal})
        tasks.append({'schema': 'resimpair_task_intent_v4', 'id': spec['id'] + '_' + request['id'], 'family': 'relocate',
            'required_skills': ['pick_place'], 'object_ids': [o['object'] for o in ops], 'goal': request['id'],
            'operations': ops, 'required_operation_precedence': [[ops[i]['id'], ops[i+1]['id']] for i in range(len(ops)-1)],
            'equivalence_groups': [], 'incidental_controller_order_is_required': False,
            'human_hand_policy': 'preserve_robot_body_side', 'non_target_policy': 'preserve_identity_and_pose',
            'release_precondition_control': True, 'scope': 'cached rigid tabletop pick/place; no scaling or handover',
            'requested_task_id': request['id']})
    placed = add_distractors(scene, spec.get('distractors', {}), occupied + goal_reservations, seed)
    scene['randomization'] = {'seed': seed, 'source_and_goal': 'user metric requests',
        'distractor_count': len(placed), 'distractor_source_ids': placed,
        'size_policy': 'fixed cached metric geometry; no runtime rescaling', 'no_success_driven_resampling': True}
    return scene, tasks, geometry_checks


def register(value, indices, containers, root, method, seed):
    from .cartesian_reachability import bind_selection
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import core
    entries = read(indices[method])['entries']; container_entries = read(containers)['assets']
    cases, rejected = [], []
    for i, request in enumerate(request_index(value)['scenes']):
        scene_dir = root / 'requests' / request['id']; write(scene_dir / 'request.json', request)
        source = binding(scene_dir / 'request.json')
        try:
            scene, tasks, checks = compile_scene(request, entries, container_entries, seed + i)
            shared = root / 'scenes' / (request['id'] + '.json'); write(shared, scene)
            write(scene_dir / 'geometry_checks.json', checks)
        except (Rejected, ValueError) as error:
            reason = error.code if isinstance(error, Rejected) else 'invalid_request_value'
            details = error.details if isinstance(error, Rejected) else {'message': str(error)}
            rejected.append({'scene': request['id'], 'reason': reason, 'details': details})
            write(scene_dir / 'rejection.json', rejected[-1])
            for task in request['tasks']:
                identity = request['id'] + '_' + task['id']
                cases.append({'id': identity, 'execution_id': identity, 'partition': 'request_rejected', 'eligible': False,
                    'preflight_reason': reason, 'source_request': source, 'requested_task_id': task['id'], 'method': method})
            continue
        for task in tasks:
            folder = root / 'inputs' / task['id']; write(folder / 'role_intent.json', task)
            selected, planned = bind_selection(scene, task, folder)
            case = {'id': task['id'], 'execution_id': task['id'], 'partition': scene['partition'],
                'source_ids': scene['split_group'], 'split_group': scene['split_group'], 'shared_scene': binding(shared),
                'method': method, 'source_coverage': 'synthetic_only', 'source_request': source,
                'requested_task_id': task['requested_task_id'], 'eligible': selected is not None,
                'arm_selection': binding(folder / 'arm_selection.json')}
            if selected is None: case['preflight_reason'] = 'no_single_arm_source_and_destination_grasp'
            else:
                write(folder / 'scene.json', selected); write(folder / 'task.json', planned)
                case.update(scene=core.bind(folder / 'scene.json'), task=core.bind(folder / 'task.json'))
            cases.append(case)
    result = {'cases': cases, 'scene_rejections': rejected, 'failures_retained': True,
              'source_coverage': 'synthetic_only', 'runtime_pose_generation': False,
              'controller': 'shared stability_motion.solve; policies recorded in experiment.json'}
    write(root / 'cohort.json', result)
    return {'registered': len(cases), 'eligible': sum(c['eligible'] for c in cases), 'scene_rejections': len(rejected),
            'optimizer_calls': 0, 'api_calls': 0}
