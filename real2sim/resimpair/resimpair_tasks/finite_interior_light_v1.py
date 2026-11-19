"""Candidate: honor a bounded interior key position instead of an infinite sun.

No materials, geometry, cameras, physical states, ambient, fill or HDRI change.
Preparation and CPU box-ray diagnosis require no simulator or GPU. Installation
is scoped to an isolated renderer worker; the retained source stays unchanged.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, resolve
from . import scene_domain_v2 as retained

VERSION = 'finite_interior_key_v1'
SHADER = ROOT / '.venv/lib/python3.12/site-packages/sapien/vulkan_shader/common/lights.glsl'


def box_ray(origin, direction, box, maximum_distance=np.inf):
    """Slab ray/AABB intersection in metres, including axis-parallel rays."""
    origin, direction = np.asarray(origin, float), np.asarray(direction, float)
    center, half = np.asarray(box['center_m'], float), np.asarray(box['size_m'], float) / 2
    if not np.isfinite(np.r_[origin, direction, center, half]).all() or np.any(half <= 0):
        raise ValueError('finite_ray_and_positive_box_required')
    if not np.isclose(np.linalg.norm(direction), 1, atol=1e-9):
        raise ValueError('unit_direction_required')
    enter, leave = 0., float(maximum_distance)
    for axis in range(3):
        lo, hi = center[axis] - half[axis], center[axis] + half[axis]
        if abs(direction[axis]) < 1e-14:
            if not lo <= origin[axis] <= hi:
                return None
        else:
            a, b = sorted(((lo - origin[axis]) / direction[axis], (hi - origin[axis]) / direction[axis]))
            enter, leave = max(enter, a), min(leave, b)
            if leave < enter:
                return None
    return {'box': box['id'], 'entry_distance_m': enter, 'exit_distance_m': leave,
            'entry_point_m': (origin + direction * enter).tolist()}


def diagnose(scene):
    d = retained.validate_scene(scene); light = d['lighting']; primary = light['primary_source']
    position, target = np.asarray(primary['position_m'], float), np.asarray(primary['target_m'], float)
    direction = -np.asarray(light['key_direction'], float)
    direction /= np.linalg.norm(direction)
    boxes = (d.get('geometry') or {}).get('boxes', [])
    origins = [target] + [np.array([x, y, .12]) for x in np.linspace(-.65, .40, 7) for y in np.linspace(-.85, .85, 9)]
    rows = []
    for origin in origins:
        delta = position - origin; distance = float(np.linalg.norm(delta))
        if distance <= .02:
            raise ValueError('finite_light_too_close_to_task_sample')
        infinite_hits = [h for b in boxes if (h := box_ray(origin, direction, b)) is not None]
        finite_hits = [h for b in boxes if (h := box_ray(origin, delta / distance, b, distance - 1e-6)) is not None]
        rows.append({'origin_m': origin.tolist(), 'infinite_direction_first_structure': min(infinite_hits, key=lambda h: h['entry_distance_m']) if infinite_hits else None,
                     'finite_segment_first_structure': min(finite_hits, key=lambda h: h['entry_distance_m']) if finite_hits else None,
                     'finite_source_distance_m': distance})
    return {'schema': VERSION, 'primary_source': deepcopy(primary), 'samples': rows,
            'infinite_structure_blocked_count': sum(r['infinite_direction_first_structure'] is not None for r in rows),
            'finite_structure_blocked_count': sum(r['finite_segment_first_structure'] is not None for r in rows),
            'scope': 'exact room-box ray intersections only; furniture, robots, task self-shadow and raster shadow-map bias require actual render',
            'renderer': binding(ROOT / 'PolicyGen/mani_skill/envs/scene.py'),
            'retained_light_installation': binding(retained.__file__),
            'shader_inverse_square': binding(SHADER), 'GPU_jobs': 0, 'physics_steps': 0, 'API_calls': 0}


def prepare(scene):
    d = retained.validate_scene(scene); light = d['lighting']; primary = light['primary_source']
    if primary['kind'] != 'ceiling_fixture':
        raise ValueError('this_candidate_only_replaces_declared_ceiling_fixture_keys')
    position, reference = np.asarray(primary['position_m'], float), np.asarray(primary['target_m'], float)
    distance = float(np.linalg.norm(position - reference))
    key = np.asarray(light['key_rgb'], float)
    if not np.isfinite(np.r_[position, reference, key, distance]).all() or distance <= .1 or np.any(key < 0):
        raise ValueError('invalid_finite_key_condition')
    diagnosis = diagnose(scene)
    if diagnosis['samples'][0]['finite_segment_first_structure'] is not None:
        raise ValueError('declared_interior_source_occluded_by_room_structure')
    if any(box_ray(position, np.array([0., 0., 1.]), b, 0.) is not None
           for b in d['geometry']['boxes']):
        raise ValueError('declared_interior_source_inside_opaque_room_box')
    result = deepcopy(scene)
    result['scene_domain']['finite_interior_key'] = {
        'schema': VERSION, 'implementation': binding(__file__), 'kind': 'point',
        'position_m': position.tolist(), 'reference_point_m': reference.tolist(),
        'reference_distance_m': distance, 'reference_unshadowed_key_rgb': key.tolist(),
        'emission_rgb': (key * distance ** 2).tolist(),
        'unit_convention': 'linear_renderer_emission_RGB; multiply retained directional RGB by reference_distance_m squared to preserve its nominal unshadowed direct contribution at the reference point; no lux/lumen/candela calibration claimed',
        'shadow': True, 'shadow_near_m': .02, 'shadow_far_m': 10., 'shadow_map_size': 2048,
        'inverse_square_shader': binding(SHADER),
        'visible_fixture_mesh_correspondence': 'none; inherited source location is a virtual fixture',
        'ambient_fill_HDRI_geometry_material_camera_physics_unchanged': True}
    check = deepcopy(result); del check['scene_domain']['finite_interior_key']
    if digest(check) != digest(scene):
        raise RuntimeError('finite_key_candidate_changed_unrelated_inputs')
    return result, diagnosis


def load_lighting(env):
    d = retained.validate_scene(env.scene_spec); light = d['lighting']; finite = d['finite_interior_key']
    if finite['schema'] != VERSION or finite['implementation'] != binding(__file__):
        raise ValueError('finite_key_implementation_binding_mismatch')
    resolve(finite['inverse_square_shader'])
    expected, _ = prepare({**env.scene_spec, 'scene_domain': {k: v for k, v in d.items() if k != 'finite_interior_key'}})
    if expected['scene_domain']['finite_interior_key'] != finite:
        raise ValueError('finite_key_values_do_not_match_declared_source')
    env.scene.set_ambient_light(light['ambient_rgb'])
    point = env.scene.add_point_light(finite['position_m'], finite['emission_rgb'], shadow=True,
        shadow_near=finite['shadow_near_m'], shadow_far=finite['shadow_far_m'], shadow_map_size=finite['shadow_map_size'])
    env.scene.add_directional_light(light['fill_direction'], light['fill_rgb'])
    resource = d['resources']['environment']
    if env.scene.can_render() and resource:
        for subscene in env.scene.sub_scenes:
            subscene.set_environment_map(str(resolve(resource['resources']['environment'])))
    env.finite_interior_key_runtime_evidence = {**deepcopy(finite), 'native_light_readback': {
        'position_m': np.asarray(point.pose.p).tolist(), 'color': np.asarray(point.color).tolist(),
        'shadow': bool(point.shadow), 'shadow_near_m': float(point.shadow_near),
        'shadow_far_m': float(point.shadow_far), 'shadow_map_size': int(point.shadow_map_size)}}


@contextmanager
def install_for_worker():
    """Wrap the retained register entry in one isolated rendering process."""
    from mani_skill.utils.registration import REGISTERED_ENVS
    original = retained.register
    def register(scene, performance=None, contact_policy='native_default'):
        value = original(scene, performance, contact_policy)
        cls = REGISTERED_ENVS[retained.ENV_ID].cls
        old = cls._load_lighting
        def lighting(env, options=None):
            if 'finite_interior_key' in env.scene_spec.get('scene_domain', {}):
                return load_lighting(env)
            return old(env, options)
        cls._load_lighting = lighting
        return value
    retained.register = register
    try:
        yield
    finally:
        retained.register = original
