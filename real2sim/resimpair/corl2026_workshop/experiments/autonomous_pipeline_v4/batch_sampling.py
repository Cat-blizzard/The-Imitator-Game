"""Seeded, bounded scene and task sampling over existing contact mechanisms.

This module is pure preparation: no simulator, image API or execution retry is
invoked. Static eligibility never implies physical qualification. Existing
``novel_catalog`` recipes remain unchanged. Runtime integration must use the
batch protocol/solver hooks for translated tool goals and target order.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .core import digest, require
from .functional_assets import geometry, preflight
from .novel_catalog import box, scene as base_scene, task as base_task

VERSION = 'parameterized_v1'
KINDS = ('spring_console', 'rocker_console', 'linear_faders',
         'rotary_indexer', 'hook_retrieval', 'granule_sweeping')
LAYOUT_PARTITIONS = {0: 'development', 1: 'development', 2: 'development',
                     3: 'heldout_layout', 4: 'heldout_layout'}
LAYOUT_CENTERS = ((0., 0.), (-.014, .012), (.014, -.012),
                  (-.020, -.015), (.020, .015))
TRAJECTORY_VARIATIONS = (
    # Relative multipliers of the retained planner's 0.40 velocity scale:
    # effective velocity scales remain 0.40, 0.34 and 0.46.
    {'speed_scale': 1.0, 'clearance_delta_m': 0.},
    {'speed_scale': .85, 'clearance_delta_m': .020},
    {'speed_scale': 1.15, 'clearance_delta_m': .040},
)


def _translate(item, field, offset):
    item[field] = np.round(np.asarray(item[field], dtype=float) + offset, 8).tolist()


def _box_bounds(item):
    lo, hi = np.asarray(item['geometry']['bounds_local_m'], dtype=float)
    require(item.get('yaw', 0.) == 0., 'batch_preflight_requires_unrotated_geometry')
    origin = np.asarray(item['position'], dtype=float)
    return lo + origin, hi + origin


def _joint_bounds(item):
    center = np.asarray(item['pivot_world'], dtype=float)
    parts = item['parts']
    lo = np.min([np.asarray(p['center']) - np.asarray(p['half_size']) for p in parts], axis=0)
    hi = np.max([np.asarray(p['center']) + np.asarray(p['half_size']) for p in parts], axis=0)
    return lo + center, hi + center


def _separated(a, b, margin=.018):
    # Planar separation intentionally ignores vertical separation: background
    # clutter may not hang over functional assets or occupy their footprint.
    alo, ahi = a; blo, bhi = b
    return bool(np.any(ahi[:2] + margin <= blo[:2]) or np.any(bhi[:2] + margin <= alo[:2]))


def _functional_variants(spec, rng):
    """Finite compatible shape candidates, with explicit selection evidence.

    Functional feature centers, lever radii and contact heights remain fixed.
    These are engineering variants of primitive assets, not downloaded assets
    and not pre-certified new asset families.
    """
    variant = int(rng.integers(0, 3))
    records = []
    kind = spec['id']
    for j in spec['joints']:
        if kind == 'spring_console':
            half = ((.035, .035), (.039, .039), (.043, .035))[variant]
            j['parts'][0]['half_size'][:2] = list(half)
            constraint = 'press cap contains fixed contact center; contact height and travel unchanged'
        elif kind == 'rocker_console':
            j['parts'][0]['half_size'][1] = (.024, .027, .030)[variant]
            constraint = 'lever radius and vertical thickness unchanged'
        elif kind == 'linear_faders':
            j['parts'][0]['half_size'][1] = (.027, .030, .033)[variant]
            constraint = 'slider half-width <= 0.033 m inside 0.038 m rail half-gap'
        else:
            # Only alter the pedestal; knob center, radius and collision
            # interface of the rotary control are retained.
            constraint = 'rotary knob radius and contact geometry unchanged'
        records.append({'entity': j['id'], 'role': 'passive_control', 'profile_index': variant,
                        'compatibility_constraint': constraint, 'native_qualification': 'pending'})
    objects = {o['id']: o for o in spec['objects']}
    if kind == 'rotary_indexer':
        objects['dial_pedestal']['half_size'][:2] = list(((.108, .108), (.115, .115), (.122, .112))[variant])
    if kind in {'hook_retrieval', 'granule_sweeping'}:
        tool = objects['hook' if kind == 'hook_retrieval' else 'paddle']
        width = ((.011, .012, .013) if kind == 'hook_retrieval' else (.012, .013, .014))[variant]
        tool['parts'][0]['half_size'][1] = width
        tool['grasp_feature']['grip_width_m'] = 2 * width
        if kind == 'hook_retrieval':
            tool['parts'][2]['half_size'][1] = (.052, .055, .058)[variant]
        else:
            tool['parts'][1]['half_size'][1] = (.074, .078, .082)[variant]
        records.append({'entity': tool['id'], 'role': 'graspable_tool', 'profile_index': variant,
                        'grip_width_m': 2 * width, 'maximum_grip_width_m': .075,
                        'functional_center_and_contact_height_preserved': True,
                        'native_qualification': 'pending'})
    return records


def _module_index(oid, kind):
    if kind in {'spring_console', 'rocker_console', 'linear_faders'}:
        if oid.startswith('housing_') or oid.startswith('rail_'):
            return int(oid.split('_')[1])
    elif kind == 'hook_retrieval':
        if oid.startswith('cargo_destination_') or oid.startswith('cargo_'):
            return int(oid.rsplit('_', 1)[1])
    elif kind == 'granule_sweeping':
        if oid.startswith('grain_'):
            return int(oid.split('_')[1])
        if oid.startswith('collector_'):
            return int(oid.rsplit('_', 1)[1])
    return None


def static_eligibility(spec):
    """Cheap structural screen; does not certify IK, contact or task success."""
    reasons = []
    try:
        preflight(spec)
    except (ValueError, KeyError) as exc:
        reasons.append('functional_asset:' + str(exc))
    targets = [o for o in spec['objects'] if not o['id'].startswith('background_')]
    nuisances = [o for o in spec['objects'] if o['id'].startswith('background_')]
    occupied = [(_box_bounds(o), o['id']) for o in targets]
    occupied += [(_joint_bounds(j), j['id']) for j in spec['joints']]
    for i, obj in enumerate(nuisances):
        for bounds, oid in occupied + [(_box_bounds(n), n['id']) for n in nuisances[:i]]:
            if not _separated(_box_bounds(obj), bounds):
                reasons.append(f'background_overlaps_functional_footprint:{obj["id"]}:{oid}')
    for obj in targets:
        if obj['kind'] == 'movable':
            x, y, z = obj['position']
            if not (-.42 <= x <= .14 and -.52 <= y <= .47 and 0. <= z <= .20):
                reasons.append('conservative_movable_workspace:' + obj['id'])
    for joint in spec['joints']:
        x, y, z = joint['pivot_world']
        if not (-.35 <= x <= .08 and -.40 <= y <= .40 and .02 <= z <= .22):
            reasons.append('conservative_control_workspace:' + joint['id'])
    return {'eligible': not reasons, 'reasons': reasons,
            'scope': 'initial geometry and conservative workspace only; native execution unverified'}


def sample_scene(kind, seed, layout_index=0):
    require(kind in KINDS, 'unsupported_batch_mechanism')
    require(type(seed) is int and 0 <= seed < 2**31, 'batch_seed_must_be_nonnegative_int32')
    require(type(layout_index) is int and layout_index in LAYOUT_PARTITIONS, 'batch_layout_index_must_be_0_to_4')
    rng = np.random.default_rng(np.random.SeedSequence([seed, layout_index, KINDS.index(kind), 20261025]))
    spec = deepcopy(base_scene(kind, seed))
    delta = np.r_[np.asarray(LAYOUT_CENTERS[layout_index]) + rng.uniform(-.004, .004, 2), 0.]
    module_offsets = {k: np.array([0., float(rng.uniform(-.017, .017)), 0.]) for k in range(2)}
    for obj in spec['objects']:
        _translate(obj, 'position', delta)
        index = _module_index(obj['id'], kind)
        if index is not None:
            _translate(obj, 'position', module_offsets[index])
    for index, joint in enumerate(spec['joints']):
        _translate(joint, 'pivot_world', delta)
        if len(spec['joints']) == 2:
            _translate(joint, 'pivot_world', module_offsets[index])
    selections = _functional_variants(spec, rng)
    for obj in spec['objects']:
        obj['geometry'] = geometry(obj)

    # Background candidates have explicit footprints. Rejected candidates are
    # recorded before physical execution; they are never hidden execution retries.
    occupied = [_box_bounds(o) for o in spec['objects']] + [_joint_bounds(j) for j in spec['joints']]
    background = []
    requested = int(rng.integers(2, 5))
    for i, y in enumerate((-.46, -.14, .23, .46)):
        half = [float(rng.uniform(.014, .023)), float(rng.uniform(.014, .023)), float(rng.uniform(.013, .028))]
        obj = box('background_' + str(i), [float(rng.uniform(.337, .353)), y + float(rng.uniform(-.010, .010)), half[2]],
                  half, ('wood', 'green', 'gray', 'white')[i], description='untouched background cuboid')
        obj['geometry'] = geometry(obj)
        compatible = all(_separated(_box_bounds(obj), bounds) for bounds in occupied)
        selected = compatible and len([x for x in background if x['selected']]) < requested
        background.append({'entity': obj['id'], 'candidate': obj, 'compatible': compatible,
                           'selected': selected, 'reason': 'clear_of_initial_functional_footprints' if compatible else 'footprint_overlap'})
        if selected:
            spec['objects'].append(obj); occupied.append(_box_bounds(obj))
    spec['appearance'] = {
        'table_rgb': rng.uniform(.32, .62, 3).round(5).tolist(),
        'background_rgb': rng.uniform(.61, .82, 3).round(5).tolist(),
        'ambient_rgb': [.38] * 3, 'key_light_rgb': [1.5] * 3,
    }
    spatial = {'objects': [{k: o[k] for k in ('id', 'position', 'shape', 'geometry')} for o in spec['objects']],
               'joints': [{k: j[k] for k in ('id', 'pivot_world', 'parts', 'limits')} for j in spec['joints']]}
    layout_hash = digest(spatial)
    instance = f'{kind}_s{seed}_l{layout_index}_{layout_hash[:8]}'
    spec.update(batch_generation=VERSION, scene_instance_id=instance, split_group=instance,
                partition=LAYOUT_PARTITIONS[layout_index],
                layout_transform={'translation_world_m': np.round(delta, 8).tolist(),
                                  'module_offsets_world_m': {str(k): v.round(8).tolist() for k, v in module_offsets.items()}},
                sampling={'version': VERSION, 'seed': seed, 'layout_index': layout_index,
                          'partition': LAYOUT_PARTITIONS[layout_index], 'layout_sha256': layout_hash,
                          'asset_variants': selections, 'background_candidates': background,
                          'requested_background_count': requested,
                          'selected_background_count': sum(x['selected'] for x in background),
                          'source_coverage': 'missing', 'physical_qualification': 'pending'})
    spec['static_eligibility'] = static_eligibility(spec)
    require(spec['static_eligibility']['eligible'], 'sampled_scene_static_ineligible:' + ','.join(spec['static_eligibility']['reasons']))
    return spec


def enumerate_tasks(scene):
    require(scene.get('batch_generation') == VERSION and scene['id'] in KINDS, 'sampled_scene_required')
    require(static_eligibility(scene)['eligible'], 'task_enumeration_requires_eligible_scene')
    tasks = []
    for variant in range(4):
        task = base_task(scene, min(variant, 2), 'sampled')
        params = deepcopy(task['planning_parameters'])
        if scene['id'] == 'rotary_indexer':
            params.update(direction=1 if variant in (0, 2) else -1, return_to_origin=variant >= 2)
            label = ('positive', 'negative', 'positive_then_return', 'negative_then_return')[variant]
            groups = [['indexed_dial']]
        else:
            selected = ([0], [1], [0, 1], [1, 0])[variant]
            params['selected'] = list(selected)
            label = ('target_0', 'target_1', 'target_0_then_1', 'target_1_then_0')[variant]
            if scene['joints']:
                groups = [[scene['joints'][i]['id']] for i in selected]
            elif scene['id'] == 'hook_retrieval':
                groups = [['cargo_' + str(i)] for i in selected]
            else:
                groups = [[f'grain_{i}_{j}' for j in range(3)] for i in selected]
        template_id = scene['scene_instance_id'] + '__' + label
        task.update(id=template_id, task_template_id=template_id, batch_generation=VERSION,
                    scene_instance_id=scene['scene_instance_id'], split_group=scene['split_group'],
                    partition=scene['partition'], branch=variant, task_variant=label,
                    planning_parameters=params, ordered_target_groups=groups,
                    source_status='designed_mechanism_probe_without_verified_real_video_window',
                    physical_qualification='pending')
        task['operations'][0]['parameters'] = deepcopy(params)
        task['goal_description'] += '; target sequence ' + ' then '.join(','.join(g) for g in groups)
        if scene['id'] == 'rotary_indexer':
            task['goal_description'] += f'; direction {params["direction"]}; return {params["return_to_origin"]}'
        tasks.append(task)
    return tasks


def vary_task(task, variation_index):
    require(task.get('batch_generation') == VERSION, 'sampled_task_required')
    require(type(variation_index) is int and variation_index in range(len(TRAJECTORY_VARIATIONS)),
            'batch_trajectory_variant_must_be_0_to_2')
    result = deepcopy(task)
    template = task['task_template_id']
    result['id'] = template + f'__v{variation_index}'
    result['trajectory_variation'] = {'index': variation_index, **TRAJECTORY_VARIATIONS[variation_index],
                                      'speed_scale_semantics': 'relative_multiplier_of_baseline_0.40',
                                      'goal_changed': False, 'initial_scene_changed': False,
                                      'physical_dynamic_difference': 'must_be_measured_after_execution'}
    return result
