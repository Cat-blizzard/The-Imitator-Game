"""Sample actual furniture outside a saved task's protected motion region.

All models retain their native metric size. Title-based role/anchor evidence is
separate from the source's sometimes wrong category. Unsupported wall/ceiling
or furniture-top anchors are recorded and skipped. The existing RenderBody-only
installer handles the resulting furniture list; no physics actor is created.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import ROOT, binding, digest, read, resolve
from . import multitask_appearance_v5 as appearance
from . import scene_domain_v2 as domain

VERSION = 'generic_fullsize_background_composition_v1'
CATALOG = ROOT / 'assets/visual_furniture_bulk_20261030_v1/catalog.json'
TITLE_RULES = (
    ('lamp', r'\b(lamp|lamps|floor\s*light|table\s*light)\b'),
    ('bed', r'\b(bed|bedframe|bed\s*frame|headboard)\b'),
    ('sofa', r'\b(sofa|couch|loveseat|love\s*seat|sectional)\b'),
    ('ottoman', r'\b(ottoman|footstool|pouf)\b'),
    ('dresser', r'\b(dresser|drawer\s*chest|chest\s*of\s*\d*\s*drawers|chest\s*of\s*drawers)\b'),
    ('clothes_rack', r'\b(clothes|clothing|garment|coat)\s*(rack|stand)\b'),
    ('storage_rack', r'\b(storage|shoe|wire)\s*(rack|shelving|shelf|shelves)\b'),
    ('shelf', r'\b(shelf|shelves|shelving|bookcase|bookshelf|bookshelves)\b'),
    ('cabinet', r'\b(cabinet|sideboard|credenza|cupboard)\b'),
    ('desk', r'\bdesk\b'),
    ('table', r'\btable\b'),
    ('bench', r'\bbench\b'),
    ('stool', r'\b(stool|barstool)\b'),
    ('chair', r'\b(chair|armchair|recliner)\b'),
    ('storage_box', r'\b(storage\s*(box|bin|chest)|toy\s*box|storage\s*trunk)\b'),
)
GROUPS = {
    'office_storage': ('shelf', 'cabinet', 'desk'),
    'living_room': ('sofa', 'chair', 'table', 'shelf'),
    'workshop_storage': ('storage_rack', 'clothes_rack', 'storage_box'),
    'home_entry': ('dresser', 'bench', 'lamp'),
}


def semantic_anchor(entry):
    title = str(entry.get('name', '')).lower().replace('-', ' ')
    matches = [(category, pattern) for category, pattern in TITLE_RULES if re.search(pattern, title)]
    # Specific multiword roles precede their general noun: table lamp is a lamp,
    # storage shelving a rack, and a desk with drawers remains a desk.
    category = matches[0][0] if matches else 'unknown'
    evidence = [pattern for _, pattern in matches]
    anchor = 'floor' if category != 'unknown' else 'unknown'
    reason = 'standing_furniture_title_and_native_geometry_candidate'
    if re.search(r'\b(ceiling|chandelier|pendant)\b', title):
        anchor = 'ceiling'; reason = 'explicit_ceiling_or_suspended_title'
    elif re.search(r'\b(floating|wall\s*mount(?:ed)?|wall\s*(shelf|shelves|cabinet|mirror)|hanging)\b', title):
        anchor = 'wall'; reason = 'explicit_wall_or_floating_title'
    elif category == 'lamp' and (entry['size_m'][2] < .95 or re.search(r'\b(table|desk|bedside)\s*lamp\b', title)):
        anchor = 'furniture_top'; reason = 'table_lamp_title_or_height_below_95cm'
    elif category == 'unknown':
        reason = 'no_supported_title_evidence_not_guessed_from_source_category'
    return {'source_category': entry['category'], 'resolved_category': category,
        'category_conflict': category != entry['category'], 'title': entry.get('name'),
        'title_patterns': evidence, 'anchor': anchor, 'anchor_reason': reason,
        'semantic_confidence': 'explicit_title_rule_not_independent_visual_confirmation',
        'supported_by_current_installer': anchor == 'floor',
        'source_placement_support': entry.get('placement_support')}


def inspect_catalog(catalog_path=CATALOG):
    catalog = read(catalog_path); rows = []
    for entry in catalog['entries']:
        role = semantic_anchor(entry); reasons = []
        if entry['scale_confidence'] != 'source_metric_record_consistent': reasons.append('metric_dimensions_not_confirmed')
        if entry['asset_partition'] not in ('development', 'heldout_asset'): reasons.append('source_partition_unresolved')
        if not role['supported_by_current_installer']: reasons.append('anchor_not_supported:' + role['anchor'])
        if not entry.get('mesh_reload_passed'): reasons.append('mesh_not_reloaded')
        if entry.get('pick_place_eligible') is not False: reasons.append('background_role_not_exclusive')
        rows.append({'id': entry['id'], 'semantics': role, 'eligible': not reasons,
                     'reasons': reasons, 'asset_partition': entry['asset_partition']})
    return {'schema': VERSION, 'catalog': binding(catalog_path), 'rows': rows,
        'source_entries': len(rows), 'eligible_floor_backgrounds': sum(r['eligible'] for r in rows),
        'eligible_by_category': dict(Counter(r['semantics']['resolved_category'] for r in rows if r['eligible'])),
        'anchor_counts': dict(Counter(r['semantics']['anchor'] for r in rows)),
        'source_category_conflicts': sum(r['semantics']['category_conflict'] for r in rows),
        'source_metadata_edited': False, 'independent_visual_semantics_inferred': False}


def _select(catalog, analysis, group, seed, partition):
    if group not in GROUPS or partition not in ('development', 'heldout_asset'):
        raise ValueError('known_background_group_and_source_partition_required')
    entries = {r['id']: r for r in catalog['entries']}; selected = []; missing = []
    for category in GROUPS[group]:
        pool = [r for r in analysis['rows'] if r['eligible'] and r['asset_partition'] == partition
                and r['semantics']['resolved_category'] == category]
        # Keep the metric room finite; this excludes oversized candidates, it
        # never rescales them to fit a shared nominal slot.
        pool = [r for r in pool if entries[r['id']]['size_m'][2] <= 2.65
                and max(entries[r['id']]['size_m'][:2]) <= 3.2]
        if not pool:
            missing.append({'category': category, 'reason': 'no_metric_floor_title_and_partition_compatible_asset'})
            continue
        chosen = min(pool, key=lambda r: digest([VERSION, seed, group, category, r['id']]))
        selected.append({'entry': entries[chosen['id']], 'semantics': chosen['semantics']})
    if len(selected) < 2:
        raise ValueError('fewer_than_two_compatible_background_furniture_assets')
    return selected, missing


def _overlaps(a, b, gap=0.):
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a[1] > b[0] - gap) and np.all(a[0] < b[1] + gap))


def compose(prepared_scene, seed, *, catalog_path=CATALOG, group='office_storage', partition=None):
    """Generic full-size floor arrangement; no per-item location whitelist."""
    domain.validate_scene(prepared_scene)
    source_partition = prepared_scene.get('partition')
    partition = source_partition if partition is None else partition
    if source_partition in ('development', 'heldout_asset') and partition != source_partition:
        raise ValueError('background_source_partition_must_match_task_partition')
    if prepared_scene['scene_domain']['geometry'] is None: raise ValueError('workspace_geometry_required')
    catalog = read(catalog_path); analysis = inspect_catalog(catalog_path)
    selected, missing = _select(catalog, analysis, group, seed, partition)
    result = deepcopy(prepared_scene); d = result['scene_domain']; g = d['geometry']
    protected = np.asarray(g['protected_bounds_m']); margin = .18
    floor_z = -.78; front_limit = protected[0, 0] - margin
    # A tall anchor is centred first; successive models extend alternatingly to
    # either side. This is size-based and independent of source IDs and task RGB.
    selected.sort(key=lambda r: (-r['entry']['size_m'][2], r['semantics']['resolved_category']))
    furniture = []; left = right = None
    for index, chosen in enumerate(selected):
        entry = chosen['entry']; resolve(entry['mesh'])
        yaw = np.pi / 2  # Native -Y front faces +X toward the workspace.
        R = Rotation.from_euler('z', yaw).as_matrix(); points = domain.corners(entry['bounds_m']) @ R.T
        bounds = np.stack([points.min(0), points.max(0)]); width = bounds[1, 1] - bounds[0, 1]
        if index == 0: y = -float(bounds[:, 1].mean())
        elif index % 2: y = left - .22 - bounds[1, 1]
        else: y = right + .22 - bounds[0, 1]
        position = np.array([front_limit - bounds[1, 0], y, floor_z - bounds[0, 2]])
        world = bounds + position
        if _overlaps(world, protected, gap=.08): raise ValueError('furniture_enters_protected_motion_region')
        if any(_overlaps(world, r['world_bounds_m'], gap=.05) for r in furniture):
            raise ValueError('furniture_intersects_other_furniture')
        left = float(world[0, 1]) if left is None else min(left, float(world[0, 1]))
        right = float(world[1, 1]) if right is None else max(right, float(world[1, 1]))
        furniture.append({'id': f'bulk_furniture_{index:02d}', 'source_id': entry['id'],
            'source': entry['source'], 'mesh': entry['mesh'], 'catalog': binding(catalog_path),
            'position_m': position.tolist(), 'yaw_rad': float(yaw), 'scale': [1., 1., 1.],
            'world_bounds_m': world.tolist(), 'source_size_m': entry['size_m'],
            'collision_shapes_added': 0, 'semantics': chosen['semantics'],
            'source_partition': entry['asset_partition'], 'split_group': entry['split_group'],
            'native_metric_scale_preserved': True, 'source_record': entry['preparation_record']})
    occupied = domain.union_bounds([r['world_bounds_m'] for r in furniture])
    rear = min(protected[0, 0] - .6, occupied[0, 0] - .16)
    front = protected[1, 0] + .85
    room_left = min(protected[0, 1] - .58, occupied[0, 1] - .16)
    room_right = max(protected[1, 1] + .58, occupied[1, 1] + .16)
    ceiling = max(2.05, occupied[1, 2] + .20, protected[1, 2] + .60)
    boxes = []
    def box(name, center, size, role):
        boxes.append({'id': name, 'kind': 'box', 'center_m': center, 'size_m': size,
                      'material_role': role, 'tint_rgba': [1., 1., 1., 1.], 'visual_only': True})
    midx, midy = (rear + front)/2, (room_left + room_right)/2
    box('floor', [midx, midy, floor_z-.04], [front-rear+.16, room_right-room_left+.16, .08], 'floor')
    box('ceiling', [midx, midy, ceiling+.04], [front-rear+.16, room_right-room_left+.16, .08], 'wall')
    box('rear_wall', [rear, midy, (floor_z+ceiling)/2], [.08, room_right-room_left, ceiling-floor_z], 'wall')
    box('left_wall', [midx, room_left, (floor_z+ceiling)/2], [front-rear, .08, ceiling-floor_z], 'wall')
    box('right_wall', [midx, room_right, (floor_z+ceiling)/2], [front-rear, .08, ceiling-floor_z], 'wall')
    box('front_far_wall', [front, midy, (floor_z+ceiling)/2], [.08, room_right-room_left, ceiling-floor_z], 'wall')
    room_bounds = np.array([[rear, room_left, floor_z], [front, room_right, ceiling]])
    camera_checks = {}
    for name, camera in result['cameras'].items():
        eye = np.asarray(camera['eye']); target = np.asarray(camera['target'])
        inside = bool(np.all(eye > room_bounds[0]+.08) and np.all(eye < room_bounds[1]-.08))
        if not inside: raise ValueError('original_camera_outside_resized_room:'+name)
        # The eye-to-workspace segment lies inside the preserved convex
        # protected prism. Every new solid AABB is outside that prism.
        protected_eye = bool(np.all(eye >= protected[0]) and np.all(eye <= protected[1]))
        protected_target = bool(np.all(target >= protected[0]) and np.all(target <= protected[1]))
        if not protected_eye or not protected_target:
            raise ValueError('original_camera_corridor_not_protected:'+name)
        camera_checks[name] = {'eye_inside_room': inside, 'eye_and_target_inside_protected_prism': True,
                              'line_of_sight_to_critical_objects': 'preserved original protected prism, not a pixel visibility certificate'}
    for item in boxes:
        center = np.asarray(item['center_m']); half = np.asarray(item['size_m'])/2
        if _overlaps([center-half, center+half], protected, gap=.06):
            raise ValueError('resized_room_enters_protected_motion_region')
    old_ids = [b['id'] for b in g['boxes']]
    g.update(boxes=boxes, furniture=furniture,
             structural_signature=digest({'boxes': boxes, 'furniture': furniture}))
    d['bulk_furniture_policy'] = {'schema': VERSION, 'implementation': binding(__file__),
        'catalog': binding(catalog_path), 'seed': seed, 'group': group, 'source_partition': partition,
        'selected': [r['source_id'] for r in furniture], 'missing_categories': missing,
        'source_category_conflicts_resolved_explicitly': [r['source_id'] for r in furniture if r['semantics']['category_conflict']],
        'unsupported_anchors_skipped': True, 'old_procedural_elements_replaced': old_ids,
        'placement_rule': 'native -Y-front toward workspace; tall first then alternate size-based left/right on floor outside complete sweep',
        'room_bounds_m': room_bounds.tolist(), 'original_camera_checks': camera_checks,
        'floor_height_m': floor_z,
        'floor_height_basis': 'inherited workspace convention: physical task tabletop z=0, room floor z=-0.78; source furniture bottom z=0',
        'native_scale_preserved': True, 'new_physical_actors': 0, 'new_collision_shapes': 0,
        'visual_semantics_requires_image_review': True, 'R2H_ready': False}
    appearance.assert_invariants(prepared_scene, result)
    domain.validate_scene(result)
    return result


def apply_overlay(scene, seed, *, motion_envelope, catalog_path=CATALOG,
                  group='office_storage', preset='office_bookcase', partition=None):
    prepared, receipt = appearance.apply_overlay(scene, seed, mode='workspace', preset=preset,
                                                motion_envelope=motion_envelope)
    if not receipt['appearance_applied']:
        return prepared, {**receipt, 'bulk_furniture_status': 'appearance_unavailable'}
    result = compose(prepared, seed, catalog_path=catalog_path, group=group, partition=partition)
    return result, {**receipt, 'implementation': binding(__file__),
        'bulk_furniture_status': 'prepared_requires_render_validation',
        'result_scene_digest': digest(result), 'bulk_policy': result['scene_domain']['bulk_furniture_policy']}
