"""Opt-in metric 3-D workspaces and bounded camera variation for saved states.

Only appearance, scene_domain and cameras may change. Room geometry is attached
as SAPIEN RenderBody entities, never as ManiSkill actors or collision bodies.
Sampling is CPU-only; register() is the explicit rendering integration boundary.
This module does not execute, qualify or count a new physical trajectory.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from itertools import product
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import ROOT, binding, digest, read, resolve

VERSION = 'resimpair_scene_domain_v2'
CATALOG_VERSION = 'resimpair_appearance_catalog_v2'
ENV_ID = 'ReSimPairSceneDomain-v2'
DEFAULT_CATALOG = ROOT / 'assets/appearance_v2/catalog.json'
URDF = ROOT / 'PolicyGen/mani_skill/assets/robots/panda/panda_v2.urdf'
MUTABLE_FIELDS = {'appearance', 'scene_domain', 'cameras'}
PRESETS = {
    'kitchen_window_wall': ('kitchen', 'window wall and low island cabinets'),
    'kitchen_l_return': ('kitchen', 'L-shaped side and rear counters'),
    'kitchen_galley': ('kitchen', 'two facing counter runs with upper cabinets'),
    'kitchen_pantry': ('kitchen', 'tall pantry and open shelf alcove'),
    'dining_corner': ('dining', 'corner wainscot and side table'),
    'dining_bay_window': ('dining', 'three-sided bay window with deep sill'),
    'dining_sideboard': ('dining', 'long sideboard and asymmetrical wall panels'),
    'dining_partition': ('dining', 'open timber slat partition and low console'),
    'office_window_desk': ('office', 'window ledge and rear desk credenza'),
    'office_dual_shelves': ('office', 'staggered shelving on perpendicular walls'),
    'office_cubicle': ('office', 'three partial-height cubicle partitions'),
    'office_bookcase': ('office', 'tall bookcase and shallow wall desk'),
    'workshop_pegboard': ('workshop', 'framed pegboard with rails and low drawers'),
    'workshop_tool_cabinets': ('workshop', 'two unequal tool cabinets and overhead rail'),
    'workshop_overhead_shelf': ('workshop', 'suspended perimeter shelf and supporting posts'),
    'workshop_assembly_bay': ('workshop', 'three structural portals and side workbench'),
    'storage_rack_aisle': ('storage', 'parallel three-level open racks'),
    'storage_corner_racks': ('storage', 'perpendicular unequal-height corner racks'),
    'storage_bins_wall': ('storage', 'rear cubby wall with staggered storage bins'),
    'storage_loading_bay': ('storage', 'loading door, jambs, pallet stack and short rack'),
}
PRESET_IDS = tuple(PRESETS)
ROLES = ('tabletop', 'wall', 'floor', 'cabinet', 'metal')
DEFAULT_TEXTURE_METERS = {'tabletop': [1., 1.], 'wall': [2., 2.],
    'floor': [2., 2.], 'cabinet': [1., 1.], 'metal': [1., 1.]}
TEMPERATURE_RGB = {3500: [1., .76, .54], 4500: [1., .86, .70],
                   6500: [1., .97, .95]}
CAMERA_LIMITS = {'eye_delta_m': [.10, .13, .09], 'target_delta_m': [.025, .04, .03],
                 'fov_delta_rad': [-.025, .13], 'max_attempts': 24,
                 'critical_margin_fraction': .012, 'minimum_object_size_pixels': 6.,
                 'robot_coverage_absolute_tolerance': .005}


def physical_scene(scene):
    return {k: v for k, v in scene.items() if k not in MUTABLE_FIELDS}


def corners(bounds):
    b = np.asarray(bounds, dtype=float)
    if b.shape != (2, 3) or not np.isfinite(b).all() or np.any(b[1] < b[0]):
        raise ValueError('invalid_metric_bounds')
    return np.array(list(product(*zip(b[0], b[1]))))


def union_bounds(rows):
    b = np.asarray(rows, dtype=float)
    return np.stack([b[:, 0].min(axis=0), b[:, 1].max(axis=0)])


def transform_bounds(bounds, transforms):
    p = corners(bounds); transforms = np.asarray(transforms)
    p = np.einsum('...ij,kj->...ki', transforms[..., :3, :3], p) + transforms[..., None, :3, 3]
    return np.stack([p.reshape(-1, 3).min(axis=0), p.reshape(-1, 3).max(axis=0)])


def _origin(node):
    t = np.eye(4)
    if node is not None:
        t[:3, 3] = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
        t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(node.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return t


@lru_cache(maxsize=128)
def _mesh_bounds(path):
    import trimesh
    mesh = trimesh.load(str(path), process=False)
    bounds = np.asarray(mesh.bounds, dtype=float)
    corners(bounds)
    return bounds


@lru_cache(maxsize=1)
def _robot_model():
    """Full native URDF chain and visual/collision bounds, including fingers."""
    root = ET.parse(URDF).getroot(); links = {}; resources = [binding(URDF)]
    for link in root.findall('link'):
        bounds = []
        for part in list(link.findall('visual')) + list(link.findall('collision')):
            geometry = part.find('geometry'); mesh = geometry.find('mesh')
            if mesh is not None:
                path = URDF.parent / mesh.get('filename')
                scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                b = _mesh_bounds(str(path)) * scale
                resources.append(binding(path))
            elif geometry.find('box') is not None:
                half = np.fromstring(geometry.find('box').get('size'), sep=' ') / 2
                b = [-half, half]
            elif geometry.find('cylinder') is not None:
                c = geometry.find('cylinder'); half = np.array([float(c.get('radius'))] * 2 + [float(c.get('length')) / 2])
                b = [-half, half]
            elif geometry.find('sphere') is not None:
                half = np.repeat(float(geometry.find('sphere').get('radius')), 3); b = [-half, half]
            else:
                raise ValueError('unsupported_robot_geometry')
            bounds.append(transform_bounds(b, _origin(part.find('origin'))))
        if bounds: links[link.get('name')] = union_bounds(bounds)
    joints = []
    for j in root.findall('joint'):
        axis = j.find('axis')
        joints.append({'name': j.get('name'), 'type': j.get('type'),
            'parent': j.find('parent').get('link'), 'child': j.find('child').get('link'),
            'origin': _origin(j.find('origin')),
            'axis': np.fromstring(axis.get('xyz'), sep=' ') if axis is not None else np.zeros(3)})
    return links, joints, resources


def _all_robot_poses(qpos, robot_root):
    _, joints, _ = _robot_model()
    qpos = np.asarray(qpos); n = len(qpos)
    index = {f'panda_joint{i+1}': i for i in range(7)}
    index.update(panda_finger_joint1=7, panda_finger_joint2=8)
    poses = {'panda_link0': np.broadcast_to(robot_root, (n, 4, 4)).copy()}
    pending = list(joints)
    while pending:
        progress = False
        for joint in list(pending):
            if joint['parent'] not in poses: continue
            delta = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            if joint['type'] in ('revolute', 'continuous', 'prismatic'):
                q = qpos[:, index[joint['name']]]
                if joint['type'] == 'prismatic': delta[:, :3, 3] = q[:, None] * joint['axis']
                else: delta[:, :3, :3] = Rotation.from_rotvec(q[:, None] * joint['axis']).as_matrix()
            elif joint['type'] != 'fixed': raise ValueError('unsupported_robot_joint')
            poses[joint['child']] = poses[joint['parent']] @ joint['origin'] @ delta
            pending.remove(joint); progress = True
        if not progress: raise ValueError('robot_joint_graph_not_connected')
    return poses


def _item_bounds(item):
    ref = item.get('asset_descriptor', item.get('container_descriptor'))
    if ref:
        descriptor = read(resolve(ref))
        if descriptor.get('visual'):
            return _mesh_bounds(str(resolve(descriptor['visual']))), {'descriptor': ref, 'visual': descriptor['visual']}
        if 'bounds_m' in descriptor: return np.asarray(descriptor['bounds_m']), {'descriptor': ref}
    half = np.asarray(item['half_size'], dtype=float)
    return np.stack([-half, half]), {'source': 'declared_primitive_half_size'}


def build_motion_envelope(scene, measurements_path):
    """CPU full-timeline mesh bounds and independent FK; no simulator or GPU.

    All source object slots must match scene order. Native saved-state exporters
    already bind that order; this helper records it explicitly. Conservative
    swept AABBs are proposals, not an occlusion or image-quality certificate.
    """
    with np.load(measurements_path, allow_pickle=False) as z: arrays = {k: z[k] for k in z.files}
    n = len(arrays['timestamps_seconds']); states = arrays['object_states']
    if states.shape != (n, len(scene['objects']), 13) or arrays['states'].shape != (n, 18):
        raise ValueError('motion_envelope_measurement_shape_mismatch')
    if any(not np.isfinite(arrays[k]).all() for k in ('states', 'object_states', 'tcp_world', 'robot_roots_world')):
        raise ValueError('motion_envelope_nonfinite_measurements')
    rows = []
    for i, item in enumerate(scene['objects']):
        local, evidence = _item_bounds(item)
        transforms = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        transforms[:, :3, :3] = Rotation.from_quat(states[:, i, [4, 5, 6, 3]]).as_matrix()
        transforms[:, :3, 3] = states[:, i, :3]
        world_corners=np.einsum('nij,kj->nki',transforms[:,:3,:3],corners(local))+transforms[:,None,:3,3]
        frame_bounds=np.stack([world_corners.min(axis=1),world_corners.max(axis=1)],axis=1)
        rows.append({'id': item['id'], 'group': 'object',
            'critical': item.get('kind') != 'static_distractor',
            'local_bounds_m': local.tolist(), 'bounds_m': transform_bounds(local, transforms).tolist(),
            'frame_bounds_m': frame_bounds.tolist(),
            'geometry': evidence, 'timeline_frames': n})
    for item in scene.get('supports', []):
        local, evidence = _item_bounds(item); transform = np.eye(4)
        transform[:3, 3] = item['position']; transform[:3, :3] = Rotation.from_euler('z', item.get('yaw', 0)).as_matrix()
        rows.append({'id': item['id'], 'group': 'container', 'critical': True,
            'local_bounds_m': local.tolist(), 'bounds_m': transform_bounds(local, transform).tolist(),
            'geometry': evidence, 'timeline_frames': n})
    links, _, resources = _robot_model(); fk_error = 0.
    for arm in range(2):
        poses = _all_robot_poses(arrays['states'][:, 9*arm:9*(arm+1)], arrays['robot_roots_world'][arm])
        fk_error = max(fk_error, float(np.max(np.abs(poses['panda_hand_tcp'] - arrays['tcp_world'][:, arm]))))
        for name, bounds in links.items():
            rows.append({'id': f'arm{arm}/{name}', 'group': 'robot', 'critical': False,
                'bounds_m': transform_bounds(bounds, poses[name]).tolist(), 'timeline_frames': n})
    if fk_error > 3e-6: raise ValueError('independent_robot_FK_does_not_match_measurements')
    swept = union_bounds([r['bounds_m'] for r in rows])
    return {'schema': 'resimpair_full_motion_envelope_v2', 'source_measurements': binding(measurements_path),
        'physical_scene_digest': digest(physical_scene(scene)), 'frame_count': n,
        'object_order': [o['id'] for o in scene['objects']], 'rows': rows,
        'full_swept_bounds_m': swept.tolist(), 'maximum_fk_matrix_error': fk_error,
        'robot_geometry': resources, 'sampling': 'every measured frame; swept bounds include complete mesh local AABB',
        'occlusion_checked': False, 'visibility_status': 'geometric_proposal_only_requires_rendered_pixel_checks'}


def camera_basis(camera):
    eye = np.asarray(camera['eye'], dtype=float); forward = np.asarray(camera['target']) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.]); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    if not np.isfinite([forward, right, down]).all(): raise ValueError('degenerate_camera')
    return np.stack([right, down, forward]), eye


def camera_projection(camera, bounds):
    """Project all eight corners, retaining behind-camera and crop diagnostics."""
    rotation, eye = camera_basis(camera); xyz = (corners(bounds) - eye) @ rotation.T
    tangent = math.tan(float(camera['fov']) / 2); aspect = camera['width'] / camera['height']
    depth_ok = bool((xyz[:, 2] > .01).all())
    if not depth_ok: return {'in_front': False, 'fully_in_frame': False, 'visible_area_fraction': 0., 'minimum_size_pixels': 0.}
    uv = .5 + xyz[:, :2] / xyz[:, 2:3] / [2 * tangent * aspect, 2 * tangent]
    low, high = uv.min(axis=0), uv.max(axis=0)
    area = np.prod(np.maximum(high-low, 0)); clipped = np.maximum(np.minimum(high, 1)-np.maximum(low, 0), 0)
    return {'in_front': True, 'fully_in_frame': bool((low >= 0).all() and (high <= 1).all()),
        'uv_min': low.tolist(), 'uv_max': high.tolist(),
        'visible_area_fraction': float(np.prod(clipped) / max(float(area), 1e-15)),
        'minimum_size_pixels': float(np.min((high-low) * [camera['width'], camera['height']]))}


def camera_coverage(camera, envelope):
    result={}
    rotation,eye=camera_basis(camera)
    factors=np.array(list(product((0.,1.),repeat=3)))
    tangent=math.tan(float(camera['fov'])/2);aspect=camera['width']/camera['height']
    for row in envelope['rows']:
        coverage={'group':row['group'],'critical':row['critical'],**camera_projection(camera,row['bounds_m'])}
        if row.get('frame_bounds_m') is not None:
            b=np.asarray(row['frame_bounds_m'])
            points=b[:,None,0,:]+factors[None]*(b[:,None,1,:]-b[:,None,0,:])
            xyz=(points-eye)@rotation.T
            if np.any(xyz[:,:,2]<=.01): coverage['minimum_size_pixels']=0.
            else:
                uv=xyz[:,:,:2]/xyz[:,:,2:]/[2*tangent*aspect,2*tangent]
                per_frame_size=(uv.max(axis=1)-uv.min(axis=1))*[camera['width'],camera['height']]
                coverage['minimum_size_pixels']=float(per_frame_size.min())
            coverage['minimum_size_scope']='minimum per-frame full mesh AABB projection, all measured frames'
        result[row['id']]=coverage
    return result


def sample_camera(scene, rng, envelope):
    if envelope is None or envelope['physical_scene_digest'] != digest(physical_scene(scene)):
        raise ValueError('bounded_camera_requires_bound_full_motion_envelope')
    result = deepcopy(scene['cameras']); source = result['opposite_high']; baseline = camera_coverage(source, envelope)
    attempts = []; accepted = None
    for index in range(CAMERA_LIMITS['max_attempts']):
        c = deepcopy(source)
        c['eye'] = (np.asarray(source['eye']) + rng.uniform(-1, 1, 3) * CAMERA_LIMITS['eye_delta_m']).tolist()
        c['target'] = (np.asarray(source['target']) + rng.uniform(-1, 1, 3) * CAMERA_LIMITS['target_delta_m']).tolist()
        c['fov'] = float(source['fov'] + rng.uniform(*CAMERA_LIMITS['fov_delta_rad']))
        coverage = camera_coverage(c, envelope); reasons = []
        for key, row in coverage.items():
            if row['critical']:
                margin = CAMERA_LIMITS['critical_margin_fraction']
                if (not row['in_front'] or not row['fully_in_frame'] or
                        min(row.get('uv_min', [-1])) < margin or max(row.get('uv_max', [2])) > 1-margin):
                    reasons.append('critical_bbox_clipped:' + key)
                if row['minimum_size_pixels'] < CAMERA_LIMITS['minimum_object_size_pixels']:
                    reasons.append('critical_bbox_too_small:' + key)
            if row['group'] == 'robot' and (row['visible_area_fraction'] + CAMERA_LIMITS['robot_coverage_absolute_tolerance'] < baseline[key]['visible_area_fraction']):
                reasons.append('robot_swept_coverage_worse_than_original:' + key)
        attempts.append({'attempt': index, 'config': c, 'reasons': reasons, 'accepted': not reasons})
        if not reasons: accepted = c; break
    if accepted is not None: result['opposite_high'] = accepted
    return result, {'mode': 'bounded', 'limits': deepcopy(CAMERA_LIMITS), 'source_configs': deepcopy(scene['cameras']),
        'actual_configs': deepcopy(result), 'modified_camera': 'opposite_high' if accepted else None,
        'status': 'accepted_geometric_proposal' if accepted else 'no_feasible_proposal_retained_original',
        'attempts': attempts, 'baseline_coverage': baseline,
        'actual_coverage': camera_coverage(result['opposite_high'], envelope),
        'all_robot_geometry_in_frame_claimed': False,
        'baseline_robot_crop_policy': 'full swept mesh bbox must not lose visible area versus original; existing crop is reported',
        'pixel_occlusion_check_required': True}


def load_catalog(path):
    value = read(path)
    if value.get('schema') != CATALOG_VERSION: raise ValueError('scene_domain_requires_catalog_v2')
    seen = set()
    for row in value['entries']:
        if row['id'] in seen or row['kind'] not in ('material', 'hdri'): raise ValueError('invalid_catalog_entry')
        seen.add(row['id'])
        required = 'environment' if row['kind'] == 'hdri' else 'base_color'
        if required not in row['resources']: raise ValueError('catalog_missing_required_channel')
    return value


def _choose_resource(catalog, rng, role, category, asset_partition):
    kind = 'hdri' if role == 'environment' else 'material'
    pool = [r for r in catalog['entries'] if r['kind'] == kind and role in r['roles']
            and (asset_partition is None or r.get('asset_partition') == asset_partition)]
    compatible = [r for r in pool if category in r.get('categories', [])]
    reason = None
    if not compatible:
        compatible = [r for r in pool if 'generic_indoor' in r.get('categories', [])]
        reason = 'no_category_specific_resource; explicit_generic_indoor_fallback'
    if not compatible:
        if role == 'environment':
            return None, 'no_semantically_compatible_HDRI; room_geometry_and_explicit_lights_only'
        raise ValueError('no_semantically_compatible_material:' + category + ':' + role)
    row = deepcopy(compatible[int(rng.integers(len(compatible)))])
    for ref in row['resources'].values(): resolve(ref)
    if kind == 'material':
        size = row.get('texture_size_m')
        if size is None:
            row['texture_size_m'] = DEFAULT_TEXTURE_METERS[role]
            row['texture_scale_used_source'] = 'explicit_role_default_source_physical_scale_unknown'
        else:
            size = np.asarray(size, dtype=float)
            if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
                raise ValueError('invalid_texture_metric_scale')
            row['texture_scale_used_source'] = row.get('texture_size_source', 'publisher_declared_metric_scale')
    return row, reason


def _workspace(preset_id, envelope, cameras):
    """Deterministic, distinct metric structures outside a protected sweep prism."""
    # Whole physical table and camera-to-workspace corridors are protected too.
    protected = np.array([[-.82, -1.02, -.10], [.60, 1.02, 1.35]])
    if envelope: protected = union_bounds([protected, envelope['full_swept_bounds_m']])
    eyes = [c['eye'] for c in cameras.values()]
    # Use the complete predeclared proposal range in every ablation, including
    # fixed-camera ones. Changing only the camera must not shift room geometry.
    opposite_eye=np.asarray(cameras['opposite_high']['eye'])
    eyes += [(opposite_eye-np.asarray(CAMERA_LIMITS['eye_delta_m'])).tolist(),
             (opposite_eye+np.asarray(CAMERA_LIMITS['eye_delta_m'])).tolist()]
    eyes = np.asarray(eyes)
    protected = union_bounds([protected, np.stack([eyes.min(0)-.12, eyes.max(0)+.12])])
    rear, front = protected[0, 0]-.60, protected[1, 0]+.85
    left, right = protected[0, 1]-.58, protected[1, 1]+.58
    floor, ceiling = -.78, max(2.05, protected[1, 2]+.60)
    center = np.array([-.12, 0., .12]); boxes = []; lights = []

    def box(name, p, size, role='wall', tint=None):
        p, size = np.asarray(p, dtype=float), np.asarray(size, dtype=float)
        if np.any(size <= 0): raise ValueError('nonpositive_workspace_box')
        boxes.append({'id': name, 'kind': 'box', 'center_m': p.tolist(), 'size_m': size.tolist(),
                      'material_role': role, 'tint_rgba': tint or [1., 1., 1., 1.], 'visual_only': True})

    def solid_back(): box('rear_wall', [rear, (left+right)/2, (ceiling+floor)/2], [.08, right-left, ceiling-floor])

    def window_back(width=1.25, height=.9, y=0., z=.65):
        lower, upper = z-height/2, z+height/2
        box('window_wall_lower', [rear, (left+right)/2, (floor+lower)/2], [.08, right-left, lower-floor])
        box('window_wall_upper', [rear, (left+right)/2, (ceiling+upper)/2], [.08, right-left, ceiling-upper])
        for side, a, b in [('left', left, y-width/2), ('right', y+width/2, right)]:
            box('window_wall_'+side, [rear, (a+b)/2, z], [.08, b-a, height])
        for yy in [y-width/2, y+width/2, y]: box('window_mullion_'+str(yy), [rear+.055, yy, z], [.08, .035, height+.06], 'cabinet')
        for zz in [lower, upper]: box('window_rail_'+str(zz), [rear+.055, y, zz], [.08, width+.06, .035], 'cabinet')
        box('window_sill', [rear+.12, y, lower-.025], [.30, width+.14, .055], 'cabinet')
        lights.append({'kind': 'window', 'position_m': [rear+.10, y, z], 'target_m': center.tolist()})

    def side_wall(side, extent=1.):
        yy = left if side == 'left' else right
        box(side+'_wall', [(rear+front)/2, yy, (floor+ceiling)/2], [(front-rear)*extent, .08, ceiling-floor])

    def cabinet(name, x, y, width=.8, depth=.32, height=.85, z=None):
        z = floor+height/2 if z is None else z
        box(name+'_body', [x,y,z], [depth,width,height], 'cabinet')
        box(name+'_recessed_plinth', [x+.02,y,z-height/2+.045],
            [depth-.07,width-.08,.09], 'metal', [.24,.24,.24,1.])
        for off in [-width/4, width/4]:
            box(name+'_door_'+str(off), [x+depth/2+.012,y+off,z], [.025,width/2-.016,height-.04], 'cabinet')
            box(name+'_handle_'+str(off), [x+depth/2+.035,y+off,z+.08], [.035,.10,.018], 'metal')
        box(name+'_top', [x,y,z+height/2+.02], [depth+.035,width+.035,.04], 'tabletop')

    def rack(name, x, y, width=.9, depth=.38, height=1.7, levels=4, rotate=False):
        for dx in [-depth/2,depth/2]:
            for dy in [-width/2,width/2]:
                xx,yy=(x+dy,y+dx) if rotate else (x+dx,y+dy)
                box(f'{name}_post_{dx}_{dy}', [xx,yy,floor+height/2], [.045,.045,height], 'metal')
        for i,zz in enumerate(np.linspace(floor+.12,floor+height-.08,levels)):
            size=[width+.05,depth+.05,.04] if rotate else [depth+.05,width+.05,.04]
            box(f'{name}_shelf_{i}', [x,y,zz], size, 'cabinet')

    box('floor', [(rear+front)/2,(left+right)/2,floor-.04], [front-rear+.3,right-left+.3,.08], 'floor')
    # Ceiling remains a real surface, while all fixtures remain above the sweep.
    box('ceiling', [(rear+front)/2,(left+right)/2,ceiling+.04], [front-rear+.1,right-left+.1,.08])
    category = PRESETS[preset_id][0]
    if preset_id in ('kitchen_window_wall','dining_bay_window','office_window_desk'): window_back()
    else: solid_back()
    side_wall('left')
    if preset_id not in ('dining_partition','office_cubicle','workshop_assembly_bay'): side_wall('right')
    # Back-facing wall is outside both cameras, enriching robot-side parallax.
    box('front_far_wall', [front,(left+right)/2,(floor+ceiling)/2], [.08,right-left,ceiling-floor])
    rx, ly, ry = rear+.25, left+.24, right-.24
    if preset_id == 'kitchen_window_wall':
        cabinet('sink_base_left',rx,-.8,.72); cabinet('sink_base_right',rx,.8,.72)
        box('backsplash',[rear+.055,0,.22],[.05,2.5,.32],'wall')
    elif preset_id == 'kitchen_l_return':
        cabinet('rear_counter',rx,0,2.4)
        rack('return_counter',-.25,ly,1.5,.33,.95,2,True)
        box('upper_cabinet',[rear+.20,.4,1.23],[.30,1.2,.55],'cabinet')
    elif preset_id == 'kitchen_galley':
        rack('left_counter',-.15,ly,1.8,.32,.85,2,True)
        rack('right_counter',.15,ry,1.4,.32,.85,2,True)
        box('galley_upper',[rear+.2,0,1.30],[.3,2.2,.50],'cabinet')
    elif preset_id == 'kitchen_pantry':
        cabinet('tall_pantry',rx,-.85,.75,.36,2.05)
        rack('pantry_open',rx,.65,1.10,.36,1.80,5)
        box('pantry_soffit',[rear+.25,0,1.65],[.48,2.7,.12],'cabinet')
    elif preset_id == 'dining_corner':
        cabinet('corner_console',rx,.55,1.30,.32,.70)
        box('wainscot_back',[rear+.05,0,-.12],[.04,3.,1.15],'cabinet')
        box('wainscot_side',[-.15,left+.05,-.12],[2.2,.04,1.15],'cabinet')
    elif preset_id == 'dining_bay_window':
        for y in [-.90,.90]:
            box('bay_pier_'+str(y),[rear+.18,y,.75],[.30,.20,1.4],'cabinet')
        box('bay_bench',[rear+.29,0,-.27],[.48,1.7,.16],'cabinet')
        cabinet('bay_sideboard',rx,1.14,.42,.30,.75)
    elif preset_id == 'dining_sideboard':
        cabinet('long_sideboard',rx,0,2.6,.35,.80)
        for i,(y,h) in enumerate([(-.8,.55),(.15,.90),(.95,.65)]):
            box('wall_panel_'+str(i),[rear+.065,y,1.05],[.05,.65,h],'cabinet')
    elif preset_id == 'dining_partition':
        for i,x in enumerate(np.linspace(rear+.20,front-.40,14)):
            box('partition_slat_'+str(i),[x,right,.65],[.045,.07,2.85],'cabinet')
        cabinet('low_console',rx,.2,1.7,.34,.62)
    elif preset_id == 'office_window_desk':
        cabinet('credenza',rx,0,2.2,.36,.75)
        box('window_high_ledge',[rear+.14,0,1.65],[.28,2.2,.06],'cabinet')
    elif preset_id == 'office_dual_shelves':
        rack('rear_shelves',rx,.45,1.6,.32,1.95,5)
        rack('side_shelves',-.25,ly,1.35,.32,1.40,3,True)
    elif preset_id == 'office_cubicle':
        for i,(x,y,sx,sy,h) in enumerate([(rear+.14,0,.06,2.7,1.7),(-.2,left+.1,1.9,.06,1.5),(.25,right,1.1,.06,1.15)]):
            box('partition_'+str(i),[x,y,floor+h/2],[sx,sy,h],'wall')
        box('cubicle_ledge',[rear+.22,0,.06],[.30,2.3,.05],'cabinet')
    elif preset_id == 'office_bookcase':
        rack('bookcase',rx,-.45,1.5,.37,2.0,6)
        cabinet('small_file',rx,.95,.55,.33,1.05)
        box('side_wall_desk',[-.2,ly,.02],[1.55,.35,.07],'cabinet')
    elif preset_id == 'workshop_pegboard':
        cabinet('workbench_drawers',rx,0,2.5,.37,.90)
        box('pegboard',[rear+.07,0,1.05],[.05,2.45,.85],'cabinet')
        for i,y in enumerate(np.linspace(-1,1,9)):
            box('pegrail_'+str(i),[rear+.13,y,1.05],[.09,.022,.65],'metal')
    elif preset_id == 'workshop_tool_cabinets':
        cabinet('tall_tool_cabinet',rx,-.9,.78,.38,1.85)
        cabinet('short_tool_cabinet',rx,.75,1.10,.38,1.15)
        box('tool_rail',[rear+.15,0,1.65],[.10,2.4,.045],'metal')
    elif preset_id == 'workshop_overhead_shelf':
        rack('rear_high_shelf',rx,0,2.4,.35,2.3,2)
        box('overhead_side',[-.1,ly,1.62],[1.7,.38,.07],'metal')
        for x in [-.9,.7]: box('hanger_'+str(x),[x,ly,1.83],[.04,.04,.43],'metal')
    elif preset_id == 'workshop_assembly_bay':
        for i,x in enumerate([rear+.15,(rear+front)/2,front-.18]):
            for y in [left,right]: box(f'portal_{i}_{y}',[x,y,.55],[.12,.12,2.66],'metal')
            box('portal_crossbar_'+str(i),[x,0,1.93],[.12,right-left,.15],'metal')
        rack('assembly_side_bench',-.2,ly,1.7,.35,.90,2,True)
    elif preset_id == 'storage_rack_aisle':
        rack('rack_left',-.1,ly,2.1,.34,2.,4,True)
        rack('rack_right',.1,ry,1.7,.34,1.8,3,True)
    elif preset_id == 'storage_corner_racks':
        rack('corner_back',rx,.4,1.7,.35,2.2,5)
        rack('corner_side',-.2,ly,1.6,.35,1.5,4,True)
        box('rack_crossbrace',[rear+.06,-.8,.5],[.05,.05,2.3],'metal')
    elif preset_id == 'storage_bins_wall':
        rack('bin_rack',rx,0,2.45,.38,2.0,5)
        for level in range(4):
            for col in range(4-level%2):
                box(f'bin_{level}_{col}',[rx+.035,-1.+col*.60+level%2*.18,floor+.32+level*.46],
                    [.26,.42,.25],'cabinet', [.8,.85,.9,1.])
    elif preset_id == 'storage_loading_bay':
        box('loading_door',[rear+.055,0,.6],[.045,1.5,2.35],'metal')
        for y in [-.84,.84]: box('door_jamb_'+str(y),[rear+.16,y,.6],[.20,.14,2.5],'metal')
        for i in range(4): box('pallet_'+str(i),[rx,.99,floor+.06+i*.12],[.35,.75,.07],'cabinet')
        rack('loading_side_rack',.15,ly,1.05,.33,1.3,3,True)
    else: raise ValueError('unknown_workspace_preset')
    if not lights:
        position = [(rear+front)/2, -.45 if category in ('kitchen','dining') else .35, ceiling-.06]
        box('ceiling_luminaire',position,[.50,.22,.06],'metal',[1.,1.,.95,1.])
        lights.append({'kind': 'ceiling_fixture', 'position_m': position, 'target_m': center.tolist()})
    margin = .06
    for b in boxes:
        low = np.asarray(b['center_m'])-np.asarray(b['size_m'])/2; high = np.asarray(b['center_m'])+np.asarray(b['size_m'])/2
        if np.all(high > protected[0]-margin) and np.all(low < protected[1]+margin):
            raise ValueError('workspace_structure_enters_protected_sweep:' + b['id'])
    signature = digest([{'center_m': b['center_m'], 'size_m': b['size_m']} for b in boxes])
    return {'preset_id': preset_id, 'category': category, 'description': PRESETS[preset_id][1],
        'boxes': boxes, 'primary_lights': lights, 'protected_bounds_m': protected.tolist(),
        'minimum_protected_margin_m': margin, 'structural_signature': signature,
        'collision_shapes_added': 0, 'maniskill_actors_added': 0,
        'geometry_source': 'procedural_metric_architecture_not_downloaded_room_meshes'}


def sample_domain(scene, seed, *, catalog_path=DEFAULT_CATALOG, preset_id=None,
                  mode='workspace', camera_mode='fixed', motion_envelope=None,
                  asset_partition=None):
    if type(seed) is not int or seed < 0 or mode not in ('catalog_only','workspace') or camera_mode not in ('fixed','bounded'):
        raise ValueError('invalid_scene_domain_configuration')
    if set(scene['cameras']) != {'robot_side','opposite_high'}: raise ValueError('two_original_cameras_required')
    if motion_envelope and motion_envelope['physical_scene_digest'] != digest(physical_scene(scene)):
        raise ValueError('motion_envelope_bound_to_different_physics')
    rng = np.random.default_rng(seed); preset_id = preset_id or PRESET_IDS[int(rng.integers(len(PRESET_IDS)))]
    if preset_id not in PRESETS: raise ValueError('unknown_workspace_preset')
    category = PRESETS[preset_id][0]; catalog = load_catalog(catalog_path); selected = {}; fallbacks = {}
    for role in (*ROLES,'environment'):
        selected[role], fallbacks[role] = _choose_resource(catalog,rng,role,category,asset_partition)
    result = deepcopy(scene)
    camera_rng=np.random.default_rng(np.random.SeedSequence([seed,1]))
    lighting_rng=np.random.default_rng(np.random.SeedSequence([seed,2]))
    if camera_mode == 'bounded': result['cameras'], camera = sample_camera(scene,camera_rng,motion_envelope)
    else: camera = {'mode':'fixed','source_configs':deepcopy(scene['cameras']),
                   'actual_configs':deepcopy(scene['cameras']),'modified_camera':None,'status':'unchanged',
                   'pixel_occlusion_check_required':True}
    # Same seed/preset gives identical resources, room layout and illumination
    # across catalog-only, workspace-fixed, and workspace-camera comparisons.
    full_geometry = _workspace(preset_id,motion_envelope,scene['cameras'])
    geometry = full_geometry if mode == 'workspace' else None
    temperature = int(lighting_rng.choice(list(TEMPERATURE_RGB))); intensity = float(lighting_rng.uniform(.75,1.25))
    source = full_geometry['primary_lights'][0]
    direction = np.asarray(source['target_m'])-np.asarray(source['position_m']); direction /= np.linalg.norm(direction)
    ambient = float(lighting_rng.uniform(.20,.32)); fill = float(lighting_rng.uniform(.16,.28))
    lighting = {'primary_source':source,'key_direction':direction.tolist(),
        'key_rgb':(np.asarray(TEMPERATURE_RGB[temperature])*intensity).tolist(),
        'ambient_rgb':[ambient]*3,'fill_direction':[.30,-.15,-.942072184], 'fill_rgb':[fill]*3,
        'temperature_k':temperature,'temperature_model':'coarse_artistic_white_balance_not_photometric',
        'environment_alignment':'unregistered_HDRI_reflections; no claim of visible_window_highlight_calibration'}
    # A legacy-compatible placeholder makes superclass construction independent
    # of appearance_v1 digests. Rendering below replaces only visual components.
    result['appearance'] = {'table_rgb':[1.,1.,1.],'background_rgb':[.85,.85,.85],
        'ambient_rgb':lighting['ambient_rgb'],'key_light_rgb':lighting['key_rgb']}
    result['scene_domain'] = {'schema':VERSION,'seed':seed,'mode':mode,'preset_id':preset_id,
        'category':category,'catalog':binding(catalog_path),'resources':selected,'resource_fallbacks':fallbacks,
        'asset_partition':asset_partition,'geometry':geometry,'lighting':lighting,'camera_sampling':camera,
        'preset_lighting_geometry_signature':full_geometry['structural_signature'],
        'visible_light_geometry_installed':mode=='workspace',
        'rng_streams':{'resources':'seed','camera':'SeedSequence([seed,1])','lighting':'SeedSequence([seed,2])'},
        'motion_envelope':deepcopy(motion_envelope),'physical_scene_digest':digest(physical_scene(scene)),
        'source_scene_digest':digest(scene),'original_appearance':deepcopy(scene.get('appearance')),
        'physical_execution_count':0,'new_trajectory':False,'visual_only':True,
        'status':'candidate_requires_independent_render_and_lineage_validation'}
    validate_scene(result)
    return result


def validate_scene(scene):
    d=scene['scene_domain']
    if d.get('schema') != VERSION or digest(physical_scene(scene)) != d['physical_scene_digest']:
        raise ValueError('scene_domain_physics_binding_mismatch')
    if (scene['cameras'] != d['camera_sampling']['actual_configs'] or
            scene['cameras']['robot_side'] != d['camera_sampling']['source_configs']['robot_side']):
        raise ValueError('scene_domain_camera_binding_mismatch')
    resolve(d['catalog'])
    for entry in d['resources'].values():
        if entry:
            for ref in entry['resources'].values(): resolve(ref)
    return d


def textured_box_mesh(size_m, texture_size_m):
    """24-vertex box with per-face metric UVs, independent normals, outward faces."""
    size=np.asarray(size_m,dtype=float); half=size/2; scale=np.asarray(texture_size_m,dtype=float)
    vertices=[]; normals=[]; uv=[]; triangles=[]
    for axis in range(3):
        axes=[i for i in range(3) if i != axis]
        for sign in (-1,1):
            normal=np.zeros(3);normal[axis]=sign
            points=[]
            for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]:
                p=normal*half;p[axes[0]]=a*half[axes[0]];p[axes[1]]=b*half[axes[1]];points.append(p)
            indices=[0,1,2,3]
            if np.dot(np.cross(points[1]-points[0],points[2]-points[0]),normal)<0: indices=[0,3,2,1]
            base=len(vertices)
            for i in indices:
                p=points[i];vertices.append(p);normals.append(normal)
                uv.append([(p[axes[0]]+half[axes[0]])/scale[0],(p[axes[1]]+half[axes[1]])/scale[1]])
            triangles.extend([[base,base+1,base+2],[base,base+2,base+3]])
    return np.asarray(vertices,dtype=np.float32),np.asarray(triangles,dtype=np.uint32),np.asarray(normals,dtype=np.float32),np.asarray(uv,dtype=np.float32)


def _physical_entity_snapshot(entity):
    import sapien
    rows=[]
    for component in entity.components:
        if isinstance(component,sapien.physx.PhysxRigidBaseComponent):
            shapes=[]
            for s in component.get_collision_shapes():
                shape={'type':type(s).__name__,'local_pose':s.local_pose.to_transformation_matrix().tolist()}
                for key in ('vertices','scale','half_size','radius','half_length'):
                    if hasattr(s,key): shape[key]=np.asarray(getattr(s,key)).tolist()
                shapes.append(shape)
            rows.append({'type':type(component).__name__,'shapes':shapes})
    return {'name':entity.name,'pose':entity.pose.to_transformation_matrix().tolist(),'physics':rows}


def _load_visuals(env):
    import sapien
    d=validate_scene(env.scene_spec)
    env.scene_domain_runtime_evidence={'schema':VERSION,'render_only_entities':[],
        'hidden_original_render_components':[],'collision_shapes_added':0,'maniskill_actors_added':0}
    evidence=env.scene_domain_runtime_evidence
    if not env.scene.can_render():
        evidence['rendering_available']=False;return
    evidence['rendering_available']=True
    def physics_inventory():
        return [[_physical_entity_snapshot(entity) for entity in subscene.entities
                 if any(isinstance(c,sapien.physx.PhysxRigidBaseComponent) for c in entity.components)]
                for subscene in env.scene.sub_scenes]
    original_physics=physics_inventory()
    # Do not call generic hide_visual: some wrappers move physical actors.
    for subscene in env.scene.sub_scenes:
        for entity in list(subscene.entities):
            name=entity.name
            hide=(name.endswith(('table_surface_visual','background_visual')) or
                  'table_scene_wall' in name or name == 'wall' or
                  (d['mode']=='workspace' and ('ground' in name)))
            if hide:
                body=entity.find_component_by_type(sapien.render.RenderBodyComponent)
                before=_physical_entity_snapshot(entity)
                if body is not None: body.visibility=0
                after=_physical_entity_snapshot(entity)
                if digest(before)!=digest(after):raise RuntimeError('hiding_visual_changed_physics')
                evidence['hidden_original_render_components'].append({'entity':name,'before':digest(before),'after':digest(after)})
    # TableSceneBuilder wall name varies by ManiSkill release; hide explicitly.
    old_wall=env.table_scene.wall.actor
    before=_physical_entity_snapshot(old_wall);body=old_wall.find_component_by_type(sapien.render.RenderBodyComponent)
    if body is not None:body.visibility=0
    after=_physical_entity_snapshot(old_wall)
    if digest(before)!=digest(after):raise RuntimeError('wall_visibility_changed_physics')
    evidence['hidden_original_render_components'].append({'entity':old_wall.name,'before':digest(before),'after':digest(after)})
    material_cache={}
    def material(role,tint):
        key=(role,tuple(tint))
        if key in material_cache:return material_cache[key]
        entry=d['resources'][role]
        mat=sapien.render.RenderMaterial(base_color=tint,
            metallic=float(entry.get('default_metallic',1. if role=='metal' else 0.)),roughness=.85)
        for channel,attribute in [('base_color','base_color_texture'),('roughness','roughness_texture'),('normal','normal_texture')]:
            if channel in entry['resources']:
                setattr(mat,attribute,sapien.render.RenderTexture2D(str(resolve(entry['resources'][channel])),mipmap_levels=5,srgb=channel=='base_color'))
        material_cache[key]=mat;return mat
    boxes=[{'id':'table_surface','center_m':[-.12,0.,.0003],'size_m':[1.16,1.88,.0004],
            'material_role':'tabletop','tint_rgba':[1.,1.,1.,1.]}]
    if d['geometry']:boxes+=d['geometry']['boxes']
    else:boxes.append({'id':'catalog_backdrop','center_m':[-1.05,0.,.3],'size_m':[.006,2.4,1.6],
                      'material_role':'wall','tint_rgba':[1.,1.,1.,1.]})
    for scene_index,subscene in enumerate(env.scene.sub_scenes):
        for box in boxes:
            role=box['material_role'];entry=d['resources'][role]
            vertices,triangles,normals,uv=textured_box_mesh(box['size_m'],entry['texture_size_m'])
            shape=sapien.render.RenderShapeTriangleMesh(vertices,triangles,normals,uv,material(role,box['tint_rgba']))
            body=sapien.render.RenderBodyComponent();body.attach(shape)
            entity=sapien.Entity();entity.name='scene_domain/'+box['id'];entity.add_component(body)
            entity.set_pose(sapien.Pose(box['center_m']));subscene.add_entity(entity)
            if any(isinstance(c,sapien.physx.PhysxRigidBaseComponent) for c in entity.components):
                raise RuntimeError('scene_domain_entity_must_be_render_only')
            evidence['render_only_entities'].append({'scene_index':scene_index,'name':entity.name,
                'component_types':[type(c).__name__ for c in entity.components],
                'material_id':entry['id'],'texture_size_m':entry['texture_size_m']})
    final_physics=physics_inventory()
    if digest(original_physics)!=digest(final_physics):
        raise RuntimeError('scene_domain_visual_install_changed_physical_entities')
    evidence.update(physical_inventory_before_sha256=digest(original_physics),
        physical_inventory_after_sha256=digest(final_physics),
        physical_entity_counts=[len(rows) for rows in final_physics],
        physical_entities_and_collision_geometry_unchanged=True)


def register(scene, performance=None, contact_policy='native_default'):
    """Register a child runtime; keep every frozen module and physical actor intact."""
    validate_scene(scene)
    from resimpair_pickplace import environment
    from mani_skill.utils.registration import REGISTERED_ENVS,register_env
    if performance is None:
        from corl2026_workshop.experiments.autonomous_pipeline_v4.timing import Accumulator
        performance=Accumulator()
    environment.register(scene,performance,contact_policy)
    parent=REGISTERED_ENVS[environment.ENV_ID].cls

    @register_env(ENV_ID,max_episode_steps=4000)
    class SceneDomainEnv(parent):
        def _load_scene(self,options):
            super()._load_scene(options)
            _load_visuals(self)

        def _load_lighting(self,options=None):
            d=validate_scene(self.scene_spec);light=d['lighting']
            self.scene.set_ambient_light(light['ambient_rgb'])
            self.scene.add_directional_light(light['key_direction'],light['key_rgb'],shadow=True,shadow_scale=5,shadow_map_size=2048)
            self.scene.add_directional_light(light['fill_direction'],light['fill_rgb'])
            environment_resource=d['resources']['environment']
            if self.scene.can_render() and environment_resource:
                for subscene in self.scene.sub_scenes:
                    subscene.set_environment_map(str(resolve(environment_resource['resources']['environment'])))
    return ENV_ID
