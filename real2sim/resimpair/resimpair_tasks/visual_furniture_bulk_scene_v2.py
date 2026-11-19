"""Full-size furniture sampling for fixed-state, fixed-camera visual derivatives.

Room purpose, source partition, native dimensions and semantic anchors drive a
single placement policy. It has no asset-ID location table. Models are real ABO
meshes installed by the existing RenderBody-only loader. No task geometry,
action, collision, physics state or camera is edited.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import multitask_appearance_v5 as appearance
from . import scene_domain_v2 as domain
from .visual_furniture_bulk_scene_v1 import _overlaps

VERSION = 'native_scale_multiwall_background_composition_v2'
CATALOG = ROOT / 'assets/visual_furniture_bulk_20261030_v2/catalog.json'
ROOMS = {
    'office': ('desk', 'chair', 'shelf', 'cabinet', 'lamp'),
    'living': ('sofa', 'chair', 'ottoman', 'table', 'shelf', 'lamp'),
    'dining': ('table', 'chair', 'stool', 'cabinet', 'bench', 'lamp'),
    'bedroom': ('bed', 'dresser', 'bench', 'ottoman', 'cabinet', 'lamp'),
    'entry': ('bench', 'dresser', 'clothes_rack', 'storage_box', 'table', 'lamp'),
    'utility': ('storage_rack', 'clothes_rack', 'storage_box', 'cabinet', 'shelf', 'stool'),
    'lounge': ('sofa', 'ottoman', 'stool', 'chair', 'table', 'lamp'),
    'studio': ('desk', 'shelf', 'chair', 'table', 'storage_rack', 'lamp'),
}
PRESETS = {'office': 'office_bookcase', 'living': 'dining_sideboard', 'dining': 'dining_sideboard',
    'bedroom': 'office_dual_shelves', 'entry': 'office_dual_shelves',
    'utility': 'workshop_assembly_bay', 'lounge': 'dining_sideboard', 'studio': 'office_bookcase'}
PATTERNS = (
    ('lamp', r'\b(lamp|lamps|floor\s*light|table\s*light|torchiere)\b'),
    ('bed', r'\b(bed|bedframe|bed\s*frame)\b'),
    ('sofa', r'\b(sofa|couch|loveseat|love\s*seat|sectional|futon)\b'),
    ('ottoman', r'\b(ottoman|footstool|pouf)\b'),
    ('dresser', r'\b(dresser|drawer\s*chest|chest\s*of\s*(?:\d+\s*)?drawers)\b'),
    ('clothes_rack', r'\b(clothes|clothing|garment|coat)\s*(rack|stand)\b'),
    ('storage_rack', r'\b(storage|shoe|wire)\s*(rack|shelving|shelf|shelves)\b'),
    ('desk', r'\bdesk\b'),
    ('cabinet', r'\b(cabinet|sideboard|credenza|cupboard)\b'),
    ('shelf', r'\b(shelf|shelves|shelving|bookcase|bookshelf|bookshelves)\b'),
    ('bench', r'\bbench\b'),
    ('stool', r'\b(stool|barstool)\b'),
    ('chair', r'\b(chair|armchair|recliner)\b'),
    ('table', r'\btable\b'),
    ('storage_box', r'\b(storage\s*(box|bin|chest)|toy\s*box|storage\s*trunk)\b'),
)


def english_title(entry):
    records = entry.get('source_metric_records', [])
    names = [item['value'] for r in records for item in (r.get('item_name') or [])
        if str(item.get('language_tag', '')).startswith('en') and item.get('value')]
    return (sorted(set(names))[0], 'publisher_english_item_name') if names else (str(entry.get('name', '')), 'source_catalog_name')


def semantic_anchor(entry):
    title, source = english_title(entry); normalized = title.lower().replace('-', ' ')
    # Accessories after "with" do not replace the head furniture role.
    head = re.split(r'\b(with|featuring|includes|including)\b', normalized, maxsplit=1)[0]
    matches = [(kind, regex) for kind, regex in PATTERNS if re.search(regex, head)]
    if not matches: matches = [(kind, regex) for kind, regex in PATTERNS if re.search(regex, normalized)]
    kind = matches[0][0] if matches else 'unknown'
    anchor = 'floor' if kind != 'unknown' else 'unknown'; reason = 'title_head_role_and_native_dimensions'
    if re.search(r'\b(ceiling|chandelier|pendant)\b', head): anchor = 'ceiling'; reason = 'explicit_suspended_role'
    elif re.search(r'\b(floating|wall\s*mount(?:ed)?|wall\s*(shelf|shelves|cabinet|mirror)|hanging)\b', head):
        anchor = 'wall'; reason = 'explicit_wall_anchor'
    elif re.search(r'\b(headboard|replacement|slipcover|furniture\s*cover|seat\s*cushion)\b', head):
        anchor = 'unsupported_accessory'; reason = 'not_a_complete_standing_furniture_item'
    elif kind == 'lamp' and (entry['size_m'][2] < .95 or re.search(r'\b(table|desk|bedside)\s*lamp\b', head)):
        anchor = 'furniture_top'; reason = 'table_lamp_title_or_native_height_below_95cm'
    return {'source_category': entry['category'], 'resolved_category': kind,
        'category_conflict': kind != entry['category'], 'title': title, 'title_source': source,
        'title_head': head, 'title_patterns': [regex for _, regex in matches],
        'anchor': anchor, 'anchor_reason': reason,
        'semantic_confidence': 'publisher_English_title_rule_engineering_inference_not_visual_ground_truth',
        'source_placement_support': entry.get('placement_support')}


def analyze(catalog):
    rows = []
    for entry in catalog['entries']:
        role = semantic_anchor(entry); reasons = []
        size = np.asarray(entry['size_m'])
        if entry['scale_confidence'] != 'source_metric_record_consistent': reasons.append('metric_dimensions_not_confirmed')
        if any(alias.get('scale_confidence')=='source_dimensions_conflict' for alias in entry.get('geometry_aliases',[])):
            reasons.append('same_geometry_alias_has_conflicting_dimensions')
        if entry['asset_partition'] not in ('development', 'heldout_asset'): reasons.append('source_partition_unresolved')
        if not entry.get('mesh_reload_passed'): reasons.append('mesh_not_reloaded')
        if entry.get('pick_place_eligible') is not False: reasons.append('background_role_not_exclusive')
        if role['anchor'] not in ('floor', 'furniture_top'): reasons.append('unsupported_anchor:' + role['anchor'])
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0): reasons.append('invalid_size')
        elif size[2] > 3. or max(size[:2]) > 4.: reasons.append('room_native_dimensions_out_of_bounds')
        rooms = [name for name, kinds in ROOMS.items() if role['resolved_category'] in kinds]
        if not rooms: reasons.append('no_supported_room_role')
        rows.append({'id': entry['id'], 'semantics': role, 'eligible': not reasons,
            'reasons': reasons, 'asset_partition': entry['asset_partition'], 'room_purposes': rooms,
            'support_requirement': 'verified_flat_native_furniture_top' if role['anchor'] == 'furniture_top' else None})
    return {'schema': VERSION, 'rows': rows, 'source_entries': len(rows),
        'eligible_floor': sum(r['eligible'] and r['semantics']['anchor'] == 'floor' for r in rows),
        'eligible_support_required': sum(r['eligible'] and r['semantics']['anchor'] == 'furniture_top' for r in rows),
        'eligible_by_category': dict(Counter(r['semantics']['resolved_category'] for r in rows if r['eligible'])),
        'anchor_counts': dict(Counter(r['semantics']['anchor'] for r in rows)),
        'source_category_conflicts': sum(r['semantics']['category_conflict'] for r in rows),
        'all_eligible_roles_have_consumer_route': all(r['room_purposes'] for r in rows if r['eligible']),
        'semantics_or_images_independently_qualified': False}


@lru_cache(maxsize=3)
def load_catalog(path, size, mtime_ns):
    catalog = read(path)
    return catalog, analyze(catalog), binding(path)


def catalog_data(path=CATALOG):
    p = Path(path).resolve(); stat = p.stat()
    return load_catalog(str(p), stat.st_size, stat.st_mtime_ns)


def sample(catalog, analysis, *, seed, room, partition, count=4, required_asset=None):
    if room not in ROOMS or partition not in ('development', 'heldout_asset'): raise ValueError('room_and_source_partition_required')
    if type(count) is not int or not 1 <= count <= 8: raise ValueError('one_to_eight_background_models_required')
    entries = {r['id']: r for r in catalog['entries']}
    pool = [r for r in analysis['rows'] if r['eligible'] and room in r['room_purposes']
        and r['asset_partition'] == partition and r['semantics']['anchor'] == 'floor']
    pool.sort(key=lambda r: digest([VERSION, seed, room, r['id']]))
    selected = []
    if required_asset:
        matching = [r for r in pool if r['id'] == required_asset]
        if len(matching) != 1: raise ValueError('required_asset_not_eligible_for_this_room_and_partition')
        selected.append(matching[0])
    # One per category first; random asset order gives every qualified geometry
    # a non-zero sampling path. A second pass fills rooms with fewer categories.
    for row in pool:
        if len(selected) == count: break
        if row['id'] not in {r['id'] for r in selected} and row['semantics']['resolved_category'] not in {
                r['semantics']['resolved_category'] for r in selected}: selected.append(row)
    for row in pool:
        if len(selected) == count: break
        if row['id'] not in {r['id'] for r in selected}: selected.append(row)
    if not selected: raise ValueError('no_qualified_floor_furniture_for_room_partition')
    return [{'entry': entries[r['id']], 'semantics': r['semantics']} for r in selected]


def floor_layout(selected, protected, seed, *, view_layout='dual_view', floor_z=-.78):
    """Place native AABBs at multiple sides; outward clearance grows with size."""
    protected = np.asarray(protected); rng = np.random.default_rng(seed)
    zones = {'rear': (0, -1, np.pi/2), 'front': (0, 1, -np.pi/2),
             'left': (1, -1, np.pi), 'right': (1, 1, 0.)}
    if view_layout == 'dual_view': order = ['rear', 'front', 'left', 'right']
    elif view_layout == 'opposite_high': order = ['rear', 'left', 'right', 'front']
    elif view_layout == 'robot_side': order = ['front', 'right', 'left', 'rear']
    else: raise ValueError('fixed_output_camera_layout_required')
    if seed % 2: order[2:] = reversed(order[2:])
    placed = []
    for index, chosen in enumerate(selected):
        entry = chosen['entry']; success = None
        for offset in range(4):
            zone = order[(index+offset)%4]; axis, sign, facing = zones[zone]; tangent = 1-axis
            yaw = facing + float(rng.uniform(-.14, .14))
            rotation = Rotation.from_euler('z', yaw).as_matrix()
            points = domain.corners(entry['bounds_m']) @ rotation.T
            bounds = np.stack([points.min(0), points.max(0)])
            extent = np.ptp(bounds, axis=0)
            for candidate in range(11):
                center = (protected[:, tangent].mean() + (0 if candidate == 0 else
                    ((candidate+1)//2)*(extent[tangent]+.24)*(1 if candidate%2 else -1)))
                position = np.zeros(3); position[tangent] = center - bounds[:, tangent].mean()
                gap = .28 + float(rng.uniform(.02, .20))
                position[axis] = protected[0 if sign < 0 else 1, axis] + sign*gap - bounds[1 if sign < 0 else 0, axis]
                position[2] = floor_z - bounds[0, 2]; world = bounds + position
                if _overlaps(world, protected, .12): continue
                if any(_overlaps(world, p['world_bounds_m'], .10) for p in placed): continue
                success = {'id': f'bulk_v2_{index:02d}', 'source_id': entry['id'],
                    'source': entry['source'], 'mesh': entry['mesh'], 'position_m': position.tolist(),
                    'yaw_rad': yaw, 'scale': [1., 1., 1.], 'world_bounds_m': world.tolist(),
                    'source_size_m': entry['size_m'], 'collision_shapes_added': 0,
                    'semantics': chosen['semantics'], 'source_partition': entry['asset_partition'],
                    'split_group': entry['split_group'], 'native_metric_scale_preserved': True,
                    'source_record': entry['preparation_record'], 'placement_zone': zone,
                    'anchor': 'floor', 'support_height_m': floor_z}
                break
            if success: break
        if success is None: raise ValueError('native_size_background_layout_cannot_fit:' + entry['id'])
        placed.append(success)
    return placed


def room_geometry(protected, furniture, cameras, floor_z=-.78):
    occupied = domain.union_bounds([protected] + [x['world_bounds_m'] for x in furniture])
    lower = np.asarray(occupied[0]) - [.28, .28, 0]
    upper = np.asarray(occupied[1]) + [.28, .28, .30]
    lower[2] = floor_z; upper[2] = max(2.05, upper[2], np.asarray(protected)[1, 2]+.6)
    mid = (lower+upper)/2; size = upper-lower; boxes = []
    def box(name, center, dimensions, role):
        boxes.append({'id': name, 'kind': 'box', 'center_m': list(map(float, center)),
            'size_m': list(map(float, dimensions)), 'material_role': role, 'tint_rgba': [1.]*4, 'visual_only': True})
    box('floor', [mid[0], mid[1], lower[2]-.04], [size[0]+.16, size[1]+.16, .08], 'floor')
    box('ceiling', [mid[0], mid[1], upper[2]+.04], [size[0]+.16, size[1]+.16, .08], 'wall')
    for name, axis, index in [('rear_wall',0,0),('front_far_wall',0,1),('left_wall',1,0),('right_wall',1,1)]:
        center=mid.copy(); center[axis]=(lower if index==0 else upper)[axis]
        dimensions=size.copy(); dimensions[axis]=.08; box(name,center,dimensions,'wall')
    checks = {}
    protected=np.asarray(protected)
    for name, camera in cameras.items():
        eye=np.asarray(camera['eye']); target=np.asarray(camera['target'])
        if not (np.all(eye>lower+.08) and np.all(eye<upper-.08)): raise ValueError('fixed_camera_outside_room:'+name)
        if not all(np.all(p>=protected[0]) and np.all(p<=protected[1]) for p in (eye,target)):
            raise ValueError('fixed_camera_ray_not_inside_protected_prism:'+name)
        checks[name]={'original_camera_unchanged': True, 'eye_inside_room': True,
            'eye_target_and_complete_physical_sweep_inside_convex_protected_prism': True,
            'pixel_visibility_certificate': False}
    for row in boxes:
        c=np.asarray(row['center_m']); h=np.asarray(row['size_m'])/2
        if _overlaps([c-h,c+h],protected,.06): raise ValueError('room_solid_intersects_physical_or_camera_corridor')
    return boxes, [lower.tolist(),upper.tolist()], checks


def _points_in_triangles(points, triangles):
    a=triangles[:,0,:2]; b=triangles[:,1,:2]; c=triangles[:,2,:2]
    v0=c-a; v1=b-a; den=v0[:,0]*v1[:,1]-v1[:,0]*v0[:,1]
    keep=np.abs(den)>1e-12; a=a[keep];v0=v0[keep];v1=v1[keep];den=den[keep]
    if not len(a): return False
    for point in points:
        v2=np.asarray(point)-a
        u=(v2[:,0]*v1[:,1]-v1[:,0]*v2[:,1])/den
        v=(v0[:,0]*v2[:,1]-v2[:,0]*v0[:,1])/den
        if not np.any((u>=-1e-6)&(v>=-1e-6)&(u+v<=1+1e-6)): return False
    return True


def support_volume_clear(triangles, center, footprint, support_height, model_height):
    """Conservative triangle-AABB test; catches spans with all vertices outside.

    This broad phase may reject an actually clear mesh, but it cannot certify a
    lamp volume merely because obstacle triangle vertices fall outside it.
    The support face itself is allowed within a two millimetre plane tolerance.
    """
    lower=np.r_[np.asarray(center)-np.asarray(footprint)/2,support_height+.002]
    upper=np.r_[np.asarray(center)+np.asarray(footprint)/2,support_height+model_height+.02]
    tlo=np.asarray(triangles).min(1); thi=np.asarray(triangles).max(1)
    intersects=np.all(thi>lower,axis=1)&np.all(tlo<upper,axis=1)
    return not bool(np.any(intersects))


@lru_cache(maxsize=256)
def native_base_footprint(path, file_hash):
    """The support footprint is the actual bottom band, not the lampshade."""
    import trimesh
    mesh=trimesh.load(resolve({'path':path,'sha256':file_hash}),force='scene',process=False).to_geometry()
    bounds=np.asarray(mesh.bounds);height=np.ptp(bounds,axis=0)[2]
    band=max(.003,min(.01,height*.02))
    vertices=mesh.vertices[mesh.vertices[:,2]<=bounds[0,2]+band]
    if len(vertices)<3: raise ValueError('native_base_contact_band_missing')
    footprint=np.stack([vertices[:,:2].min(0),vertices[:,:2].max(0)])
    if np.any(np.ptp(footprint,axis=0)<.01): raise ValueError('native_base_support_footprint_degenerate')
    return footprint,{'native_bottom_band_height_m':band,'native_bottom_band_vertex_count':len(vertices),
        'native_base_bounds_xy':footprint.tolist(),'base_inferred_from_mesh_not_physical_stability_test':True}


def try_furniture_top(furniture, candidates, seed):
    """A small lamp needs actual flat support triangles, not merely a top AABB."""
    import trimesh
    for support in furniture:
        if support['semantics']['resolved_category'] not in ('desk','table','cabinet','dresser'): continue
        mesh=trimesh.load(resolve(support['mesh']),force='scene',process=False).to_geometry()
        triangles=mesh.triangles; normals=mesh.face_normals
        upward=(normals[:,2]>.999)&(mesh.area_faces>1e-7)&(np.ptp(triangles[:,:,2],axis=1)<1e-4)
        heights=np.round(triangles[:,:,2].mean(1),2)
        levels=[h for h in set(heights[upward]) if .45<=h<=1.25]
        levels.sort(key=lambda h: -mesh.area_faces[upward&(heights==h)].sum())
        for height in levels[:4]:
            surface=triangles[upward&(heights==height)]
            lo=surface[:,:,:2].min((0,1)); hi=surface[:,:,:2].max((0,1))
            for chosen in candidates:
                entry=chosen['entry']; bounds=np.asarray(entry['bounds_m'])
                try:base,base_evidence=native_base_footprint(entry['mesh']['path'],entry['mesh']['sha256'])
                except ValueError:continue
                footprint=np.ptp(base,axis=0)+.06
                if np.any(footprint>hi-lo): continue
                # A rectangle with all corners inside one convex native face
                # is wholly supported. This conservative proof needs no extra
                # geometry package and cannot bridge a hole between triangles.
                support_faces=surface[np.argsort(-np.linalg.norm(
                    np.cross(surface[:,1]-surface[:,0],surface[:,2]-surface[:,0]),axis=1))[:32]]
                center=None; actual_height=None
                for tri in support_faces:
                    trial=tri[:,:2].mean(0)
                    points=np.array([[trial[0]+x*footprint[0]/2,trial[1]+y*footprint[1]/2] for x in (-1,1) for y in (-1,1)])
                    if _points_in_triangles(points,tri[None]):
                        center=trial;actual_height=float(tri[:,2].max());break
                if center is None: continue
                local_xy=center-base.mean(0)
                body_center=bounds[:,:2].mean(0)+local_xy
                if not support_volume_clear(triangles,body_center,np.ptp(bounds,axis=0)[:2]+.04,actual_height,entry['size_m'][2]): continue
                yaw=support['yaw_rad']; R=Rotation.from_euler('z',yaw).as_matrix()
                local=np.r_[local_xy,actual_height-bounds[0,2]]
                position=np.asarray(support['position_m'])+R@local
                world=domain.corners(bounds)@R.T+position
                wb=np.stack([world.min(0),world.max(0)])
                if any(_overlaps(wb,x['world_bounds_m'],.02) for x in furniture if x is not support): continue
                return {'id': 'bulk_v2_supported_lamp', 'source_id':entry['id'],'source':entry['source'],
                    'mesh':entry['mesh'],'position_m':position.tolist(),'yaw_rad':yaw,'scale':[1.,1.,1.],
                    'world_bounds_m':wb.tolist(),'source_size_m':entry['size_m'],'collision_shapes_added':0,
                    'semantics':chosen['semantics'],'source_partition':entry['asset_partition'],
                    'split_group':entry['split_group'],'native_metric_scale_preserved':True,
                    'source_record':entry['preparation_record'],'placement_zone':support['placement_zone'],
                    'anchor':'furniture_top','support_id':support['id'],'support_height_m':float(position[2]),
                    'support_evidence':{'native_support_plane_z':actual_height,'flat_triangle_count':len(surface),
                        'full_footprint_covered_by_single_convex_native_triangle':True,'margin_m':.03,
                        'base_footprint':base_evidence,
                        'clearance_triangle_AABB_broadphase':True,'conservative_clearance_can_reject_valid_support':True,
                        'new_physical_support_claimed':False}}
    return None


def compose(prepared_scene, seed, *, catalog_path=CATALOG, room='office', partition=None,
            count=4, view_layout='dual_view', required_asset=None, add_supported_lamp=True):
    domain.validate_scene(prepared_scene)
    source_partition=prepared_scene.get('partition'); partition=source_partition if partition is None else partition
    if source_partition in ('development','heldout_asset') and partition!=source_partition:
        raise ValueError('background_partition_must_match_task_partition')
    catalog,analysis,catalog_ref=catalog_data(catalog_path)
    selected=sample(catalog,analysis,seed=seed,room=room,partition=partition,count=count,required_asset=required_asset)
    result=deepcopy(prepared_scene); d=result['scene_domain']; g=d['geometry']
    protected=np.asarray(g['protected_bounds_m'])
    furniture=floor_layout(selected,protected,seed,view_layout=view_layout)
    support_status='not_requested'
    if add_supported_lamp and 'lamp' in ROOMS[room]:
        entries={r['id']:r for r in catalog['entries']}
        support_pool=[r for r in analysis['rows'] if r['eligible'] and r['asset_partition']==partition
            and r['semantics']['anchor']=='furniture_top' and room in r['room_purposes']]
        support_pool.sort(key=lambda r:digest([seed,'supported',r['id']]))
        candidates=[{'entry':entries[r['id']],'semantics':r['semantics']} for r in support_pool[:12]]
        lamp=try_furniture_top(furniture,candidates,seed)
        if lamp is not None:
            if _overlaps(lamp['world_bounds_m'],protected,.12): raise ValueError('supported_furniture_enters_protected_region')
            furniture.append(lamp); support_status='supported_by_native_flat_surface'
        else: support_status='no_verified_native_support_optional_lamp_skipped'
    boxes,bounds,camera_checks=room_geometry(protected,furniture,result['cameras'])
    old_ids=[b['id'] for b in g['boxes']]
    for item in furniture: item['catalog']=catalog_ref; resolve(item['mesh'])
    g.update(boxes=boxes,furniture=furniture,structural_signature=digest({'boxes':boxes,'furniture':furniture}))
    d['bulk_furniture_policy']={'schema':VERSION,'implementation':binding(__file__),'catalog':catalog_ref,
        'seed':seed,'room_purpose':room,'source_partition':partition,'view_layout':view_layout,
        'selected':[r['source_id'] for r in furniture],'floor_model_count':len(selected),
        'supported_model_status':support_status,'native_scale_preserved':True,
        'protected_region':g['protected_bounds_m'],'room_bounds_m':bounds,
        'original_camera_checks':camera_checks,'camera_parameters_modified':False,
        'old_procedural_elements_replaced':old_ids,'floor_height_m':-.78,
        'floor_height_basis':'existing tabletop z=0, room floor z=-0.78 convention; source base is z=0',
        'placement_rule':'native-sized multiple walls, inward yaw plus bounded variation, full physical/camera prism protected',
        'new_physical_actors':0,'new_collision_shapes':0,'independent_semantics_and_video_quality':'unreviewed',
        'R2H_ready':False}
    appearance.assert_invariants(prepared_scene,result); domain.validate_scene(result)
    return result


def apply_overlay(scene, seed, *, motion_envelope, catalog_path=CATALOG, room='office', preset=None,
                  partition=None, count=4, view_layout='dual_view', required_asset=None):
    prepared,receipt=appearance.apply_overlay(scene,seed,mode='workspace',preset=preset or PRESETS[room],
                                             motion_envelope=motion_envelope)
    if not receipt['appearance_applied']: return prepared,{**receipt,'bulk_furniture_status':'appearance_unavailable','R2H_ready':False}
    try:
        result=compose(prepared,seed,catalog_path=catalog_path,room=room,partition=partition,
            count=count,view_layout=view_layout,required_asset=required_asset)
    except (FileNotFoundError,OSError) as exc:
        # Missing optional appearance resources cannot become a physics failure.
        appearance.assert_invariants(scene,scene)
        return deepcopy(scene),{**receipt,'appearance_applied':False,'bulk_furniture_status':'missing_optional_resource',
            'visual_degraded':True,'R2H_ready':False,'reason':str(exc),'result_scene_digest':digest(scene)}
    return result,{**receipt,'implementation':binding(__file__),'result_scene_digest':digest(result),
        'bulk_furniture_status':'prepared_requires_full_video_validation','bulk_policy':result['scene_domain']['bulk_furniture_policy']}


def qualify_layout_reachability(catalog_path, protected_bounds, *, cameras, seed=2026103021):
    """Exercise the actual sampler/placer for every qualified floor geometry.

    This proves a consumable layout exists, not that it appears in a rendered
    image. Full video visibility/semantics remain a separately sampled review.
    """
    catalog,analysis,ref=catalog_data(catalog_path); rows=[]
    for row in analysis['rows']:
        if not row['eligible'] or row['semantics']['anchor']!='floor': continue
        room=row['room_purposes'][0]
        try:
            selected=sample(catalog,analysis,seed=seed,room=room,partition=row['asset_partition'],count=4,required_asset=row['id'])
            placed=floor_layout(selected,protected_bounds,seed)
            _,room_bounds,camera_checks=room_geometry(protected_bounds,placed,cameras)
            passed=any(p['source_id']==row['id'] for p in placed)
            rows.append({'id':row['id'],'passed':passed,'room':room,'partition':row['asset_partition'],
                'layout_digest':digest(placed),'placement':next(p for p in placed if p['source_id']==row['id']),
                'room_bounds_m':room_bounds,'fixed_camera_checks':camera_checks,
                'other_furniture':[p['source_id'] for p in placed if p['source_id']!=row['id']]})
        except Exception as exc: rows.append({'id':row['id'],'passed':False,'error':type(exc).__name__+':'+str(exc)})
    return {'schema':VERSION,'catalog':ref,'implementation':binding(__file__),'analysis':analysis,
        'protected_bounds_m':protected_bounds,'rows':rows,'actual_sampler_and_placer_attempts':len(rows),
        'reachable_unique_geometry_count':sum(r['passed'] for r in rows),'rendered_or_visually_validated_count':0,
        'support_required_assets_not_counted_as_floor_reachable':analysis['eligible_support_required'],
        'new_physical_executions':0,'API_calls':0}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('inspect','qualify'))
    parser.add_argument('--catalog',type=Path,default=CATALOG)
    parser.add_argument('--sources',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('new_analysis_output_required')
    if args.command=='inspect':
        catalog,analysis,ref=catalog_data(args.catalog);result={**analysis,'catalog':ref,'implementation':binding(__file__)}
    else:
        if args.sources is None: raise ValueError('frozen_real_source_recipes_required')
        specs=read(args.sources)['sources'];envelopes=[];source_refs=[];all_cameras={}
        seen=set()
        for spec in specs:
            key=digest(spec['execution'])
            if key in seen: continue
            seen.add(key);actual=appearance.replay.validate_source(resolve(spec['execution']).parent)
            baseline,_=appearance.render.baseline_scene(actual)
            envelope=domain.build_motion_envelope(baseline,actual['source']/'measurements.npz')
            prepared,_=appearance.apply_overlay(baseline,1,mode='workspace',preset='office_bookcase',motion_envelope=envelope)
            envelopes.append(prepared['scene_domain']['geometry']['protected_bounds_m'])
            for name,camera in baseline['cameras'].items():all_cameras[key[:8]+'_'+name]=camera
            source_refs.append(actual['bindings']['execution_bundle.json'])
        protected=domain.union_bounds(envelopes).tolist()
        result=qualify_layout_reachability(args.catalog,protected,cameras=all_cameras)
        result.update(source_recipes=binding(args.sources),source_executions=source_refs,
            qualification_region='union of complete physical/camera protected regions of every retained source; no generated training pair')
    write(args.output,result)
    print({k:result[k] for k in ('source_entries','eligible_floor','eligible_support_required','reachable_unique_geometry_count','actual_sampler_and_placer_attempts') if k in result})
