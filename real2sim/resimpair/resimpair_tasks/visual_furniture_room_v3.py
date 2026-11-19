"""Candidate room-function and surface-role policy over real local resources.

This deliberately narrower office/dining experiment separates metric mesh
qualification from evidence that an asset serves a particular room function.
Neither asset IDs nor task-object colors determine selection. Original metre
scales, texture bytes, actions, physics and cameras are preserved.
"""
from collections import Counter,defaultdict
from copy import deepcopy
import re
import numpy as np
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import ROOT,binding,digest,read,resolve
from . import visual_furniture_bulk_scene_v2 as base
from . import scene_material_policy_v3 as materials

VERSION='functional_room_and_surface_policy_v3'
CATALOG=base.CATALOG
ROOMS={'office':('shelf','desk','chair','cabinet','lamp'),
       'dining':('cabinet','chair','table','stool','bench','lamp')}
PRESETS={'office':'office_bookcase','dining':'dining_sideboard'}
domain=base.domain
appearance=base.appearance


def room_fit(entry,semantics,room):
    if room not in ROOMS:raise ValueError('candidate_policy_currently_covers_office_and_dining')
    title=semantics['title'].lower().replace('-',' ');kind=semantics['resolved_category'];reasons=[]
    if re.search(r'\b(patio|outdoor|garden|camping|balcony)\b',title):reasons.append('explicit_outdoor_function')
    patterns={
        'office':{'desk':r'\b(desk|workstation)\b','shelf':r'\b(bookcase|bookshelf|bookshelves)\b',
            'chair':r'\b(office|desk|task|ergonomic|computer|executive|drafting|conference)\b',
            'cabinet':r'\b(file|filing|office|printer|document)\b','lamp':r'\b(floor|desk|task)\s*lamp\b'},
        'dining':{'table':r'\b(dining|kitchen|bistro|pub|bar|breakfast)\b',
            'chair':r'\b(dining|kitchen|bistro|cafe|café)\b','stool':r'\b(bar|counter|kitchen|dining|pub)\b',
            'cabinet':r'\b(sideboard|buffet|hutch|china|kitchen|dining)\b',
            'bench':r'\b(dining|kitchen|bistro)\b','lamp':r'\bfloor\s*lamp\b'}}
    pattern=patterns[room].get(kind)
    matched=bool(pattern and re.search(pattern,title))
    if not matched:reasons.append('room_function_not_supported_by_publisher_title')
    if room=='office' and re.search(r'\b(wardrobe|closet|sliding\s*door|vanity|dressing)\b',title):
        reasons.append('wardrobe_or_dressing_function_not_workplace_storage')
    if room=='office' and kind=='cabinet' and (max(entry['size_m'][:2])>1.8 or entry['size_m'][2]>2.4):
        reasons.append('large_closet_scale_not_office_storage')
    if room=='dining' and kind=='table' and not .65<=entry['size_m'][2]<=1.15:
        reasons.append('native_height_not_dining_or_bar_table')
    return {'eligible':not reasons,'room':room,'functional_title_pattern':pattern,
        'reasons':reasons,'scope':'publisher-title and native-size inference; actual room coherence still needs image review'}


def select(catalog,analysis,room,partition,seed,count=4):
    entries={r['id']:r for r in catalog['entries']};groups=defaultdict(list)
    for row in analysis['rows']:
        if not row['eligible'] or row['asset_partition']!=partition or row['semantics']['anchor']!='floor':continue
        evidence=room_fit(entries[row['id']],row['semantics'],room)
        if evidence['eligible']:groups[row['semantics']['resolved_category']].append((row,evidence))
    result=[]
    for kind in ROOMS[room]:
        pool=sorted(groups[kind],key=lambda p:digest([VERSION,seed,room,p[0]['id']]))
        if pool:
            row,evidence=pool[0];result.append({'entry':entries[row['id']],'semantics':row['semantics'],'room_fit':evidence})
        if len(result)==count:break
    if len(result)<count:
        pool=sorted((p for values in groups.values() for p in values),key=lambda p:digest([VERSION,seed,'fill',p[0]['id']]))
        for row,evidence in pool:
            if row['id'] not in {r['entry']['id'] for r in result}:
                result.append({'entry':entries[row['id']],'semantics':row['semantics'],'room_fit':evidence})
            if len(result)==count:break
    if len(result)<2:raise ValueError('fewer_than_two_room_function_qualified_models')
    return result,{k:len(v) for k,v in groups.items()}


def paired_view_layout(selected,protected,seed):
    """Pack two native-sized backgrounds per viewing direction, without scaling."""
    protected=np.asarray(protected);rng=np.random.default_rng(seed);placed=[]
    for group,side in [(selected[::2],-1),(selected[1::2],1)]:
        prepared=[]
        for chosen in group:
            entry=chosen['entry'];yaw=(np.pi/2 if side<0 else -np.pi/2)+float(rng.uniform(-.08,.08))
            points=domain.corners(entry['bounds_m'])@Rotation.from_euler('z',yaw).as_matrix().T
            bounds=np.stack([points.min(0),points.max(0)])
            prepared.append((chosen,yaw,bounds))
        total=sum(np.ptp(b,axis=0)[1] for _,_,b in prepared)+.16*max(0,len(prepared)-1)
        cursor=-total/2+float(rng.uniform(-.10,.10))
        for chosen,yaw,bounds in prepared:
            entry=chosen['entry'];position=np.zeros(3)
            position[0]=protected[0 if side<0 else 1,0]+side*float(rng.uniform(.28,.40))-bounds[1 if side<0 else 0,0]
            position[1]=cursor-bounds[0,1];position[2]=-.78-bounds[0,2]
            world=bounds+position;cursor=world[1,1]+.16
            if base._overlaps(world,protected,.12) or any(base._overlaps(world,p['world_bounds_m'],.05) for p in placed):
                raise ValueError('native_room_pair_enters_protected_region_or_other_furniture')
            placed.append({'id':f'room_v3_{len(placed):02d}','source_id':entry['id'],'source':entry['source'],
                'mesh':entry['mesh'],'position_m':position.tolist(),'yaw_rad':yaw,'scale':[1.,1.,1.],
                'world_bounds_m':world.tolist(),'source_size_m':entry['size_m'],'collision_shapes_added':0,
                'semantics':chosen['semantics'],'room_function_evidence':chosen['room_fit'],
                'source_partition':entry['asset_partition'],'split_group':entry['split_group'],
                'native_metric_scale_preserved':True,'source_record':entry['preparation_record'],
                'placement_zone':'rear' if side<0 else 'front','anchor':'floor','support_height_m':-.78})
    return placed


def surface_fit(entry,role):
    text=' '.join([entry['id'],entry.get('name',''),*entry.get('source_tags',[])]).lower()
    family=entry.get('source_material_family');reasons=[]
    if role=='tabletop':
        if family!='wood' or not re.search(r'veneer|worktop|desktop|countertop|table\b',text):
            reasons.append('this_candidate_uses_explicit_finished_wood_worktops')
        if re.search(r'tile|grout|floor|paving|pavement|climbing|wall',text):reasons.append('floor_or_wall_product_not_worktop')
    elif role=='wall':
        if family!='plaster':reasons.append('this_candidate_uses_ordinary_plastered_walls')
        if re.search(r'climbing|anti.?skid|anti.?slip|floor|patio|ceiling|roof|grip',text):
            reasons.append('special_activity_or_nonwall_product')
    elif role=='floor':
        if not re.search(r'floor|flooring|parquet',text):reasons.append('no_explicit_floor_function')
        if re.search(r'climbing|anti.?skid|anti.?slip|pavement|paving|patio|outdoor|garden',text):reasons.append('special_activity_or_exterior_floor')
    return {'eligible':not reasons,'reasons':reasons,'role':role,'publisher_tags':entry.get('source_tags',[]),
        'physical_texture_scale_preserved':True,'RGB_or_normal_map_modified':False}


def apply_surfaces(scene,room,seed,partition='development'):
    index=materials.load_index(materials.DEFAULT_INDEX);result=deepcopy(scene);d=result['scene_domain'];evidence={}
    for role in ('tabletop','wall','floor'):
        pool=[r for r in index['rows'] if r['entry']['asset_partition']==partition
            and materials.eligibility(r['entry'],r['descriptor'],role,room)['eligible']
            and surface_fit(r['entry'],role)['eligible']]
        if not pool:raise ValueError('room_surface_pool_empty:'+room+':'+role+':'+partition)
        chosen=min(pool,key=lambda r:digest([VERSION,seed,room,role,r['id']]))
        d['resources'][role]=deepcopy(chosen['entry'])
        evidence[role]={'selected':chosen['id'],'eligible_count':len(pool),
            'functional_surface_evidence':surface_fit(chosen['entry'],role),'descriptor':chosen['descriptor']}
    d['room_surface_policy']={'schema':VERSION,'index':binding(materials.DEFAULT_INDEX),
        'implementation':binding(__file__),'room':room,'material_partition':partition,
        'selections':evidence,'inherited_material_choices_superseded_for':['tabletop','wall','floor'],
        'metre_UV_implementation_unchanged':True,'source_texture_dimensions_unchanged':True,
        'lighting_RGB_and_HDRI_unchanged':True,'task_object_colors_not_used_for_selection':True}
    return result


def apply_overlay(scene,seed,*,motion_envelope,catalog_path=CATALOG,room='office',preset=None,
                  partition=None,count=4,view_layout='dual_view',required_asset=None):
    if room not in ROOMS:raise ValueError('only_office_and_dining_candidate_rooms_supported')
    if view_layout!='dual_view':raise ValueError('functional_candidate_requires_original_two_fixed_view_directions')
    if required_asset is not None:raise ValueError('candidate_room_selection_has_no_per_asset_whitelist')
    source_partition=scene.get('partition');partition=source_partition if partition is None else partition
    if partition!=source_partition:raise ValueError('source_partition_must_match')
    catalog,analysis,catalog_ref=base.catalog_data(catalog_path)
    if catalog.get('pending'):raise ValueError('final_closed_source_catalog_required')
    prepared,receipt=appearance.apply_overlay(scene,seed,mode='workspace',preset=preset or PRESETS[room],
        motion_envelope=motion_envelope,material_partition='development')
    if not receipt['appearance_applied']:return prepared,{**receipt,'room_candidate_status':'visual_degraded','R2H_ready':False}
    selected,coverage=select(catalog,analysis,room,partition,seed,count)
    result=apply_surfaces(prepared,room,seed);d=result['scene_domain'];g=d['geometry'];protected=g['protected_bounds_m']
    furniture=paired_view_layout(selected,protected,seed)
    for item in furniture:item['catalog']=catalog_ref;resolve(item['mesh'])
    boxes,room_bounds,camera_checks=base.room_geometry(protected,furniture,result['cameras'])
    g.update(boxes=boxes,furniture=furniture,structural_signature=digest({'boxes':boxes,'furniture':furniture}))
    # The inherited light is virtual; no generated furniture/light mesh is
    # substituted to make an untrue visible-fixture claim.
    d['visible_light_geometry_installed']=False
    d['lighting']['visible_geometry_correspondence']='virtual_light_no_corresponding_visible_fixture'
    d['bulk_furniture_policy']={'schema':VERSION,'implementation':binding(__file__),'catalog':catalog_ref,
        'seed':seed,'room_purpose':room,'source_partition':partition,'selected':[x['source_id'] for x in furniture],
        'room_function_pool_by_category':coverage,'floor_model_count':len(furniture),'supported_model_status':'not_requested',
        'view_layout':'two_native_sized_models_per_fixed_view_direction','native_scale_preserved':True,
        'protected_region':protected,'room_bounds_m':room_bounds,'original_camera_checks':camera_checks,
        'floor_height_m':-.78,'floor_height_basis':'existing tabletop z=0 / room floor z=-0.78 convention',
        'new_physical_actors':0,'new_collision_shapes':0,'R2H_ready':False,
        'joint_changes':['functional furniture filtering','two-sided size-aware packing','surface-role filtering'],
        'attribution_limit':'paired combined candidate, no claim that background count alone caused an improvement'}
    appearance.assert_invariants(scene,result);domain.validate_scene(result)
    return result,{**receipt,'implementation':binding(__file__),'room_candidate_status':'prepared_requires_independent_render',
        'result_scene_digest':digest(result),'bulk_policy':d['bulk_furniture_policy'],'surface_policy':d['room_surface_policy'],
        'evaluation_scope':'heldout task/mesh where applicable, development material palette; not heldout-appearance evaluation'}
