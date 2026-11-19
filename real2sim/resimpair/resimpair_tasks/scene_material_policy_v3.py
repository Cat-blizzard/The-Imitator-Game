"""CPU-only, task-independent indoor surface selection from measured textures.

This is a candidate material sampler, not an image-quality acceptance function.
It measures retained source RGB files, uses explicit room/surface semantics, and
changes no geometry, object color, light, camera, task, contact or trajectory.
No source texture is recolored, flattened, regenerated, or overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write

VERSION = 'scene_material_policy_v3'
INDEX_VERSION = 'measured_indoor_surface_index_v3'
ROLES = ('tabletop', 'wall', 'floor', 'cabinet', 'metal')
INTERVENTION_ROLES = ('tabletop', 'wall', 'floor', 'cabinet')
CATEGORIES = ('kitchen', 'dining', 'office', 'workshop', 'storage')
DEFAULT_CATALOG = ROOT / 'assets/appearance_v2/curated/catalog.json'
DEFAULT_INDEX = ROOT / 'assets/appearance_v3/material_policy_v3/index.json'
MODULE = Path(__file__).resolve()

# Fixed design bounds, independent of task objects and of rendered target colors.
# L* is CIELAB D65 in [0,100]. Texture RMS is computed on L*/100 source patches,
# not simulated illumination or a guarantee of task visibility.
LIMITS = {
    'tabletop': {'L50': [35., 73.], 'chroma50_max': 38., 'L90_L10_max': 38.,
                 'metric_highpass_sigma4_rms_max': .055, 'metric_highpass_sigma16_rms_max': .10},
    'wall': {'L50': [42., 87.], 'chroma50_max': 26., 'L90_L10_max': 33.,
             'metric_highpass_sigma4_rms_max': .045, 'metric_highpass_sigma16_rms_max': .075},
    'cabinet': {'L50': [28., 80.], 'chroma50_max': 42., 'L90_L10_max': 43.,
                'metric_highpass_sigma4_rms_max': .070, 'metric_highpass_sigma16_rms_max': .12},
    'floor': {'L50': [22., 82.], 'chroma50_max': 45., 'L90_L10_max': 50.,
              'metric_highpass_sigma4_rms_max': .095, 'metric_highpass_sigma16_rms_max': .15},
    'metal': {'L50': [25., 88.], 'chroma50_max': 35., 'L90_L10_max': 45.,
              'metric_highpass_sigma4_rms_max': .09, 'metric_highpass_sigma16_rms_max': .14},
}
METRIC_PATCH = {'size_m': 1., 'pixels_per_metre': 256,
                'normalized_phases': [[0., 0.], [.25, .25], [.5, .5], [.75, .75]],
                'gaussian_sigma_pixels': [1., 4., 16.]}
EXTERIOR = {'pavement', 'paving', 'paver', 'pavers', 'sidewalk', 'cobble', 'cobbles',
            'cobblestone', 'cobblestones', 'asphalt', 'facade', 'roof', 'roofing',
            'seaworn', 'gravel', 'terrain', 'bark', 'path', 'pathway', 'road', 'streets'}
DAMAGED = {'rust', 'rusty', 'rotting', 'rotten', 'mold', 'mould', 'moss', 'damaged',
           'broken', 'peeling', 'chipped', 'dirty', 'weathered', 'worn', 'cracked', 'abandoned'}
FINISHED_WOOD = {'veneer', 'veneers', 'laminate', 'laminated', 'varnished', 'polished',
                 'coated', 'stained', 'table', 'desktop', 'worktop', 'countertop', 'furniture'}
FLOOR_WORDS = {'floor', 'flooring', 'floors', 'parquet', 'plank', 'planks', 'deck', 'decking'}
ROUGH = {'rough', 'splintered', 'unfinished', 'coarse', 'roughness', 'roughcut'}


def _words(entry):
    fields = [entry['id'], entry.get('name', ''), *entry.get('source_categories', []), *entry.get('source_tags', [])]
    return set(re.findall(r'[a-z]+', ' '.join(fields).lower()))


def rgb_to_lab(rgb):
    """sRGB -> CIELAB, D65, no camera-specific exposure or white-balance fit."""
    rgb = np.asarray(rgb, dtype=np.float64)
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    xyz = linear @ np.array([[.4124564,.3575761,.1804375],
                             [.2126729,.7151522,.0721750],
                             [.0193339,.1191920,.9503041]]).T
    xyz /= [.95047, 1., 1.08883]
    delta = 6/29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz/(3*delta**2)+4/29)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],axis=-1)


def _metric_patch(rgb, texture_size_m, phase):
    n = int(METRIC_PATCH['size_m'] * METRIC_PATCH['pixels_per_metre'])
    spacing = np.arange(n, dtype=float) / METRIC_PATCH['pixels_per_metre']
    u = (phase[0] + spacing / texture_size_m[0]) % 1.
    v = (phase[1] + spacing / texture_size_m[1]) % 1.
    yy, xx = np.meshgrid(v*rgb.shape[0], u*rgb.shape[1], indexing='ij')
    return np.stack([map_coordinates(rgb[:,:,c], [yy,xx], order=1, mode='grid-wrap', prefilter=False)
                     for c in range(3)], axis=-1)


def describe_material(entry):
    """Measure complete RGB plus four fixed metric patches from one source."""
    image_path = resolve(entry['resources']['base_color'])
    with Image.open(image_path) as im:
        im.load(); width,height=im.size
        full_rgb=np.asarray(im.convert('RGB'),dtype=np.float32)/255
        # Statistics from an area resample of the full resource; no crop/seed selection.
        rgb=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BOX),dtype=np.float64)/255
    lab=rgb_to_lab(rgb); L=lab[:,:,0]; chroma=np.linalg.norm(lab[:,:,1:],axis=-1)
    luminance=rgb@[.2126,.7152,.0722]
    p=np.percentile(L,[5,10,25,50,75,90,95]); median=np.median(lab.reshape(-1,3),axis=0)
    size=entry.get('texture_size_m'); scale_source=entry.get('texture_size_source')
    scale_known=size is not None
    if size is None: size=[1.,1.];scale_source='assumed_1m_only_for_descriptor_not_qualified_physical_scale'
    size=np.asarray(size,dtype=float)
    if size.shape!=(2,) or not np.isfinite(size).all() or np.any(size<=0):
        raise ValueError('invalid_published_texture_dimensions:'+entry['id'])
    patches=[]
    for phase in METRIC_PATCH['normalized_phases']:
        plane=rgb_to_lab(_metric_patch(full_rgb,size,phase))[:,:,0]/100
        row={'phase':phase,'luminance_rms_contrast':float(plane.std()),
             'L90_L10':float(np.percentile(plane,90)-np.percentile(plane,10))}
        for sigma in METRIC_PATCH['gaussian_sigma_pixels']:
            residual=plane-gaussian_filter(plane,sigma=sigma,mode='reflect')
            row[f'highpass_sigma{int(sigma)}_rms']=float(np.sqrt(np.mean(residual**2)))
        gx,gy=np.gradient(plane); gradient=np.hypot(gx,gy)
        row['gradient_p95']=float(np.percentile(gradient,95))
        row['gradient_fraction_above_002']=float(np.mean(gradient>.02))
        patches.append(row)
    # Worst fixed-phase patch prevents an otherwise quiet texture from hiding a
    # local high-contrast knot/seam, without searching target-dependent patches.
    texture={f'metric_{key}':max(row[key] for row in patches) for key in patches[0] if key!='phase'}
    color_bins=np.minimum((rgb*8).astype(int),7); codes=color_bins[:,:,0]*64+color_bins[:,:,1]*8+color_bins[:,:,2]
    counts=np.bincount(codes.ravel(),minlength=512); order=np.argsort(-counts,kind='stable')[:5]
    palette=[{'rgb_bin_center':[(int(code)//64+.5)/8,((int(code)//8)%8+.5)/8,(int(code)%8+.5)/8],
              'fraction':float(counts[code]/counts.sum())} for code in order if counts[code]]
    return {'source_base_color':entry['resources']['base_color'],'source_dimensions_px':[width,height],
        'analysis_full_image_px':[256,256],'texture_size_m':size.tolist(),'metric_scale_known':scale_known,
        'analysis_metric_input_px':[width,height],
        'metric_scale_source':scale_source,'mean_srgb':rgb.mean(axis=(0,1)).tolist(),
        'median_lab':median.tolist(),'mean_lab':lab.mean(axis=(0,1)).tolist(),
        'L_percentiles':dict(zip(['p05','p10','p25','p50','p75','p90','p95'],map(float,p))),
        'L90_L10':float(p[5]-p[1]),'L_rms_contrast':float(L.std()/100),
        'chroma50':float(np.median(chroma)),'chroma90':float(np.percentile(chroma,90)),
        'srgb_luma_percentiles':dict(zip(['p05','p50','p95'],map(float,np.percentile(luminance,[5,50,95])))),
        'rgb_palette_bins':palette,'metric_patch_protocol':deepcopy(METRIC_PATCH),
        'metric_patches':patches,**texture,
        'limits':'albedo RGB proxy; does not measure rendered shadows, specularity, object contrast or task visibility'}


def semantic_reasons(entry, role, category):
    """No visual target or task arguments are accepted here."""
    if role not in ROLES or category not in CATEGORIES:raise ValueError('unsupported_indoor_slot')
    words=_words(entry); family=entry.get('source_material_family');reasons=[]
    if role not in entry.get('roles',[]):reasons.append('source_role_not_declared')
    if category not in entry.get('categories',[]) and 'generic_indoor' not in entry.get('categories',[]):
        reasons.append('source_scene_category_not_declared')
    exterior=sorted(words & EXTERIOR)
    if exterior:reasons.append('exterior_surface:'+','.join(exterior))
    damage=sorted(words & DAMAGED)
    if damage and (category in ('kitchen','dining','office') or role in ('tabletop','metal')):
        reasons.append('damaged_surface_in_daily_work_slot:'+','.join(damage))
    finished=bool(words & FINISHED_WOOD or ('fine' in words and 'grained' in words))
    floor_words=bool(words & FLOOR_WORDS)
    rough=bool(words & ROUGH)
    if role=='tabletop':
        if family=='wood':
            if floor_words:reasons.append('floorboard_or_deck_is_not_worktop')
            if rough:reasons.append('rough_wood_is_not_finished_worktop')
            if not finished:reasons.append('no_finish_or_worktop_semantic_evidence')
        elif family=='stone':
            if not words & {'marble','granite','terrazzo'}:reasons.append('no_smooth_stone_worktop_evidence')
        elif family=='metal':
            if category not in ('workshop','storage'):reasons.append('metal_worktop_reserved_for_workshop_storage')
            if words & {'corrugated','grate','mesh','perforated','rust','rusty'}:reasons.append('textured_metal_not_flat_worktop')
        else:reasons.append('worktop_family_not_supported')
    elif role=='wall':
        if category in ('kitchen','dining','office'):
            if family=='wood' and (floor_words or rough or not words & {'panel','panels','paneled','panelling','panelled','veneer'}):
                reasons.append('wood_wall_requires_finished_panel_or_veneer')
            if family in ('brick','concrete','metal'):reasons.append('industrial_surface_not_primary_daily_room_wall')
            if family=='tile' and category=='office':reasons.append('tile_not_primary_office_wall')
        if family=='stone' and role=='wall':reasons.append('stone_worktop_not_primary_room_wall')
    elif role=='cabinet':
        if family=='wood' and (rough or floor_words or not (finished or words & {'panel','panels','paneled','panelled'})):
            reasons.append('cabinet_requires_finished_joinery_surface')
        if family=='metal' and category in ('dining','office'):reasons.append('metal_not_primary_daily_cabinet_face')
        if family not in ('wood','metal'):reasons.append('cabinet_family_not_supported')
    elif role=='floor':
        if family=='wood' and not words & {'floor','flooring','floors','parquet','plank','planks'}:
            reasons.append('wood_veneer_or_worktop_not_floor')
        if family in ('plaster','metal','brick'):reasons.append('primary_floor_family_not_supported')
        if family=='concrete' and category in ('kitchen','dining','office'):reasons.append('concrete_reserved_for_workshop_storage_floor')
    elif role=='metal':
        if family!='metal':reasons.append('hardware_requires_metal')
        if words & {'corrugated','grate','mesh','perforated','rust','rusty','coarse','shutter'}:
            reasons.append('coarse_or_damaged_sheet_not_small_hardware')
    return sorted(set(reasons))


def rgb_reasons(descriptor, role):
    lim=LIMITS[role]; reasons=[]; median=descriptor['L_percentiles']['p50']
    if not lim['L50'][0]<=median<=lim['L50'][1]:reasons.append('median_lightness_outside_slot_range')
    for key in ('chroma50','L90_L10','metric_highpass_sigma4_rms','metric_highpass_sigma16_rms'):
        if descriptor[key]>lim[key+'_max']:reasons.append(key+'_above_slot_limit')
    if not descriptor['metric_scale_known']:reasons.append('physical_texture_scale_unknown')
    return reasons


def eligibility(entry, descriptor, role, category, mode='semantic_rgb'):
    if mode not in ('semantic_only','semantic_rgb'):raise ValueError('unsupported_material_policy_mode')
    semantic=semantic_reasons(entry,role,category)
    rgb=rgb_reasons(descriptor,role) if mode=='semantic_rgb' else []
    return {'eligible':not semantic and not rgb,'semantic_reasons':semantic,'rgb_reasons':rgb}


def build_index(catalog_path, output_path):
    output_path=Path(output_path)
    if output_path.exists():raise ValueError('material_index_output_must_be_new')
    start=time.monotonic();catalog=read(catalog_path)
    if catalog.get('schema')!='resimpair_appearance_catalog_v2':raise ValueError('curated_appearance_catalog_v2_required')
    rows=[]
    for entry in sorted(catalog['entries'],key=lambda r:r['id']):
        if entry['kind']!='material':continue
        descriptor=describe_material(entry)
        rows.append({'id':entry['id'],'entry':deepcopy(entry),'descriptor':descriptor,
            'eligibility':{mode:{cat:{role:eligibility(entry,descriptor,role,cat,mode) for role in ROLES}
                                for cat in CATEGORIES} for mode in ('semantic_only','semantic_rgb')}})
    value={'schema':INDEX_VERSION,'source_catalog':binding(catalog_path),'implementation':binding(MODULE),
        'policy_limits':deepcopy(LIMITS),'metric_patch_protocol':deepcopy(METRIC_PATCH),'rows':rows,
        'materials_measured':len(rows),'HDRIs_unchanged':sum(e['kind']=='hdri' for e in catalog['entries']),
        'elapsed_seconds':time.monotonic()-start,'api_calls':0,'gpu_initializations':0,
        'threshold_design':'fixed task-independent slot design; not downstream image acceptance or per-target contrast fit',
        'resource_bytes_modified':False,'renders_generated':0}
    write(output_path,value)
    return value


def load_index(path):
    value=read(path)
    if value.get('schema')!=INDEX_VERSION:raise ValueError('unsupported_material_index')
    resolve(value['source_catalog']);resolve(value['implementation'])
    if value['policy_limits']!=LIMITS:raise ValueError('material_policy_limits_changed_since_index')
    return value


def _stratum(row):
    d=row['descriptor'];L=d['L_percentiles']['p50'];chroma=d['chroma50']
    light='dark' if L<45 else 'middle' if L<65 else 'light'
    color='neutral' if chroma<8 else 'warm' if d['median_lab'][2]>=0 else 'cool'
    texture='subtle' if d['metric_highpass_sigma4_rms']<.018 else 'moderate'
    return (row['entry'].get('source_material_family','unknown'),light,color,texture)


def choose(index, category, role, seed, *, mode='semantic_rgb', asset_partition=None):
    if type(seed)is not int or seed<0:raise ValueError('nonnegative_integer_material_seed_required')
    eligible=[r for r in index['rows'] if (asset_partition is None or r['entry']['asset_partition']==asset_partition)
              and eligibility(r['entry'],r['descriptor'],role,category,mode)['eligible']]
    if not eligible:raise ValueError('no_qualified_material_slot:'+category+':'+role+':'+str(asset_partition))
    # Uniform strata prevent the abundant wood family from silently replacing
    # every eligible alternative. All candidates retain their original RGB/PBR.
    groups=defaultdict(list)
    for row in eligible:groups[_stratum(row)].append(row)
    rng_seed=int.from_bytes(hashlib.sha256(f'{VERSION}:{seed}:{category}:{role}'.encode()).digest()[:8],'big')
    rng=np.random.default_rng(rng_seed);strata=sorted(groups);key=strata[int(rng.integers(len(strata)))]
    members=sorted(groups[key],key=lambda r:r['id']); selected=members[int(rng.integers(len(members)))]
    entry=deepcopy(selected['entry'])
    for ref in entry['resources'].values():resolve(ref)
    return entry,{'eligible_assets':len(eligible),'eligible_strata':len(strata),'selected_stratum':list(key),
        'selected_id':entry['id'],'descriptor':deepcopy(selected['descriptor']),
        'eligibility':eligibility(entry,selected['descriptor'],role,category,mode),
        'sampling':'uniform measured family/lightness/color/texture strata then uniform asset; no task inputs'}


def apply_policy(prepared_scene, seed, index_path=DEFAULT_INDEX, *, mode='semantic_rgb',
                 asset_partition='development', roles=INTERVENTION_ROLES):
    """Replace only surface resource choices of a frozen domain scene.

    Scene/task content is deliberately not passed into choose(): one room
    category, partition and seed selects the same materials for every task.
    """
    from .scene_domain_v2 import validate_scene
    validate_scene(prepared_scene);index=load_index(index_path)
    domain=prepared_scene['scene_domain'];category=domain['category']
    if asset_partition not in ('development','heldout_appearance'):
        raise ValueError('explicit_material_partition_required_no_cross_partition_fallback')
    if not roles or len(set(roles))!=len(roles) or set(roles)-set(ROLES):raise ValueError('invalid_material_intervention_roles')
    if 'material_policy' in domain:raise ValueError('apply_material_policy_to_original_prepared_scene_only')
    if domain['catalog']!=index['source_catalog']:raise ValueError('material_policy_source_catalog_differs_from_prepared_scene')
    partition=asset_partition
    result=deepcopy(prepared_scene);selected={};evidence={}
    for role in roles:
        selected[role],evidence[role]=choose(index,category,role,seed,mode=mode,asset_partition=partition)
    # Existing HDRI, key/fill/ambient lighting, geometry and camera remain exact.
    previous=deepcopy(domain['resources'])
    result['scene_domain']['resources'].update(selected)
    provenance_label='mixed_inherited_and_'+partition+'_materials'
    result['scene_domain']['asset_partition']=provenance_label
    result['scene_domain']['material_policy']={'schema':VERSION,'mode':mode,'seed':seed,
        'index':binding(index_path),'implementation':binding(MODULE),'category':category,
        'asset_partition':provenance_label,'selected_material_partition':partition,
        'previous_asset_partition':domain.get('asset_partition'),
        'resource_partitions':{role:e.get('asset_partition') if e else None
                               for role,e in result['scene_domain']['resources'].items()},
        'intervention_roles':list(roles),'inherited_roles':sorted(set(previous)-set(roles)),
        'source_prepared_scene_digest':digest(prepared_scene),
        'previous_resources':previous,'selections':evidence,'limits':deepcopy(LIMITS),
        'environment_resource_unchanged':True,'lighting_geometry_camera_unchanged':True,
        'reads_task_objects_or_target_colors':False,'resource_RGB_bytes_modified':False,
        'downstream_render_validation':'required_not_performed_by_sampler'}
    validate_scene(result)
    for key in ('geometry','lighting','camera_sampling'):
        if result['scene_domain'][key]!=domain[key]:raise RuntimeError('material_policy_changed_nonmaterial_field')
    return result


def coverage(index):
    records={}
    for mode in ('semantic_only','semantic_rgb'):
        records[mode]={}
        for category in CATEGORIES:
            records[mode][category]={}
            for role in ROLES:
                qualified=[r for r in index['rows'] if eligibility(r['entry'],r['descriptor'],role,category,mode)['eligible']]
                records[mode][category][role]={'count':len(qualified),
                    'by_partition':dict(Counter(r['entry']['asset_partition'] for r in qualified)),
                    'by_family':dict(Counter(r['entry'].get('source_material_family') for r in qualified)),
                    'by_stratum':{'/'.join(key):value for key,value in Counter(_stratum(r) for r in qualified).items()},
                    'ids':[r['id'] for r in qualified]}
    return {'schema':'material_policy_coverage_v3','materials_measured':len(index['rows']),
        'coverage':records,'counts_are_eligible_sources_not_render_successes':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('build-index');build.add_argument('--catalog',type=Path,default=DEFAULT_CATALOG);build.add_argument('--output',type=Path,required=True)
    summary=sub.add_parser('coverage');summary.add_argument('--index',type=Path,default=DEFAULT_INDEX);summary.add_argument('--output',type=Path,required=True)
    sample=sub.add_parser('apply');sample.add_argument('--scene',type=Path,required=True);sample.add_argument('--index',type=Path,default=DEFAULT_INDEX)
    sample.add_argument('--output',type=Path,required=True);sample.add_argument('--seed',type=int,required=True)
    sample.add_argument('--mode',choices=['semantic_only','semantic_rgb'],default='semantic_rgb')
    args=parser.parse_args()
    if args.command=='build-index':print({'materials_measured':build_index(args.catalog,args.output)['materials_measured']})
    elif args.command=='coverage':write(args.output,coverage(load_index(args.index)))
    else:write(args.output,apply_policy(read(args.scene),args.seed,args.index,mode=args.mode))


if __name__=='__main__':main()
