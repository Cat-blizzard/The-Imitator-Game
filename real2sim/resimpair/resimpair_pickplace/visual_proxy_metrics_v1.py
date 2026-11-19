"""Read-only, selected-view RGB diagnostics, not human-video quality labels.

Actor-ID masks define the regions; color and texture measurements use actual
RGB. These proxies neither alter the frozen physical/lineage rules nor qualify
an asset, a task, or a human video. Missing evidence remains unknown.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from resimpair_grasp.common import binding, read, resolve

VERSION='independent_rgb_proxy_diagnostics_v1'
CAMERA='opposite_high'
RULES={'selected_camera':CAMERA,'ring_inner_radius_px':1.,'ring_outer_radius_px':8.,
    'minimum_foreground_sample_pixels':8,'minimum_background_sample_pixels':32,
    'foreground_erode_radius_px':1,'table_erode_radius_px':12,'minimum_table_pixels':1024,
    'Lab_conversion':'OpenCV float32 RGB in [0,1] -> CIELAB L in [0,100]',
    'deltaE':'Euclidean distance between foreground and neighborhood median Lab vectors; descriptive CIE76 proxy',
    'luminance_CNR':'abs(median L_fg - median L_bg)/sqrt((1.4826*MAD_fg)^2+(1.4826*MAD_bg)^2+1)',
    'table_highpass_sigma_px':[1.,3.],'gradient_edge_threshold_L_per_px':3.,
    'pca_minor_width':'5th-to-95th percentile visible-mask projection on minor principal axis, plus one pixel',
    'RGB_frames':'frozen initial/first native held-and-lifted/final; video frames decoded from MP4, screen frames from raw PNG',
    'pixel_geometry_frames':'every available frame in the bound actual segmentation',
    'no_universal_quality_threshold_or_aggregated_quality_score':True,
    'missing_evidence':'unknown; no inferred pixels or substituted source frames'}


def require(condition,message):
    if not condition:raise ValueError(message)


def geometry(mask):
    y,x=np.nonzero(mask);count=len(x)
    if not count:return {'visible_pixels':0,'status':'not_visible','visible_bbox':None,
        'bbox_short_edge_px':0.,'pca_minor_width_px':0.}
    bounds=[[int(x.min()),int(y.min())],[int(x.max()+1),int(y.max()+1)]]
    width=int(x.max()-x.min()+1);height=int(y.max()-y.min()+1)
    points=np.c_[x,y].astype(float)
    if count<3:minor=float(min(width,height))
    else:
        centered=points-points.mean(axis=0)
        _,basis=np.linalg.eigh(centered.T@centered/count)
        projected=centered@basis[:,0];minor=float(np.percentile(projected,95)-np.percentile(projected,5)+1)
    return {'visible_pixels':count,'fraction_of_image':count/mask.size,'status':'observed',
        'visible_bbox':bounds,'bbox_width_px':width,'bbox_height_px':height,
        'bbox_short_edge_px':float(min(width,height)),'pca_minor_width_px':minor}


def _erode(mask,radius):
    return cv2.erode(mask.astype(np.uint8),np.ones((radius*2+1,radius*2+1),np.uint8),
                     borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)


def _contrast(lab,foreground,background):
    n_fg=int(foreground.sum());n_bg=int(background.sum())
    if n_fg<RULES['minimum_foreground_sample_pixels'] or n_bg<RULES['minimum_background_sample_pixels']:
        return {'status':'unknown','reason':'insufficient_actual_region_pixels','foreground_pixels':n_fg,'background_pixels':n_bg}
    fg=lab[foreground];bg=lab[background];m_fg=np.median(fg,axis=0);m_bg=np.median(bg,axis=0)
    mad_fg=float(np.median(np.abs(fg[:,0]-m_fg[0])));mad_bg=float(np.median(np.abs(bg[:,0]-m_bg[0])))
    delta=float(np.linalg.norm(m_fg-m_bg));delta_l=float(abs(m_fg[0]-m_bg[0]))
    cnr=delta_l/float(np.sqrt((1.4826*mad_fg)**2+(1.4826*mad_bg)**2+1.))
    distances=np.linalg.norm(fg-m_bg,axis=1)
    return {'status':'measured','foreground_pixels':n_fg,'background_pixels':n_bg,
        'median_foreground_Lab':m_fg.tolist(),'median_background_Lab':m_bg.tolist(),
        'median_color_deltaE76':delta,'median_luminance_difference_L':delta_l,
        'foreground_luminance_MAD':mad_fg,'background_luminance_MAD':mad_bg,
        'luminance_CNR':cnr,'foreground_to_background_median_deltaE_q10':float(np.percentile(distances,10)),
        'foreground_to_background_median_deltaE_q50':float(np.median(distances))}


def rgb_metrics(rgb,mask,table_mask,surface_mask,segmentation):
    require(rgb.shape[:2]==mask.shape==table_mask.shape==surface_mask.shape==segmentation.shape,
            'RGB_and_segmentation_region_shapes_disagree')
    lab=cv2.cvtColor(rgb.astype(np.float32)/255.,cv2.COLOR_RGB2LAB)
    result={'pixel_geometry':geometry(mask)}
    if mask.any():
        distance=distance_transform_edt(~mask)
        ring=(distance>RULES['ring_inner_radius_px'])&(distance<=RULES['ring_outer_radius_px'])
        interior=_erode(mask,RULES['foreground_erode_radius_px'])
        use_interior=int(interior.sum())>=RULES['minimum_foreground_sample_pixels']
        foreground=interior if use_interior else mask
        result['foreground_sampling']='one_pixel_eroded_mask' if use_interior else 'full_mask_thin_or_small_object'
        result['local_context_contrast']=_contrast(lab,foreground,ring)
        result['local_surface_contrast']=_contrast(lab,foreground,ring&surface_mask)
        ids,counts=np.unique(segmentation[ring],return_counts=True)
        result['local_context_entity_pixels']={str(int(k)):int(v) for k,v in zip(ids,counts)}
    else:
        result.update(local_context_contrast={'status':'unknown','reason':'target_not_visible'},
                      local_surface_contrast={'status':'unknown','reason':'target_not_visible'})
    interior=_erode(table_mask,RULES['table_erode_radius_px']);n=int(interior.sum())
    table={'table_visible_pixels':int(table_mask.sum()),'interior_pixels':n}
    if n<RULES['minimum_table_pixels']:
        table.update(status='unknown',reason='insufficient_actual_table_interior_pixels')
    else:
        L=lab[...,0];table.update(status='measured',luminance_percentiles_L=np.percentile(L[interior],[10,50,90]).tolist())
        table['highpass']={}
        for sigma in RULES['table_highpass_sigma_px']:
            high=L-cv2.GaussianBlur(L,(0,0),sigmaX=sigma,borderType=cv2.BORDER_REFLECT101)
            values=high[interior]
            table['highpass'][str(sigma)]={'RMS_L':float(np.sqrt(np.mean(values**2))),
                'absolute_q95_L':float(np.percentile(np.abs(values),95))}
        gx=cv2.Sobel(L,cv2.CV_32F,1,0,ksize=3,scale=1/8,borderType=cv2.BORDER_REFLECT101)
        gy=cv2.Sobel(L,cv2.CV_32F,0,1,ksize=3,scale=1/8,borderType=cv2.BORDER_REFLECT101)
        magnitude=np.sqrt(gx**2+gy**2)[interior]
        table['gradient_RMS_L_per_px']=float(np.sqrt(np.mean(magnitude**2)))
        table['gradient_q95_L_per_px']=float(np.percentile(magnitude,95))
        table['gradient_edge_fraction_above_3L_per_px']=float(np.mean(magnitude>RULES['gradient_edge_threshold_L_per_px']))
    result['table_texture']=table
    return result


def _table_ids(mapping):
    domain=mapping.get('scene_domain',{})
    if 'table_surface' in domain:return domain['table_surface'],'actual_scene_domain_table_surface'
    ids=[row['entity_id'] for row in mapping['all_entities'] if row['name'].endswith('_table_surface_visual')]
    return ids,'actual_original_table_surface_visual' if ids else 'unknown_no_table_surface_entity_mapping'


def _rgb_keyframes(directory,worker,indices,key_indices,camera):
    if worker['stage']=='screen':
        frames=read(directory/'frame_mapping.json')['frames'];result={}
        for row in frames:
            if row['source_state_index'] in key_indices:
                ref=row['images'][camera]
                with Image.open(resolve(ref)) as image:rgb=np.asarray(image.convert('RGB'))
                result[row['source_state_index']]=(rgb,{'encoding':'original_png','binding':ref})
        return result
    camera_info=read(directory/'execution_bundle.json')['cameras'][camera]
    reference=camera_info['video'];result={};wanted={indices.index(i):i for i in key_indices}
    with av.open(str(resolve(reference))) as video:
        for frame_index,frame in enumerate(video.decode(video=0)):
            if frame_index in wanted:result[wanted[frame_index]]=(frame.to_ndarray(format='rgb24'),
                {'encoding':'decoded_mp4','binding':reference,'output_frame_index':frame_index,'PTS_seconds':float(frame.pts*frame.time_base)})
    return result


def analyze(directory):
    directory=Path(directory).resolve();worker=read(directory/'worker_result.json')
    protocol=read(resolve(worker['render_protocol']));case=next(c for c in protocol['cases'] if c['id']==worker['render_case_id'])
    source=next(s for s in protocol['sources'] if s['id']==case['source_id'])
    native=read(resolve(source['source_execution']));targets=list(native['task_intent']['object_ids'])
    key_indices=source['screen_state_indices'];manifest=read(resolve(worker['visibility_manifest']))
    require(manifest['RGB_and_segmentation_same_get_obs'] is True,'RGB_and_segmentation_not_synchronous')
    require(manifest['render_case_id']==case['id'],'foreign_visibility_manifest')
    indices=manifest['state_indices'];mapping=manifest['entity_mapping'];ids,table_basis=_table_ids(mapping)
    surface_ids=ids+[i for values in mapping['supports'].values() for i in values]
    key_rgb=_rgb_keyframes(directory,worker,indices,key_indices,CAMERA)
    require(set(key_rgb)==set(key_indices),'missing_frozen_RGB_keyframes')
    frames=[];all_geometry=[]
    with np.load(resolve(manifest['segmentation']),allow_pickle=False) as archive:
        require(np.array_equal(archive['source_state_indices'],indices),'segmentation_state_indices_changed')
        segmentation=archive[manifest['arrays'][CAMERA]]
        require(len(segmentation)==len(indices),'segmentation_frame_count_changed')
        for local,index in enumerate(indices):
            raster=segmentation[local]
            target_metrics={}
            for entity in targets:
                mask=np.isin(raster,mapping['objects'][entity]);g=geometry(mask)
                all_geometry.append({'source_state_index':index,'entity':entity,**g})
                if index in key_indices:
                    rgb,_=key_rgb[index]
                    target_metrics[entity]=rgb_metrics(rgb,mask,np.isin(raster,ids),np.isin(raster,surface_ids),raster)
            if index in key_indices:
                frames.append({'source_state_index':index,'state_label':source['screen_state_labels'][key_indices.index(index)],
                    'RGB':key_rgb[index][1],'targets':target_metrics})
    return {'schema':VERSION,'metric_code':binding(__file__),'rules':RULES,'case_id':case['id'],
        'source_id':case['source_id'],'stage':case['stage'],'variant':case['variant'],
        'quality_condition':case.get('quality_condition'),'camera':CAMERA,
        'source_execution':source['source_execution'],'render_protocol':worker['render_protocol'],
        'worker_result':binding(directory/'worker_result.json'),'visibility_manifest':worker['visibility_manifest'],
        'table_mask_basis':table_basis,'table_entity_ids':ids,'keyframes':frames,
        'all_frame_target_geometry':all_geometry,'native_frame_count':len(indices),
        'status':'diagnostic_only','human_video_quality':'unmeasured','task_semantic_quality':'not_inferred',
        'physical_and_lineage_rules_changed':False,'new_renders':0,'api_calls':0}


if __name__=='__main__':
    import json
    parser=argparse.ArgumentParser();parser.add_argument('--execution',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(analyze(args.execution),indent=2))
