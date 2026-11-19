"""Bind visual readiness to the camera actually consumed by R2H.

All frozen v3 source/state/dual-camera integrity checks remain mandatory. The
auxiliary view cannot rescue an invisible reference, and its existing crop is
reported without pretending it is a crop in the selected input. Native physical
rules and all numerical pixel thresholds remain unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from resimpair_grasp.common import binding, read, resolve
from . import render_lineage_validator_v5_core as base

VERSION = 'independent_selected_reference_rolling_visibility_v5'
REFERENCE_CAMERA = 'opposite_high'
FROZEN_CORE_SHA256 = 'de5b8e82fc4835121ba5d574d417dcf6a3664902a2d208e0d5b8e3d99819641e'
RULE = {'reference_camera': REFERENCE_CAMERA,
        'dual_camera_integrity_unchanged': True,
        'full_visual_bbox_required_in_selected_camera_every_frame': True,
        'pixel_thresholds_apply_to_selected_camera_only': {'target_object': 16, 'active_hand': 16,
                                                          'target_container': 64, '3D_structure': 1024},
        'auxiliary_view_cannot_rescue_missing_selected_pixels': True,
        'auxiliary_crop_is_retained_as_diagnostic': True,
        'natural_terminal_containment_occlusion': 'unknown_not_ready',
        'scope': 'Minimum selected-input visibility; not aesthetic, semantic or human-generation quality.'}


def reference_visibility(previous, task, variant):
    """Pure policy function over independently measured per-view evidence."""
    rows = previous.get('per_frame_geometry_and_visible_pixels', [])
    if not rows:
        return {'status':'unknown','reference_camera':REFERENCE_CAMERA,
                'reason':'missing_validated_per_view_pixel_evidence','prior_evidence_status':previous.get('status')}
    operated=set(task['object_ids']);operations=task['operations']
    targets={op['target']['support'] for op in operations if op['target']['support']!='table'}
    critical=operated|targets;arms={'arm'+str(op['arm']) for op in operations}
    keyframes=sorted((row for row in rows if row['keyframe']),key=lambda row:row['source_state_index'])
    if len(keyframes)!=3 or any(REFERENCE_CAMERA not in row['views'] for row in rows):
        return {'status':'unknown','reference_camera':REFERENCE_CAMERA,'reason':'missing_registered_selected_view_keyframes'}
    bbox_ok=all(row['views'][REFERENCE_CAMERA]['entities'][entity]['fully_in_frame']
                for row in rows for entity in critical)
    auxiliary={}
    for camera in rows[0]['views']:
        if camera==REFERENCE_CAMERA:continue
        crops=[{'frame':row['source_state_index'],'entity':entity,
                'bbox_pixels':row['views'][camera]['entities'][entity]['bbox_pixels']}
               for row in rows for entity in critical if not row['views'][camera]['entities'][entity]['fully_in_frame']]
        auxiliary[camera]={'critical_crop_observations':len(crops),
                           'frames_with_critical_crop':len({r['frame'] for r in crops}),
                           'affected_entities':sorted({r['entity'] for r in crops}),
                           'first_crop':crops[0] if crops else None,
                           'used_as_generation_input':False,'all_frame_details_retained':True}
    findings=[];failure=not bbox_ok;unknown=False
    for position,row in enumerate(keyframes):
        view=row['views'][REFERENCE_CAMERA];index=row['source_state_index']
        for entity in sorted(operated):
            count=view['entities'][entity]['visible_pixels']
            contained=any(op['object']==entity and op['target']['support']!='table' for op in operations)
            status='confirmed' if count>=16 else 'unknown' if position==2 and contained else 'failed'
            unknown|=status=='unknown';failure|=status=='failed'
            findings.append({'entity':entity,'source_state_index':index,'camera':REFERENCE_CAMERA,
                'visible_pixels':count,'minimum_pixels':16,'status':status,
                'reason':'natural_terminal_container_occlusion_possible' if status=='unknown' else None})
        for entity in sorted(targets):
            count=view['entities'][entity]['visible_pixels'];failure|=count<64
            findings.append({'entity':entity,'source_state_index':index,'camera':REFERENCE_CAMERA,
                'visible_pixels':count,'minimum_pixels':64,'status':'confirmed' if count>=64 else 'failed'})
        if position==1:
            for arm in sorted(arms):
                count=view['active_hand_pixels'][arm];failure|=count<16
                findings.append({'entity':arm+'_hand','source_state_index':index,'camera':REFERENCE_CAMERA,
                    'visible_pixels':count,'minimum_pixels':16,'status':'confirmed' if count>=16 else 'failed'})
    structure_required=variant in ('environment_3d','environment_3d_camera')
    structure_ok=not structure_required or all(row['views'][REFERENCE_CAMERA]['structural_pixels']>=1024 for row in keyframes)
    failure|=not structure_ok
    return {'status':'failed' if failure else 'unknown' if unknown else 'confirmed',
        'reference_camera':REFERENCE_CAMERA,'selected_full_visual_bbox_in_frame':bbox_ok,
        'selected_actual_3d_structural_pixels_present':structure_ok,'keyframe_findings':findings,
        'auxiliary_view_diagnostics':auxiliary,'per_frame_geometry_and_visible_pixels':rows,
        'scope':RULE['scope'],'geometry_basis':previous.get('geometry_basis'),
        'no_hard_reference_failure':not failure}


def validate(directory):
    actual_base=binding(base.__file__)
    if actual_base['sha256']!=FROZEN_CORE_SHA256:
        return {'schema':VERSION,'validator':binding(__file__),'base_validator':actual_base,
                'passed':False,'errors':[{'reason':'frozen_v5_integrity_validator_changed'}],
                'pixel_readability':{'status':'unknown'},'generation_ready':False,'screen_gate_passed':False}
    result=base.validate(directory)
    result['schema']=VERSION;result['base_validator']=result['validator'];result['validator']=binding(__file__)
    result['reference_camera']=REFERENCE_CAMERA;result['selected_reference_rule']=deepcopy(RULE)
    result['retained_v3_decision']={'passed':result['passed'],
        'pixel_readability_status':result['pixel_readability']['status'],
        'screen_gate_passed':result['screen_gate_passed'],'generation_ready':result['generation_ready'],
        'original_code_and_original_reports_unchanged':True}
    if 'independent_quality_rules' in result:
        result['base_independent_quality_rules']=result.pop('independent_quality_rules')
    if not result['passed']:return result
    parent=read(resolve(result['source_native_execution']))
    try:
        visibility=reference_visibility(result['pixel_readability'],parent['task_intent'],result['variant'])
    except Exception as exc:
        visibility={'status':'unknown','reference_camera':REFERENCE_CAMERA,
                    'reason':'selected_view_evidence_unverifiable','error':{'type':type(exc).__name__,'reason':str(exc)}}
    result['pixel_readability']=visibility
    result['generation_ready']=visibility['status']=='confirmed'
    result['screen_gate_passed']=visibility.get('no_hard_reference_failure') is True
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execution',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(validate(args.execution),indent=2,ensure_ascii=False))
