"""Compile selected scene components into consistent, source-bound candidate cases.

This CPU interface deliberately copies the physical blueprint rather than stale
source/factor metadata. It creates no simulation and grants no qualification.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from .planning_contract import FROZEN_SPEC, bind
from .multi_task_candidate import validate_case


def build_case(*, case_id, task, seed, split, human_window_id, background,
               proposal=None, added_distractors=(), human_spec=FROZEN_SPEC,
               selection_manifest=None, comparison=None):
    spec=json.loads(Path(human_spec).read_text())
    task_spec=next(t for t in spec['tasks'] if t['id']==task)
    human=next(w for w in task_spec['human_windows'] if w['id']==human_window_id)
    if human['split'] != split:
        raise ValueError('Requested scene would move a human source across splits')
    items=deepcopy(list(added_distractors))
    case=dict(schema='rsh_scene_candidate_v2',case_id=case_id,task=task,seed=seed,split=split,
        human_spec=bind(human_spec),human_window_ids=[human_window_id],background=deepcopy(background),
        background_split=split,scene_inventory=dict(static_distractor_count=len(items),static_distractors=[
            {k:item[k] for k in ('instance_id','asset_uid','category')} for item in items]),
        training_admitted=False,production_approved=False,execution_status='prepared_not_executed',
        source_note='Previously reviewed frozen source; no new human source or task type is implied')
    if task=='apple_bowl_transfer':
        if proposal is None:
            raise ValueError('An explicit physical layout blueprint is required')
        if proposal['seed']!=seed or proposal['split']!=split:
            raise ValueError('Do not silently change physical seed or source split')
        keys=('seed','split','active_arm','roles','camera','critical_event_probe_points','layout_mode','camera_mode')
        physical={k:deepcopy(proposal[k]) for k in keys}
        physical.update(case_id=case_id,level='candidate_asset_background_expansion',added_distractors=items,
            human_window_id=human['id'],human_video=deepcopy(human['video']),human_source_window=deepcopy(human),
            factor_design=dict(layout_seed=seed,clutter_count=len(items),camera_condition='preserved_blueprint',
                appearance_condition='explicit_background' if background else 'native_original_background',
                new_dynamic_trajectory_not_assumed=True),
            planner_requirements=dict(obstacle_instance_ids=['source_container','destination_container']+[x['instance_id'] for x in items],
                include_attached_object_collision=True,strict_full_q_path_validation_before_execution=True,
                collision_mask_changes_after_failure_forbidden=True,phase_labels_do_not_verify_contact=True),
            simulation_executed=False,training_admitted=False)
        case['proposal']=physical
    elif items:
        raise ValueError('Native book/cabinet experts do not qualify new physical clutter')
    if selection_manifest is not None:
        case['selection_manifest']=bind(selection_manifest)
        if 'proposal' in case:
            case['proposal']['source_asset_selection']=case['selection_manifest']
    if comparison is not None:case['controlled_comparison']=deepcopy(comparison)
    validate_case(case)
    return case
