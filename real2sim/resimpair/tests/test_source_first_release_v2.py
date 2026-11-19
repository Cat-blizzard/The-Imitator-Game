from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from resimpair_grasp.common import ROOT, binding, digest, read, resolve
from resimpair_tasks import source_first_task_selection_v1 as old
from resimpair_tasks import source_first_task_selection_v2 as new
from resimpair_tasks import source_first_factory_adapter_v2 as adapter

CASE = ROOT/'runs/raw_source_tasks_20261031_v1/factory/cases/00'
TRAY = binding(ROOT/'runs/raw_container_population_20261031_v2/jobs/5926d85cfdaed413c42c0e11/descriptor.json')


def actual_group():
    rows = read(CASE/'selection.json')['selected_assets']
    return [({'asset':r['descriptor'],'cache':r['cache'],'variant_id':r['variant_id']},read(resolve(r['descriptor'])))for r in rows]


def actual_pool():
    ref=read(CASE/'factory/protocol.json')['bundle']
    return adapter.coordinator.source.admitted_pool(resolve(ref))


def test_original_deep_receiver_rejects_same_first12_and_real_shallow_tray_passes():
    group=actual_group();target=read(CASE/'target_plan.json')['targets'][0]
    checker=new.ReleaseCompatibility();before=checker.check(group[1],target['descriptor'],read(resolve(target['descriptor'])))
    after=checker.check(group[1],TRAY,read(resolve(TRAY)))
    assert before['tested_candidates']==after['tested_candidates']==12
    assert before['passed_candidates']==0 and after['passed_candidates']==12
    assert before['failure_reasons']=={'required_drop_exceeds_declared_maximum':12}
    cache=read(resolve(group[1][0]['cache']))
    assert before['candidate_ids']==after['candidate_ids']==[r['id']for r in cache['candidates'][:12]]
    assert before['policy']==after['policy']==new.RELEASE_POLICY
    assert before['cache']==after['cache'] and before['asset']==after['asset']


def test_new_plan_preserves_source_uniform_group_and_filters_before_smallest_receiver():
    _,_,population,bins=actual_pool();pool={s:xs for s,xs in population.items()if xs[0][1]['partition']=='development'}
    seed=read(CASE/'selection.json')['seed'];group,receipt=new.choose_objects(pool,3,'development',seed)
    old_group,old_receipt=old.choose_objects(pool,3,'development',seed)
    assert group==old_group and receipt==old_receipt
    frozen=digest(group);contract=read(CASE/'target_plan.json')['contract']
    before=old.plan_targets(group,bins,'development',seed,contract=contract)
    after=new.plan_targets(group,bins,'development',seed,contract=contract)
    assert digest(group)==frozen and sorted(i for t in after['targets']for i in t['indices'])==[0,1,2]
    assert after['actual_scene_objects']==after['requested_scene_objects']==3
    assert all(r['passed'] for t in after['targets']for r in t['release_geometry'])
    assert any(not r['passed'] for r in after['release_preflight']['checks'])
    bad=read(CASE/'target_plan.json')['targets'][0]['descriptor']
    assert any(t.get('descriptor')==bad for t in before['targets'])
    assert all(t.get('descriptor')!=bad for t in after['targets'])


def test_original_fixed_xy_revision_reports_real_no_fit_without_moving_payloads(tmp_path):
    original=(CASE/'requests.json').read_bytes()
    result=adapter.revise_receivers(CASE,tmp_path/'related')
    assert result['prepared'] is False and result['reason']=='no_release_and_footprint_receiver_for_fixed_scene'
    rows=read(tmp_path/'related/RECEIVER_SELECTION.json')['rows']
    assert len(rows)==121
    assert {r['reason']for r in rows}=={'fixed_original_goal_offsets_do_not_fit',
        'receiver_release_geometry_not_qualified','footprint_outside_table'}
    assert result['IK_calls']==result['physics_steps']==result['GPU_jobs']==result['API_calls']==0
    assert (CASE/'requests.json').read_bytes()==original and not (tmp_path/'related/factory').exists()


def test_fixed_receiver_footprint_failure_is_retained_without_resampling(tmp_path):
    ref=read(CASE/'target_plan.json')['targets'][1]['descriptor']
    values=actual_pool();limited=(*values[:3],[(ref,read(resolve(ref)))])
    before=(CASE/'requests.json').read_bytes()
    with patch.object(adapter.coordinator.source,'admitted_pool',return_value=limited):
        result=adapter.revise_receivers(CASE,tmp_path/'blocked')
    assert result['prepared']is False and result['reason']=='no_release_and_footprint_receiver_for_fixed_scene'
    rows=read(tmp_path/'blocked/RECEIVER_SELECTION.json')['rows']
    assert rows and any(r.get('reason')in {'footprint_outside_table','footprint_overlap'}for r in rows)
    assert not (tmp_path/'blocked/factory').exists() and (CASE/'requests.json').read_bytes()==before


def test_new_adapter_actually_compiles_and_checks_the_same_sources_at_new_targets(tmp_path):
    bundle=resolve(read(CASE/'factory/protocol.json')['bundle'])
    config=read(CASE/'selection.json')
    result=adapter.build(bundle,tmp_path/'new',schedule=[{'partition':'development','N':3,'seed':config['seed']}],
        selection_config={'mode':'source_uniform'},task_capacity_contract=read(CASE/'target_plan.json')['contract'])
    assert result['compiled_scenes']==1,result
    chosen=read(tmp_path/'new/cases/00/selection.json')
    assert chosen==config
    report=read(tmp_path/'new/cases/00/RELEASE_PREFLIGHT.json')
    assert report['passed'] and all(op['passed']for row in report['rows']for op in row['operations'])
    assert result['new_IK']==result['new_physics']==result['new_GPU']==result['API_calls']==0


def test_first12_limit_is_not_relaxed_when_a_later_pose_would_pass():
    case=read(CASE/'factory/cohort.json')['cases'][0];scene=read(resolve(case['scene']));op=read(resolve(case['task']))['operations'][0]
    obj=next(o for o in scene['objects']if o['id']==op['object'])
    cache=deepcopy(new.load_cache(scene['grasp_caches'][obj['id']],obj['asset_descriptor']))
    while len(cache['candidates'])<13:cache['candidates'].append(deepcopy(cache['candidates'][0]))
    calls=[]
    def geometry(*args):
        calls.append(1)
        if len(calls)<=12:raise ValueError('required_drop_exceeds_declared_maximum')
        return None,{'would_pass_after_original_limit':True}
    with patch.object(new,'load_cache',return_value=cache),patch.object(new.release,'release_geometry',side_effect=geometry):
        result=new.check_operation(scene,op)
    assert result['passed']is False and len(calls)==12


def test_unexpected_implementation_failure_stops_before_second_selection(tmp_path):
    bundle=resolve(read(CASE/'factory/protocol.json')['bundle'])
    with patch.object(new,'plan_targets',side_effect=KeyError('unexpected_internal_field')) as planner:
        with pytest.raises(KeyError,match='unexpected_internal_field'):
            adapter.build(bundle,tmp_path/'stopped',schedule=[
                {'partition':'development','N':3,'seed':2026103101},
                {'partition':'development','N':3,'seed':2026103102}])
    assert planner.call_count==1 and not (tmp_path/'stopped/cases/01').exists()
    receipt=read(tmp_path/'stopped/cases/00/RESULT.json')
    assert receipt['reason']=='unknown_system_failure' and receipt['new_admission_stopped']
