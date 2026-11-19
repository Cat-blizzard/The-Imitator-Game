from pathlib import Path
import pytest
from resimpair_grasp.common import ROOT,read,resolve
from resimpair_tasks import metric_production_batch_v3 as entry

BUNDLE=ROOT/'runs/semantic_scale_consumer_20261030_v1/final_consumer_v2/BUNDLE.json'
CONTAINERS=ROOT/'runs/container_affordance_consumer_20261030_v2/sampler_catalog.json'
CAMPAIGN=ROOT/'runs/production_integration_20261030_v1'
COHORT=CAMPAIGN/'unified_metric_v3_N3_N8_r1'
FACTORY=CAMPAIGN/'prepared_inputs/unified_metric_v3_N3_N8_r1/sampled/factory'


def test_published_final_semantic_and_strict_container_bindings_are_consumable():
    b=entry.consumer_inputs(BUNDLE,CONTAINERS)
    assert read(resolve(b['consumer_semantic_revision']))['consumer_qualified'] is True
    assert len(read(CONTAINERS)['assets'])==159


def test_old_unfiltered_snapshot_is_not_admitted_by_new_entry():
    old=ROOT/'runs/metric_assets_20261030_v7/bundles/snapshots/05524_01567_1793371238953079903/BUNDLE.json'
    with pytest.raises(ValueError,match='final_bound_semantic_and_container_consumer_required'):
        entry.consumer_inputs(old,CONTAINERS)


def test_mismatching_container_catalog_cannot_silently_restore_13_failures():
    old=ROOT/'runs/metric_assets_20261030_v7/container_union_v6/sampler_catalog.json'
    with pytest.raises(ValueError,match='explicit_strict_container_catalog_must_match_final_bundle'):
        entry.consumer_inputs(BUNDLE,old)


def test_actual_N3_N8_compilation_preserves_full_source_partition_and_unique_tasks():
    result=entry.verify_compiled(FACTORY)
    assert sorted(s['target_object_count'] for s in result['scenes'])==[3,8]
    assert {s['partition'] for s in result['scenes']}=={'development','heldout_asset'}
    assert result['cases']==8
    assert result['unique_task_IDs'] and result['original_source_partition_preserved']


def test_actual_prepared_cohort_dispatches_to_driver4_without_enabling_closure(monkeypatch):
    from resimpair_tasks import production_integration_v4 as stable,production_integration_v6 as experiment
    calls=[]
    monkeypatch.setattr(stable,'run',lambda cohort,limit:calls.append((Path(cohort),limit)) or {'dispatch_only':True})
    monkeypatch.setattr(experiment,'run',lambda *a,**k:pytest.fail('experimental closure must not run by default'))
    assert entry.run(COHORT,1)=={'dispatch_only':True}
    assert calls==[(COHORT,1)]
    assert len(list((COHORT/'executions').glob('*/reservation.json')))==0
