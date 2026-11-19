"""Regression for the real unexecuted-v2 interface failure and its v3 repair."""
import inspect
import pytest
from resimpair_grasp.common import ROOT,read
from resimpair_tasks import visual_furniture_bulk_render_v4 as batch


def test_retained_real_incomplete_preparation_is_rejected():
    with pytest.raises(ValueError,match='frozen_evaluation_dependencies_missing'):
        batch.validate_prepared_interface(ROOT/'runs/visual_furniture_bulk_native_videos_20261030_v2')


def test_repaired_real_preparation_has_all_required_evidence_fields():
    result=batch.validate_prepared_interface(ROOT/'runs/visual_furniture_bulk_native_videos_20261030_v3')
    assert result['complete_frozen_validator_interface']
    assert not result['physical_checks_or_thresholds_modified']


def test_future_allocation_needs_explicit_previous_budget_and_identity():
    parameters=inspect.signature(batch.prepare).parameters
    assert parameters['prior_budget'].default is inspect.Parameter.empty
    assert parameters['allocation_id'].default is inspect.Parameter.empty


def test_negative_visual_repeats_do_not_silently_slice_the_cohort(tmp_path):
    with pytest.raises(ValueError,match='zero_to_three'):
        batch.source_specs_from_cohort(ROOT/'runs/multitask_render_20261030_v2/selection_protocol.json',
            tmp_path/'recipes.json',repeats=-1)
    assert not (tmp_path/'recipes.json').exists()


def test_real_provisional_catalog_with_later_split_conflict_is_not_admitted_to_fresh_batches():
    with pytest.raises(ValueError,match='source_scope_not_closed'):
        batch.require_completed_catalog(read(ROOT/'assets/visual_furniture_bulk_20261030_v2/snapshots/render_catalog_first_full_qualification.json'))
    assert batch.require_completed_catalog(read(ROOT/'assets/visual_furniture_bulk_20261030_v2/catalog.json'))
