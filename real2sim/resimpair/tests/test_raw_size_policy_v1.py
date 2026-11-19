import pytest
from resimpair_assets.raw_size_policy_v1 import decide_raw, metric_policy


def row(source='objaverse', name='toy car', category='toy'):
    return {'id':source+':example','source':source,'name':name,'category':category}


def test_metric_source_keeps_original_large_size_and_exits_grasp_role():
    d = decide_raw(row('ycb_official'), [.2,.4,.9])
    assert d['source_unit_to_m'] == 1 and d['role'] == 'background'


def test_unknown_units_use_category_not_input_numeric_magnitude_or_old_cache():
    a = decide_raw(row(), [.5,1.,2.])
    b = decide_raw({**row(),'old_longest_m':.035}, [50.,100.,200.])
    assert a['target_longest_m'] == pytest.approx(b['target_longest_m'])
    assert a['source_unit_to_m'] == pytest.approx(b['source_unit_to_m'] * 100)
    assert .126 <= a['target_longest_m'] <= .154


def test_deterministic_source_variation_and_source_partition_irrelevance():
    a = decide_raw(row(), [1,2,3])
    assert a == decide_raw(row(), [1,2,3])
    assert a['source_unit_to_m'] == decide_raw({**row(),'partition':'heldout_asset'}, [1,2,3])['source_unit_to_m']


def test_raw_byte_aliases_share_engineering_size():
    a = decide_raw({**row(),'source_sha':'shared'}, [1,2,3])
    b = decide_raw({**row(),'id':'objaverse:alias','source_sha':'shared'}, [1,2,3])
    assert a['target_longest_m'] == b['target_longest_m']


def test_wide_container_anchor_never_overrides_known_metric_source():
    a = decide_raw(row(name='serving tray',category='container'), [1,2,3], role_override='container')
    b = decide_raw(row('google_scanned_objects_original_fuel','serving tray','container'), [.01,.2,.3], role_override='container')
    assert .378 <= a['target_longest_m'] <= .462
    assert b['source_unit_to_m'] == 1 and b['target_longest_m'] == .3


def test_furniture_is_not_forced_into_container_role():
    with pytest.raises(ValueError, match='role_evidence'):
        decide_raw(row(name='chair', category='furniture'), [1,2,3], role_override='container')


def test_unit_unknown_policy_does_not_claim_source_metric_truth():
    d = decide_raw(row(), [1,2,3]); p = metric_policy(d, {'path':'evidence','sha256':'x'})
    assert p['size']['mode'] == 'inferred_category'
    assert d['size_evidence']['source_metric_truth'] == 'unknown'


@pytest.mark.parametrize('dimensions', [[0,1,2],[1,2,float('nan')],[1,2]])
def test_bad_original_geometry_not_rescaled_into_validity(dimensions):
    with pytest.raises(ValueError): decide_raw(row(), dimensions)
