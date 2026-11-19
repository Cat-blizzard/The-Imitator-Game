import pytest
from resimpair_assets.raw_candidate_bundle_v3 import ROLLING_FIELDS,rolling_contract


@pytest.mark.parametrize('key',['native_support','native_properties','geometry_qualified','source_to_asset'])
def test_rolling_input_cannot_replace_producer_evidence(key):
    entry={k:None for k in ROLLING_FIELDS};entry[key]={'belongs_to_another_source':True}
    with pytest.raises(ValueError,match='override_native'):
        rolling_contract({'entries':[entry]})


def test_missing_rolling_bindings_do_not_pass():
    entry={k:None for k in ROLLING_FIELDS if k!='source_state'}
    with pytest.raises(ValueError):rolling_contract({'entries':[entry]})
