from pathlib import Path
from unittest.mock import Mock

import pytest

from resimpair_assets import raw_native_input_repairs_v1 as repair
from resimpair_grasp.common import binding, read, write


def fake_source(tmp_path, monkeypatch, qualification_sequence):
    original = tmp_path / 'original_job.json'
    write(original, {'raw_row': {'id': 'test:source'}, 'ingestion_group': 'test'})
    folder = tmp_path / 'compiled'
    write(folder / 'parent.json', {})
    write(folder / 'metric_policy.json', {})
    write(folder / 'source_state.json', {})
    write(folder / 'size_decision.json', {})
    parent = binding(folder / 'parent.json')
    monkeypatch.setattr(repair, 'prepare_raw', lambda *_: (parent, {}, {}))

    def materialize(reference, policy, output):
        write(output / 'descriptor.json', {})
        return {'variants': [{'descriptor': binding(output / 'descriptor.json')}]}

    def compile_variant(reference, output):
        write(output / 'descriptor.json', {})
        value = {'asset': binding(output / 'descriptor.json'), 'source_id': 'test:source',
            'geometry_qualified': next(qualification_sequence), 'surface': {'unchanged_absolute_tolerance_m': .003}}
        write(output / 'result.json', value)
        return value

    def replay(entry, output):
        write(output / 'RESULT.json', {'passed': True})
        return {'passed': True}

    monkeypatch.setattr(repair.metric_variants_v2, 'materialize', materialize)
    native = Mock(side_effect=compile_variant)
    monkeypatch.setattr(repair.metric_native_v5, 'compile_variant', native)
    refine = Mock(return_value=parent)
    monkeypatch.setattr(repair.metric_native_v6, 'refine', refine)
    monkeypatch.setattr(repair, 'inspect', lambda *_: {'passed': True, 'reasons': []})
    monkeypatch.setattr(repair, '_fresh_replay', replay)
    return original, folder, native, refine


def test_qualified_single_convex_is_selected_without_decomposition(tmp_path, monkeypatch):
    original, folder, native, refine = fake_source(tmp_path, monkeypatch, iter([True]))
    result = repair.compile_source(original, folder)
    assert result['status'] == 'native_ready'
    assert result['selected_representation'] == 'single_convex_hull'
    assert native.call_count == 1 and refine.call_count == 0
    assert len(result['candidates']) == 1


def test_failed_single_hull_is_retained_then_only_one_fallback(tmp_path, monkeypatch):
    original, folder, native, refine = fake_source(tmp_path, monkeypatch, iter([False, False]))
    result = repair.compile_source(original, folder)
    assert result['status'] == 'native_qualification_failed'
    assert native.call_count == 2 and refine.call_count == 1
    assert len(result['candidates']) == 2
    assert all(read(Path(repair.ROOT) / c['result']['path'])['status'] == 'native_qualification_failed'
               for c in result['candidates'])
    assert not (folder / 'native_entry.json').exists()


def test_dynamic_budget_refuses_fifth_settle_before_call(tmp_path, monkeypatch):
    calls = Mock(return_value={'frames': 360})
    monkeypatch.setattr(repair.metric_native_v4, 'settle', calls)
    with repair.settle_budget(tmp_path):
        for index in range(4):
            repair.metric_native_v4.settle({'part': index}, tmp_path / f'settle_{index}.json')
        with pytest.raises(ValueError, match='budget_exhausted'):
            repair.metric_native_v4.settle({'part': 4}, tmp_path / 'extra.json')
    assert calls.call_count == 4
    assert repair.metric_native_v4.settle is calls
    assert len(list((tmp_path / 'settle_reservations').glob('*.json'))) == 4
