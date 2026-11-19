from pathlib import Path
from unittest.mock import patch

import pytest

from resimpair_assets import raw_manipulation_v2 as adapter
from resimpair_assets import metric_native_v3, metric_native_v6, raw_manipulation_v1
from resimpair_assets import stable_cook_confirmed_v2
from resimpair_grasp import assets
from resimpair_grasp.common import binding, read, resolve, write


def raw_job(tmp_path):
    path = tmp_path / 'original_job.json'
    write(path, {'raw_row': {'id': 'test:source'}, 'seed': 19,
                 'ingestion_group': 'test', 'output': str(tmp_path / 'original_output'),
                 'metadata_records': [{'keep_original_metadata': True}]})
    return path


def test_prepare_rebases_only_output_and_refuses_existing_attempt(tmp_path):
    source = raw_job(tmp_path)
    before = binding(source)
    output = tmp_path / 'new_attempt'
    prepared = adapter.prepare(source, output)
    job = read(resolve(prepared['job']))
    derived = read(resolve(job['compile_input']))
    original = read(source)
    assert {k: v for k, v in derived.items() if k != 'output'} == {k: v for k, v in original.items() if k != 'output'}
    assert derived['output'] == str(output / 'compiled')
    assert binding(source) == before
    assert metric_native_v3._stable_cook is assets._stable_cook
    assert not (output / 'compiled').exists()
    with pytest.raises(ValueError, match='no_retry'):
        adapter.prepare(source, output)


def test_original_cooker_is_restored_after_failure():
    original = metric_native_v3._stable_cook
    with pytest.raises(RuntimeError, match='sentinel'):
        with adapter._confirmed_cook_hook():
            assert metric_native_v3._stable_cook is stable_cook_confirmed_v2.stable_cook
            assert assets._stable_cook is original
            raise RuntimeError('sentinel')
    assert metric_native_v3._stable_cook is original is assets._stable_cook


def test_worker_hook_rejects_relaxed_convergence_threshold(monkeypatch):
    monkeypatch.setattr(stable_cook_confirmed_v2, 'CONVERGENCE_M', 1e-5)
    with pytest.raises(ValueError, match='original_thresholds_and_budget_required'):
        with adapter._confirmed_cook_hook():
            raise AssertionError('relaxed hook must not be installed')
    assert metric_native_v3._stable_cook is assets._stable_cook


def test_worker_requires_process_marker_before_any_compilation(tmp_path):
    prepared = adapter.prepare(raw_job(tmp_path), tmp_path / 'attempt')
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError, match='separate_confirmed_cook_worker_required'):
            adapter.worker(resolve(prepared['job']))
    assert not (tmp_path / 'attempt/worker_started.json').exists()


def test_worker_calls_original_raw_flow_with_confirmed_hook_and_counts_failed_probes(tmp_path, monkeypatch):
    prepared = adapter.prepare(raw_job(tmp_path), tmp_path / 'attempt')
    job_path = resolve(prepared['job'])
    monkeypatch.setenv(adapter.WORKER_JOB_ENV, prepared['job']['sha256'])

    def fake_compile(derived_job_path):
        assert metric_native_v3._stable_cook is stable_cook_confirmed_v2.stable_cook
        job = read(derived_job_path)
        folder = Path(job['output']) / 'native_stage'
        write(folder / 'hull_0_confirmation_v2/probe_0/job.json', {'failed_before_result': True})
        write(folder / 'initial_settle.json', {'frames': 360})
        (folder / 'hull_0_source.obj').write_text('source')
        (folder / 'hull_0_cooked_0.obj').write_text('conversion')
        raise ValueError('native_file_not_confirmed')

    with patch.object(raw_manipulation_v1, 'compile_raw', side_effect=fake_compile) as original:
        result = adapter.worker(job_path)
    assert original.call_count == 1
    assert result['status'] == 'geometry_or_policy_failed'
    assert result['cpu_accounting']['fresh_process_probe_attempts'] == 1
    assert result['cpu_accounting']['completed_native_probes'] == 0
    assert result['cpu_accounting']['native_geometry_conversion_exports'] == 1
    assert result['cpu_accounting']['recorded_settle_steps'] == 360
    assert metric_native_v3._stable_cook is assets._stable_cook
    with pytest.raises(ValueError, match='no_retry'):
        adapter.worker(job_path)


def test_descriptor_entry_rejects_old_native_geometry(tmp_path):
    path = tmp_path / 'old_native.json'
    write(path, {'source_id': 'test:source', 'semantic_role': 'manipulation',
                 'native_compiler': {'input_metric_descriptor': {'path': 'parent.json'}}})
    with pytest.raises(ValueError, match='use_original_pre_native_metric_descriptor'):
        adapter.prepare(path, tmp_path / 'attempt', kind='metric_descriptor')
    assert not (tmp_path / 'attempt').exists()


def test_descriptor_worker_calls_retained_native_chain_without_raw_reimport(tmp_path, monkeypatch):
    mesh = tmp_path / 'part.obj'
    mesh.write_text('metric mesh input')
    descriptor_path = tmp_path / 'metric.json'
    write(descriptor_path, {'source_id': 'test:source', 'semantic_role': 'manipulation',
                           'visual': binding(mesh), 'collisions': [binding(mesh)]})
    prepared = adapter.prepare(descriptor_path, tmp_path / 'attempt', kind='metric_descriptor')
    monkeypatch.setenv(adapter.WORKER_JOB_ENV, prepared['job']['sha256'])

    def fake_native(path, output):
        assert path == descriptor_path
        assert metric_native_v3._stable_cook is stable_cook_confirmed_v2.stable_cook
        value = {'asset': binding(descriptor_path), 'geometry_qualified': False}
        write(output / 'result.json', value)
        return value

    with patch.object(metric_native_v6, 'compile_variant', side_effect=fake_native) as original:
        with patch.object(raw_manipulation_v1, 'compile_raw', side_effect=AssertionError('unexpected raw import')):
            result = adapter.worker(resolve(prepared['job']))
    assert original.call_count == 1
    assert result['status'] == 'native_qualification_failed'
    assert result['new_grasp_required'] is True
    assert result['grasp_cache_reuse_authorized'] is False
    assert metric_native_v3._stable_cook is assets._stable_cook
