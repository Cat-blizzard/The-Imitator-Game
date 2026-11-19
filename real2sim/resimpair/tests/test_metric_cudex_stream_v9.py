import pytest

from resimpair_grasp.common import binding, write
from resimpair_tasks import metric_cudex_stream_v9 as candidate


def setup_ledger(tmp_path, monkeypatch, **updates):
    root = tmp_path / 'ledger'
    write(root / 'protocol.json', {'schema': 'test_original_budget_ledger'})
    write(tmp_path / 'source_plan.json', {'source_ids': ['source:one']})
    plan = binding(tmp_path / 'source_plan.json')
    monkeypatch.setattr(candidate.retained, 'validate', lambda _: {
        'source_plan': plan, 'maximum_cumulative_optimizer_jobs': 3080,
        'gpu_indices': [3, 1, 0], 'maximum_gpu_parallelism': 3})
    receipt = {'parent_protocol': binding(root / 'protocol.json'), 'source_plan': plan,
        'driver': binding(candidate.__file__), 'retained_supervisor': binding(candidate.retained.__file__),
        'supplemental_native_indexes': [], 'maximum_cumulative_optimizer_jobs': 3080,
        'additional_optimizer_allowance': 0, 'gpu_indices': [0, 1, 2, 3],
        'maximum_gpu_parallelism': 4, 'GPU2_explicit_root_admission': True, **updates}
    write(root / 'resource_continuation_v9.json', receipt)
    return root


def test_four_cards_preserve_original_cumulative_budget(tmp_path, monkeypatch):
    root = setup_ledger(tmp_path, monkeypatch)
    result = candidate.amended_config(root)
    assert result['maximum_gpu_parallelism'] == 4
    assert result['maximum_cumulative_optimizer_jobs'] == 3080


def test_resource_amendment_cannot_add_attempt_allowance(tmp_path, monkeypatch):
    root = setup_ledger(tmp_path, monkeypatch, additional_optimizer_allowance=1)
    with pytest.raises(ValueError, match='cannot_increase_budget'):
        candidate.amended_config(root)


def test_gpu2_requires_recorded_authorization(tmp_path, monkeypatch):
    root = setup_ledger(tmp_path, monkeypatch, GPU2_explicit_root_admission=False)
    with pytest.raises(ValueError, match='explicit_resource_admission'):
        candidate.amended_config(root)


def test_duplicate_card_and_unbounded_parallelism_rejected(tmp_path, monkeypatch):
    root = setup_ledger(tmp_path, monkeypatch, gpu_indices=[0, 1, 1, 3], maximum_gpu_parallelism=5)
    with pytest.raises(ValueError, match='bounded_per_card'):
        candidate.amended_config(root)
