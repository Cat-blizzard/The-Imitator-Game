import pytest
from scipy.spatial import QhullError

from resimpair_assets.metric_container_population_v5 import classify_error, prepare
from resimpair_grasp.common import write


def test_empty_cavity_and_degenerate_mesh_are_per_source_failures():
    assert classify_error(ValueError('no_upright_container_support_pose_at_current_metric_scale')) == 'cavity_or_numeric_failed'
    assert classify_error(QhullError('degenerate points')) == 'cavity_or_numeric_failed'


def test_binding_and_infrastructure_errors_do_not_masquerade_as_empty_cavity():
    assert classify_error(ValueError('Artifact hash mismatch')) == 'unknown_system_failure'
    assert classify_error(ModuleNotFoundError('pyembree')) == 'unknown_system_failure'
    assert classify_error(MemoryError()) == 'unknown_system_failure'


def test_container_stage_waits_for_frozen_parent_scope(tmp_path):
    cpu = tmp_path / 'cpu'
    write(cpu / 'status.json', {'counts': {'completed': 10, 'pending': 1, 'running': 0}})
    with pytest.raises(ValueError, match='finish_current_5524'):
        prepare(cpu, tmp_path / 'containers')
    assert not (tmp_path / 'containers').exists()
