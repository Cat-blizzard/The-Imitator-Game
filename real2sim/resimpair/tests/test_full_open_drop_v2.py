import pytest
from resimpair_pickplace.full_open_drop_v2 import install
from resimpair_pickplace import whole_path_motion_v1 as preview, stability_motion as execution


def test_full_aperture_checked_and_commanded_from_first_open_step_then_restored():
    task = {'operations': [{'target': {'support': 'container_0'},
        'release': {'mode': 'drop_from_above', 'allow_drop': True, 'fragile': False}}]}
    before = preview.opening_width, execution.opening_width, execution.aperture_command
    with install(task):
        assert preview.opening_width(.035) == .08
        assert execution.opening_width(.035) == .08
        assert execution.aperture_command(.041, -.01, .04) == pytest.approx(1.)
        assert execution.aperture_command(.08, -.01, .04) == pytest.approx(1.)
    assert before == (preview.opening_width, execution.opening_width, execution.aperture_command)


def test_near_floor_or_undeclared_tasks_cannot_use_full_drop_adapter():
    with pytest.raises(ValueError, match='explicit_container_drop'):
        with install({'operations': [{'target': {'support': 'container_0'}}]}): pass
