"""A native success must not bypass a failed original user goal."""
import importlib
import json

import pytest


@pytest.mark.parametrize('name', ['condition_export', 'condition_export_v2'])
def test_requested_failure_blocks_both_export_bridges(tmp_path, monkeypatch, name):
    module = importlib.import_module('resimpair_pickplace.' + name)
    from resimpair_pickplace import requested_goal_audit
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import generic_compiler
    (tmp_path / 'execution_bundle.json').write_text(json.dumps({'case': {'source_request': {}}}))
    monkeypatch.setattr(module, 'evaluate', lambda _: {'passed': True})
    monkeypatch.setattr(requested_goal_audit, 'evaluate', lambda *_: {'passed': False})
    def forbidden(*_):
        pytest.fail('A failed metric goal reached the human-generation compiler')
    monkeypatch.setattr(generic_compiler, 'compile_bundle', forbidden)
    with pytest.raises(ValueError, match='requested metric goal acceptance'):
        module.export(tmp_path)
    assert not (tmp_path / 'compiled').exists()
