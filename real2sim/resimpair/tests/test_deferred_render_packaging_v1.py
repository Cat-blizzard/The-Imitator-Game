import os

import pytest

from corl2026_workshop.experiments.autonomous_pipeline_v4.core import checked
from resimpair_grasp.common import ROOT, binding, read, write
from resimpair_tasks.deferred_render_v1 import COPY_EVIDENCE
from resimpair_tasks.deferred_render_packaging_v1 import package


def test_legacy_binding_repair_preserves_every_artifact_and_derivative_identity(tmp_path):
    source = tmp_path / 'raw'; source.mkdir()
    artifact = source / 'artifact.bin'; artifact.write_bytes(b'original bytes')
    relative = {**binding(artifact), 'path': os.path.relpath(artifact, ROOT)}
    with pytest.raises(ValueError, match='canonical path changed'):
        checked(relative)
    camera = {'video': relative, 'initial_image': relative, 'final_image': relative,
              'frame_count': 398, 'fps': 20.0}
    bundle = {'saved_state_rerender': True, 'native_execution': False,
        'physical_execution_count': 0, 'new_trajectory': False,
        'source_execution': relative, 'source_execution_id': 'same_physical_execution',
        'measurements': relative, 'planned_paths': relative, 'event_protocol': relative,
        'planning_declaration': relative, 'cameras': {'robot_side': camera, 'opposite_high': camera}}
    for name in (*COPY_EVIDENCE, 'frame_mapping.json', 'source_artifacts.json', 'audit.json'):
        write(source / name, {'fixture': name})
    write(source / 'execution_bundle.json', bundle)
    original_binding = binding(source / 'execution_bundle.json')
    result = package(source, tmp_path / 'bound'); fixed = read(result['path'])
    assert original_binding == binding(source / 'execution_bundle.json')
    assert fixed['source_execution_id'] == bundle['source_execution_id']
    assert fixed['physical_execution_count'] == 0 and fixed['native_execution'] is False
    assert fixed['saved_state_rerender'] is True and fixed['new_trajectory'] is False
    assert checked(fixed['measurements']).read_bytes() == artifact.read_bytes()
    for c in fixed['cameras'].values():
        assert c['video']['sha256'] == relative['sha256']
        assert checked(c['video']).read_bytes() == artifact.read_bytes()
    assert read(tmp_path / 'bound/binding_revision.json')['media_reencoded'] is False


def test_repair_does_not_relabel_fresh_native_executions(tmp_path):
    raw = tmp_path / 'raw'
    write(raw / 'execution_bundle.json', {'saved_state_rerender': False,
        'physical_execution_count': 1, 'new_trajectory': True})
    with pytest.raises(ValueError, match='only_explicit_zero_execution'):
        package(raw, tmp_path / 'output')
