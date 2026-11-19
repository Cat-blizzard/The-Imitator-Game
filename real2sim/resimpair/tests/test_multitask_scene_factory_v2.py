import json
from pathlib import Path

import numpy as np
import pytest

from resimpair_grasp.common import ROOT, read, resolve, write
from resimpair_tasks import multitask_scene_factory_v2 as factory


def test_rotated_slots_do_not_get_rejected_by_coordinate_rounding():
    for count in [2, 3, 4]:
        for angle in np.linspace(-np.pi, np.pi, 1001):
            result = factory.simple_slots([.03] * count, .13, angle=float(angle))
            assert result['passed']
            assert result['minimum_inter_payload_gap_m'] >= .025 - 1e-10


def test_real_build_records_rejected_layout_then_writes_final_request_once(tmp_path, monkeypatch):
    """Real selected meshes/caches/compiler and immutable writes; no physics.

    This optional integration fixture comes from the retained failed v1 build.
    Production validation additionally executes all ten frozen source groups.
    """
    retained = ROOT / 'runs/multitask_factory_20261030_v1'
    if not retained.exists(): pytest.skip('retained native asset integration fixture unavailable')
    selection_path = sorted(retained.glob('scenes/*/source_selection.json'))[0]
    selection = read(selection_path)
    poses = read(retained / 'snapshots/index.json')
    sources = {o['source_id'] for o in selection['objects']}
    selected = [e for e in poses['entries'] if e['source_id'] in sources]
    partition = read(resolve(selected[0]['asset']))['partition']
    bin_sources = {c['source_id'] for c in selection['containers']}
    bins = [b for b in read(retained / 'snapshots/containers.json')['assets']
            if read(resolve(b))['source_id'] in bin_sources]
    native = read(retained / 'snapshots/static_index.json')
    # Reuse enough same-partition native clutter without loading the whole pool.
    clutter = []
    for e in native['entries']:
        d = read(resolve(e['asset']))
        if d['partition'] == partition and d['source_id'] not in sources | bin_sources:
            clutter.append(e)
        if len(clutter) == 30: break
    write(tmp_path / 'poses.json', {'entries': selected})
    write(tmp_path / 'containers.json', {'assets': bins})
    write(tmp_path / 'native.json', {'entries': clutter})
    original = factory.workspace_checks
    calls = []

    def first_layout_rejected(scene, tasks, radius):
        answer = original(scene, tasks, radius)
        calls.append(answer)
        if len(calls) == 1:
            # Exercise a second proposal through the real immutable write path.
            return {**answer, 'passed': False, 'rejected_operations': 1,
                    'test_control': 'forced_first_layout_rejection'}
        return answer

    monkeypatch.setattr(factory, 'workspace_checks', first_layout_rejected)
    output = tmp_path / 'output'
    arguments = (tmp_path / 'poses.json', tmp_path / 'containers.json', tmp_path / 'native.json', output)
    result = factory.build(*arguments, scenes=1, tasks_per_scene=8, seed=202610300800,
                           heldout_fraction=1. if partition == 'heldout_asset' else 0.)
    assert result['geometry_passed_scenes'] == 1
    row = read(output / 'scenes.json')['scenes'][0]
    attempts = read(resolve(row['attempts']))['attempts']
    assert not attempts[0]['accepted'] and attempts[-1]['accepted']
    assert all('overwrite' not in str(a) for a in attempts)
    assert len(read(resolve(row['request']))['tasks']) == 8
    assert len(read(resolve(row['compiled_tasks']))['tasks']) == 8
    assert result['physical_executions'] == result['API_calls'] == result['optimizer_calls'] == 0
    with pytest.raises(ValueError, match='preserve_existing_factory'):
        factory.build(*arguments, scenes=1)
