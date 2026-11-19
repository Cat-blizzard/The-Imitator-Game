from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from corl2026_workshop.experiments.autonomous_pipeline_v4.library import make_scene
from resimpair_grasp.common import ROOT, read
from resimpair_tasks import appearance_v1 as appearance


def test_legacy_scene_inputs_remain_identical():
    rows = read(ROOT / 'validation/appearance_20261029/baseline/scenes.json')['scenes']
    for row in rows:
        assert make_scene('baseline', row['seed'], nuisance=row['nuisance'], layout=row['layout']) == row['scene']


def test_physics_tasks_and_cameras_unchanged_but_visual_sources_vary():
    scene = make_scene('fixture', 42, nuisance=True, layout=True)
    rows = [appearance.sample(scene, seed) for seed in range(20)]
    assert scene == make_scene('fixture', 42, nuisance=True, layout=True)
    for row in rows:
        assert {k: v for k, v in row.items() if k != 'appearance'} == {
            k: v for k, v in scene.items() if k != 'appearance'}
        appearance.validate_scene(row)
        assert np.isclose(np.linalg.norm(row['appearance']['key_direction']), 1.)
        assert row['appearance']['key_direction'][2] < 0
    assert len({r['appearance']['table_surface']['id'] for r in rows}) >= 4
    assert len({r['appearance']['environment']['id'] for r in rows}) >= 4
    assert {r['appearance']['background_mode'] for r in rows} == {'wall', 'environment'}
    assert appearance.sample(scene, 7) == appearance.sample(scene, 7)


def test_catalog_is_self_contained_and_all_declared_resources_resolve():
    catalog = appearance.validate_catalog(appearance.DEFAULT_CATALOG)
    assert len(catalog['entries']) == 25
    assert sum('environment' in e['roles'] for e in catalog['entries']) == 9
    for entry in catalog['entries']:
        assert entry['license'] == 'CC0-1.0'
        for resource in entry['resources'].values():
            assert not Path(resource['path']).is_absolute()
            assert appearance.resolve(resource).is_relative_to(ROOT)


@pytest.mark.parametrize('config', [
    {'table_ids': ['does_not_exist']}, {'key_intensity': [2, 1]},
    {'ambient_intensity': [-1, 1]}, {'temperature_k': [100000]},
    {'background_mode': 'unimplemented_room_mesh'}, {'camera_randomize': True}])
def test_bad_configuration_fails_before_environment(config):
    with pytest.raises(ValueError):
        appearance.sample(make_scene('fixture', 1), 1, config=config)


def test_mutated_task_scene_requires_resampling_visual_binding():
    scene = appearance.sample(make_scene('fixture', 1), 1)
    scene['objects'][0]['position'][0] += .02
    with pytest.raises(ValueError, match='physics_or_camera_changed'):
        appearance.validate_scene(scene)


def test_library_opt_in_uses_same_sampler_without_touching_physics():
    old = make_scene('fixture', 3)
    candidate = make_scene('fixture', 3, appearance_config={})
    assert candidate == appearance.sample(old, 2003)
