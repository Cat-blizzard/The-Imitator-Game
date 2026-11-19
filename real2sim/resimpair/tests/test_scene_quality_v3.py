from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from resimpair_grasp.common import read, resolve
from resimpair_tasks import scene_quality_v3 as q
from resimpair_tasks import scene_domain_v2 as base
from resimpair_tasks.furniture_metric_repair_v3 import publisher_scale

ROOT=Path('runs/scene_quality_20261029_v3')


def test_publisher_uniform_units_and_anisotropic_counterexample():
    assert publisher_scale([10.974,5.021,21.405],[1097.4,502.1,2140.5])==pytest.approx(.1)
    with pytest.raises(ValueError,match='uniform_scale'):
        publisher_scale([10.974,5.021,21.405],[1000,502.1,2140.5])


@pytest.mark.parametrize('source_id',['source_0','source_1','source_2'])
def test_prepared_real_ablation_preserves_physics_camera_light(source_id):
    scenes={c:read(ROOT/'cases'/f'screen_{source_id}_{c}'/'render_scene.json') for c in q.CONDITIONS}
    original=scenes['baseline_v2']
    for scene in scenes.values():
        assert base.physical_scene(scene)==base.physical_scene(original)
        assert scene['cameras']==original['cameras']
        assert scene['scene_domain']['lighting']==original['scene_domain']['lighting']
    assert scenes['material_only']['scene_domain']['geometry']==original['scene_domain']['geometry']
    assert scenes['furniture_only']['scene_domain']['resources']==original['scene_domain']['resources']
    assert scenes['combined']['scene_domain']['geometry']==scenes['furniture_only']['scene_domain']['geometry']
    assert scenes['combined']['scene_domain']['resources']==scenes['material_only']['scene_domain']['resources']


@pytest.mark.parametrize('source_id',['source_0','source_1','source_2'])
def test_real_furniture_protects_sweep_and_preserves_scale(source_id):
    scene=read(ROOT/'cases'/f'screen_{source_id}_combined'/'render_scene.json')
    g=scene['scene_domain']['geometry'];sweep=np.asarray(g['protected_bounds_m'])
    for f in g['furniture']:
        b=np.asarray(f['world_bounds_m']);assert f['scale']==[1.,1.,1.]
        assert not (np.all(b[1]>sweep[0]-.06) and np.all(b[0]<sweep[1]+.06))
        assert resolve(f['mesh']).is_file()


def test_visual_physical_corruption_is_rejected():
    scene=read(ROOT/'cases/screen_source_0_combined/render_scene.json')
    changed=deepcopy(scene);changed['objects'][0]['position'][0]+=.1
    with pytest.raises(ValueError,match='physics_binding_mismatch'):base.validate_scene(changed)
