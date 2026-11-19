from pathlib import Path
import numpy as np
import pytest
import trimesh
from PIL import Image

from resimpair_grasp.common import read, sha, write
from resimpair_tasks import visual_furniture_bulk_v1 as bulk


def test_native_metric_record_disagreement_is_retained_without_scale_fix():
    good = {'item_id': 'good', 'dimensions_m': {'height': 1., 'length': 2., 'width': .5}}
    bad = {'item_id': 'bad', 'dimensions_m': {'height': 1., 'length': .2, 'width': .5}}
    assert bulk.dimension_evidence([2, 1, .5], [good])['status'] == 'source_metric_record_consistent'
    mixed = bulk.dimension_evidence([2, 1, .5], [good, bad])
    assert mixed['status'] == 'source_dimensions_conflict' and mixed['uniform_scale_applied'] == 1.
    assert len(mixed['comparisons']) == 2
    assert bulk.dimension_evidence([2, 1, .5], [])['status'] == 'source_gltf_units_only_not_independently_confirmed'


def test_native_axis_transform_preserves_lengths_and_front_direction():
    R = bulk.Y_TO_Z[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.)
    assert np.array_equal(R @ [0., 1., 0.], [0., 0., 1.])
    assert np.array_equal(R @ [0., 0., 1.], [0., -1., 0.])


def test_geometry_dedup_ignores_material_and_index_order_but_not_metric_size():
    mesh = trimesh.creation.box(extents=[.4, .7, 1.2])
    a = trimesh.Scene(mesh)
    reordered = mesh.copy(); reordered.faces = reordered.faces[::-1, ::-1]
    reordered.visual.face_colors = [255, 0, 0, 255]
    b = trimesh.Scene(reordered)
    assert bulk.geometry_signature(a) == bulk.geometry_signature(b)
    c = mesh.copy(); c.apply_scale(1.1)
    assert bulk.geometry_signature(a) != bulk.geometry_signature(trimesh.Scene(c))


def test_real_export_path_preserves_geometry_embedded_textures_and_grounding(tmp_path):
    mesh = trimesh.creation.box(extents=[2., 1., .5])
    image = Image.new('RGB', (2048, 1024), (70, 110, 150))
    material = trimesh.visual.material.PBRMaterial(name='test', baseColorTexture=image,
                                                  roughnessFactor=.65, metallicFactor=.1)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)), material=material)
    source = tmp_path / 'source.glb'; source.write_bytes(trimesh.Scene(mesh).export(file_type='glb'))
    before = sha(source)
    row = {'asset_id': 'fixture:test', 'source_id': 'test', 'name': 'fixture cabinet',
        'geometry_path': str(source), 'sha256': before, 'raw_dimensions_y_up': [2., 1., .5],
        'background_category': 'cabinet', 'categories': ['CABINET'],
        'source_url': 'https://example.invalid/test', 'license': 'fixture', 'license_url': 'fixture',
        'attribution': 'unit-test fixture'}
    job = tmp_path / 'job.json'
    write(job, {'source': row, 'output': str(tmp_path / 'output'), 'texture_max_pixels': 1024,
                'source_metric_records': [], 'source_scale': 1.})
    result = bulk.convert(job)
    assert sha(source) == before
    assert result['native_source_scale'] == 1.
    assert result['size_m'] == pytest.approx([2., .5, 1.])
    assert result['grounding_error_m'] < 1e-6 and result['mesh_reload_passed']
    assert result['pick_place_eligible'] is False and result['allowed_roles'] == ['background']
    assert result['GPU_render_validated'] is False
    assert result['texture_conversion'][0]['input_size_px'] == [2048, 1024]
    assert result['textures_after_reload'][0]['output_size_px'] == [1024, 512]
    assert result['split_group'] == 'geometry:' + result['geometry_signature']
    with pytest.raises(ValueError, match='no_asset_conversion_retry'):
        bulk.convert(job)


def test_balanced_batch_has_each_category_before_repeated_categories():
    records = [{'asset_id': f'{c}{i}', 'sha256': f'{c}{i}', 'background_category': c}
               for c in bulk.CATEGORIES.values() for i in range(3)]
    chosen = bulk.balanced(records, 28, 2026103009)
    assert len(chosen) == 28
    assert len({r['background_category'] for r in chosen[:14]}) == 14
    assert chosen == bulk.balanced(records, 28, 2026103009)


@pytest.mark.parametrize('value,expected', [
    ({'unit': 'inches', 'value': 12}, .3048),
    ({'normalized_value': {'unit': 'centimeters', 'value': 30}}, .3),
    ({'unit': 'unknown', 'value': 30}, None),
    ({'unit': 'meters', 'value': float('nan')}, None),
])
def test_unit_evidence(value, expected):
    result = bulk.unit_value(value, bulk.LENGTH)
    assert result is None if expected is None else result == pytest.approx(expected)
