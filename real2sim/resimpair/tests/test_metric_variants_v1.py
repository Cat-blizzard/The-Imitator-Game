import copy

import numpy as np
import pytest
import trimesh

from resimpair_assets.metric_variants_v1 import (
    VERSION, compound_properties, mass_plan, materialize, size_plan,
)
from resimpair_grasp.common import binding, read, write


@pytest.fixture
def source(tmp_path):
    # A raw 10x20x30 cm item was previously shrunk by half.
    mesh = trimesh.creation.box(extents=[.05, .10, .15])
    mesh.visual.vertex_colors = np.tile([170, 50, 20, 255], (len(mesh.vertices), 1))
    visual = tmp_path / 'visual.glb'; mesh.export(visual)
    collision = tmp_path / 'collision.obj'; mesh.export(collision)
    evidence = tmp_path / 'source_dimensions.json'
    write(evidence, {'source_id': 'test:item', 'dimensions_m': [.1, .2, .3], 'mass_kg': .6})
    d = {'schema': 'resimpair_mesh_asset_v1', 'id': 'legacy', 'source_id': 'test:item',
         'name': 'box', 'category': 'box', 'partition': 'heldout_asset',
         'visual': binding(visual), 'collisions': [binding(collision)],
         'source_to_asset': np.diag([.5, .5, .5, 1]).tolist(),
         'qualification': {'geometry': 'pass', 'training_admitted': True}}
    p = {'schema': VERSION, 'source_id': 'test:item', 'semantic_role': 'manipulation',
         'semantic_category': 'box', 'source_object_role': 'household_movable',
         'role_rationale': 'test fixture',
         'size': {'mode': 'source_metric', 'source_unit_to_m': 1,
                  'dimensions_m': [.1, .2, .3], 'evidence': [binding(evidence)]},
         'scale_factors': [.9, 1., 1.1], 'variant_ratio_bounds': [.9, 1.1],
         'mass': {'mode': 'measured_base_mass', 'base_mass_kg': .6,
                  'evidence': [binding(evidence)], 'rationale': 'bound source record'}}
    return d, p, mesh


def test_metric_geometry_mass_and_partition_are_bound_to_each_variant(source, tmp_path):
    d, p, mesh = source
    dp, pp = tmp_path / 'descriptor.json', tmp_path / 'policy.json'
    write(dp, d); write(pp, p)
    before = [binding(dp), binding(pp), d['visual'], d['collisions'][0]]
    result = materialize(dp, pp, tmp_path / 'variants')
    assert len(result['variants']) == 3
    for row in result['variants']:
        new = read(tmp_path / 'variants' / row['variant_id'] / 'descriptor.json')
        assert np.allclose(new['dimensions_m'], np.array([.1, .2, .3]) * row['relative_scale'])
        assert new['mass_kg'] == pytest.approx(.6 * row['relative_scale'] ** 3)
        assert new['density_kg_m3'] == pytest.approx(100.)
        assert new['grasp_cache_invalidated'] and new['no_parent_pose_transferred']
        assert new['qualification']['geometry'] == 'pending_independent_check'
        assert new['qualification']['training_admitted'] is False
        assert new['partition'] == 'heldout_asset'
        assert new['lineage']['split_group_source_id'] == 'test:item'
        props = read(tmp_path / 'variants' / row['variant_id'] / 'physical_properties.json')
        assert np.isfinite(props['inertia_kg_m2']).all()
        assert props['native_mass_calibration'] == 'pending'
    from resimpair_grasp.common import resolve
    for ref in before:
        resolve(ref)
    with pytest.raises(ValueError, match='preserve_existing'):
        materialize(dp, pp, tmp_path / 'variants')


def test_source_dimension_conflict_is_not_fixed_by_distorting_shape(source):
    d, p, mesh = source
    p['size']['dimensions_m'] = [.1, .2, .6]
    with pytest.raises(ValueError, match='dimension_geometry_disagreement'):
        size_plan(d, mesh, p)


def test_furniture_role_cannot_be_miniaturized_as_manipulation(source):
    d, p, mesh = source
    p['source_object_role'] = 'furniture'
    with pytest.raises(ValueError, match='furniture'):
        size_plan(d, mesh, p)
    p['semantic_role'] = 'background'
    assert size_plan(d, mesh, p)['base_multiplier_from_parent'] == 2.


def test_unknown_units_allow_declared_inference_not_silent_normalization(source):
    d, p, mesh = source
    p['size'] = {'mode': 'inferred_category', 'base_longest_m': .12,
                 'plausible_longest_m': [.08, .16], 'rationale': 'hypothetical category range'}
    plan = size_plan(d, mesh, p)
    assert plan['confidence'] == 'engineering_inference_not_measured'
    assert plan['base_obb_m'][-1] == pytest.approx(.12)
    p['size']['plausible_longest_m'] = [.08, .125]
    with pytest.raises(ValueError, match='category_size_interval'):
        size_plan(d, mesh, p)


def test_fixed_mass_assumption_can_be_configured_without_claiming_measurement(source):
    _, p, _ = source
    p['mass'] = {'mode': 'assumed_base_mass', 'base_mass_kg': .15,
                 'rationale': 'bounded household surrogate for absent weight record'}
    plan = mass_plan(p, .0003)
    assert plan['effective_density_kg_m3'] == pytest.approx(500.)
    assert plan['confidence'] == 'engineering_assumption_not_measured'


def test_effective_density_and_overlap_model_are_explicit(source):
    _, p, mesh = source
    translated = mesh.copy(); translated.apply_translation([.04, 0, 0])
    props = compound_properties([mesh, translated])
    assert props['center_of_mass_m'] == pytest.approx([.02, 0, 0])
    assert props['collision_shape_volume_sum_m3'] == pytest.approx(2 * mesh.volume)
    assert props['overlap_correction'].startswith('not_applied')
    p['mass'] = {'mode': 'effective_density', 'density_kg_m3': 400.,
                 'rationale': 'effective shape density, not solid material density'}
    assert mass_plan(p, .001)['base_mass_kg'] == pytest.approx(.4)


def test_nonfinite_or_open_mesh_mass_is_rejected_before_output(source, tmp_path):
    d, p, mesh = source
    open_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[:-1], process=False)
    with pytest.raises(ValueError, match='closed_consistently'):
        compound_properties([open_mesh])
    p['scale_factors'] = [1, float('nan')]
    with pytest.raises(ValueError, match='positive_finite'):
        size_plan(d, mesh, p)


def test_binding_change_and_anisotropic_source_transform_rejected(source, tmp_path):
    d, p, mesh = source
    d['source_to_asset'][0][0] = .7
    with pytest.raises(ValueError, match='uniform_similarity'):
        size_plan(d, mesh, p)
    d['source_to_asset'][0][0] = .5
    p['size']['evidence'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash changed'):
        size_plan(d, mesh, p)
