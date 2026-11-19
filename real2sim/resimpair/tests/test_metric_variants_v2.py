import copy

import numpy as np
import pytest
import trimesh

from resimpair_assets import metric_variants_v2 as candidate
from resimpair_grasp.common import binding, read, resolve, write


@pytest.fixture
def source(tmp_path):
    mesh = trimesh.creation.box(extents=[.05, .1, .15])
    visual, collision = tmp_path / 'visual.glb', tmp_path / 'collision.obj'
    mesh.export(visual); mesh.export(collision)
    evidence = tmp_path / 'measurements.json'; write(evidence, {'dimensions_m': [.1, .2, .3], 'mass_kg': .6})
    descriptor = {'schema': 'resimpair_mesh_asset_v1', 'id': 'test_box', 'source_id': 'test:box',
        'visual': binding(visual), 'collisions': [binding(collision)], 'source_to_asset': np.diag([.5, .5, .5, 1.]).tolist(),
        'partition': 'heldout_asset', 'name': 'box', 'category': 'unlisted_generic_object'}
    policy = {'schema': candidate.VERSION, 'source_id': 'test:box', 'semantic_role': 'manipulation',
        'semantic_category': 'unlisted_generic_object', 'source_object_role': 'household_movable',
        'role_rationale': 'Explicit generic tabletop engineering fixture', 'variant_ratio_bounds': [.9, 1.1],
        'scale_factors': [.9, 1., 1.1],
        'size': {'mode': 'source_metric', 'source_unit_to_m': 1., 'dimensions_m': [.1, .2, .3],
                 'evidence': [binding(evidence)]},
        'mass': {'mode': 'assumed_base_mass', 'base_mass_kg': .6, 'rationale': 'engineering fixture, not a measurement'}}
    dp, pp = tmp_path / 'descriptor.json', tmp_path / 'policy.json'
    write(dp, descriptor); write(pp, policy)
    return dp, pp, tmp_path


def test_v2_materializes_real_geometry_with_cubic_mass_and_fifth_power_inertia(source):
    dp, pp, directory = source
    before = binding(dp), binding(pp)
    result = candidate.materialize(dp, pp, directory / 'v2')
    assert len(result['variants']) == 3
    moments = []
    for row in result['variants']:
        d = read(resolve(row['descriptor'])); props = read(resolve(d['physical_properties']))
        report = candidate.audit_variant(resolve(row['descriptor']))
        assert report['passed'] and not report['actual_variant_mass_measured']
        assert not report['profile']['matched_category_profiles']  # No category whitelist.
        assert d['mass_kg'] == pytest.approx(.6 * row['relative_scale'] ** 3)
        assert d['simulation_usage']['counts_toward_robot_payload'] is True
        assert d['qualification']['physical_execution'] == 'unknown'
        assert d['qualification']['training_admitted'] is False
        moments.append(np.array(props['inertia_kg_m2']) / row['relative_scale'] ** 5)
    assert np.allclose(moments[0], moments[1]) and np.allclose(moments[1], moments[2])
    assert before == (binding(dp), binding(pp))


@pytest.mark.parametrize('mass', [0, -1, 1000, float('nan'), float('inf')])
def test_abnormal_dynamic_mass_rejected_before_any_geometry_output(source, mass):
    dp, pp, directory = source
    policy = read(pp); policy['mass']['base_mass_kg'] = mass
    if np.isfinite(mass):
        changed = directory / 'bad.json'; write(changed, policy)
        with pytest.raises(ValueError): candidate.materialize(dp, changed, directory / 'must_not_exist')
    else:
        with pytest.raises(ValueError): candidate.retained.mass_plan(policy, .006)
    assert not (directory / 'must_not_exist').exists()


def test_self_declared_huge_randomization_is_rejected_not_clipped(source):
    dp, pp, directory = source
    policy = read(pp); policy['scale_factors'] = [1., 2.]; policy['variant_ratio_bounds'] = [.5, 2.]
    changed = directory / 'unbounded.json'; write(changed, policy)
    with pytest.raises(ValueError, match='variant_ratios_outside'):
        candidate.preflight(dp, changed)
    assert read(changed)['scale_factors'] == [1., 2.]


def test_full_sized_furniture_uses_static_profile_not_robot_payload(source):
    dp, pp, directory = source
    policy = read(pp)
    policy.update(semantic_role='background', semantic_category='kitchen_cart', source_object_role='furniture')
    policy['size'] = {'mode': 'inferred_category', 'base_longest_m': .81, 'plausible_longest_m': [.6, 1.],
                      'rationale': 'full sized cart fixture'}
    policy['mass']['base_mass_kg'] = 14.1
    changed = directory / 'cart_policy.json'; write(changed, policy)
    proof = candidate.preflight(dp, changed)
    assert all(x['simulation_usage']['body_type'] == 'static_visual' for x in proof['checks'])
    assert all(x['simulation_usage']['robot_payload_mass_kg'] == 0 for x in proof['checks'])
    assert proof['checks'][1]['mass_kg'] == 14.1
    policy['semantic_role'] = 'manipulation'; changed = directory / 'cart_bad_role.json'; write(changed, policy)
    with pytest.raises(ValueError, match='furniture'): candidate.preflight(dp, changed)


def values():
    mesh = trimesh.creation.box(extents=[.1, .1, .1])
    props = candidate.retained.compound_properties([mesh]); density = 500.
    props.update(mass_kg=.5, effective_density_kg_m3=density,
                 inertia_kg_m2=(np.asarray(props['unit_density_inertia_kg_m2']) * density).tolist())
    policy = {'semantic_role': 'manipulation', 'semantic_category': 'generic', 'variant_ratio_bounds': [.9, 1.1],
              'mass': {'mode': 'assumed_base_mass', 'rationale': 'engineering fixture'}}
    size = {'scale_factors': [1.]}
    mass = {'base_mass_kg': .5, 'effective_density_kg_m3': density, 'confidence': 'engineering_assumption_not_measured'}
    return policy, size, mass, props, [.1, .1, .1], mesh.bounds, 1., candidate.load_bounds()


@pytest.mark.parametrize('failure', ['asymmetric', 'negative', 'triangle', 'too_large', 'outside_COM', 'density'])
def test_inertia_density_and_COM_pathologies_rejected(failure):
    args = list(values()); props = args[3]; matrix = np.array(props['inertia_kg_m2'])
    if failure == 'asymmetric': matrix[0, 1] = .01
    elif failure == 'negative': matrix[0, 0] = -.01
    elif failure == 'triangle': matrix = np.diag([.001, .001, .01])
    elif failure == 'too_large': matrix *= 1000
    elif failure == 'outside_COM': props['center_of_mass_m'][0] = 1.
    elif failure == 'density': props['effective_density_kg_m3'] = 1e9
    props['inertia_kg_m2'] = matrix.tolist()
    props['unit_density_inertia_kg_m2'] = (matrix / props['effective_density_kg_m3']).tolist()
    with pytest.raises(ValueError): candidate.inspect_values(*args)


def test_source_mass_cannot_be_substituted_by_self_consistent_provenance(source):
    dp, pp, directory = source
    result = candidate.materialize(dp, pp, directory / 'v2')
    d = read(resolve(result['variants'][1]['descriptor'])); props = read(resolve(d['physical_properties']))
    d['mass_kg'] *= 2; d['density_kg_m3'] *= 2
    d['mass_provenance']['base_mass_kg'] *= 2; d['mass_provenance']['effective_density_kg_m3'] *= 2
    props['mass_kg'] *= 2; props['effective_density_kg_m3'] *= 2
    props['inertia_kg_m2'] = (np.asarray(props['inertia_kg_m2']) * 2).tolist()
    physical = directory / 'changed_props.json'; write(physical, props); d['physical_properties'] = binding(physical)
    changed = directory / 'changed_descriptor.json'; write(changed, d)
    with pytest.raises(ValueError, match='mass_provenance:base_mass_kg'):
        candidate.audit_variant(changed)


def test_cache_bound_to_other_size_rejected_before_execution(source):
    from resimpair_grasp.grasps import GRIPPER
    dp, pp, directory = source
    result = candidate.materialize(dp, pp, directory / 'v2')
    a, b = result['variants'][:2]
    path = directory / 'wrong_cache.json'
    write(path, {'signature': {'descriptor': a['descriptor'], 'gripper': GRIPPER}, 'candidates': []})
    with pytest.raises(ValueError, match='different mesh/scale/gripper'):
        candidate.load_for_manipulation(binding(path), b['descriptor'])
