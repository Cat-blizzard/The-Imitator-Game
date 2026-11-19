"""Geometry/split/manifest counterexamples for expanded candidate sampling."""
import json

import numpy as np
import pytest

from resimpair_grasp.common import binding, read, write
from resimpair_pickplace import source_role_catalog_v1 as roles
from resimpair_pickplace.requested_tasks import Rejected, add_distractors
from resimpair_tasks import asset_sampling_v3 as factory


def asset(source='small', radius=.025, height=.03, partition='development'):
    return {'source_id': source, 'partition': partition, 'radius': radius,
        'floor_radial_extent_m': radius, 'vertices': np.array([[radius, 0., 0.], [-radius, 0., height]])}


def container(source='bowl', floor=.10, wall=.11, depth=.08, partition='development'):
    return {'source_id': source, 'partition': partition, 'descriptor': {'cavity': {
        'flat_floor_radius_m': floor, 'minimum_wall_radius_m': wall, 'floor_z_m': .01, 'rim_z_m': depth + .01}}}


def test_real_inner_geometry_matters_even_if_outer_box_is_large():
    assert factory.compatibility(asset(), container())['passed']
    assert not factory.compatibility(asset(radius=.06), container(floor=.05))['passed']
    assert not factory.compatibility(asset(radius=.04), container(wall=.045))['passed']
    assert factory.compatibility(asset(height=.12), container(), 'supported_inside')['passed']
    assert not factory.compatibility(asset(height=.12), container(), 'fully_inside')['passed']


def test_pair_packing_is_disjoint_for_unequal_radii_and_many_angles():
    first, second, bowl = asset(radius=.021), asset(radius=.032), container()
    for angle in np.linspace(-np.pi, np.pi, 31):
        offsets = np.array(factory.pack_pair(first, second, bowl, angle))
        assert np.linalg.norm(offsets[0] - offsets[1]) > .021 + .032 + .015
        assert all(np.linalg.norm(o) + r + .008 <= .10 for o, r in zip(offsets, [.021, .032]))
    assert factory.pack_pair(asset(radius=.05), asset(radius=.05), bowl, 0.) is None


def test_original_partition_and_seeded_balancing_are_preserved():
    from collections import Counter
    pool = [asset('a'), asset('b'), asset('heldout', partition='heldout_asset')]
    bins = [container('c'), container('d')]; uses, bin_uses = Counter(), Counter()
    for number in range(4):
        group, station, _ = factory.choose_pair(pool, bins, 'development', uses, bin_uses, 8, number, 'supported_inside')
        assert all(a['partition'] == 'development' for a in group)
        for a in group: uses[a['source_id']] += 1
        bin_uses[station['source_id']] += 1
    assert bin_uses == {'c': 2, 'd': 2}
    schedule = factory.partition_schedule(100, {'development', 'heldout_asset'}, .2, 31)
    assert schedule == factory.partition_schedule(100, {'development', 'heldout_asset'}, .2, 31)
    assert schedule.count('heldout_asset') == 20


def test_live_index_update_cannot_change_frozen_snapshot(tmp_path):
    source = tmp_path / 'live.json'; source.write_text('{"entries": [1]}')
    evidence = factory.snapshot(source, tmp_path / 'frozen.json')
    source.write_text('{"entries": [1, 2]}')
    assert read(tmp_path / 'frozen.json')['entries'] == [1]
    assert binding(tmp_path / 'frozen.json') == evidence['snapshot']


@pytest.fixture
def role_catalog(tmp_path):
    descriptors = []
    for i in range(4):
        path = tmp_path / f'{i}.json'
        write(path, {'id': f'id{i}', 'source_id': f'source{i}', 'partition': 'development',
            'dimensions_m': [.05, .05, .05], 'roles': {'static_distractor': True},
            'category': f'cat{i}', 'half_size': [.025]*3, 'mass_kg': .08, 'name': f'name{i}'})
        descriptors.append(binding(path))
    catalog = tmp_path / 'catalog.json'; write(catalog, {'assets': descriptors})
    sources = {f'other{i}': {'allowed_as_distractor': False} for i in range(6000)}
    sources.update({f'source{i}': {'partitions': ['development'], 'allowed_as_distractor': i != 3,
                                 'descriptors': [descriptors[i]]} for i in range(4)})
    index = tmp_path / 'roles.json'; write(index, {'schema': roles.SCHEMA, 'catalog': binding(catalog), 'sources': sources})
    return catalog, index


def test_more_than_4096_roles_use_bound_index_and_existing_clutter_sampler(role_catalog):
    catalog, index = role_catalog
    settings = {'catalog': binding(catalog), 'source_role_index': binding(index), 'count': 3}
    scene = {'partition': 'development', 'objects': []}
    selected = add_distractors(scene, settings, [], 8)
    assert set(selected) == {'source0', 'source1', 'source2'}
    assert len(scene['objects']) == 3
    assert roles.select(binding(catalog), binding(index), 'heldout_asset', .1, set()) == []


def test_changed_index_cannot_use_cached_admission(role_catalog):
    catalog, index = role_catalog; original = binding(index)
    assert len(roles.select(binding(catalog), original, 'development', .1, set())) == 3
    value = read(index); value['sources']['source3']['allowed_as_distractor'] = True
    index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='hash changed'):
        roles.select(binding(catalog), original, 'development', .1, set())


def test_catalog_alone_cannot_bypass_split_role_policy(role_catalog):
    catalog, _ = role_catalog
    with pytest.raises(Rejected, match='requires_bound_source_role_index'):
        add_distractors({'partition': 'development', 'objects': []}, {'catalog': binding(catalog)}, [], 9)
