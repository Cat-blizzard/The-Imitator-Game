"""Real portable geometry and immutable request/coverage boundary checks."""
from pathlib import Path

import pytest

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from resimpair_pickplace.pose_bundle import materialize
from resimpair_pickplace.requested_tasks import (
    Rejected, add_distractors, compile_scene, request_index, MAX_DISTRACTOR_EXCLUSIONS)
from resimpair_tasks.coverage_audit_v1 import audit
from resimpair_tasks import task_factory_v1 as factory


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    folder = tmp_path_factory.mktemp('factory_native_library')
    indices = materialize(ROOT / 'assets/precomputed_grasps_v1', folder)
    return indices['cudex'], folder / 'container_catalog.json'


@pytest.fixture(scope='module')
def batch(library, tmp_path_factory):
    output = tmp_path_factory.mktemp('task_factory_parent') / 'batch'
    status = factory.build(*library, output, scenes=9, min_unique_assets=9, seed=20261027)
    return output, status


def test_real_geometry_five_goals_per_scene_and_bounded_chunks(batch):
    folder, status = batch
    assert status['status'] == 'ready_for_independent_ik'
    assert status['generated_scenes'] == 9 and status['generated_tasks'] == 45
    assert status['distinct_anchor_sources'] == 9
    assert status['physical_executions'] == status['api_calls'] == status['optimizer_calls'] == 0
    chunks = read(folder / 'chunks.json')['chunks']
    assert [c['task_count'] for c in chunks] == [40, 5]
    assert [c['seed_start'] for c in chunks] == [20261027, 20261035]
    object_counts = set(); container_counts = set()
    for chunk in chunks:
        requests = request_index(read(resolve(chunk['requests'])))
        for scene in requests['scenes']:
            assert len(scene['tasks']) == 5
            assert all(c['source_id'] for c in scene['containers'])
            object_counts.add(len(scene['objects'])); container_counts.add(len(scene['containers']))
    assert object_counts == {2, 3, 4} and container_counts == {1, 2}


def test_independent_auditor_derives_more_than_three_real_goal_structures(batch):
    folder, _ = batch
    rows = read(folder / 'coverage_manifest.json')['rows']
    result = audit(rows, expected_count=45)
    assert result['passed'], result['errors']
    assert result['counts']['unique_relation_structure'] > 3
    assert result['counts']['unique_quantized_metric'] == 45
    assert result['counts']['contact_mechanisms'] == 1
    assert result['counts']['manipulated_source_assets'] >= 9
    assert all(row['physical_status'] == row['human_video_status'] == 'not_evaluated' for row in rows)
    assert all(row['source_coverage'] == 'synthetic_only' for row in rows)
    assert all(row.get('source_novelty') != 'global_unseen_asset' for row in rows)
    for scene in read(folder / 'scenes.json')['scenes']:
        related = [r for r in rows if r['scene_id'] == scene['scene_id']]
        assert len(related) == 5
        assert all(r['source_request'] == scene['request'] for r in related)
    assert not result['source_partition_leaks']
    assert all(len(p) == 1 for p in read(folder / 'source_splits.json')['source_partitions'].values())


def test_registration_seed_reproduces_actual_compiler_distractors(batch, library):
    folder, _ = batch
    record = read(folder / 'scenes.json')['scenes'][0]
    request = read(resolve(record['request']))
    compiled, tasks, checks = compile_scene(request, read(library[0])['entries'], read(library[1])['assets'], record['seed'])
    saved = read(folder / 'scenes' / request['id'] / 'compiled_scene.json')
    assert saved == compiled
    assert tasks == read(folder / 'scenes' / request['id'] / 'compiled_tasks.json')['tasks']
    assert checks == read(folder / 'scenes' / request['id'] / 'geometry_preflight.json')['checks']


def test_short_pool_reports_gap_without_fabricating_a_hundred_sources(library, tmp_path):
    output = tmp_path / 'insufficient'
    status = factory.build(*library, output, scenes=100, min_unique_assets=100)
    assert status['status'] == 'insufficient_assets'
    assert status['ready_unique_assets'] < 100
    assert status['requested_unique_assets'] == 100
    assert status['generated_tasks'] == status['generated_scenes'] == 0
    assert not (output / 'coverage_manifest.json').exists()
    assert (output / 'inventory.json').exists()


def test_rejected_geometry_keeps_five_tasks_and_every_finite_proposal(library, tmp_path, monkeypatch):
    calls = []
    def reject(spec, *_args):
        calls.append(spec)
        raise Rejected('deliberate_obstacle_counterexample', detail='retained test failure')
    monkeypatch.setattr(factory, 'compile_scene', reject)
    output = tmp_path / 'rejected'
    status = factory.build(*library, output, scenes=1, min_unique_assets=1)
    assert status['status'] == 'geometry_rejections'
    assert len(calls) == len(factory.LAYOUTS) == 6
    assert len({tuple(o['source_id'] for o in s['objects']) for s in calls}) == 1
    rows = read(output / 'coverage_manifest.json')['rows']
    assert len(rows) == 5 and all(r['geometry_status'] == 'rejected' for r in rows)
    record = read(output / 'scenes.json')['scenes'][0]
    evidence = read(resolve(record['geometry_preflight']))
    assert all(a['reason'] == 'deliberate_obstacle_counterexample' for a in evidence['attempts'])
    assert all(resolve(a['request']).is_file() for a in evidence['attempts'])


def test_native_cache_geometry_is_required_before_source_admission(library, tmp_path):
    value = read(library[0]); entry = value['entries'][0]
    cache = read(resolve(entry['cache']))
    cache['signature']['geometry_scope'] = 'proxy'
    write(tmp_path / 'cache.json', cache)
    write(tmp_path / 'index.json', {'entries': [{'asset': entry['asset'], 'cache': binding(tmp_path / 'cache.json')}]})
    pool, _, rejected, _ = factory.load_pool(tmp_path / 'index.json', library[1])
    assert not pool
    assert rejected[0]['details']['message'] == 'native_collision_geometry_cache_required'


def test_existing_output_cannot_be_rewritten(library, batch):
    folder, _ = batch
    before = binding(folder / 'coverage_manifest.json')
    with pytest.raises(ValueError, match='already exists'):
        factory.build(*library, folder, scenes=9, seed=999)
    assert before == binding(folder / 'coverage_manifest.json')


def test_full_input_pools_are_excluded_from_clutter_in_every_scene(batch, library):
    folder, _ = batch
    expected = {read(resolve(e['asset']))['source_id'] for e in read(library[0])['entries']}
    expected.update(read(resolve(b))['source_id'] for b in read(library[1])['assets'])
    policy = read(folder / 'source_role_policy.json')
    assert set(policy['exclude_source_ids']) == expected
    for record in read(folder / 'scenes.json')['scenes']:
        spec = read(resolve(record['request']))
        assert set(spec['distractors']['exclude_source_ids']) == expected
        scene = read(folder / 'scenes' / spec['id'] / 'compiled_scene.json')
        identities = scene['randomization']['distractor_source_ids']
        assert len(identities) == spec['distractors']['count']
        assert not set(identities) & expected
        assert expected > {o['source_id'] for o in spec['objects']}


def test_legacy_missing_exclusion_and_explicit_empty_policy_match():
    first = {'objects': [], 'partition': 'heldout_asset'}
    second = {'objects': [], 'partition': 'heldout_asset'}
    a = add_distractors(first, {'count': 4, 'max_dimension_m': .13}, [], 20261027)
    b = add_distractors(second, {'count': 4, 'max_dimension_m': .13, 'exclude_source_ids': []}, [], 20261027)
    assert len(a) == 4 and a == b and first == second


def test_excluded_source_cannot_reappear_under_a_static_catalog_role():
    baseline = {'objects': [], 'partition': 'heldout_asset'}
    original = add_distractors(baseline, {'count': 4, 'max_dimension_m': .13}, [], 20261027)
    excluded = original[0]
    after = {'objects': [], 'partition': 'heldout_asset'}
    selected = add_distractors(after, {'count': 4, 'max_dimension_m': .13,
                                     'exclude_source_ids': [excluded]}, [], 20261027)
    assert len(selected) == 4 and excluded not in selected
    assert excluded not in {read(resolve(o['asset_descriptor']))['source_id'] for o in after['objects']}


@pytest.mark.parametrize('invalid', [None, 'source:one', {}, [1], [None], [['source:one']], [''],
    [' source:one'], ['source:one\n'], ['source:one', 'source:one'], ['x' * 513],
    [f'source:{i}' for i in range(MAX_DISTRACTOR_EXCLUSIONS + 1)]])
def test_exclusion_policy_is_bounded_and_strict(invalid):
    with pytest.raises(Rejected, match='invalid_distractor_exclude_source_ids'):
        add_distractors({'objects': [], 'partition': 'heldout_asset'}, {'exclude_source_ids': invalid}, [], 1)


def test_container_cross_role_partition_conflict_is_rejected_without_relabeling(library, tmp_path):
    entry = read(library[0])['entries'][0]
    active = read(resolve(entry['asset']))
    value = read(resolve(read(library[1])['assets'][0]))
    value.update(source_id=active['source_id'], partition='heldout_asset' if active['partition'] == 'development' else 'development')
    write(tmp_path / 'conflicting_container.json', value)
    write(tmp_path / 'containers.json', {'assets': [binding(tmp_path / 'conflicting_container.json')]})
    before = binding(tmp_path / 'conflicting_container.json')
    _, containers, _, rejected = factory.load_pool(library[0], tmp_path / 'containers.json')
    assert not containers
    assert rejected[0]['reason'] == 'container_active_source_partition_conflict'
    assert rejected[0]['details']['active_partitions'] == [active['partition']]
    assert binding(tmp_path / 'conflicting_container.json') == before


def test_same_container_identity_cannot_enter_two_partitions(library, tmp_path):
    value = read(resolve(read(library[1])['assets'][0]))
    value['source_id'] = 'regression:shared_container_identity'
    values = []
    for partition in ['development', 'heldout_asset']:
        value['partition'] = partition
        path = tmp_path / (partition + '.json'); write(path, value); values.append(binding(path))
    write(tmp_path / 'containers.json', {'assets': values})
    _, containers, _, rejected = factory.load_pool(library[0], tmp_path / 'containers.json')
    assert not containers and len(rejected) == 2
    assert all(r['reason'] == 'container_source_partition_conflict' for r in rejected)
