"""A rejected stage retains the collision evidence already computed by execution."""

from mani_skill.examples.motionplanning.dual.sync.solve import _compact_stage


def test_candidate_rejection_keeps_first_witness_without_full_trace():
    pair = {'object_name1': 'left', 'link_name1': 'finger', 'object_name2': 'target'}
    witness = {'time_s': .15, 'control_interval': 3, 'collisions': [pair]}
    report = _compact_stage('descend', {'status': 'rejected', 'reason': 'candidate_collision',
        'validation': {'status': 'collision', 'first_collision': witness, 'sample_count': 8},
        'trace': []})
    assert report['collision_validation']['first_collision'] == witness
    assert 'trace' not in report and 'sample_count' not in report['collision_validation']


def test_initial_rejection_keeps_pair_without_changing_status():
    collisions = [{'link_name1': 'finger', 'object_name2': 'target'}]
    report = _compact_stage('retreat', {'status': 'rejected', 'reason': 'initial_collision',
        'initial_collisions': collisions, 'trace': []})
    assert report['initial_collisions'] == collisions
    assert report['status'] == 'rejected' and report['reason'] == 'initial_collision'


def test_success_remains_compact_and_does_not_include_diagnostic_trace():
    report = _compact_stage('descend', {'status': 'passed', 'initial_collisions': [],
        'validation': {'status': 'passed', 'first_collision': None},
        'trace': [{'measurements': {'large': [1, 2, 3]}}]})
    assert report == {'name': 'descend', 'status': 'passed',
                      'collision_validation': {'status': 'passed'}}
