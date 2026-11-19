import copy
import pytest
from resimpair_grasp.common import ROOT, read, resolve
from resimpair_pickplace.rolling_pickplace_v3 import operation_task
from corl2026_workshop.experiments.autonomous_pipeline_v4 import planning_protocols, simulation


def test_real_multitask_protocol_fragments_preserve_full_task():
    # Actual frozen input which exposed the missing target/goal adapter bug.
    root = ROOT / 'runs/multitask_10x8_20261030_v2/candidate'
    if not (root / 'cohort.json').exists():
        pytest.skip('real integration evidence is not installed')
    cases = read(root / 'cohort.json')['cases']
    for count in (2, 3, 4):
        case = next(c for c in cases if c['operation_count'] == count)
        scene, task = read(resolve(case['scene'])), read(resolve(case['task']))
        original = copy.deepcopy(task)
        ids = [o['id'] for o in scene['objects']]
        supports = ids + [s['id'] for s in scene['supports']] + ['table']
        full = planning_protocols.pick_place(scene, task, ids, supports, simulation.targets(scene, task))
        for op in task['operations']:
            one = operation_task(task, op)
            fragment = planning_protocols.pick_place(scene, one, ids, supports, simulation.targets(scene, one))
            assert len(fragment['events']) == 6
            assert {e['id'] for e in fragment['events']} == {e['id'] for e in full['events'] if e['id'].startswith(op['id'] + '.')}
            assert [e for e in fragment['entities'] if e['role'] == 'target'][0]['id'] == op['object']
        assert task == original
        broken = copy.deepcopy(task); broken['operations'] = task['operations'][:1]
        with pytest.raises(ValueError, match='target_has_no_declared_goal'):
            planning_protocols.pick_place(scene, broken, ids, supports, simulation.targets(scene, broken))
