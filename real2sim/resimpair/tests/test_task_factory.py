from pathlib import Path
import copy
import pytest
from resimpair_grasp.common import ROOT,read,resolve
from resimpair_tasks import programs,factory


def sample(tmp_path):
    index=ROOT/'assets/grasp_library/grasp_cache_v3/development_index.json'
    pool,rejected=factory.entries([index],'development')
    a=next(e for e in pool if read(resolve(e['asset']))['category']=='fruit')
    b=next(e for e in pool if read(resolve(e['asset']))['category']=='bottle')
    return programs.create_scene(ROOT/'assets/grasp_library/v2/catalog.json',[a,b],20261026,0,tmp_path,['select','clear','arrange','ordered'])


def test_same_scene_has_distinct_goals_and_unchanged_non_targets(tmp_path):
    scene,tasks=sample(tmp_path)
    assert len(tasks)==5 and len({t['id'] for t in tasks})==5
    select=next(t for t in tasks if t['composition_template']=='select_0')
    protocol=programs.validate_task(scene,select)
    assert any(c['id']=='object_b.unchanged' for c in protocol['constraints'])
    ordered=next(t for t in tasks if t['composition_template']=='ordered_pair')
    clear=next(t for t in tasks if t['composition_template']=='clear_pair')
    assert ordered['operations'][0]['object']!=clear['operations'][0]['object']
    assert ordered['required_operation_precedence'] and not clear['required_operation_precedence']
    active={read(resolve(o['asset_descriptor']))['source_id'] for o in scene['objects'][:2]}
    for o in scene['objects'][2:]:
        d=read(resolve(o['asset_descriptor']))
        assert d['source_id'] not in active and d['partition'] in {'development','shared_distractor'}


def test_new_object_names_cannot_grant_pouring_or_static_manipulation(tmp_path):
    scene,tasks=sample(tmp_path);task=copy.deepcopy(tasks[0]);task['required_skills']=['pour']
    with pytest.raises(ValueError,match='Unsupported capability'):programs.validate_task(scene,task)
    task=copy.deepcopy(tasks[0]);oid=scene['objects'][2]['id'];task['object_ids']=[oid];task['operations'][0]['object']=oid
    with pytest.raises(ValueError,match='Static-only'):programs.validate_task(scene,task)


def test_repeated_transfers_require_a_different_event_adapter(tmp_path):
    scene,tasks=sample(tmp_path);task=copy.deepcopy(tasks[0]);task['operations'].append(copy.deepcopy(task['operations'][0]))
    with pytest.raises(ValueError,match='Repeated transfers'):programs.validate_task(scene,task)
