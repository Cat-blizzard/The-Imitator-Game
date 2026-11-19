from pathlib import Path
import pytest
from resimpair_grasp.common import write,binding
from resimpair_tabletop.extend import previous_identities


def test_extension_skips_failures_too_and_refuses_unfinished_batch(tmp_path):
    terminal=tmp_path/'terminal.json';write(terminal,{'status':'failed'})
    plan=tmp_path/'plan.json';write(plan,{'jobs':[{'id':'one','source_sha256':'identity_a'}]})
    complete=tmp_path/'complete.json';write(complete,{'plan':binding(plan),'attempted':1,'scheduled':1,
        'counts':{'failed':1},'rows':[{'id':'one','terminal':binding(terminal)}]})
    seen,parents=previous_identities([complete]);assert seen=={'identity_a'} and len(parents)==1
    pending=tmp_path/'pending.json';write(pending,{'plan':binding(plan),'attempted':0,'scheduled':1,
        'counts':{'pending':1},'rows':[]})
    with pytest.raises(ValueError,match='incomplete'):previous_identities([pending])


def test_extension_inherits_all_prior_processed_identities(tmp_path):
    previous=[]
    for i in range(2):
        terminal=tmp_path/f't{i}.json';write(terminal,{'status':'completed','id':i})
        plan=tmp_path/f'p{i}.json';write(plan,{'jobs':[{'id':str(i),'source_sha256':str(i)}],
            'reused_parent_catalogs':[binding(previous[-1])] if previous else []})
        catalog=tmp_path/f'c{i}.json';write(catalog,{'plan':binding(plan),'attempted':1,'scheduled':1,
            'counts':{'completed':1},'rows':[{'id':str(i),'terminal':binding(terminal)}]})
        previous.append(catalog)
    assert previous_identities([previous[-1]])[0]=={'0','1'}
