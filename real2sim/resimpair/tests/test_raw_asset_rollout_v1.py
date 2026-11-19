"""Recovery and lineage checks for a finite raw queue; no solver substitutes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from resimpair_grasp.common import binding, digest, read, write
from resimpair_tasks.raw_asset_rollout_v1 import (
    deadline, selected_paths, unfinished_gpu_jobs, validate_gpu_job,
)


def bound_job(tmp_path):
    root=tmp_path/'queue';root.mkdir()
    sid='objaverse:bound-source'
    cpu=root/'cpu'/digest(sid)[:24]/'job.json'
    write(cpu,{'raw_row':{'id':sid}})
    asset=root/'asset.json';write(asset,{'actual_geometry_identity':'fixture-only'})
    entry={'asset':binding(asset),'source_id':sid}
    write(cpu.parent/'native_entry.json',entry)
    write(root/'population.json',{'sources':[{'source_id':sid,'job':binding(cpu)}]})
    p={'job_paths':[str(cpu)],'qualification_job_paths':[str(cpu)],'population':binding(root/'population.json')}
    write(root/'protocol.json',p)
    path=root/'gpu'/digest(sid)[:24]/'job.json'
    value={'source_id':sid,'protocol':binding(root/'protocol.json'),'output':str(path.parent),
           'asset':entry['asset'],'entry':entry}
    write(path,value)
    return root,p,sid,path,value


def test_native_job_is_bound_to_registered_source(tmp_path):
    root,p,sid,path,value=bound_job(tmp_path)
    assert validate_gpu_job(path,root,p,{sid})==value


@pytest.mark.parametrize('mutation',['foreign_directory','asset','scope','protocol','output'])
def test_misbound_gpu_jobs_are_rejected(tmp_path,mutation):
    root,p,sid,path,value=bound_job(tmp_path)
    selected={sid}
    if mutation=='foreign_directory':path=root/'gpu'/'foreign'/'job.json'
    elif mutation=='scope':selected=set()
    elif mutation=='asset':value['asset']={'path':'unrelated','sha256':'wrong'}
    elif mutation=='protocol':value['protocol']['sha256']='wrong'
    elif mutation=='output':value['output']=str(root/'elsewhere')
    # This is isolated invalid input, not a mutation of production evidence.
    path.parent.mkdir(parents=True,exist_ok=True)
    import json
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):validate_gpu_job(path,root,p,selected)


def test_deferred_resource_does_not_mean_finished(tmp_path):
    job=tmp_path/'job.json';write(job,{'fixture':'not_submitted'})
    assert unfinished_gpu_jobs([job])==[job]
    write(job.parent/'terminal.json',{'status':'completed','retained':0})
    assert unfinished_gpu_jobs([job])==[]


def test_population_requires_exact_independent_gate(tmp_path):
    root,p,sid,path,value=bound_job(tmp_path)
    with pytest.raises(FileNotFoundError):selected_paths(root,p,'population')
    review=root/'review.json';write(review,{'decision':'continue'})
    write(root/'continuation_gate.json',{'continue_population':True,
        'protocol':binding(root/'protocol.json'),'independent_review':binding(review)})
    assert selected_paths(root,p,'population')==[Path(p['job_paths'][0])]


def test_CPU_and_GPU_share_one_deadline(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        values=list(pool.map(lambda _:deadline(tmp_path,{'maximum_wall_seconds':60}),range(16)))
    assert len(set(values))==1
    assert values[0]==read(tmp_path/'START.json')['deadline_unix']
