"""No native calls: protect finite quotas and attempted identities on resume."""
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1]/'human_to_robot_pipeline/resume.py'
spec = importlib.util.spec_from_file_location('human_sim_resume', SOURCE)
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


def test_partial_native_or_binding_only_is_never_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(resume, 'GEN', tmp_path)
    jobs = [dict(id=str(i), output=str(tmp_path/'native'/str(i))) for i in range(4)]
    Path(jobs[0]['output']).mkdir(parents=True)
    (tmp_path/'scale192_bindings_v1'/'1').mkdir(parents=True)
    Path(jobs[2]['output']).mkdir(parents=True)
    (tmp_path/'scale192_bindings_v1'/'2').mkdir(parents=True)
    remaining, attempted = resume.remaining_jobs(dict(jobs=jobs))
    assert [j['id'] for j in remaining] == ['3']
    assert [j['id'] for j in attempted] == ['0', '1', '2']


def test_first_batch_covers_both_assets_and_arms_without_duplicates():
    jobs = [dict(id=f'{asset}_{arm}_{i}',asset=dict(id=asset),acting_arm_index=arm)
            for asset in ['bowl1','bowl2'] for arm in [0,1] for i in range(4)]
    result = resume.balanced_order(jobs)
    assert len(result)==len({j['id'] for j in result})==16
    assert {(j['asset']['id'],j['acting_arm_index']) for j in result[:4]} == {
        ('bowl1',0),('bowl1',1),('bowl2',0),('bowl2',1)}


@pytest.mark.parametrize('change', [dict(compute_owners=['another process']),
                                  dict(memory_MiB=2048),dict(utilization=80)])
def test_foreign_compute_is_not_treated_as_available(change):
    snapshot=dict(compute_owners=[],memory_MiB=21,utilization=0)
    snapshot.update(change)
    with pytest.raises(RuntimeError):
        resume.assert_idle(snapshot)
