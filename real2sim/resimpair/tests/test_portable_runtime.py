"""Relocation, evidence integrity and budget preservation; no paid operations."""
import json
from pathlib import Path
from unittest.mock import patch
import pytest

from resimpair_runtime import ROOT, evidence_path, prepare_asset_catalog
from corl2026_workshop.experiments.autonomous_pipeline_v4.core import activate, checked


def test_source_activation_uses_only_repository():
    # Imitator 2.0: the frozen tree's code is vendored (and byte-checked); its images, logs and results are not
    result=activate()
    assert Path(result['source_root']) == ROOT/'data_generation_methods'
    assert result['verified_file_count']+result['not_vendored_non_code_files']==619
    assert result['verified_file_count']>=199


def test_relocated_evidence_preserves_original_bytes_and_bindings():
    config=json.loads((ROOT/'configs/providers.json').read_text())
    bound=config['gemini']['review_schema_qualification']
    path=checked(bound)
    assert path.is_relative_to(ROOT)
    assert str(path)!=bound['path']
    with pytest.raises(ValueError,match='bytes changed'):
        checked(dict(bound,sha256='0'*64))


def test_asset_geometry_is_local_and_hash_verified():
    catalog=json.loads(prepare_asset_catalog().read_text())
    assert len(catalog['assets'])>=12
    for row in catalog['assets']:
        for name,sha in row['files'].items():
            p=checked({'path':name,'sha256':sha})
            assert p.is_relative_to(ROOT/'assets')


def test_historical_counts_are_not_zeroed(tmp_path):
    from resimpair import historical_lineage
    # Test a clean clone, never inspect or mutate a live paid-call ledger.
    with patch('resimpair.REGISTRY', tmp_path/'registry'):
        counts,evidence=historical_lineage()
    assert counts['image']>=29 and counts['video']>=22 and counts['review']>=50
    assert evidence


def test_new_campaign_blocks_unresolved_previous_submission(tmp_path):
    from resimpair import historical_lineage
    registry=tmp_path/'registry'; registry.mkdir()
    ledger=tmp_path/'ledger'; stage=ledger/'stages/image_test'; stage.mkdir(parents=True)
    with patch('resimpair.REGISTRY', registry):
        baseline,_=historical_lineage()
    spec={'ledger_path':str(ledger)}
    (registry/'prior.json').write_text(json.dumps(spec))
    (ledger/'budget.json').write_text(json.dumps(spec))
    (stage/'call_started.json').write_text(json.dumps({'kind':'image'}))
    with patch('resimpair.REGISTRY', registry):
        with pytest.raises(ValueError,match='prior_submission_unknown'):
            historical_lineage()
        (stage/'result.json').write_text('{}')
        counts,_=historical_lineage()
    assert counts == dict(baseline,image=baseline['image']+1)


def test_validation_history_survives_clone_without_double_counting(tmp_path):
    from resimpair import historical_lineage
    from corl2026_workshop.experiments.autonomous_pipeline_v4.core import bind
    provenance=tmp_path/'provenance'; provenance.mkdir()
    original=provenance/'lineage_snapshot.json'
    original.write_text(json.dumps({'lineage_started_calls':{'image':29,'video':22,'review':50}}))
    registry=tmp_path/'runtime/campaign_registry'; registry.mkdir(parents=True)
    ledger=tmp_path/'runs/probe/ledger'; stage=ledger/'stages/image_probe'; stage.mkdir(parents=True)
    marker=stage/'call_started.json'; marker.write_text(json.dumps({'kind':'image'}))
    (stage/'result.json').write_text('{}')
    (provenance/'portable_validation_lineage.json').write_text(json.dumps({
        'base_snapshot_sha256':bind(original)['sha256'],'unresolved_submissions':0,
        'additional_started_calls':{'image':1,'video':0,'review':0},
        'campaigns':[{'campaign_id':'probe','markers':[{'stage':stage.name,'kind':'image',
             'sha256':bind(marker)['sha256'],'resolved':True}]}]}))
    with patch('resimpair.ROOT',tmp_path),patch('resimpair.REGISTRY',registry):
        clean,_=historical_lineage()
        spec={'campaign_id':'probe','ledger_path':str(ledger)}
        (registry/'probe.json').write_text(json.dumps(spec)); (ledger/'budget.json').write_text(json.dumps(spec))
        local,_=historical_lineage()
        assert local == clean == {'image':30,'video':22,'review':50}
        marker.write_text(json.dumps({'kind':'review'}))
        with pytest.raises(ValueError,match='carried_call_marker_changed'):
            historical_lineage()


def test_gpu_guard_rejects_existing_owner():
    from corl2026_workshop.experiments.autonomous_pipeline_v4.resources import check_idle
    with pytest.raises(RuntimeError,match='occupied'):
        check_idle({'compute_owners':['existing job'],'memory_MiB':1,'utilization':0})


def test_credentials_are_never_in_the_repository():
    # Imitator 2.0 policy (reverses ReSimPair's): keys come from the environment; private/ is git-ignored
    import subprocess
    tracked=subprocess.run(['git','ls-files','private'],cwd=ROOT,capture_output=True,text=True).stdout.split()
    assert tracked==[]
    ignored=subprocess.run(['git','check-ignore','-q','private/.env.local'],cwd=ROOT).returncode
    assert ignored==0
