import hashlib
import json
from pathlib import Path
import sqlite3
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import trimesh

from resimpair_assets import raw_metric_containers_v1 as intake


def raw_fixture(tmp_path):
    package = tmp_path / 'original'; meshes = package / 'meshes'; meshes.mkdir(parents=True)
    textures = package / 'materials/textures'; textures.mkdir(parents=True)
    source = meshes / 'model.obj'
    source.write_text('mtllib model.mtl\nv 0 0 0\nv 0.2 0 0\nv 0 0.1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl surface\nf 1/1 2/2 3/3\n')
    (meshes / 'model.mtl').write_text('newmtl surface\nKd 1 1 1\nmap_Kd texture.png\n')
    Image.new('RGB', (3,3), (37,104,211)).save(textures/'texture.png')
    return {'id':'google_scanned_objects_original_fuel:test_fixture',
            'source':'google_scanned_objects_original_fuel','name':'test_fixture','partition':'development',
            'path':str(source),'source_sha':hashlib.sha256(source.read_bytes()).hexdigest(), 'record':'{}'}


def prepare_fixture(tmp_path):
    row = raw_fixture(tmp_path)
    database = tmp_path/'raw.sqlite'
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE assets(id TEXT,source TEXT,name TEXT,partition TEXT,path TEXT,source_sha TEXT,record TEXT)')
        connection.execute('INSERT INTO assets VALUES(?,?,?,?,?,?,?)',tuple(row.values()))
    selection = tmp_path/'selection.json'; selection.write_text(json.dumps({'source_ids':[row['id']]}))
    root = tmp_path/'candidate'
    with patch.object(intake, 'sufficient_disk', return_value=100*2**30):
        intake.prepare(root,selection,database)
    return root, row


def test_package_relative_GSO_material_resolver_preserves_texture_and_original_scale(tmp_path):
    row = raw_fixture(tmp_path)
    before = Path(row['path']).read_bytes()
    deps = intake.raw_dependencies(row)
    resolver = {name:intake.resolve(ref).read_bytes() for name,ref in deps['resolver'].items()}
    scene = trimesh.load(row['path'],force='scene',process=False,resolver=resolver)
    assert np.allclose(scene.to_geometry().extents,[.2,.1,0])
    material = intake.texture_summary(scene)[0]
    assert material['texture'] and material['has_UV']
    assert material['texture']['shape'] == [3,3,4]
    assert Path(row['path']).read_bytes() == before
    assert deps['raw_files_modified'] is False


def test_protocol_freezes_bounded_resources_and_uses_direct_raw_source(tmp_path):
    root,_ = prepare_fixture(tmp_path)
    protocol = intake.validate(root)
    assert protocol['maximum_sources']==37 and protocol['maximum_CPU_workers']==2
    assert protocol['per_source_timeout_seconds']==120 and protocol['maximum_wall_seconds_from_first_run']==1200
    assert protocol['retries']==0 and protocol['source_scale_factor']==1.0
    assert protocol['API_calls']==protocol['GPU_jobs']==protocol['physics_steps']==0
    job = intake.read(intake.resolve(protocol['jobs'][0]))
    assert job['source_scale_factor']==1.0 and job['preflight_error'] is None
    assert 'native_descriptor' not in job


def test_unknown_reservation_cannot_be_resubmitted(tmp_path):
    root,_ = prepare_fixture(tmp_path)
    job = intake.resolve(intake.read(root/'PROTOCOL.json')['jobs'][0])
    intake.write(job.parent/'reservation.json',{'attempt':1})
    with patch.object(intake, 'sufficient_disk', return_value=100*2**30):
        with pytest.raises(RuntimeError,match='recover_before_admission'):
            intake.run(root)
    assert not (job.parent/'worker_started.json').exists()
    assert intake.status(root)['counts']=={'unknown_reserved_no_terminal':1}


def test_frozen_scope_rejects_foreign_worker_and_protocol_reuse(tmp_path):
    root,row = prepare_fixture(tmp_path)
    foreign=tmp_path/'foreign.json';foreign.write_text('{}')
    with pytest.raises(ValueError,match='not_in_frozen_source_scope'):
        intake.worker(root,foreign)
    registered = intake.resolve(intake.read(root/'PROTOCOL.json')['jobs'][0])
    with pytest.raises(ValueError,match='requires_prior_registered_reservation'):
        intake.worker(root,registered)
    intake.write(registered.parent/'reservation.json',{'source_id':row['id'],'job':intake.binding(registered),'attempt':2})
    with pytest.raises(ValueError,match='reservation_does_not_match'):
        intake.worker(root,registered)
    with pytest.raises(FileExistsError,match='preserve_existing'):
        intake.prepare(root,tmp_path/'selection.json',tmp_path/'raw.sqlite')


def test_source_hash_and_missing_material_fail_instead_of_gray_fallback(tmp_path):
    row=raw_fixture(tmp_path)
    bad={**row,'source_sha':'0'*64}
    with pytest.raises(ValueError,match='raw_source_hash_changed'):
        intake.raw_dependencies(bad)
    (tmp_path/'original/materials/textures/texture.png').unlink()
    with pytest.raises(ValueError,match='texture_missing'):
        intake.raw_dependencies(row)


def test_low_disk_stops_intake_before_source_submission():
    with patch.object(intake.shutil, 'disk_usage', return_value=SimpleNamespace(free=49*2**30)):
        with pytest.raises(RuntimeError,match='minimum_50GiB'):
            intake.sufficient_disk('/unused')
