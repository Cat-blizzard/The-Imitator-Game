import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import trimesh

from resimpair_assets import raw_container_population_v2 as population
from resimpair_pickplace.containers import cavity


def tray(depth=.04):
    pieces = [trimesh.creation.box([.30,.30,.01])]
    for axis in [0,1]:
        for side in [-1,1]:
            extent = [.30,.30,depth]; extent[axis] = .01
            wall = trimesh.creation.box(extent)
            offset = [0,0,depth/2+.005]; offset[axis] = side*.145
            wall.apply_translation(offset); pieces.append(wall)
    return trimesh.util.concatenate(pieces)


def fixture(tmp_path, suffix='.glb'):
    mesh = tray()
    uv = (mesh.vertices[:, :2] + .15) / .30
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.new('RGBA',(4,4),(37,104,211,255)))
    scene = trimesh.Scene(mesh)
    transform = np.eye(4); transform[:3,3] = [.1,.2,.3]
    scene.apply_transform(transform)
    source = tmp_path / ('source' + suffix)
    if suffix == '.gltf':
        exports = trimesh.exchange.gltf.export_gltf(scene)
        gltf_key = next(k for k in exports if k.endswith('.gltf'))
        for name, value in exports.items():
            path = source if name == gltf_key else tmp_path/name
            path.write_bytes(value)
    else: scene.export(source)
    return {'id':'objaverse:container_fixture', 'source':'objaverse', 'name':'Wide serving tray',
        'category':'container', 'partition':'development', 'path':str(source),
        'source_sha':hashlib.sha256(source.read_bytes()).hexdigest(),
        'record':json.dumps({'categories':['containers'],'license':'test fixture'})}


def prepare_fixture(tmp_path):
    row = fixture(tmp_path); database = tmp_path/'raw.sqlite'
    with sqlite3.connect(database) as connection:
        names = ','.join(f'{key} TEXT' for key in row)
        connection.execute(f'CREATE TABLE assets({names})')
        connection.execute('INSERT INTO assets VALUES('+','.join('?' for _ in row)+')',tuple(row.values()))
    selection = tmp_path/'selection.json'; selection.write_text(json.dumps({'source_ids':[row['id']]}))
    metadata = tmp_path/'metadata.json'; metadata.write_text('{"matches":{}}')
    root = tmp_path/'batch'
    with patch.object(population,'sufficient_disk',return_value=100*2**30):
        population.prepare(root,source_selection=selection,database=database,metadata=metadata)
    return root,row


def test_diagnostic_trace_preserves_original_cavity_and_restores_trace():
    mesh = tray(); expected = cavity(mesh)
    previous = sys.gettrace()
    result, diagnostics = population.cavity_with_diagnostics(mesh)
    assert result == expected
    assert len(diagnostics['center_attempts']) == 5
    assert diagnostics['branch_counts'] == {'retained_cavity_candidate':5}
    assert sys.gettrace() is previous


@pytest.mark.parametrize('mesh,expected',[(trimesh.creation.box([.3,.3,.004]),'depth_below_8mm'),
                                        (tray(depth=.2),'depth_above_150mm')])
def test_original_failure_is_split_by_actual_executed_depth_branch(mesh,expected):
    with pytest.raises(ValueError,match='no_supported_open_cavity'): cavity(mesh)
    result, diagnostics = population.cavity_with_diagnostics(mesh)
    assert result is None and diagnostics['retained_error'] == 'no_supported_open_cavity'
    assert any(expected in row.get('failed_conditions',[]) for row in diagnostics['center_attempts'])


def test_GLTF_local_resources_are_hash_bound_and_nodes_preserved(tmp_path):
    row = fixture(tmp_path,'.gltf'); deps = population.raw_dependencies(row)
    assert deps['resolver']
    scene = trimesh.load(row['path'],force='scene',process=False,
        resolver={k:population.resolve(v).read_bytes() for k,v in deps['resolver'].items()})
    assert np.allclose(scene.to_geometry().bounds.mean(axis=0)[:2],[.1,.2])
    assert population.material_summary(scene)[0]['textures']
    target = population.resolve(next(iter(deps['resolver'].values())))
    target.write_bytes(b'changed')
    with pytest.raises(ValueError): population.resolve(next(iter(deps['resolver'].values())))


@pytest.mark.parametrize('uri',['https://example.invalid/texture.png','../elsewhere.bin'])
def test_GLTF_network_or_parent_resources_are_rejected(tmp_path,uri):
    source=tmp_path/'asset.gltf';source.write_text(json.dumps({'buffers':[{'uri':uri}]}))
    with pytest.raises(ValueError,match='nonlocal_GLTF|outside_package'):
        population.raw_dependencies({'path':str(source),'source_sha':'','source':'polyhaven'})


def test_UV_values_and_all_supported_texture_channels_are_checked(tmp_path):
    row=fixture(tmp_path);scene=trimesh.load(row['path'],force='scene',process=False)
    other=scene.copy(); mesh=next(iter(other.geometry.values()));mesh.visual.uv[0,0]+=.01
    with pytest.raises(ValueError,match='UV_values_changed'):population.verify_materials(scene,other)
    other=scene.copy();material=next(iter(other.geometry.values())).visual.material
    material.normalTexture=Image.new('RGB',(4,4),'red')
    with pytest.raises(ValueError,match='texture_channel_pixels'):population.verify_materials(scene,other)


def test_frozen_budget_worker_scope_and_reservation_boundaries(tmp_path):
    root,_=prepare_fixture(tmp_path); protocol=population.validate(root)
    assert protocol['maximum_sources']==2859 and protocol['maximum_CPU_workers']==4
    assert protocol['per_source_timeout_seconds']==120 and protocol['maximum_wall_seconds_from_first_run']==14400
    assert protocol['retries']==protocol['API_calls']==protocol['GPU_jobs']==protocol['physics_steps']==0
    job=population.resolve(protocol['jobs'][0]); foreign=tmp_path/'foreign.json';foreign.write_text('{}')
    with pytest.raises(ValueError,match='not_in_frozen_source_scope'):population.worker(root,foreign)
    with pytest.raises(ValueError,match='requires_prior_registered_reservation'):population.worker(root,job)
    with pytest.raises(ValueError,match='recorded_qualification_gate'):population.run(root,phase='population')
    population.write(job.parent/'reservation.json',{'attempt':1})
    with patch.object(population,'sufficient_disk',return_value=100*2**30):
        with pytest.raises(RuntimeError,match='recover_before_admission'):population.run(root)
    assert not (job.parent/'worker_started.json').exists()


def test_real_static_mesh_compile_uses_shared_policy_original_surface_and_consumer(tmp_path):
    root,_=prepare_fixture(tmp_path); protocol=population.validate(root)
    job=population.resolve(protocol['jobs'][0]); result=population.compile_one(job)
    assert result['status']=='candidate_geometry_ready'
    descriptor=population.read(population.resolve(result['asset']))
    decision=population.read(job.parent/'SIZE_DECISION.json')
    assert .378 <= decision['target_longest_m'] <= .462
    assert decision['size_depends_on_gripper_or_container_fit'] is False
    assert descriptor['physical_status']=='unmeasured' and descriptor['native_policy'].startswith('static_triangle')
    assert result['support']['passed'] and result['physics_steps']==result['GPU_jobs']==0
    population.write(job.parent/'result.json',result)
    population.write(job.parent/'terminal.json',{'source_id':result['source_id'],'status':result['status'],
        'result':population.binding(job.parent/'result.json')})
    summary=population.publish(root,'qualification')
    assert summary['consumer_candidates']==1


@pytest.mark.parametrize('error',[KeyError('unexpected_code_field'),OSError('disk_write_failed')])
def test_unexpected_implementation_or_resource_failure_is_not_asset_rejection(tmp_path,error):
    root,_=prepare_fixture(tmp_path);protocol=population.validate(root)
    job=population.resolve(protocol['jobs'][0]);data=population.read(job)
    population.write(job.parent/'reservation.json',{'source_id':data['source_id'],
        'job':protocol['jobs'][0],'attempt':1})
    with patch.object(population,'compile_one',side_effect=error):
        result=population.worker(root,job)
    assert result['status']=='unknown_system_failure'
