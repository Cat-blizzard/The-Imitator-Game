"""Read-only task target bindings for complete action replay.

The two semantic telemetry names map through explicit saved target0/target1
geometry roles to native actor names and H5 state keys. Geometry file bytes,
vertices and initial poses are checked; positions never select an identity.
No H5 key, source state, restore policy or physical acceptance gate is changed.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _name(value: Any) -> str:
    _require(isinstance(value, str) and bool(value) and value.strip() == value,
             'Expected an explicit nonempty identity')
    return value


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    _require(array.shape == shape and array.dtype.kind == 'f'
             and bool(np.isfinite(array).all()), f'Invalid finite floating {label} {shape}')
    return array


@dataclass(frozen=True)
class TargetBinding:
    side: int
    semantic_name: str
    actor_name: str
    state_name: str


def load_target_bindings(directory: Path, report: Mapping[str, Any],
                         actor_arrays: Mapping[str, np.ndarray]
                         ) -> tuple[tuple[TargetBinding, TargetBinding], dict[str, Any]]:
    """Bind two roles to native [T+1,13] states via hash-checked saved geometry.

    State layout is world XYZ/WXYZ, world linear velocity, angular velocity.
    The recorded role pose must equal native H5 frame0 after its existing dtype
    conversion. Full later target states are checked separately at frame i+1.
    """
    directory = Path(directory)
    names = report.get('target_names')
    _require(isinstance(names, list) and len(names) == 2, 'Exactly two semantic target names required')
    names = [_name(name) for name in names]
    _require(len(set(names)) == 2, 'Semantic target names must be unique')
    path = directory/'task-geometry.json'
    manifest = json.loads(path.read_text())
    # Canonical serialization also rejects nonfinite JSON, without rewriting it.
    encoded = json.dumps(manifest, sort_keys=True, allow_nan=False)
    _require(encoded == json.dumps(report.get('task_geometry'), sort_keys=True, allow_nan=False),
             'Geometry JSON and report manifest differ')
    _require(isinstance(manifest, dict) and isinstance(manifest.get('actors'), dict),
             'Missing explicit task geometry actor roles')
    digest = manifest.get('geometry_npz_sha256')
    _require(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest) is not None
             and _sha(directory/'task-geometry.npz') == digest, 'Geometry NPZ file SHA differs')
    roles = manifest['actors']
    _require({key for key in roles if re.fullmatch(r'target[0-9]+', key)} == {'target0', 'target1'},
             'Exactly target0 and target1 roles required')
    bindings: list[TargetBinding] = []
    geometry_hashes: dict[str, str] = {}
    frames: int | None = None
    with np.load(directory/'task-geometry.npz', allow_pickle=False) as geometry:
        for side, semantic in enumerate(names):
            role = roles[f'target{side}']
            _require(isinstance(role, dict), 'Target geometry role must be an object')
            binding = TargetBinding(side, semantic, _name(role.get('actor_name')), _name(role.get('state_name')))
            _require(binding.state_name in actor_arrays, 'Explicit target state key absent from native H5')
            states = np.asarray(actor_arrays[binding.state_name])
            _require(states.ndim == 2 and states.shape[0] >= 2, 'Target states need T+1 frames')
            _array(states, (len(states), 13), 'target states')
            _require(frames is None or frames == len(states), 'Target T+1 lengths differ')
            frames = len(states)
            _require(bool(np.all(np.abs(np.linalg.norm(states[:, 3:7], axis=1)-1.) <= 1e-3)),
                     'Nonunit native target state quaternion')
            vk, pk = _name(role.get('local_vertices_array')), _name(role.get('world_pose_array'))
            _require(vk in geometry and pk in geometry, 'Target geometry arrays missing')
            vertices, pose = geometry[vk], geometry[pk]
            _require(vertices.ndim == 2 and len(vertices) > 0, 'Nonempty target solid vertices required')
            _array(vertices, (len(vertices), 3), 'target vertices')
            _array(pose, (7,), 'target world pose')
            _require(role.get('shape') == list(vertices.shape) and role.get('vertex_count') == len(vertices),
                     'Target vertex shape/count manifest differs')
            vh = hashlib.sha256(np.ascontiguousarray(vertices, dtype=np.float64).tobytes(order='C')).hexdigest()
            _require(role.get('local_vertices_sha256') == vh, 'Target vertices SHA differs')
            _require(bool(abs(np.linalg.norm(pose[3:])-1.) <= 1e-5), 'Nonunit geometry quaternion')
            _require(np.array_equal(pose.astype(states.dtype), states[0, :7]),
                     'Explicit target role pose differs from native H5 frame0')
            geometry_hashes[f'target{side}'] = vh
            bindings.append(binding)
    _require(len({b.actor_name for b in bindings}) == len({b.state_name for b in bindings}) == 2,
             'Target actor/state bindings must be one-to-one')
    result = (bindings[0], bindings[1])
    return result, {'bindings': [asdict(b) for b in result], 'state_frames': frames,
        'task_geometry_json_sha256': _sha(path), 'geometry_npz_sha256': digest,
        'geometry_json_equals_report_manifest': True, 'target_vertex_sha256': geometry_hashes,
        'initial_native_poses_exact': True, 'mapping_source': 'explicit saved target0/target1 roles',
        'native_state_keys_preserved': True, 'position_based_identity_inference': False}


def target_state(state: Mapping[str, Any], binding: TargetBinding) -> np.ndarray:
    """Read the native actor state[13]; never rename a stored physical state."""
    actors = state.get('actors', {})
    _require(binding.state_name in actors, 'Bound native actor state key missing')
    return _array(actors[binding.state_name], (13,), 'bound target state')


def validate_target_telemetry(bindings: Sequence[TargetBinding],
                              states: Sequence[Mapping[str, Any]],
                              ticks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require semantic tick i values to equal all13 native H5 frame i+1 values."""
    _require(len(bindings) == 2 and [b.side for b in bindings] == [0, 1], 'Ordered target pair required')
    _require(len(states) == len(ticks)+1 and len(ticks) > 0, 'Expected T ticks and T+1 states')
    checked = 0
    for index, tick in enumerate(ticks):
        _require(tick.get('environment_step') == index+1, 'Nonsequential target telemetry')
        objects = tick.get('objects')
        _require(isinstance(objects, list) and len(objects) == 2, 'Exactly two telemetry targets required')
        for binding, obj in zip(bindings, objects):
            _require(obj.get('actor') == binding.semantic_name, 'Telemetry semantic target pairing differs')
            measured = np.r_[_array(obj.get('pose'), (7,), 'pose'),
                             _array(obj.get('linear_velocity_m_s'), (3,), 'linear velocity'),
                             _array(obj.get('angular_velocity_rad_s'), (3,), 'angular velocity')]
            expected = target_state(states[index+1], binding)
            _require(np.array_equal(measured.astype(expected.dtype), expected),
                     'Target telemetry differs from native H5 frame i+1')
            checked += 1
    return {'target_state_checks': checked, 'components_per_check': 13,
            'frame_alignment': 'tick i -> native H5 state i+1', 'exact_after_existing_dtype_cast': True}


def validate_live_bindings(bindings: Sequence[TargetBinding], semantic_names: Sequence[str],
                           targets: Sequence[Any], scene_actors: Mapping[str, Any],
                           state: Mapping[str, Any]) -> dict[str, Any]:
    """Validate actual object identity and native registry/state key after reset."""
    _require(len(bindings) == len(targets) == 2 and len({id(actor) for actor in targets}) == 2,
             'Two distinct live physical targets required')
    _require(list(semantic_names) == [b.semantic_name for b in bindings], 'Live semantic target names differ')
    for binding, actor in zip(bindings, targets):
        _require(getattr(actor, 'name', None) == binding.actor_name, 'Live native actor name differs')
        keys = [key for key, value in scene_actors.items() if value is actor]
        _require(keys == [binding.state_name], 'Live target native state registry identity differs')
        target_state(state, binding)
    return {'live_actor_names_exact': True, 'native_registry_object_identity_exact': True,
            'native_state_keys_exact': True, 'physical_state_renamed': False}


def preflight(directory: Path) -> dict[str, Any]:
    """Read an existing recording only; no simulator or runtime module import."""
    import h5py
    directory = directory.resolve()
    report = json.loads((directory/'report.json').read_text())
    metadata = json.loads((directory/'trajectory.json').read_text())
    ticks = [json.loads(line) for line in (directory/'ticks.jsonl').read_text().splitlines() if line.strip()]
    files = ('report.json', 'trajectory.json', 'trajectory.h5', 'ticks.jsonl',
             'task-geometry.json', 'task-geometry.npz')
    before = {name: _sha(directory/name) for name in files}
    with h5py.File(directory/'trajectory.h5', 'r') as handle:
        episode = metadata['episodes'][0]['episode_id']
        arrays = {name: dataset[:] for name, dataset in handle[f'traj_{episode}/env_states/actors'].items()}
    bindings, receipt = load_target_bindings(directory, report, arrays)
    states = [{'actors': {name: value[index] for name, value in arrays.items()}}
              for index in range(len(ticks)+1)]
    receipt.update(validate_target_telemetry(bindings, states, ticks))
    unchanged = all(_sha(directory/name) == digest for name, digest in before.items())
    _require(unchanged, 'Source evidence changed during alias preflight')
    return {'status': 'passed', 'input_dir': str(directory), 'source_sha256': before,
            'receipt': receipt, 'source_evidence_unchanged': True, 'environment_steps': 0,
            'simulation_created': False, 'native_physics_replay': 'NOT_RUN',
            'helper_sha256': _sha(Path(__file__))}


def self_test() -> dict[str, Any]:
    """Synthetic tests cover explicit aliases, temporal equality and bad identities."""
    rejected: list[str] = []
    def reject(label: str, call: Any) -> None:
        try:
            call()
        except (ValueError, KeyError, TypeError):
            rejected.append(label)
        else:
            raise AssertionError(f'Expected rejection: {label}')

    with tempfile.TemporaryDirectory(prefix='tig-task-replay-alias-cpu-') as temporary:
        path = Path(temporary)
        arrays, geometry, roles = {}, {}, {}
        for side in range(2):
            native = f'native-bottle-{side}'
            pose = np.array([float(side), 0., .02, 1., 0., 0., 0.])
            vertices = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.], [0., 0., .01]])
            value = np.tile(np.r_[pose, np.zeros(6)], (3, 1))
            value[1, 0] += .001; value[2, 0] += .002
            arrays[native] = value
            geometry[f'target{side}_vertices'] = vertices
            geometry[f'target{side}_pose'] = pose
            roles[f'target{side}'] = {'actor_name': native, 'state_name': native,
                'local_vertices_array': f'target{side}_vertices', 'world_pose_array': f'target{side}_pose',
                'vertex_count': 4, 'shape': [4, 3],
                'local_vertices_sha256': hashlib.sha256(vertices.tobytes()).hexdigest()}
        np.savez(path/'task-geometry.npz', **geometry)
        manifest = {'actors': roles, 'geometry_npz_sha256': _sha(path/'task-geometry.npz')}
        def write(value: dict[str, Any]) -> dict[str, Any]:
            (path/'task-geometry.json').write_text(json.dumps(value, allow_nan=False))
            return {'target_names': ['pill_a', 'pill_b'], 'task_geometry': copy.deepcopy(value)}
        report = write(manifest)
        original = {key: value.copy() for key, value in arrays.items()}
        bindings, receipt = load_target_bindings(path, report, arrays)
        assert receipt['native_state_keys_preserved'] and bindings[0].state_name == 'native-bottle-0'
        states = [{'actors': {name: value[index].copy() for name, value in arrays.items()}} for index in range(3)]
        ticks = []
        for index in range(2):
            objects = []
            for binding in bindings:
                value = states[index+1]['actors'][binding.state_name]
                objects.append({'actor': binding.semantic_name, 'pose': value[:7].tolist(),
                    'linear_velocity_m_s': value[7:10].tolist(), 'angular_velocity_rad_s': value[10:].tolist()})
            ticks.append({'environment_step': index+1, 'objects': objects})
        assert validate_target_telemetry(bindings, states, ticks)['target_state_checks'] == 4
        live = [SimpleNamespace(name=b.actor_name) for b in bindings]
        registry = {b.state_name: actor for b, actor in zip(bindings, live)}
        assert validate_live_bindings(bindings, ['pill_a', 'pill_b'], live, registry, states[0])['native_state_keys_exact']
        identity_report = dict(report, target_names=[b.actor_name for b in bindings])
        identity, _ = load_target_bindings(path, identity_report, arrays)
        assert all(b.semantic_name == b.state_name for b in identity)
        for label, mutate in (
            ('duplicate_actor', lambda m: m['actors']['target1'].update(actor_name='native-bottle-0')),
            ('duplicate_state', lambda m: m['actors']['target1'].update(state_name='native-bottle-0')),
            ('missing_native_state', lambda m: m['actors']['target1'].update(state_name='pill_b')),
            ('vertices_digest', lambda m: m['actors']['target0'].update(local_vertices_sha256='0'*64)),
            ('npz_digest', lambda m: m.update(geometry_npz_sha256='0'*64)),
            ('extra_target_role', lambda m: m['actors'].update(target2=copy.deepcopy(m['actors']['target0'])))):
            bad = copy.deepcopy(manifest); mutate(bad)
            reject(label, lambda bad=bad: load_target_bindings(path, write(bad), arrays))
        report = write(manifest)
        reject('json_report_mismatch', lambda: load_target_bindings(path, dict(report, task_geometry={}), arrays))
        bad_arrays = {key: value.copy() for key, value in arrays.items()}
        bad_arrays['native-bottle-0'][0, 0] += .001
        reject('frame0_pose', lambda: load_target_bindings(path, report, bad_arrays))
        bad_arrays['native-bottle-0'] = arrays['native-bottle-0'][:, :12]
        reject('state_width', lambda: load_target_bindings(path, report, bad_arrays))
        bad_arrays['native-bottle-0'] = arrays['native-bottle-0'].copy()
        bad_arrays['native-bottle-0'][1, 7] = np.nan
        reject('nonfinite_state', lambda: load_target_bindings(path, report, bad_arrays))
        altered = copy.deepcopy(ticks); altered[0]['objects'][0]['actor'] = 'native-bottle-0'
        reject('telemetry_semantics', lambda: validate_target_telemetry(bindings, states, altered))
        altered = copy.deepcopy(ticks); altered[0]['objects'][0]['linear_velocity_m_s'][0] = .000001
        reject('full13_telemetry', lambda: validate_target_telemetry(bindings, states, altered))
        reject('frame_shift', lambda: validate_target_telemetry(bindings, states[::-1], ticks))
        reject('live_actor_swapped', lambda: validate_live_bindings(bindings, ['pill_a', 'pill_b'], live[::-1], registry, states[0]))
        reject('live_registry_alias', lambda: validate_live_bindings(bindings, ['pill_a', 'pill_b'], live,
            {'pill_a': live[0], 'pill_b': live[1]}, states[0]))
        reject('live_registry_duplicate', lambda: validate_live_bindings(bindings, ['pill_a', 'pill_b'], live,
            dict(registry, duplicate=live[0]), states[0]))
        assert set(arrays) == set(original) and all(np.array_equal(arrays[key], value) for key, value in original.items())
    return {'status': 'passed', 'actor_state_shape': [3, 13], 'vertex_shape': [4, 3],
        'explicit_alias_and_identity_mapping_passed': True, 'target_state_checks': 4,
        'native_arrays_unchanged': True, 'rejected_inputs': rejected, 'environment_steps': 0,
        'simulation_created': False, 'helper_sha256': _sha(Path(__file__))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--output-json', type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.input_dir is not None:
        result = preflight(args.input_dir)
    else:
        parser.error('Provide --self-test or an existing --input-dir')
    text = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n'
    if args.output_json is not None:
        with args.output_json.open('x') as handle:
            handle.write(text)
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
