"""TIG client for the isolated, official cuRobo planner.

Only JSON crosses the Python/CUDA ABI boundary. Poses use XYZ/WXYZ, joint
positions radians, and timestamps seconds. No simulator or executor is owned
by this client. Full paired collision validation remains the caller's duty.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import selectors
import subprocess
import time
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from .trajectory_schedule import ArmTrajectory

ROOT = Path(__file__).resolve().parents[7]


def urdf_tcp_pose(urdf: Path, joints: dict[str, float], *,
                  base_link: str = 'panda_link0', tcp_link: str = 'panda_hand_tcp') -> np.ndarray:
    """Independent CPU URDF FK, base XYZ/WXYZ[7], without touching physics."""
    from scipy.spatial.transform import Rotation
    tree = ET.parse(urdf).getroot()
    by_child = {node.find('child').attrib['link']: node for node in tree.findall('joint')}
    chain, current = [], tcp_link
    while current != base_link:
        node = by_child[current]
        chain.append(node)
        current = node.find('parent').attrib['link']
    transform = np.eye(4)
    for node in reversed(chain):
        origin = node.find('origin')
        step = np.eye(4)
        if origin is not None:
            step[:3, 3] = np.fromstring(origin.attrib.get('xyz', '0 0 0'), sep=' ')
            rpy = np.fromstring(origin.attrib.get('rpy', '0 0 0'), sep=' ')
            step[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
        transform = transform @ step
        if node.attrib['type'] != 'fixed':
            angle = float(joints[node.attrib['name']])
            axis_node = node.find('axis')
            axis = np.fromstring('1 0 0' if axis_node is None else axis_node.attrib['xyz'], sep=' ')
            if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isfinite(angle):
                raise ValueError('Invalid URDF axis/joint value')
            axis = axis / np.linalg.norm(axis)
            step = np.eye(4)
            if node.attrib['type'] in ('revolute', 'continuous'):
                step[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            elif node.attrib['type'] == 'prismatic':
                step[:3, 3] = axis * angle
            else:
                raise ValueError('Unsupported URDF joint type')
            transform = transform @ step
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return np.r_[transform[:3, 3], quaternion[3], quaternion[:3]]


def pose_error(actual: Any, expected: Any) -> dict[str, float]:
    actual = checked_array(actual, (7,), 'actual pose')
    expected = checked_array(expected, (7,), 'expected pose')
    for pose in (actual, expected):
        if not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-5, rtol=0):
            raise ValueError('Pose quaternion must be normalized')
    dot = abs(float(np.dot(actual[3:], expected[3:]))) / (np.linalg.norm(actual[3:]) * np.linalg.norm(expected[3:]))
    return {'position_m': float(np.linalg.norm(actual[:3] - expected[:3])),
            'rotation_rad': float(2 * np.arccos(np.clip(dot, -1., 1.)))}


def checked_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name}: expected finite {shape}, got {result.shape}")
    return result


def trajectory_from_result(result: dict[str, Any], *, action_key: str,
                           joint_names: tuple[str, ...], start: np.ndarray,
                           velocity_limit: float = .9,
                           acceleration_limit: float = .9) -> tuple[ArmTrajectory, dict[str, Any]]:
    """Validate raw cuRobo arrays[N,7], preserve time, reorder only by names."""
    if result.get('success') is not True:
        raise RuntimeError(f"cuRobo planning failed: {result.get('status')}")
    names = tuple(result['joint_names'])
    if len(names) != 7 or len(set(names)) != 7 or set(names) != set(joint_names):
        raise ValueError('cuRobo/controller joint identities differ')
    order = [names.index(name) for name in joint_names]
    times = np.asarray(result['times_s'], dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('cuRobo must return increasing actual timestamps from zero')
    arrays = {name: checked_array(result[name], (len(times), 7), name)[:, order]
              for name in ('position', 'velocity', 'acceleration', 'jerk')}
    start = checked_array(start, (7,), 'measured start')
    delta = float(np.max(np.abs(arrays['position'][0] - start)))
    if delta > 1e-6:
        raise ValueError(f'cuRobo trajectory does not start at measured state: {delta}')
    peaks = {name: float(np.max(np.abs(value))) for name, value in arrays.items()
             if name != 'position'}
    # Float32 native arithmetic gets a numerical tolerance, not a changed limit.
    if peaks['velocity'] > velocity_limit + 1e-5 or peaks['acceleration'] > acceleration_limit + 1e-5:
        raise ValueError(f'cuRobo derivatives exceed the shared limits: {peaks}')
    trajectory = ArmTrajectory(action_key, joint_names, times, arrays['position'], 1.)
    return trajectory, {'joint_reorder': order, 'start_error_rad': delta,
                        'raw_derivative_peaks': peaks, 'retiming_applied': False}


class CuroboProcess:
    """One task-owned planner process; native diagnostics go to a separate log."""
    def __init__(self, gpu_uuid: str, log_path: Path, *, timeout_s: float = 180.,
                 python: Path | None = None) -> None:
        python = python or ROOT / 'infrastructure/envs/curobo-tig/bin/python'
        worker = ROOT / 'infrastructure/curobo_tig_worker.py'
        if not python.exists() or not worker.exists():
            raise FileNotFoundError('Independent cuRobo environment/worker is not installed')
        if not gpu_uuid.startswith('GPU-') or timeout_s <= 0:
            raise ValueError('Explicit GPU UUID and positive timeout required')
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        environment.update(CUDA_VISIBLE_DEVICES=gpu_uuid, PYTHONNOUSERSITE='1',
                           PYTHONDONTWRITEBYTECODE='1')
        cache = ROOT / 'infrastructure/tmp/curobo-v2-runtime-cache'
        for variable, subdirectory in [('CUDA_CACHE_PATH', 'cuda'),
                                       ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                                       ('WARP_CACHE_PATH', 'warp')]:
            directory = cache / subdirectory
            directory.mkdir(parents=True, exist_ok=True)
            environment[variable] = str(directory)
        self._log = log_path.open('x')
        self.process = subprocess.Popen([str(python), '-B', '-u', str(worker), '--serve'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            env=environment, cwd=ROOT, text=True, bufsize=1)
        self.timeout_s, self.sequence = timeout_s, 0
        self.failed = False
        self.receipts: list[dict[str, Any]] = []

    def request(self, op: str, **payload: Any) -> dict[str, Any]:
        self.sequence += 1
        data = json.dumps({'id': self.sequence, 'op': op, **payload}, allow_nan=False)
        before = time.monotonic()
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(data + '\n')
        self.process.stdin.flush()
        # First-use compilation belongs to initialization, outside solve latency.
        timeout = max(self.timeout_s, 600.) if op == 'configure' else self.timeout_s
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                self.failed = True
                self.receipts.append({'id': self.sequence, 'op': op, 'ok': False,
                    'wall_s': time.monotonic() - before, 'reason': 'process_timeout', 'timeout_s': timeout})
                raise TimeoutError(f'cuRobo {op} exceeded {timeout}s; see worker log')
            line = self.process.stdout.readline()
        if not line:
            self.failed = True
            raise RuntimeError(f'cuRobo worker exited during {op}: {self.process.poll()}')
        response = json.loads(line)
        if response.get('id') != self.sequence:
            raise RuntimeError('cuRobo protocol response ID mismatch')
        self.receipts.append({'id': self.sequence, 'op': op,
                              'wall_s': time.monotonic() - before, 'ok': response.get('ok')})
        if response.get('ok') is not True:
            raise RuntimeError(f"cuRobo {op}: {response.get('error')}")
        return response['result']

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                if self.failed:
                    raise RuntimeError('Terminate unresponsive task-owned planner')
                self.request('shutdown')
                self.process.wait(timeout=10)
            except Exception:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self._log.close()


def self_test() -> dict[str, Any]:
    names = tuple(f'j{i}' for i in range(7))
    q = np.arange(7, dtype=float) * .01
    result = {'success': True, 'joint_names': list(reversed(names)), 'times_s': [0., .1],
              'position': [list(reversed(q)), list(reversed(q + .01))],
              'velocity': [[.1]*7]*2, 'acceleration': [[0.]*7]*2, 'jerk': [[0.]*7]*2}
    trajectory, audit = trajectory_from_result(result, action_key='left', joint_names=names, start=q)
    assert np.array_equal(trajectory.positions_rad[0], q) and audit['retiming_applied'] is False
    rejected = 0
    for field, value in [('times_s', [0., 0.]), ('joint_names', ['x']*7),
                         ('position', [[float('nan')]*7]*2), ('velocity', [[1.]*7]*2),
                         ('acceleration', [[1.]*7]*2), ('success', False)]:
        try:
            trajectory_from_result({**result, field: value}, action_key='left', joint_names=names, start=q)
        except (ValueError, RuntimeError):
            rejected += 1
    assert rejected == 6
    pose = [1., 2., 3., 1., 0., 0., 0.]
    assert pose_error(pose, pose) == {'position_m': 0., 'rotation_rad': 0.}
    assert pose_error([1., 2., 3., -1., 0., 0., 0.], pose)['rotation_rad'] == 0
    urdf = ROOT / 'infrastructure/development/tig-dual-arm/mani_skill/assets/robots/panda/panda_v3.urdf'
    zero_fk = urdf_tcp_pose(urdf, {f'panda_joint{i}': 0. for i in range(1, 8)})
    assert zero_fk.shape == (7,) and np.isfinite(zero_fk).all()
    assert np.isclose(np.linalg.norm(zero_fk[3:]), 1.)
    return {'status': 'passed', 'shape': list(trajectory.positions_rad.shape), 'rejected_cases': rejected}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(self_test(), allow_nan=False))
