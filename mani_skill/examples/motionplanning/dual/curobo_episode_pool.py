"""One bounded cuRobo worker (four planners), reused across compatible episodes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

SCHEMA = 'tig-curobo-episode-reuse-v1'


def compatibility_key(snapshots: Any, *, task_id: str, seed: int,
                      interpolation_dt: float, gpu_uuid: str,
                      worker_script: Path | None, context: Any) -> str:
    # Use exactly the worker's allowed dynamic fields. A new obstacle geometry,
    # payload capacity or fixed configuration requires construction, not reset.
    from infrastructure.curobo_tig_session_worker import robot_topology, world_topology
    from infrastructure.curobo_tig_task_payload_worker import episode_model_topology
    worker = Path(worker_script) if worker_script else Path(__file__).resolve().parents[4]/'infrastructure/curobo_tig_task_payload_worker.py'
    value = {'task': task_id, 'seed': seed, 'dt': interpolation_dt, 'gpu': gpu_uuid,
             'config_profiling_enabled': os.environ.get('TIG_CUROBO_PROFILE_CONFIG') == '1',
             'worker': str(worker.resolve()), 'worker_sha256': hashlib.sha256(worker.read_bytes()).hexdigest(),
             'context': context,
             'arms': [{'robot': robot_topology(s['robot_cfg']), 'world': world_topology(s['world']),
                       'model': episode_model_topology(s['payload_model'])} for s in snapshots]}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _client(gpu_uuid: str, log_path: Path, *, task_id: str, worker_script: Path | None) -> Any:
    from .curobo_task_payload import CuroboTaskPayloadSessions, SelectedWorkerProcess
    transport = SelectedWorkerProcess(gpu_uuid, log_path, worker_script, task_id=task_id) if worker_script else None
    return CuroboTaskPayloadSessions(gpu_uuid, log_path, task_id=task_id, transport=transport)


class CuroboEpisodePool:
    """Own at most one process. Incompatible configurations replace it explicitly.

    A failed physical episode can reuse the process after a complete reset.
    A broken transport requires replacement. A reset rejection never silently
    retries the same episode or turns it into a successful result.
    """
    def __init__(self, *, client_factory: Callable[..., Any] = _client) -> None:
        self.worker: Any = None
        self._factory = client_factory
        self._key: str | None = None
        self._generation = 0
        self._receipt_start = 0
        self._log_path: Path | None = None
        self._construction_episode: str | None = None
        self._report: dict[str, Any] | None = None

    def __enter__(self) -> CuroboEpisodePool:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        if self.worker is not None:
            try:
                self.worker.close()
            finally:
                self.worker = None
                self._key = None
                self._report = None

    def acquire(self, snapshots: Any, *, episode_id: str, task_id: str, seed: int,
                gpu_uuid: str, log_path: Path, worker_script: Path | None,
                context: Any, report: dict[str, Any], interpolation_dt: float = .02) -> Any:
        self._report = None
        key = compatibility_key(snapshots, task_id=task_id, seed=seed,
            interpolation_dt=interpolation_dt, gpu_uuid=gpu_uuid,
            worker_script=worker_script, context=context)
        reason = ('cold' if self.worker is None else 'transport_failed' if self.worker._rpc.failed
                  else 'configuration_changed' if key != self._key else 'compatible')
        reuse = reason == 'compatible'
        if not reuse:
            self.close()
            self.worker = self._factory(gpu_uuid, log_path, task_id=task_id, worker_script=worker_script)
            self._generation += 1
            self._key = key
            self._log_path = log_path
            self._construction_episode = episode_id
        self._receipt_start = len(self.worker.receipts)
        self._report = report
        report['episode_reuse'] = {'schema': SCHEMA, 'episode_id': episode_id,
            'reused': reuse, 'selection_reason': reason, 'worker_generation': self._generation,
            'compatibility_sha256': key, 'construction_episode': self._construction_episode,
            'worker_log': str(self._log_path), 'planner_count': 4,
            'reset_status': 'running', 'planner_seed': seed,
            'seed_semantics': 'restore constructor sampler seed; no bitwise-equivalence claim'}
        report['initialization'] = []
        try:
            if not reuse:
                report['initialization'] = self.worker.initialize_pair(snapshots, seed=seed,
                    interpolation_dt=interpolation_dt, warmup=True)
            # Also reset a newly constructed worker: cold and warm comparisons
            # start with the same public sampler/reset protocol.
            result = self.worker.begin_episode(snapshots, episode_id=episode_id,
                seed=seed, interpolation_dt=interpolation_dt)
            report['episode_reuse'].update(reset_status='passed', reset=result)
        except BaseException:
            report['episode_reuse']['reset_status'] = 'failed'
            self.finish(report)
            if not reuse:  # Partial construction cannot be completed by reset.
                self.close()
            raise
        return self.worker

    def finish(self, report: dict[str, Any]) -> None:
        if self.worker is not None and report is self._report:
            report['worker_requests'] = self.worker.receipts[self._receipt_start:]


def self_test() -> dict[str, Any]:
    import copy
    from types import SimpleNamespace
    class Fake:
        instances: list[Any] = []
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.receipts: list[Any] = []
            self._rpc = SimpleNamespace(failed=False)
            self.closed = False
            self.instances.append(self)
        def initialize_pair(self, *args: Any, **kwargs: Any) -> list[Any]:
            self.receipts.extend([{'op': 'init_session'}]*4)
            return [{'init': True}]*4
        def begin_episode(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            self.receipts.append({'op': 'begin_episode'})
            return {'episode_id': kwargs['episode_id']}
        def close(self) -> None:
            self.closed = True
    snapshot = {'robot_cfg': {'kinematics': {'lock_joints': {}, 'cspace': {'default_joint_position': []}}},
                'world': {'cuboid': {'box': {'dims': [1, 1, 1], 'pose': [0]*7}}},
                'payload_model': {'source_geometry_sha256': 'a', 'source_world_shapes':
                                  {'box': {'dims': [1, 1, 1], 'pose': [0]*7}}}}
    pair = [snapshot, copy.deepcopy(snapshot)]
    common = dict(task_id='fake', seed=0, gpu_uuid='GPU-fake', log_path=Path('/tmp/fake.log'),
                  worker_script=Path(__file__), context={'quality': 'unchanged'})
    with CuroboEpisodePool(client_factory=Fake) as pool:
        a: dict[str, Any] = {}; first = pool.acquire(pair, episode_id='a', report=a, **common)
        pool.finish(a)
        b = copy.deepcopy(pair)
        b[0]['world']['cuboid']['box']['pose'][0] = 2
        b[0]['payload_model']['source_world_shapes']['box']['pose'][0] = 2
        b[0]['payload_model']['source_geometry_sha256'] = 'b'
        for ident, scene in [('b', b), ('a2', pair)]:
            report: dict[str, Any] = {}
            assert pool.acquire(scene, episode_id=ident, report=report, **common) is first
            pool.finish(report)
            assert report['episode_reuse']['reused'] and len(report['worker_requests']) == 1
        assert len(a['worker_requests']) == 5 and len(Fake.instances) == 1
        unrelated: dict[str, Any] = {}; pool.finish(unrelated)
        assert 'worker_requests' not in unrelated
        changed = copy.deepcopy(pair); changed[0]['world']['cuboid']['box']['dims'][0] = 2
        r: dict[str, Any] = {}; second = pool.acquire(changed, episode_id='new', report=r, **common)
        assert second is not first and first.closed and r['episode_reuse']['selection_reason'] == 'configuration_changed'
        second._rpc.failed = True
        r = {}; third = pool.acquire(changed, episode_id='broken', report=r, **common)
        assert third is not second and second.closed and r['episode_reuse']['selection_reason'] == 'transport_failed'
    assert third.closed
    return {'status': 'passed', 'checks': ['pose-only A-B-A reuses', 'geometry change replaces',
        'broken transport replaces', 'cold receipts retain four init calls', 'one bounded worker closes'], 'GPU_initialized': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), allow_nan=False))
