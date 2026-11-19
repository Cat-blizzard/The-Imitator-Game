"""Endpoint and dispatch tests only: no environment, actuator or physics steps."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from resimpair_grasp.common import binding, digest, read, write
from resimpair_pickplace import above_rim_release_v1 as legacy
from resimpair_pickplace import above_rim_release_v2 as candidate
from resimpair_pickplace import whole_path_motion_v1 as whole
from resimpair_pickplace.cartesian_reachability import INITIAL, pose
from resimpair_tasks.source_first_height_profile_audit_v1 import common_initializations
from resimpair_tasks.source_first_reachable_layout_v3 import WorkspaceReleaseSelector


FIXTURE = Path(__file__).parent / 'fixtures/height_profile_false_negative_v1.json'


class Pose:
    def __init__(self, transform):
        self.transform = np.asarray(transform, float).copy()
    def to_transformation_matrix(self):
        return self.transform.copy()


def operation():
    return {'id': 'transfer_0', 'primitive': 'pick_place', 'object': 'object_0', 'arm': None,
        'target': {'support': 'receiver', 'offset_xy': [0., 0.], 'yaw': 0.},
        'release': {'mode': 'drop_from_above', 'allow_drop': True, 'fragile': False,
                    'bottom_above_rim_m': .03, 'hand_above_rim_m': .01, 'maximum_fall_m': .15}}


def inputs(tmp_path, count=2):
    candidates = [{'id': f'c{i}', 'tcp_in_asset': np.eye(4).tolist(), 'width_m': .04} for i in range(count)]
    signature = {'descriptor': {'path': 'unit_dispatch_fixture', 'sha256': 'unused'}}
    cache = {'signature': signature, 'key': digest(signature), 'candidates': candidates}
    path = tmp_path / 'original_cache.json'; write(path, cache)
    scene = {'robot_roots': [[-.6, -.35, 0.], [-.6, .35, 0.]], 'seed': 17,
        'objects': [{'id': 'object_0', 'position': [-.2, 0., .05], 'yaw': 0.}],
        'supports': [], 'grasp_caches': {'object_0': binding(path)}}
    task = {'operations': [operation()], 'object_ids': ['object_0'], 'requested_task_id': 'single'}
    rec = SimpleNamespace(scene_spec=deepcopy(scene), directory=tmp_path / 'execution',
        object_ids=['object_0'], objects=[SimpleNamespace(pose=Pose(pose([-.2, 0., .05])))],
        drop_operations={operation()['id']: operation()}, drop_release_targets={})
    return scene, task, rec


def proposal(arm, cid, heights, cost=1., accepted=True):
    return {'arm': arm, 'candidate': cid, 'height_profile_m': list(heights), 'accepted': accepted,
            'joint_travel_proxy': cost, 'checks': []}


def prepared(tmp_path, monkeypatch, *, preview=None):
    scene, task, rec = inputs(tmp_path)
    calls = []
    class FakeSelector:
        def __init__(self, *args):
            pass
        def select(self, *args):
            return {'passed': True, 'decisions': [proposal(1, 'c0', (.20, .18), accepted=False),
                proposal(1, 'c0', (.12, .12)), proposal(0, 'c1', (.26, .24), accepted=False)]}
    class Guard:
        def install(self, index):
            self.index = index
    def make_solver(rec, arm, directory):
        guard = Guard(); guard.arm = arm; guard.directory = directory
        return guard
    def preview_route(guard, grasp, desired, grasp_candidate, heights):
        calls.append((guard.arm, grasp_candidate['id'], heights))
        if preview is not None:
            return preview(guard, grasp, desired, grasp_candidate, heights)
        return {'passed': True, 'height_profile_m': list(heights), 'physical_grasp_status': 'unmeasured'}
    with monkeypatch.context() as patch:
        patch.setattr(candidate, 'ReleaseSelector', FakeSelector)
        patch.setattr(whole, 'make_solver', make_solver)
        patch.setattr(legacy, 'DropMotion', Guard)
        patch.setattr(whole, 'preview_route', preview_route)
        patch.setattr(legacy, 'release_geometry', lambda *args: (pose([.1, 0., .1]), {'mode': 'drop_from_above'}))
        chosen = candidate.prepare(rec, task)
    return scene, task, rec, chosen, calls


@pytest.mark.parametrize('case_index', range(3))
def test_saved_real_false_negative_reaches_low_profile(case_index, tmp_path, monkeypatch):
    """Real mplib IK at the three recorded failures, with the same saved starts."""
    fixture = read(FIXTURE)['cases'][case_index]
    source = np.asarray(fixture['source_object_transform'])
    cache_path = tmp_path / 'cache.json'; write(cache_path, {'candidates': [fixture['candidate']]})
    scene = {'objects': [{'id': fixture['object'], 'position': source[:3, 3].tolist(), 'yaw': 0.,
                         'measured_world_transform': source.tolist()}],
             'grasp_caches': {fixture['object']: binding(cache_path)}}
    monkeypatch.setattr(legacy, 'release_geometry', lambda *args: (
        np.asarray(fixture['release_object_transform']), deepcopy(fixture['release_geometry'])))
    bank = fixture['common_random_initializations']
    def with_saved_source(selector):
        native_ik = selector.ik
        def ik(arm, target, seed_qpos):
            if arm != fixture['arm']:
                return None, 'other_arm_not_in_this_recorded_counterfactual'
            for check in fixture['saved_prefix']:
                if np.allclose(target, check['world_tcp'], rtol=0, atol=1e-12):
                    return np.asarray(check['qpos'], float), check['status']
            with common_initializations(selector.planners[arm], bank):
                return native_ik(arm, target, seed_qpos)
        selector.ik = ik
        return selector
    old = with_saved_source(WorkspaceReleaseSelector(fixture['robot_roots'], fixture['seed']))
    new = with_saved_source(candidate.ReleaseSelector(fixture['robot_roots'], fixture['seed']))
    assert not old.select(scene, fixture['operation'])['passed']
    report = new.select(scene, fixture['operation'])
    low = next(r for r in report['decisions'] if r['arm'] == fixture['arm']
               and r['height_profile_m'] == [.12, .12])
    assert low['accepted']
    assert [c['stage'] for c in low['checks']] == ['pregrasp', 'grasp', 'lift', 'above_goal', 'release']
    assert all(c['status'] == 'Success' for c in low['checks'])
    assert low['candidate'] == fixture['candidate']['id']
    assert report['physical_success'] == 'unmeasured'


def test_all_failures_stay_bounded_to_two_arms_first_twelve(tmp_path, monkeypatch):
    scene, task, _ = inputs(tmp_path, count=13)
    selector = object.__new__(candidate.ReleaseSelector); selector.planners = [None, None]; selector.bindings = []
    selector.ik = lambda *args: (None, 'source_ik_failed')
    monkeypatch.setattr(legacy, 'release_geometry', lambda *args: (pose([.1, 0., .1]), {}))
    result = selector.select(scene, task['operations'][0])
    assert not result['passed'] and len(result['decisions']) == 2 * 12 * 3
    assert {r['candidate'] for r in result['decisions']} == {f'c{i}' for i in range(12)}
    assert result['actual_endpoint_IK_calls'] == 24
    assert all(r['checks'][-1]['status'] == 'source_ik_failed' for r in result['decisions'])


def test_original_workspace_and_profile_limits_are_not_relaxed(monkeypatch):
    selector = object.__new__(candidate.ReleaseSelector)
    def unexpected(*args):
        raise AssertionError('out-of-workspace target reached native IK')
    monkeypatch.setattr(legacy.OriginalSelector, 'ik', unexpected)
    q, status = selector.ik(0, pose([-.451, 0., .2]), INITIAL)
    assert q is None and status == 'outside_retained_complete_path_workspace'
    for value in ([.06, .06], [.2, .12], [np.nan, .12], [.12], None, 'invalid'):
        with pytest.raises(ValueError, match='original_whole_path_height_profile'):
            candidate.profile(value)
    assert tuple(candidate.profile(h) for h in candidate.PROFILES) == whole.HEIGHTS


def test_profile_rows_do_not_consume_distinct_grasp_budget():
    rows = [proposal(0, 'c0', h, 1.) for h in candidate.PROFILES]
    rows += [proposal(0, f'c{i}', (.12, .12), i + 1.) for i in range(1, 4)]
    rows += [proposal(1, 'd0', (.12, .12), 1.5)]
    result = candidate.ordered_choices({'decisions': rows})
    assert [(r['arm'], r['candidate']) for r in result] == [(0, 'c0'), (1, 'd0'), (0, 'c1'), (0, 'c2')]
    assert [tuple(r['height_profile_m']) for r in result[0]['profiles']] == list(candidate.PROFILES)


def test_prepare_passes_only_eligible_arm_grasp_profile_to_retained_preview(tmp_path, monkeypatch):
    original, task, rec, chosen, calls = prepared(tmp_path, monkeypatch)
    assert calls == [(1, 'c0', (.12, .12))]
    assert task['operations'][0]['arm'] is None
    assert chosen['operations'][0] == {**task['operations'][0], 'arm': 1}
    assert rec.whole_path_choices == {'transfer_0': (.12, .12)}
    cache = read(Path(rec.scene_spec['grasp_caches']['object_0']['path']))
    assert cache['signature']['height_profile_m'] == [.12, .12]
    assert cache['signature']['parent_cache'] == original['grasp_caches']['object_0']
    assert [c['id'] for c in cache['candidates']] == ['c0']
    assert binding(tmp_path / 'original_cache.json') == original['grasp_caches']['object_0']
    proof = candidate.execution_choice(rec, task, 1, tmp_path / 'transfer_0_motion')
    assert proof['height_profile_m'] == [.12, .12] and proof['candidate'] == 'c0'


@pytest.mark.parametrize('mode', ['guard_rejects', 'wrong_profile'])
def test_endpoint_pass_cannot_bypass_original_complete_path_guard(mode, tmp_path, monkeypatch):
    def reject(*args):
        if mode == 'guard_rejects':
            raise RuntimeError('independent_guard_rejected:above_bowl')
        return {'passed': True, 'height_profile_m': [.20, .18]}
    with pytest.raises(RuntimeError, match='no_complete_pick_place_proposal'):
        prepared(tmp_path, monkeypatch, preview=reject)
    decision = read(tmp_path / 'execution/whole_path_selection/transfer_0/selection.json')
    assert decision['selected'] is None
    assert len(decision['decisions']) == 1 and not decision['decisions'][0]['passed']
    assert not (tmp_path / 'execution/whole_path_selection/transfer_0/selected_cache.json').exists()


def worker_spec():
    from resimpair_pickplace.rolling_pickplace_v3 import VERSION as ROLLING
    return {'candidate_release_selector': candidate.VERSION, 'candidate_motion_override': whole.VERSION,
        'motion_policy': 'transport_fallback_v2', 'execution_selection_policy': ROLLING,
        'code': {'resimpair_pickplace/whole_path_motion_v1.py': binding(whole.__file__),
                 'resimpair_pickplace/above_rim_release_v2.py': binding(candidate.__file__)}}


def test_nested_legacy_worker_consumes_committed_profile_in_actual_motion_chain(tmp_path, monkeypatch):
    """Run original worker contexts and DropMotion -> WholePathMotion math.

    Only the final native planner is replaced by a capture function; no command
    is sent. This protects dispatch and height propagation, not path success.
    """
    import sys
    from resimpair_pickplace import stability_motion, transport_fallback_v2, runner
    from resimpair_pickplace import payload_descent_v2 as descent, certified_prefix_v1 as prefix
    _, task, rec, _, _ = prepared(tmp_path, monkeypatch)
    guard_method = legacy.DropMotion.check
    old_whole_class, old_whole_plan = whole.WholePathMotion, whole.WholePathMotion.plan
    class Guard:
        pass
    def solver(recorder, arm, directory):
        guard = Guard(); guard.rec = recorder; guard.arm = arm; guard.directory = Path(directory)
        guard.selected_actor = recorder.objects[0]; guard.counter = 0; guard.preview = False; guard.use_attach = False
        guard.agent = SimpleNamespace(robot=SimpleNamespace(qpos=INITIAL.copy()), tcp=recorder.objects[0])
        return guard
    captured = []
    def native_plan(guard, waypoint, stage, current=None):
        captured.append((stage, waypoint.to_transformation_matrix()))
        return {'dispatch_only': True}
    monkeypatch.setattr(runner, 'verify', lambda root: worker_spec())
    monkeypatch.setattr(stability_motion, 'make_solver', solver)
    monkeypatch.setattr(whole.ContinuationMotion, 'plan', native_plan)
    monkeypatch.setattr(legacy, 'release_geometry', lambda *args: (pose([.1, 0., .1]), {'mode': 'drop_from_above'}))
    monkeypatch.setitem(sys.modules, 'sapien', SimpleNamespace(Pose=Pose))
    with candidate.install_for_worker(tmp_path, task):
        assert whole.WholePathMotion is old_whole_class
        assert whole.WholePathMotion.plan is old_whole_plan
        # The retained production driver enters v1 itself. It must still see
        # the new prepare without changing its descent -> drop -> prefix order.
        with descent.install(), legacy.install(task), prefix.install(task):
            assert whole.prepare is candidate.prepare and legacy.prepare is candidate.prepare
            with whole.install_for_worker(tmp_path):
                guard = stability_motion.make_solver(rec, 1, tmp_path / 'transfer_0_motion')
                guard.__class__ = transport_fallback_v2.TransportFallbackMotion
                assert guard.height_profile == (.12, .12)
                assert type(guard).check is guard_method
                guard.plan(Pose(pose([-.2, 0., .05 + .20])), 'lift', INITIAL)
                guard.plan(Pose(pose([.1, 0., .9])), 'above_bowl', INITIAL)
                guard.plan(Pose(pose([.1, 0., .9])), 'release_above_rim', INITIAL)
        assert whole.WholePathMotion is old_whole_class
        assert whole.WholePathMotion.plan is old_whole_plan
    assert [s for s, _ in captured] == ['lift', 'above_bowl', 'release_above_rim']
    assert [T[2, 3] for _, T in captured] == pytest.approx([.05 + .12, .1 + .12, .1])
    receipt = read(tmp_path / 'transfer_0_motion/height_profile_execution.json')
    assert (receipt['arm'], receipt['candidate'], receipt['height_profile_m']) == (1, 'c0', [.12, .12])


@pytest.mark.parametrize('mismatch', ['arm', 'profile', 'grasp', 'missing_choice'])
def test_execution_rejects_changed_selection_before_solver_or_movement(mismatch, tmp_path, monkeypatch):
    from resimpair_pickplace import stability_motion, runner
    _, task, rec, _, _ = prepared(tmp_path, monkeypatch)
    arm = 1
    if mismatch == 'arm':
        arm = 0
    elif mismatch == 'profile':
        rec.whole_path_choices['transfer_0'] = (.2, .18)
    elif mismatch == 'grasp':
        cache = read(Path(rec.scene_spec['grasp_caches']['object_0']['path']))
        cache['candidates'][0]['id'] = 'not_an_original_grasp'
        path = tmp_path / 'changed_cache.json'; write(path, cache)
        rec.scene_spec['grasp_caches']['object_0'] = binding(path)
    else:
        rec.whole_path_choices = {}
    calls = []
    monkeypatch.setattr(stability_motion, 'make_solver', lambda *args: calls.append(args))
    monkeypatch.setattr(runner, 'verify', lambda root: worker_spec())
    with candidate.install(task), whole.install_for_worker(tmp_path):
        with pytest.raises(ValueError):
            stability_motion.make_solver(rec, arm, tmp_path / 'transfer_0_motion')
    assert not calls


def test_adapter_restores_every_retained_hook_after_exception(tmp_path, monkeypatch):
    from resimpair_pickplace import stability_motion, rolling_pickplace_v3, transport_fallback_v2, runner
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import release_control
    task = {'operations': [operation()]}
    before = (legacy.prepare, whole.prepare, whole.WholePathMotion, legacy.DropMotion, stability_motion.make_solver,
              rolling_pickplace_v3.PlanningView, transport_fallback_v2.TransportFallbackMotion, release_control.readiness)
    code = candidate.retained_bindings()
    monkeypatch.setattr(runner, 'verify', lambda root: worker_spec())
    with pytest.raises(RuntimeError, match='probe'):
        with candidate.install(task), legacy.install(task), whole.install_for_worker(tmp_path):
            raise RuntimeError('probe')
    after = (legacy.prepare, whole.prepare, whole.WholePathMotion, legacy.DropMotion, stability_motion.make_solver,
             rolling_pickplace_v3.PlanningView, transport_fallback_v2.TransportFallbackMotion, release_control.readiness)
    assert before == after and candidate.retained_bindings() == code


def test_missing_original_worker_context_cannot_default_to_fixed_height(tmp_path, monkeypatch):
    from resimpair_pickplace import stability_motion
    _, task, rec, _, _ = prepared(tmp_path, monkeypatch)
    with candidate.install(task):
        with pytest.raises(ValueError, match='retained_whole_path_worker_context'):
            stability_motion.make_solver(rec, 1, tmp_path / 'transfer_0_motion')


def test_historical_experiment_is_not_implicitly_opted_in(tmp_path, monkeypatch):
    from resimpair_pickplace import runner
    spec = worker_spec(); spec.pop('candidate_release_selector')
    monkeypatch.setattr(runner, 'verify', lambda root: spec)
    before = whole.prepare
    with pytest.raises(ValueError, match='explicit_profile_selector'):
        with candidate.install_for_worker(tmp_path, {'operations': [operation()]}):
            raise AssertionError('historical experiment unexpectedly entered new adapter')
    assert whole.prepare is before
