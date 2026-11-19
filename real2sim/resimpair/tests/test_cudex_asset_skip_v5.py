"""Execution safety and data-failure routing, not physical grasp evidence."""
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from resimpair_grasp.common import binding, write
from resimpair_grasp import cudex_finite_input_v4 as finite
from resimpair_tasks import cudex_asset_skip_v5 as candidate


def mesh(com=(0., 0., 0.), volume=1.):
    return SimpleNamespace(vertices=np.array([[0., 0., 0.], [1., 1., 1.]]),
        faces=np.array([[0, 1, 0]]), center_mass=np.asarray(com), volume=volume,
        bounds=np.array([[0., 0., 0.], [1., 1., 1.]]), is_watertight=False)


def test_nonfinite_COM_is_rejected_without_guessing_replacement():
    report = finite.inspect_mesh(mesh((np.nan, np.nan, np.nan), 0.), 0.)
    assert report['status'] == 'asset_invalid'
    assert report['reasons'] == ['nonfinite_mesh_center_mass']
    assert report['center_mass_m'] is None and report['signed_volume_m3'] == 0.


def test_finite_open_mesh_not_silently_repaired_or_rejected_as_ungraspable():
    report = finite.inspect_mesh(mesh(), 0.)
    assert report['status'] == 'finite_input' and report['is_watertight'] is False
    assert report['finite_does_not_certify_physical_mass_or_grasp'] is True


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_vertices_and_support_rejected(bad):
    data = mesh(); data.vertices[0, 0] = bad
    assert 'nonfinite_mesh_vertices' in finite.inspect_mesh(data, 0.)['reasons']
    assert 'nonfinite_native_table_support' in finite.inspect_mesh(mesh(), bad)['reasons']


@pytest.fixture
def failed(tmp_path):
    folder = tmp_path / 'job'; write(folder / 'job.json', {'test': True})
    ref = binding(folder / 'job.json')
    write(folder / 'terminal.json', {'status': 'failed', 'failure_stage': 'gpu', 'returncode': 1})
    write(folder / 'optimizer_terminal.json', {'returncode': 1, 'error': None})
    write(folder / 'reservation.json', {'job': ref})
    (folder / 'worker.log').write_text('/curobo/opt/newton/newton_base.py\n'
        'q = q.view(self.n_problems, self.action_horizon * self.d_action)\n'
        + candidate.EMPTY_INPUT_ERROR + '\n')
    return folder, {'bulk_jobs': [ref]}, ref


def test_known_empty_input_is_failed_not_success_and_no_overwrite(failed):
    folder, protocol, ref = failed
    before = binding(folder / 'terminal.json')
    result = candidate.classify_terminal(protocol, ref)
    assert result['retry'] is False and result['cache_admitted'] is False
    assert result['reason_code'] == 'empty_newton_input_runtime_error'
    assert before == binding(folder / 'terminal.json')


@pytest.mark.parametrize('filename', ['optimizer_result.json', 'result.json', 'export_reservation.json', 'raw.npz'])
def test_ambiguous_output_not_skipped(failed, filename):
    folder, protocol, ref = failed
    (folder / filename).write_text('unresolved')
    with pytest.raises(ValueError, match='requires_recovery'):
        candidate.classify_terminal(protocol, ref)


def test_unknown_runtime_error_is_not_relabelled_asset_failure(failed):
    folder, protocol, ref = failed
    (folder / 'worker.log').write_text('RuntimeError: CUDA out of memory\n')
    with pytest.raises(ValueError, match='unrecognized_terminal_error'):
        candidate.classify_terminal(protocol, ref)


def test_nonfinite_state_required_not_error_text_alone(failed, monkeypatch):
    folder, protocol, ref = failed
    (folder / 'worker.log').write_text(candidate.NONFINITE_ERRORS[0] + '\n')
    monkeypatch.setattr(finite, 'inspect_job', lambda _path: {'status': 'finite_input', 'reasons': []})
    np.savez_compressed(folder / 'solver_state_before_diagnostics.npz', q_unsorted=np.zeros((256, 9)))
    with pytest.raises(ValueError, match='finite_output_does_not_match'):
        candidate.classify_terminal(protocol, ref)
    q = np.zeros((256, 9)); q[12, 3] = np.nan
    np.savez_compressed(folder / 'solver_state_before_diagnostics.npz', q_unsorted=q)
    result = candidate.classify_terminal(protocol, ref)
    assert result['solver_state']['nonfinite_rows'] == 1 and result['cache_admitted'] is False


def test_known_bad_assets_continue_to_next_without_old_failure_quota():
    observed, receipts = [], []
    def execute(ref):
        observed.append(ref['path'])
        return {'status': 'completed' if ref['path'] == '34' else 'failed'}
    result = candidate.dispatch([{'path': str(i)} for i in range(35)], execute, 1,
        time.time() + 5, threading.Event(), lambda *_: None, lambda: None,
        lambda ref, _r: receipts.append(ref['path']))
    assert observed == [str(i) for i in range(35)] and result['error'] is None
    assert result['known_asset_failures'] == 34 and len(receipts) == 34


def test_unknown_submission_and_deadline_stop_without_retry():
    called = []
    result = candidate.dispatch([{'path': str(i)} for i in range(5)],
        lambda ref: called.append(ref) or {'status': 'needs_recovery', 'error': 'unknown'},
        1, time.time() + 5, threading.Event(), lambda *_: None, lambda: None, lambda *_: None)
    assert len(called) == 1 and result['error']
    result = candidate.dispatch([{'path': 'x'}], lambda _: pytest.fail('must not submit'),
        1, time.time() - 1, threading.Event(), lambda *_: None, lambda: None, lambda *_: None)
    assert result['submitted'] == 0


def test_unknown_error_drains_inflight_without_further_admission():
    barrier = threading.Barrier(2); called = []; finished = []
    def execute(ref):
        called.append(ref['path']); barrier.wait(timeout=2)
        if ref['path'] == '0': return {'status': 'failed'}
        time.sleep(.05); return {'status': 'completed'}
    def classify(*_): raise ValueError('not_a_known_asset_error')
    result = candidate.dispatch([{'path': str(i)} for i in range(5)], execute, 2,
        time.time() + 5, threading.Event(), lambda ref, _r: finished.append(ref['path']), lambda: None, classify)
    assert result['error'] and sorted(called) == sorted(finished) == ['0', '1']
