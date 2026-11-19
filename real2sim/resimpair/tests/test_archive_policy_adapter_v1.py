import ast
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from resimpair_tasks import archive_policy_adapter_v1 as candidate


def test_default_is_original_deflate6_and_global_numpy_is_untouched(tmp_path):
    original = np.savez_compressed
    original_renderer = candidate.retained.render.np
    folder = tmp_path / 'default'; folder.mkdir()
    trace = []
    with patch.object(candidate.archive, 'write_npz', side_effect=AssertionError('default switched')):
        with candidate.install(receipts=trace):
            proxy = candidate.retained.render.np
            assert np.savez_compressed is original
            assert proxy.arange is np.arange and proxy.int32 is np.int32
            assert np.array_equal(proxy.stack([np.arange(3), np.arange(3)]), np.stack([np.arange(3)] * 2))
            proxy.savez_compressed(folder / 'segmentation.npz', data=np.arange(5, dtype=np.int32))
    assert candidate.retained.render.np is original_renderer
    assert trace[0]['policy'] == 'deflate6'
    assert np.savez_compressed is original


def test_opt_in_reaches_writer_and_restores_on_exception(tmp_path):
    original = candidate.retained.render.np
    trace = []
    with pytest.raises(RuntimeError, match='after-save'):
        with candidate.install('deflate1', trace):
            candidate.retained.render.np.savez_compressed(tmp_path / 'segmentation.npz', data=np.arange(7))
            raise RuntimeError('after-save')
    assert candidate.retained.render.np is original
    with np.load(tmp_path / 'segmentation.npz', allow_pickle=False) as archive:
        assert np.array_equal(archive['data'], np.arange(7))
    assert trace[0]['policy'] == 'deflate1'


def test_rejects_unknown_policy_and_non_segmentation_writes_without_leaking(tmp_path):
    original = candidate.retained.render.np
    with pytest.raises(ValueError, match='explicit_known'):
        with candidate.install('lossy'): pass
    with pytest.raises(ValueError, match='only_registered'):
        with candidate.install('deflate1'):
            candidate.retained.render.np.savez_compressed(tmp_path / 'different.npz', data=np.arange(4))
    assert candidate.retained.render.np is original
    assert not (tmp_path / 'different.npz').exists()


def test_existing_archive_is_never_overwritten(tmp_path):
    path = tmp_path / 'segmentation.npz'; path.write_bytes(b'original')
    for policy in candidate.POLICIES:
        with pytest.raises(FileExistsError):
            with candidate.install(policy):
                candidate.retained.render.np.savez_compressed(path, data=np.arange(3))
        assert path.read_bytes() == b'original'


def test_module_local_dispatch_does_not_redirect_resource_subprocess():
    original = subprocess.Popen; renderer = candidate.retained.render.subprocess
    with candidate.dispatch():
        assert subprocess.Popen is original
        assert subprocess.check_output([sys.executable, '-c', 'print("resource-ok")'], text=True).strip() == 'resource-ok'
    assert candidate.retained.render.subprocess is renderer


def test_only_renderer_worker_module_is_rewritten():
    command = [sys.executable, '-m', 'resimpair_tasks.scene_quality_render_v3', 'worker', '--root', '/never', '--case', 'case']
    original = command.copy()
    with patch.object(subprocess, 'Popen', return_value='sentinel') as popen:
        with candidate.dispatch():
            assert candidate.retained.render.subprocess.Popen(command, env={'test': 'value'}) == 'sentinel'
            sent = popen.call_args.args[0]
            assert sent[2] == 'resimpair_tasks.archive_policy_adapter_v1'
            assert sent[:2] == command[:2] and sent[3:] == command[3:]
            assert popen.call_args.kwargs == {'env': {'test': 'value'}}
    assert command == original


def test_unexpected_dispatch_restores_original_subprocess_reference():
    original = candidate.retained.render.subprocess
    with pytest.raises(ValueError, match='unexpected_renderer_worker_command'):
        with candidate.dispatch():
            candidate.retained.render.subprocess.Popen(['unregistered-command'])
    assert candidate.retained.render.subprocess is original


def test_original_renderer_has_one_exact_segmentation_archive_call_site():
    tree = ast.parse(Path(candidate.retained.render.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == 'np' and n.func.attr == 'savez_compressed']
    assert len(calls) == 1
    assert ast.unparse(calls[0].args[0]) == "folder / 'segmentation.npz'"
