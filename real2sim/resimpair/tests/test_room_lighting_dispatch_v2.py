import subprocess
import sys
from unittest.mock import patch
import pytest
from resimpair_tasks import room_lighting_dispatch_v2 as candidate


def test_read_only_resource_subprocess_survives_worker_dispatch_scope():
    original = subprocess.Popen
    original_module = candidate.retained.render.subprocess
    with candidate.dispatch():
        assert subprocess.Popen is original
        assert candidate.retained.render.subprocess is not subprocess
        # A real CPU child, standing in for nvidia-smi resource readback.
        assert subprocess.check_output([sys.executable, '-c', 'print("query-ok")'], text=True).strip() == 'query-ok'
    assert candidate.retained.render.subprocess is original_module


def test_renderer_worker_redirects_only_child_module_without_mutating_command():
    command = [sys.executable, '-m', 'resimpair_tasks.scene_quality_render_v3', 'worker', '--root', '/tmp/never-run', '--case', 'x']
    with patch.object(subprocess, 'Popen', return_value='sentinel') as spawn:
        with candidate.dispatch():
            assert candidate.retained.render.subprocess.Popen(command, env={'x': 'y'}) == 'sentinel'
            assert spawn.call_args.args[0][2] == 'resimpair_tasks.room_lighting_replay_v1'
            assert spawn.call_args.kwargs == {'env': {'x': 'y'}}
    assert command[2] == 'resimpair_tasks.scene_quality_render_v3'


def test_unexpected_renderer_launch_is_rejected_without_leaking_patch():
    original = candidate.retained.render.subprocess
    with pytest.raises(ValueError, match='unexpected_renderer_child_command'):
        with candidate.dispatch():
            candidate.retained.render.subprocess.Popen(['unexpected-program'])
    assert candidate.retained.render.subprocess is original
