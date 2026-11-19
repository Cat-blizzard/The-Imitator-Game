from fractions import Fraction

import numpy as np
import pytest

from corl2026_workshop.experiments.autonomous_pipeline_v4 import media
from resimpair_tasks.deferred_render_v1 import NativeClockVideo, validate_timeline


def fixture_data(n=5):
    states = np.zeros((n, 1, 101), dtype=np.float32)
    times = np.arange(n) / 20.
    arrays = {'timestamps_seconds': times.copy(), 'native_commands': np.zeros((n-1, 16)),
              'states': np.zeros((n, 18)), 'tcp_world': np.zeros((n, 2, 4, 4)),
              'object_states': np.zeros((n, 3, 13))}
    return states, times, arrays


def test_saved_clock_is_preserved_exactly():
    assert validate_timeline(*fixture_data()) == Fraction(20, 1)


@pytest.mark.parametrize('damage', ['drop_state', 'time_shift', 'permuted_times', 'missing_action', 'nonuniform', 'missing_qpos'])
def test_misaligned_or_nonuniform_saved_timeline_is_rejected(damage):
    states, times, arrays = fixture_data()
    if damage == 'drop_state': states = states[:-1]
    if damage == 'time_shift': times = times + .05
    if damage == 'permuted_times': times = times[::-1]
    if damage == 'missing_action': arrays['native_commands'] = arrays['native_commands'][:-1]
    if damage == 'nonuniform':
        times[3] += .01; arrays['timestamps_seconds'] = times.copy()
    if damage == 'missing_qpos': arrays['states'] = arrays['states'][:-1]
    with pytest.raises(ValueError): validate_timeline(states, times, arrays)


def test_actual_mp4_encoding_retains_all_native_PTS(tmp_path):
    path = tmp_path / 'fixture.mp4'
    writer = NativeClockVideo(path, 32, 32, Fraction(20, 1))
    for i in range(17): writer.append(np.full((32, 32, 3), i*12, dtype=np.uint8), i)
    writer.close()
    packet = media.packet(path, tmp_path / 'decoded', frame_budget=3)
    assert packet['decoded_frames'] == 17
    np.testing.assert_allclose(packet['all_native_pts_seconds'], np.arange(17)/20, rtol=0, atol=1e-10)


def test_writer_refuses_missing_or_repeated_frames(tmp_path):
    writer = NativeClockVideo(tmp_path / 'fixture.mp4', 32, 32, Fraction(20, 1))
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    writer.append(rgb, 0)
    with pytest.raises(ValueError): writer.append(rgb, 2)
    with pytest.raises(ValueError): writer.append(rgb, 0)
    writer.close()
