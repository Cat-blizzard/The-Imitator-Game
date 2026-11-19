"""Optional native motion reference with exact aspect and explicit frame mapping.

This candidate does not submit model calls, replace frozen requests, or claim a
human video. All output frames are decoded native frames, uniformly resampled in
time and resized isotropically. The old square-only derivative stays unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from corl2026_workshop.experiments.autonomous_pipeline_v4.core import (
    bind, checked, read, require, write,
)
from corl2026_workshop.experiments.autonomous_pipeline_v4.media import packet

VERSION = 'aspect_preserving_native_motion_reference_v2'


def dimensions(width, height, short_edge=720, long_edge=1280):
    """Exact rational aspect, even H.264 dimensions, no crop/pad/stretch."""
    require(all(type(x) is int and x > 0 for x in (width, height, short_edge, long_edge)),
            'positive_integer_dimensions_required')
    divisor = math.gcd(width, height)
    w, h = width // divisor, height // divisor
    factor = 2 * int(min(short_edge / min(w, h), long_edge / max(w, h)) // 2)
    require(factor >= 2, 'exact_even_aspect_cannot_fit_requested_limits')
    return w * factor, h * factor


def frame_indices(times, duration, fps=24):
    times = np.asarray(times, dtype=float)
    require(times.ndim == 1 and len(times) > 1 and np.isfinite(times).all()
            and np.all(np.diff(times) > 0), 'finite_monotone_source_clock_required')
    require(type(duration) is int and 4 <= duration <= 30 and type(fps) is int and 1 <= fps <= 60,
            'bounded_integer_duration_and_fps_required')
    target = np.linspace(times[0], times[-1], duration * fps)
    right = np.minimum(np.searchsorted(times, target, side='left'), len(times) - 1)
    left = np.maximum(right - 1, 0)
    chosen = np.where(target - times[left] <= times[right] - target, left, right)
    require(chosen[0] == 0 and chosen[-1] == len(times) - 1 and np.all(np.diff(chosen) >= 0),
            'full_native_interval_must_be_retained')
    return chosen, target


def camera_binding(execution, video):
    bundle = read(checked(execution))
    matches = [(name, c) for name, c in bundle['cameras'].items()
               if c.get('video') and checked(c['video']) == checked(video)]
    require(len(matches) == 1, 'unique_native_camera_video_binding_required')
    name, camera = matches[0]
    require('intrinsic_cv' in camera['parameters'] and 'extrinsic_cv' in camera['parameters'],
            'actual_camera_calibration_required')
    return name, camera


def derivative(contract, output, *, short_edge=720, long_edge=1280, fps=24):
    output = Path(output)
    receipt = output.with_suffix('.json')
    native = checked(contract['simulation_video'])
    execution = contract['execution']
    checked(execution)
    duration = contract['generation_information']['duration_seconds']
    settings = {'short_edge_limit': short_edge, 'long_edge_limit': long_edge, 'fps': fps,
                'duration_seconds': duration, 'resize_policy': 'exact_rational_isotropic_even_dimensions'}
    if receipt.exists():
        old = read(receipt)
        require(old['schema'] == VERSION and old['native_video'] == contract['simulation_video']
                and old['native_execution'] == execution and old['settings'] == settings,
                'cached_motion_reference_inputs_changed')
        for field in ('video', 'frame_mapping', 'native_packet', 'reference_packet', 'spatial_condition'):
            checked(old[field])
        return old
    require(not output.exists(), 'partial_motion_reference_requires_new_output_or_diagnosis')
    folder = output.parent / (output.stem + '_evidence')
    require(not folder.exists(), 'partial_motion_evidence_requires_diagnosis_no_retry')
    folder.mkdir(parents=True)
    name, camera = camera_binding(execution, contract['simulation_video'])
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams',
        '-of', 'json', str(native)], check=True, capture_output=True, text=True, timeout=30)
    stream = json.loads(probe.stdout)['streams'][0]
    require(stream.get('sample_aspect_ratio') in (None, 'N/A', '1:1')
            and float(stream.get('tags', {}).get('rotate', 0)) == 0
            and not any(float(d.get('rotation', 0)) != 0 for d in stream.get('side_data_list', [])),
            'non_square_pixels_or_rotation_require_explicit_calibration_adapter')
    info = packet(native, folder / 'native_packet', frame_budget=5)
    w, h = dimensions(info['width'], info['height'], short_edge, long_edge)
    require((info['width'], info['height']) == (camera['config']['width'], camera['config']['height']),
            'encoded_source_dimensions_differ_from_camera')
    chosen, target = frame_indices(info['all_native_pts_seconds'], duration, fps)
    scale = w / info['width']
    affine = np.array([[scale, 0, (scale - 1) / 2], [0, scale, (scale - 1) / 2], [0, 0, 1]])
    intrinsic = np.asarray(camera['parameters']['intrinsic_cv'], dtype=float)
    require(intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all(), 'invalid_source_intrinsic')
    spatial = {'schema': VERSION, 'source_execution': execution, 'source_camera_name': name,
        'source_camera_config': deepcopy(camera['config']), 'source_camera_parameters': deepcopy(camera['parameters']),
        'source_dimensions': [info['width'], info['height']], 'output_dimensions': [w, h],
        'exact_aspect_ratio': [w // math.gcd(w, h), h // math.gcd(w, h)],
        'pixel_center_resize_affine': affine.tolist(), 'resized_intrinsic_cv': (affine @ intrinsic).tolist(),
        'extrinsic_cv': deepcopy(camera['parameters']['extrinsic_cv']),
        'resampling_convention': 'OpenCV pixel-center map: output = scale*(source+0.5)-0.5',
        'new_simulator_camera': False, 'source_calibration_is_actual_native_export': True,
        'crop': False, 'padding': False, 'anisotropic_resize': False}
    write(folder / 'spatial_condition.json', spatial)
    rows = [{'output_frame_index': j, 'output_pts_seconds': j / fps,
             'source_video_frame_index': int(i), 'source_video_pts_seconds': info['all_native_pts_seconds'][int(i)],
             'target_native_pts_seconds': float(target[j]),
             'source_state_index': int(i) if camera.get('source_state_index_equals_video_frame_index') else None}
            for j, i in enumerate(chosen)]
    write(folder / 'frame_mapping.json', {'schema': VERSION, 'source_video': contract['simulation_video'],
        'source_execution': execution, 'source_camera_name': name, 'rows': rows,
        'selection': 'nearest_native_pts_ties_choose_earlier_no_interpolated_frames',
        'first_and_last_native_frame_retained': True, 'native_states_unchanged': True})
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s:v', f'{w}x{h}', '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2',
        '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    start = time.monotonic()
    capture = cv2.VideoCapture(str(native))
    require(capture.isOpened(), 'source_video_cannot_be_decoded')
    source_index = 0
    emitted = 0
    process = None
    try:
        with (folder / 'encode_stderr.log').open('wb') as stderr:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr)
            while True:
                require(time.monotonic() - start < 120, 'bounded_reference_encoding_timeout')
                ok, frame = capture.read()
                if not ok:
                    break
                require(frame.shape[:2] == (info['height'], info['width']), 'decoded_source_size_changed')
                if emitted < len(chosen) and chosen[emitted] == source_index:
                    resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
                    raw = resized.tobytes()
                    while emitted < len(chosen) and chosen[emitted] == source_index:
                        process.stdin.write(raw)
                        emitted += 1
                source_index += 1
            process.stdin.close()
            require(process.wait(timeout=max(1, 120 - (time.monotonic() - start))) == 0,
                    'reference_encoder_failed_preserve_diagnostic')
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        raise
    finally:
        capture.release()
    require(source_index == info['decoded_frames'] and emitted == len(chosen), 'native_frame_mapping_incomplete')
    decoded = packet(output, folder / 'reference_packet', frame_budget=5)
    require(decoded['decoded_frames'] == duration * fps and (decoded['width'], decoded['height']) == (w, h)
            and abs(decoded['duration_seconds'] - duration) < .001
            and abs(decoded['fps'] - fps) < 1e-6, 'reference_output_clock_or_dimensions_mismatch')
    checked(contract['simulation_video'])
    value = {'schema': VERSION, 'video': bind(output), 'native_video': contract['simulation_video'],
        'native_execution': execution, 'settings': settings, 'implementation': bind(__file__),
        'native_packet': bind(folder / 'native_packet/packet.json'),
        'reference_packet': bind(folder / 'reference_packet/packet.json'),
        'frame_mapping': bind(folder / 'frame_mapping.json'), 'spatial_condition': bind(folder / 'spatial_condition.json'),
        'transformation': 'complete native interval, explicit nearest-frame retiming, isotropic resize; no crop or motion editing',
        'source_pts_interval': [info['all_native_pts_seconds'][0], info['all_native_pts_seconds'][-1]],
        'output_duration_seconds': duration, 'output_frames': duration * fps, 'native_states_unchanged': True,
        'elapsed_seconds': time.monotonic() - start, 'source_joint_paths_are_not_mandatory_human_paths': True,
        'human_generation_started': False, 'API_calls': 0, 'default_promoted': False}
    write(receipt, value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--camera', choices=('robot_side', 'opposite_high'), required=True)
    parser.add_argument('--duration', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    camera = read(args.execution)['cameras'][args.camera]
    result = derivative({'execution': bind(args.execution), 'simulation_video': camera['video'],
        'generation_information': {'duration_seconds': args.duration}}, args.output)
    print({key: result[key] for key in ('schema', 'video', 'output_frames', 'elapsed_seconds', 'API_calls')})
