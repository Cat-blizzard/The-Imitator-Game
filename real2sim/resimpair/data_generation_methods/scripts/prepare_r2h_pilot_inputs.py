#!/usr/bin/env python3
"""Prepare six CPU-only, unmodified robot-video inputs; never admit training data."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import av
import numpy as np
from PIL import Image

BASE = Path('/mnt/sdf/dataset/zhouxunzhe/zhiqi/data_generation_methods')
SOURCE = Path('/mnt/sdf/datasets/hypericl_human_robot_task_library_20261019_v1/generation/pilot_delivery_v1/policy_manifest.json')
TASKS = {'apple': 'real_bowl_transfer', 'book': 'closed_book_to_open_box', 'cab': 'cabinet_right_door'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def bind(path, expected=None):
    if not path:
        return {'path': None, 'available': False, 'declared_sha256': expected}
    path = Path(path).resolve()
    if not path.is_file():
        return {'path': str(path), 'available': False, 'declared_sha256': expected}
    digest = sha(path)
    if expected and digest != expected:
        raise ValueError(f'Source hash mismatch: {path}')
    return {'path': str(path), 'available': True, 'bytes': path.stat().st_size,
            'sha256': digest, 'declared_sha256': expected,
            'declared_sha256_matches': digest == expected if expected else None}


def activity(row):
    with np.load(row['arrays'], allow_pickle=False) as z:
        p = z['poses'][..., :3, 3]
        span = np.linalg.norm(p - p[:1], axis=-1).max(axis=0)
    return {'maximum_TCP_displacement_from_initial_m': span.tolist(),
            'arm_slots_above_3cm': np.flatnonzero(span > .03).tolist(),
            'arm_slot_convention': '0=left, 1=right; measured TCP displacement, not a semantic arm label'}


def feature(row, receipt, motion):
    if row['task'] == TASKS['apple']:
        asset = receipt.get('receptacle', {}).get('uid')
    elif row['task'] == TASKS['book']:
        asset = next((a.get('uid') for a in receipt.get('asset_specs', [])
                      if 'book' in a.get('name', '')), None)
    else:
        # Cabinet pilot varies layout; do not invent another door asset from distractors.
        asset = 'same_native_right_cabinet_asset'
    return {'task_asset_identity': asset,
            'active_arm_pattern': motion['arm_slots_above_3cm'],
            'program': row.get('program')}


def select(rows):
    selected, coverage = [], {}
    for family, task in TASKS.items():
        pool = []
        for row in sorted(rows, key=lambda r: r['id']):
            if row.get('task') != task or row.get('split') != 'train':
                continue
            receipt = json.loads(Path(row['collection_receipt']).read_text())
            motion = activity(row)
            pool.append((row, receipt, motion, feature(row, receipt, motion)))
        pairs = [pair for pair in combinations(pool, 2) if pair[0][0]['group'] != pair[1][0]['group']]
        if not pairs:
            raise ValueError(f'Fewer than two independent train groups: {family}')
        def score(pair):
            return (len({x[3]['task_asset_identity'] for x in pair}),
                    len({tuple(x[3]['active_arm_pattern']) for x in pair}),
                    len({x[3]['program'] for x in pair}))
        pair = max(pairs, key=score)
        selected.extend((family, *x) for x in pair)
        coverage[family] = {'candidate_train_rows': len(pool),
                            'selected_ids': [x[0]['id'] for x in pair],
                            'selected_features': [x[3] for x in pair],
                            'selection_score': list(score(pair)),
                            'limitation': ('Cabinet has one native asset and left-active-arm pattern; '
                                           'the two groups are layout variants, not two mechanisms or assets.'
                                           if family == 'cab' else None)}
    return selected, coverage


def video_check(path, expected_n, expected_fps, preview_dir=None):
    picks = {0: 'first', expected_n // 2: 'middle', expected_n - 1: 'last'}
    times, previews, pts_values, dimensions = [], [], [], set()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        average_rate, time_base = stream.average_rate, stream.time_base
        advertised_frames = stream.frames
        for i, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ValueError(f'Missing decoded frame PTS: {path}:{i}')
            timestamp = float(frame.pts * frame.time_base)
            times.append(timestamp)
            pts_values.append(int(frame.pts))
            dimensions.add((frame.width, frame.height))
            if preview_dir is not None and i in picks:
                name = f'{picks[i]}_frame{i:06d}.png'
                output = preview_dir / name
                rgb = frame.to_ndarray(format='rgb24')
                Image.fromarray(rgb).save(output, format='PNG')
                with Image.open(output) as image:
                    restored = np.asarray(image.convert('RGB'))
                    np.testing.assert_array_equal(restored, rgb)
                previews.append({'name': picks[i], 'path': str(output.resolve()),
                                 'frame_index_0based': i, 'pts': int(frame.pts),
                                 'time_base': str(frame.time_base), 'pts_seconds': timestamp,
                                 'sha256': sha(output),
                                 'decoded_RGB_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                                 'is_reviewed_key_event': False})
    if len(times) != expected_n:
        raise ValueError(f'Decoded count differs from source manifest: {path}')
    dt = np.diff(times)
    if not np.all(dt > 0):
        raise ValueError(f'Non-monotonic presentation timestamps: {path}')
    errors = np.abs(np.asarray(times) - np.arange(expected_n) / expected_fps)
    tolerance = max(float(time_base) / 2 + 1e-9, 1e-8)
    if float(errors.max()) > tolerance:
        raise ValueError(f'Source clock differs from native labels: {path}')
    if preview_dir is not None and len(previews) != 3:
        raise ValueError('Expected three native PNG previews')
    return {'full_video_decoded': True, 'decoded_frames': len(times),
            'advertised_stream_frames': advertised_frames, 'average_rate': str(average_rate),
            'native_fps_manifest': expected_fps, 'time_base': str(time_base),
            'first_pts': pts_values[0], 'last_pts': pts_values[-1],
            'first_pts_seconds': times[0], 'last_pts_seconds': times[-1],
            'end_exclusive_seconds_from_cadence': times[-1] + float(np.median(dt)),
            'delta_seconds_min_median_max': [float(dt.min()), float(np.median(dt)), float(dt.max())],
            'max_abs_pts_vs_native_index_seconds': float(errors.max()),
            'all_decoded_pts_sha256': hashlib.sha256(json.dumps(pts_values).encode()).hexdigest(),
            'decoded_dimensions': [list(x) for x in sorted(dimensions)], 'previews': previews}


def array_check(path, n, fps):
    with np.load(path, allow_pickle=False) as z:
        fields = {k: {'shape': list(z[k].shape), 'dtype': str(z[k].dtype)} for k in z.files}
        for key, shape in {'states': (n, 18), 'poses': (n, 2, 4, 4),
                           'target_poses': (n, 2, 4, 4), 'native_commands': (n, 16),
                           'eef_mask': (n, 14)}.items():
            if z[key].shape != shape:
                raise ValueError(f'Native array shape mismatch: {path}:{key}')
        t = z['timestamps_seconds']
        np.testing.assert_allclose(t, np.arange(n) / fps, atol=1e-8, rtol=0)
        runs = []
        if 'stage' in z:
            stages = z['stage'].astype(str)
            starts = [0, *(np.flatnonzero(stages[1:] != stages[:-1]) + 1).tolist(), n]
            runs = [{'label': str(stages[a]), 'start_frame_inclusive': a,
                     'end_frame_exclusive': b, 'start_seconds': a / fps,
                     'end_seconds_exclusive': b / fps} for a, b in zip(starts[:-1], starts[1:])]
        return {'available_fields': fields, 'native_frame_count': n,
                'timestamp_count': len(t), 'timestamp_first_seconds': float(t[0]),
                'timestamp_last_seconds': float(t[-1]), 'dt_seconds': 1 / fps,
                'stage_available': 'stage' in z, 'native_controller_stage_runs': runs,
                'stage_semantics': 'Original controller labels; not independent human key-event approval.',
                'eef_mask_valid_per_dimension': z['eef_mask'].sum(axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=BASE / 'pilot_inputs')
    args = parser.parse_args()
    output = args.output.resolve()
    permitted = (BASE / 'pilot_inputs').resolve()
    if output != permitted and permitted not in output.parents:
        raise ValueError('Output must remain under data_generation_methods/pilot_inputs')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    manifest = args.manifest.resolve()
    source_hash = sha(manifest)
    source = json.loads(manifest.read_text())
    if len(source['episodes']) != 24:
        raise ValueError('This input pilot expects the immutable reviewed 24-row source block')
    chosen, coverage = select(source['episodes'])
    output.mkdir(parents=True, exist_ok=False)
    inputs = []
    for family, row, receipt, motion, features in chosen:
        case_dir = output / row['id']
        case_dir.mkdir()
        files = {'native512_video': bind(row['reference_video']['path'], row.get('reference_video_sha256')),
                 'policy224_video': bind(row['video']['path'], row.get('source_video_sha256')),
                 'arrays': bind(row['arrays'], receipt.get('sha256', {}).get('arrays.npz')),
                 'receipt': bind(row['collection_receipt'], row.get('receipt_sha256')),
                 'optional_commit': bind(row.get('collection_commit')),
                 'prior_native_verification': bind(row.get('collection_verification'))}
        if not all(files[k]['available'] for k in ('native512_video', 'policy224_video', 'arrays', 'receipt')):
            raise ValueError(f'Required source input missing: {row["id"]}')
        native = video_check(files['native512_video']['path'], row['length'], row['fps'], case_dir)
        if native['decoded_dimensions'] != [[512, 512]]:
            raise ValueError('Expected native 512x512 reference stream')
        policy = video_check(files['policy224_video']['path'], row['length'], row['fps'])
        arrays = array_check(row['arrays'], row['length'], row['fps'])
        entry = {'input_id': row['id'], 'family': family, 'scope': 'inputs_only',
                 'training_admitted': False, 'synthetic_output_created': False,
                 'source_manifest': str(manifest), 'source_manifest_sha256': source_hash,
                 'source_episode_row_unchanged': row, 'source_files': files,
                 'source_row_flags_original': {k: v for k, v in row.items() if isinstance(v, bool)},
                 'source_receipt_flags_original': {k: v for k, v in receipt.items() if isinstance(v, bool)},
                 'flags_note': 'Historical pending/approval flags are copied verbatim; this preparation grants no new approval.',
                 'source_asset_specs': receipt.get('asset_specs', []),
                 'source_receptacle': receipt.get('receptacle'), 'selection_features': features,
                 'measured_arm_activity': motion, 'source_receipt_events_original': receipt.get('events', []),
                 'source_action_space_definition': source['action_spaces'][row['action_space']],
                 'native512_decode': native, 'policy224_decode': policy, 'arrays_metadata': arrays,
                 'optional_existing_QA_locators': {k: receipt.get(k) for k in
                                                  ('qa_visibility', 'qa_visibility_sha256', 'contact_data', 'contact_sha256')},
                 'preview_semantics': 'First/middle/last native frame; not reviewed stage keyframes.'}
        write(case_dir / 'input.json', entry)
        inputs.append(entry)
        print(json.dumps({'input_id': row['id'], 'decoded_frames': row['length'], 'fps': row['fps'], 'passed': True}), flush=True)
    if sha(manifest) != source_hash:
        raise ValueError('Source manifest changed during read-only preparation')
    doc = {'schema': 'r2h_native_input_pilot_v1', 'scope': 'inputs_only', 'training_admitted': False,
           'source_manifest': str(manifest), 'source_manifest_sha256': source_hash,
           'source_top_level_metadata_unchanged': {k: v for k, v in source.items() if k not in ('episodes', 'action_spaces')},
           'selection': coverage, 'inputs': inputs}
    write(output / 'input_manifest.json', doc)
    with (output / 'input_manifest.jsonl').open('x', encoding='utf-8') as handle:
        for entry in inputs:
            handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
    readme = '''# Robot-to-human方法验证：原生输入小样本\n\n本目录只有6条train机器人候选的元数据和18张原生PNG，尚未合成、调用API或加入训练。\n\n每类apple/book/cab各选2个不同group，优先覆盖任务资产和实际活动臂。柜门源候选只有同一种柜门资产和左活动臂，因此只覆盖不同布局，不声称两种资产/双臂。\n\n视频、arrays、receipt均留在源目录；这里没有复制完整视频/动作、生成新视频或改写源文件。input_manifest.json中source_files记录绝对路径、SHA256和声明哈希核验。全部六条的512和224视频逐帧解码，记录真实帧数、FPS、PTS摘要和native array字段/时间。\n\n每条PNG取0、N//2、N-1帧，不是已审核的抓取、接触或释放关键事件。stage runs和receipt events仅复用原控制器记录，不自动升级为视觉关键事件。\n\nsource_episode_row_unchanged及source_*_flags_original保留原flag，包括历史pending、production_approved、reference_pool_eligible等值。inputs_only/training_admitted=false仅约束本次输入用途；不改写源24条的历史状态，也不否定另行存在的训练reader审核。\n\n没有安装后端或检查密钥。后续视频编辑的动作/腕部标签是否仍然有效需要另行验证，不能直接继承。输出目录已存在时脚本默认拒绝覆盖。\n'''
    (output / 'README.md').write_text(readme, encoding='utf-8')
    report = {'passed': True, 'scope': 'inputs_only', 'training_admitted': False,
              'selected_inputs': len(inputs), 'selected_by_family': dict(Counter(x['family'] for x in inputs)),
              'distinct_groups': len({x['source_episode_row_unchanged']['group'] for x in inputs}),
              'full_video_decodes': len(inputs) * 2, 'native_preview_PNGs': len(inputs) * 3,
              'source_manifest_unchanged': True, 'source_video_or_action_modified': False,
              'full_video_copies': 0, 'synthetic_images_or_videos_generated': 0,
              'API_calls': 0, 'GPU_or_simulator_started': False,
              'python_executable': sys.executable,
              'producer': bind(__file__),
              'available_CPU_backend_modules': {k: importlib.util.find_spec(k) is not None for k in ('av', 'numpy', 'PIL')},
              'input_manifest': bind(output / 'input_manifest.json'),
              'input_manifest_jsonl': bind(output / 'input_manifest.jsonl'), 'README': bind(output / 'README.md')}
    write(output / 'VERIFICATION.json', report)
    print(json.dumps({'output': str(output), **report}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
