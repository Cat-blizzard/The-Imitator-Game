"""Four predeclared multi-operation examples from actual audited CPU rollouts.

Reuse the frozen saved-state renderer and GPU-sharing scheduler. Preserve the
original scene, both cameras, every state, and the native clock. Extract all
operation events from the encoded videos afterward; no R2H quality gate,
human condition export, model call, or new physical execution is performed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import os
from pathlib import Path
import shutil
import time

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import deferred_render_v1 as original
from . import scene_quality_render_v3 as render

VERSION = 'multitask_native_replay_examples_v1'
ALLOCATION_ID = 'multitask_10x8_video_examples_20261030_v1'
REGISTRY = ROOT / 'runtime/multitask_render_registry_v1'
PRIOR_BUDGET = ROOT / 'runs/unified_pickplace_render_20261030_v1/budget/limits.json'
GROUPS = [('collect', ('collect_group',)), ('mixed', ('split_group', 'distribute_group')),
          ('clear', ('clear_then_place',)), ('arrange', ('arrange_group',))]
EVENTS = ('grasp', 'lift', 'arrive', 'release', 'supported', 'withdraw')
LABELS = {'collect': '多物收集', 'mixed': '混合目标／分装', 'clear': '腾空位置后放置', 'arrange': '多物排列',
          'grasp': '抓住', 'lift': '抬起', 'arrive': '到达', 'release': '释放', 'supported': '获得支撑', 'withdraw': '退开'}


def select_sources(audit):
    """Select the first accepted request per structure in the frozen order."""
    main = [r for r in audit['rows'] if r['arm'] == 'candidate']
    if len(main) != 80 or not all(r.get('terminal') for r in main):
        raise ValueError('all_80_candidate_requests_must_be_resolved_before_example_selection')
    selected = []; missing = []
    for group, structures in GROUPS:
        matches = [r for r in main if r['task_structure'] in structures and r.get('dataset_physical_passed') is True]
        if matches:
            selected.append({'group': group, 'audit_row': matches[0]})
        else:
            missing.append({'group': group, 'structures': list(structures), 'status': 'no_actual_qualified_source'})
    return selected, missing


def event_keyframes(task, event_frames, frame_count):
    event_map = {row['id']: row['frame'] for row in event_frames}
    operations = []; indices = {0, frame_count - 1}
    if len(task['operations']) < 2:
        raise ValueError('multi_operation_example_required')
    for op in task['operations']:
        values = {name: event_map.get(op['id'] + '.' + name) for name in EVENTS}
        if any(type(f) is not int or not 0 <= f < frame_count for f in values.values()):
            raise ValueError('each_operation_requires_all_six_native_event_frames')
        if not (values['grasp'] <= values['lift'] <= values['arrive'] <= values['release'] <= values['withdraw']):
            raise ValueError('native_operation_event_order_invalid')
        indices.update(values.values())
        operations.append({'operation_id': op['id'], 'object_id': op['object'], 'arm': op['arm'],
            'target': op['target'], 'frames': values})
    operations.sort(key=lambda r: (r['frames']['grasp'], r['operation_id']))
    return {'frame_indices': sorted(indices), 'operations_in_measured_time_order': operations,
            'native_event_rule_unchanged': True, 'source': 'retained audit.json event_frames',
            'visibility_or_human_generation_qualification': False}


def register_budget(root, source_count, prior_budget=PRIOR_BUDGET, registry=REGISTRY):
    prior_budget = Path(prior_budget).resolve(); registry = Path(registry)
    prior = read(prior_budget)
    history = prior['historical_actual_render_jobs'] + len(list((prior_budget.parent / 'reservations').glob('*.json')))
    if prior['maximum_lineage_render_jobs_including_history'] != 60 or history > 60:
        raise ValueError('prior_render_lineage_60_required')
    if not 1 <= source_count <= 4: raise ValueError('one_to_four_actual_sources_required')
    registry.mkdir(parents=True, exist_ok=True)
    with (registry / '.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        old = [read(p) for p in (registry / 'registrations').glob('*.json')]
        same = [r for r in old if r['allocation_id'] == ALLOCATION_ID]
        if any(r['allocation_limit'] != 4 for r in same) or sum(r['maximum_attempts'] for r in same) + source_count > 4:
            raise ValueError('same_render_allowance_cannot_reset_with_new_directory')
        for row in same: resolve(row['budget'])
        limits = {'schema': VERSION, 'maximum_attempts': source_count,
            'allocation_id': ALLOCATION_ID, 'allocation_limit': 4,
            'allocation': {'screen': 0, 'video': source_count},
            'prior_budget': binding(prior_budget), 'historical_actual_render_jobs': history,
            'prior_lineage_limit': 60, 'maximum_lineage_render_jobs_including_history': 64,
            'unspent_prior_allowance_not_reassigned': True,
            'physical_execution_limit': 0, 'api_limit': 0, 'gpu': 2, 'minimum_free_gpu_mib': 12288,
            'parallel_workers': 1, 'timeout_seconds': {'screen': 240, 'video': 240},
            'resource_wait_seconds': 180, 'automatic_retries': 0,
            'unknown_submission_must_be_recovered': True}
        write(root / 'budget/limits.json', limits)
        record = {'allocation_id': ALLOCATION_ID, 'allocation_limit': 4, 'maximum_attempts': source_count,
                  'root': str(root), 'budget': binding(root / 'budget/limits.json'), 'created_unix': time.time()}
        target = registry / 'registrations' / (digest([ALLOCATION_ID, str(root)]) + '.json')
        write(target, record)
    return binding(target), limits


def prepare(audit_path, output, *, prior_budget=PRIOR_BUDGET):
    audit_path, output = Path(audit_path).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('preserve_existing_multitask_render_directory')
    if shutil.disk_usage(ROOT).free < 10 * 2**30: raise ValueError('insufficient_disk_preserve_history')
    audit = read(audit_path); selected, missing = select_sources(audit)
    if not selected: raise ValueError('no_actual_qualified_multi_operation_source')
    independent = read(resolve(audit['protocol']))
    rows = {(r['arm'], r['case_id']): r for r in independent['rows']}
    values = []; source_rows = []; cases = []
    for i, choice in enumerate(selected):
        audited = choice['audit_row']; original_row = rows['candidate', audited['case_id']]
        source = original.validate_source(original_row['directory'])
        if audited.get('bundle') != source['bindings']['execution_bundle.json']:
            raise ValueError('audited_source_bundle_changed')
        task = source['task']; keys = event_keyframes(task, read(source['source'] / 'audit.json')['event_frames'], len(source['states']))
        row = {'id': f'source_{i:02d}', 'group': choice['group'], 'source_case_id': audited['case_id'],
            'task_structure': audited['task_structure'], 'operation_count': len(task['operations']),
            'source_execution': source['bindings']['execution_bundle.json'], 'artifacts': source['bindings'],
            'source_execution_code': source['bundle']['code'], 'dataset_partition': original_row['case']['partition'],
            'split_group': original_row['case']['split_group'], 'frame_count': len(source['states']),
            'fps': float(source['fps']), 'event_keyframes': keys}
        case_id = f'video_{i:02d}_{choice["group"]}'; folder = output / 'cases' / case_id
        scene, repair = render.baseline_scene(source)
        render.validate_visual_derivative(source['scene'], scene, 'baseline')
        write(folder / 'render_scene.json', scene); write(folder / 'source_artifacts.json', source['bindings'])
        write(folder / 'operation_keyframes.json', keys)
        for name in original.COPY_EVIDENCE: shutil.copy2(source['source'] / name, folder / name)
        case = {'id': case_id, 'source_id': row['id'], 'stage': 'video', 'variant': 'baseline',
            'scene': binding(folder / 'render_scene.json'), 'source_execution': row['source_execution'],
            'source_partition': row['dataset_partition'], 'split_group': row['split_group'],
            'predeclared_control': True, 'screen_case_id': None,
            'visualization_only': True, 'not_a_single_operation_quality_screen': True,
            'state_indices': list(range(len(source['states']))), 'event_keyframes': binding(folder / 'operation_keyframes.json')}
        if repair:
            write(folder / 'metadata_repair.json', repair); case['metadata_repair'] = binding(folder / 'metadata_repair.json')
        values.append(source); source_rows.append(row); cases.append(case)
    receipt, limits = register_budget(output, len(cases), prior_budget)
    selection = {'schema': VERSION, 'author': 'root_adapter_using_independently_accepted_physics',
        'source_audit': binding(audit_path), 'source_protocol': audit['protocol'],
        'selection_policy': 'first_qualified_candidate_in_original_request_order_per_predeclared_structure',
        'requested_structure_groups': [{'id': g, 'task_structures': list(s)} for g, s in GROUPS],
        'missing_structure_groups': missing, 'sources': source_rows, 'cases': cases,
        'predeclared_control_video_case_ids': [c['id'] for c in cases],
        'qualification_scope': 'actual qualified simulation replay for human inspection only',
        'uses_single_operation_R2H_gate': False, 'human_generation_ready': False,
        'new_physical_executions': 0, 'API_calls': 0}
    write(output / 'selection_protocol.json', selection)
    modules = [Path(__file__), Path(render.__file__), Path(original.__file__),
        ROOT / 'resimpair_tasks/scene_quality_v3.py', ROOT / 'resimpair_tasks/scene_domain_v2.py',
        ROOT / 'resimpair_tasks/scene_material_policy_v3.py', ROOT / 'resimpair_tasks/appearance_v1.py',
        ROOT / 'resimpair_pickplace/environment.py', ROOT / 'resimpair_tasks/cudex_shared_gpu_v1.py',
        ROOT / 'corl2026_workshop/experiments/autonomous_pipeline_v4/simulation.py',
        ROOT / 'corl2026_workshop/experiments/autonomous_pipeline_v4/resources.py',
        ROOT / 'PolicyGen/mani_skill/render/shaders.py', ROOT / 'PolicyGen/mani_skill/sensors/camera.py']
    protocol = {'schema': render.VERSION, 'multitask_adapter': binding(__file__),
        'independent_protocol': binding(output / 'selection_protocol.json'),
        'selection_protocol_author': 'root; only upstream source physics audit is independent',
        'sources': source_rows, 'cases': cases, 'budget': binding(output / 'budget/limits.json'),
        'budget_registration': receipt, 'code': {str(p.relative_to(ROOT)): binding(p) for p in modules},
        'state_roundtrip_tolerance': 1e-6, 'TCP_tolerance_m': 1e-5,
        'physical_execution_count': 0, 'native_execution': False, 'new_trajectory': False, 'api_calls': 0,
        'full_state_videos_without_resampling': True, 'raw_entity_segmentation_all_rendered_frames': True,
        'single_operation_quality_gate_used': False, 'R2H_conditions_exported': 0, 'default_promoted': False}
    write(output / 'protocol.json', protocol)
    refs = {'protocol': binding(output / 'protocol.json')}
    for i, ref in enumerate(render.bindings_in(protocol)):
        resolve(ref); refs[f'dependency_{i}'] = ref
    for case in cases:
        for ref in render.bindings_in(read(resolve(case['scene']))):
            resolve(ref); refs['scene_resource_' + digest(ref)] = ref
    write(output / 'frozen_inputs.json', refs)
    render.verify(output)
    report = {'schema': VERSION, 'selected_sources': len(cases), 'missing_structure_groups': missing,
        'maximum_render_jobs': limits['maximum_attempts'], 'lineage_render_ceiling': 64,
        'new_render_jobs': 0, 'new_physical_executions': 0, 'API_calls': 0,
        'preserved_original_appearance_and_cameras': True, 'human_generation_ready': False}
    write(output / 'prepared.json', report); return report


def validate_frame_mapping(mapping, source):
    rows = mapping['frames']; n = len(source['states'])
    if (len(rows) != n or mapping['env_step_calls'] != 0 or not mapping['all_states_rendered_once']
            or not mapping['original_actions_and_timeline_preserved']):
        raise ValueError('complete_native_state_replay_required')
    for i, row in enumerate(rows):
        if row['frame'] != i or row['source_state_index'] != i or set(row['cameras']) != {'robot_side', 'opposite_high'}:
            raise ValueError('source_state_camera_frame_mapping_changed')
        if abs(row['source_timestamp_seconds'] - source['timestamps'][i]) > 1e-10:
            raise ValueError('source_timestamp_changed')
        expected = hashlib.sha256(np.ascontiguousarray(source['states'][i]).tobytes()).hexdigest()
        if row['state_sha256'] != expected: raise ValueError('rendered_state_sha_changed')
        render.validate_restored(row['errors'], 1e-6, 1e-5)


def decode_keyframes(video, output, indices, timestamps, camera):
    import av
    from PIL import Image
    selected = set(indices); count = 0; evidence = []
    output.mkdir(parents=True, exist_ok=True)
    with av.open(str(video)) as container:
        if len(container.streams.video) != 1: raise ValueError('one_encoded_video_stream_required')
        for frame in container.decode(video=0):
            if count >= len(timestamps) or frame.pts is None:
                raise ValueError('encoded_frame_count_or_timestamp_invalid')
            timestamp = float(frame.pts * frame.time_base)
            if abs(timestamp - float(timestamps[count])) > 1e-7:
                raise ValueError('encoded_PTS_differs_from_native_clock')
            if count in selected:
                target = output / f'{camera}_state_{count:06d}.png'
                if target.exists(): raise ValueError('preserve_existing_keyframe_image')
                Image.fromarray(frame.to_ndarray(format='rgb24')).save(target)
                evidence.append({'source_state_index': count, 'timestamp_seconds': timestamp, 'image': binding(target),
                    'origin': 'decoded_from_exact_frame_in_encoded_simulation_video'})
            count += 1
    if count != len(timestamps) or len(evidence) != len(selected):
        raise ValueError('encoded_native_clock_video_incomplete')
    return {'video': binding(video), 'decoded_frame_count': count, 'clock_matches_all_source_timestamps': True,
            'images': evidence}


def postprocess(root):
    root = Path(root).resolve(); protocol, _ = render.verify(root)
    if protocol['multitask_adapter'] != binding(__file__): raise ValueError('multitask_render_adapter_changed')
    rows = []
    for case in protocol['cases']:
        folder = root / 'cases' / case['id']; terminal_path = folder / 'terminal.json'
        if not terminal_path.exists():
            rows.append({'case_id': case['id'], 'status': 'not_rendered'}); continue
        terminal = read(terminal_path)
        if terminal['returncode'] != 0:
            rows.append({'case_id': case['id'], 'status': 'render_failed', 'terminal': binding(terminal_path)}); continue
        proof_path = folder / 'media_integrity.json'
        if proof_path.exists():
            rows.append({'case_id': case['id'], 'status': 'media_verified', 'proof': binding(proof_path)}); continue
        source_row = next(s for s in protocol['sources'] if s['id'] == case['source_id'])
        source = original.validate_source(resolve(source_row['source_execution']).parent)
        validate_frame_mapping(read(folder / 'frame_mapping.json'), source)
        result = read(folder / 'worker_result.json'); keys = read(resolve(case['event_keyframes']))
        if result['frames_per_camera'] != len(source['states']) or result['env_step_calls'] != 0:
            raise ValueError('render_worker_counts_changed')
        cameras = {name: decode_keyframes(folder / (name + '.mp4'), folder / 'keyframes',
                   keys['frame_indices'], source['timestamps'], name) for name in ('robot_side', 'opposite_high')}
        proof = {'schema': VERSION, 'source_execution': source_row['source_execution'],
            'frame_mapping': binding(folder / 'frame_mapping.json'), 'worker_result': binding(folder / 'worker_result.json'),
            'operation_events': case['event_keyframes'], 'cameras': cameras,
            'all_operations_have_keyframes': True, 'physical_execution_count': 0,
            'quality_scope': 'state/frame/clock integrity only; no new visual or human quality acceptance',
            'human_generation_ready': False, 'API_calls': 0}
        write(proof_path, proof); rows.append({'case_id': case['id'], 'status': 'media_verified', 'proof': binding(proof_path)})
    report = {'schema': VERSION, 'protocol': binding(root / 'protocol.json'), 'outputs': rows,
        'render_attempts': len(list((root / 'budget/reservations').glob('*.json'))),
        'successful_dual_view_sources': sum(r['status'] == 'media_verified' for r in rows),
        'new_physical_executions': 0, 'API_calls': 0, 'human_videos': 0, 'human_conditions': 0}
    if all(r['status'] != 'not_rendered' for r in rows): write(root / 'complete.json', report)
    return report


def gallery(root, output):
    root, output = Path(root).resolve(), Path(output).resolve(); protocol, _ = render.verify(root)
    if output.exists(): raise ValueError('preserve_existing_gallery')
    esc = html.escape
    def link(p): return esc(os.path.relpath(p, output.parent), quote=True)
    sections = []
    for case in protocol['cases']:
        folder = root / 'cases' / case['id']; source = next(s for s in protocol['sources'] if s['id'] == case['source_id'])
        proof = read(folder / 'media_integrity.json') if (folder / 'media_integrity.json').exists() else None
        content = f'<section><h2>{esc(LABELS[source["group"]])}</h2><p>{esc(source["source_case_id"])} · {source["operation_count"]} 次操作 · {source["frame_count"]} 帧 / {source["fps"]:g} FPS</p>'
        if not proof:
            content += '<p>尚无完整视频及时间轴核验结果。</p></section>'; sections.append(content); continue
        content += '<div class="videos">' + ''.join(f'<figure><video controls preload="metadata" src="{link(folder / (name + ".mp4"))}"></video><figcaption>{label}</figcaption></figure>' for name, label in [('robot_side', '机械臂侧'), ('opposite_high', '桌子对面')]) + '</div>'
        images = {row['source_state_index']: row for row in proof['cameras']['opposite_high']['images']}
        for step, operation in enumerate(source['event_keyframes']['operations_in_measured_time_order'], 1):
            target = operation['target']; content += f'<h3>步骤 {step}：{esc(operation["object_id"])} → {esc(target["support"])}（机械臂 {operation["arm"]}）</h3><div class="frames">'
            for name in ('grasp', 'lift', 'release', 'withdraw'):
                row = images[operation['frames'][name]]
                content += f'<figure><img loading="lazy" src="{link(resolve(row["image"]))}"><figcaption>{esc(LABELS[name])} · {row["timestamp_seconds"]:.2f}s · 帧 {row["source_state_index"]}</figcaption></figure>'
            content += '</div>'
        content += f'<p><a href="{link(folder / "media_integrity.json")}">来源、状态与时间轴核验</a></p></section>'; sections.append(content)
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>多物体、多任务仿真样例</title><style>body{font:16px/1.65 "Noto Sans CJK SC","Microsoft YaHei",sans-serif;background:#f0f3f6;color:#192d40;margin:0}main{max-width:1400px;margin:auto;padding:24px}header,section{background:white;padding:24px;margin-bottom:24px;border-radius:10px}.videos{display:grid;grid-template-columns:1fr 1fr;gap:16px}.frames{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}figure{margin:0}video,img{width:100%;background:#dfe6ed}figcaption{font-size:13px}h2{margin-top:0}p{overflow-wrap:anywhere}@media(max-width:800px){.videos,.frames{grid-template-columns:1fr 1fr}}a{color:#145e98}</style><main><header><h1>一个场景，多物体与组合任务</h1><p>从本轮实际独立审核通过的轨迹中，按预定任务结构选择首个成功例。双视角复用同一次物理执行的全部保存状态，保留原外观与原始时间轴。每步截图来自视频对应帧。</p><p>本页是仿真回放展示；不是总体成功率证明，也未生成人类视频或人类生成条件。新物理执行 0，API 调用 0。</p></header>' + ''.join(sections) + '</main></html>'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream: stream.write(document)
    return {'gallery': binding(output), 'sources': len(protocol['sources']), 'new_render_jobs': 0}


def run(root, limit=4):
    root = Path(root).resolve(); protocol, limits = render.verify(root)
    if protocol['multitask_adapter'] != binding(__file__): raise ValueError('multitask_adapter_changed')
    if type(limit) is not int or not 1 <= limit <= 4: raise ValueError('finite_maximum_four_video_jobs_required')
    result = render.run(root, 'video', min(limit, limits['allocation']['video']), video_group='controls')
    proof = postprocess(root)
    return {'render': result, 'media': proof}


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--audit', type=Path, required=True); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prior-budget', type=Path, default=PRIOR_BUDGET)
    r = sub.add_parser('run'); r.add_argument('--root', type=Path, required=True); r.add_argument('--limit', type=int, default=4)
    post = sub.add_parser('postprocess'); post.add_argument('--root', type=Path, required=True)
    g = sub.add_parser('gallery'); g.add_argument('--root', type=Path, required=True); g.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.audit, args.root, prior_budget=args.prior_budget) if args.command == 'prepare' else (
        run(args.root, args.limit) if args.command == 'run' else postprocess(args.root) if args.command == 'postprocess' else gallery(args.root, args.output))
    print(result)


if __name__ == '__main__': main()
