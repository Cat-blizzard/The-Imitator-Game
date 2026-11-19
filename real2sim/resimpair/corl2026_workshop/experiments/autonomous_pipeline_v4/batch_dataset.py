"""Materialize a traceable index from independent admission, never worker flags."""
from __future__ import annotations

import argparse
from collections import Counter
import html
from pathlib import Path
import json

from .core import bind, checked, read, require, write, write_bytes


def gallery(output, index):
    """A local inspection page backed only by the admitted dataset index."""
    cards = []
    for row in index['items']:
        case = html.escape(row['case'], quote=True)
        manifest = read(checked(row['manifest']))
        task = read(checked(manifest['files']['task.json']))
        label = html.escape(str(task.get('goal_description', task.get('goal', task.get('family', row['mechanism'])))))
        views = ''.join(f'<div><p>{title}</p><video controls preload="none" src="{case}/{name}"></video></div>'
                        for name, title in [('simulation_opposite.mp4', '桌面对侧'),
                                            ('simulation_robot_side.mp4', '机器人一侧')])
        cards.append(f'<details data-kind="{html.escape(row["mechanism"], quote=True)}"><summary>{case}</summary>'
                     f'<p>{label} · {html.escape(row["partition"])} · {row["seconds"]:.1f} 秒</p>'
                     f'<div class="views">{views}</div><a href="{case}/manifest.json">动作、状态、相机及验收绑定</a></details>')
    options = ''.join(f'<option>{html.escape(k)}</option>' for k in sorted(index['counts']['by_mechanism']))
    counts = index['counts']
    page = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>批量仿真任务数据</title>'
            '<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 20px;line-height:1.65;background:#f5f6f8;color:#1b2938}'
            'details{margin:14px 0;padding:16px;background:white;border:1px solid #cdd7e1;border-radius:8px}summary{cursor:pointer;overflow-wrap:anywhere}'
            '.views{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}video{width:100%}a{color:#125b99}select{padding:8px}</style>'
            f'<h1>已通过独立复查的仿真任务</h1><p>{counts["qualified_native_executions"]} 次执行 · '
            f'{counts["scene_instances"]} 个场景配置 · {counts["mechanism_families"]} 类机制 · '
            f'{counts["scene_task_programs"]} 个场景任务组合。每项两路视频来自同一次物理执行。</p>'
            '<p>本页仅包含仿真合格数据。新机制尚无真实人类来源绑定；人类生成和审核另行统计。速度／高度变体不计作新任务种类。</p>'
            '<p><a href="index.json">完整索引与排除记录</a></p><label>机制 <select id="kind"><option value="">全部</option>'
            + options + '</select></label>' + ''.join(cards)
            + '<script>document.getElementById("kind").onchange=function(){document.querySelectorAll("details").forEach(e=>e.hidden=!!this.value&&e.dataset.kind!==this.value)}</script></html>')
    write_bytes(Path(output) / 'index.html', page.encode('utf-8'))


def build(root, audit, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    report = read(audit)
    require(report['schema'] == 'batch_native_independent_acceptance_20261025_v1', 'batch_audit_required')
    require(Path(report['root']).resolve() == root, 'audit_root_mismatch')
    require(not output.exists(), 'dataset_revision_must_be_new')
    output.mkdir(parents=True)
    holdback_path = root / 'admission_holdbacks.json'
    holdbacks = {}
    if holdback_path.exists():
        updates = read(holdback_path)
        checked(updates['evidence'])
        holdbacks = {(r['case'], r['attempt']): r for r in updates['case_attempts']}
    rows = []; failures = []
    for row in report['cases']:
        heldback = holdbacks.get((row['case'], row.get('attempt')))
        if heldback:
            checked(heldback['execution']); checked(heldback['revised_protocol'])
            failures.append({'case': row['case'], 'attempt': row.get('attempt'),
                'status': 'held_back_under_revised_rule', 'errors': [heldback['reason']],
                'current_rule_evidence': bind(holdback_path), 'old_audit_unchanged': True})
            continue
        if not row['independent_qualified']:
            failures.append({'case': row['case'], 'attempt': row.get('attempt'),
                             'status': row['status'], 'errors': row['errors']})
            continue
        bundle = read(checked(row['execution']))
        case = row['case_record']; folder = output / case['id']
        task = read(checked(case['task']))
        folder.mkdir()
        files = {'simulation_opposite.mp4': bundle['cameras']['opposite_high']['video'],
                 'simulation_robot_side.mp4': bundle['cameras']['robot_side']['video'],
                 'actions_states.npz': bundle['measurements'],
                 'event_protocol.json': bundle['event_protocol'],
                 'execution_bundle.json': row['execution'],
                 'scene.json': case['scene'], 'task.json': case['task']}
        for name, bound in files.items():
            (folder / name).symlink_to(checked(bound))
        manifest = {'schema': 'resimpair_native_dataset_item_v1', 'case': case['id'],
            'attempt': row['attempt'], 'source_coverage': 'missing', 'source_human_video': None,
            'synthetic_human_video': None, 'formal_real_sim_human_pair': False,
            'native_qualified': True, 'independent_audit': bind(audit),
            'split_group': case['split_group'], 'partition': case['partition'],
            'semantic_key': case['semantic_key'],
            'trajectory_variation': task.get('trajectory_variation'),
            'physical_scene_sha256': row['physical_scene_sha256'],
            'raw_robot_dynamic_sha256': row['raw_robot_dynamic_sha256'],
            'quantized_robot_dynamic_sha256': row['quantized_robot_dynamic_sha256'],
            'frames': row['frames'], 'control_hz': bundle['cameras']['opposite_high']['fps'],
            'cameras': bundle['cameras'], 'files': files,
            'native_worker_seconds': row['worker_terminal']['elapsed_seconds'],
            'human_generation_is_a_separate_queue': True, 'training_started': False}
        write(folder / 'manifest.json', manifest)
        rows.append({'case': case['id'], 'manifest': bind(folder / 'manifest.json'),
                     'mechanism': case['group'], 'partition': case['partition'],
                     'scene': case['scene_instance_id'], 'seconds': manifest['native_worker_seconds'],
                     'semantic_key': case['semantic_key'],
                     'trajectory_variation': task.get('trajectory_variation'),
                     'dynamic_fingerprint': manifest['quantized_robot_dynamic_sha256']})
    result = {'schema': 'resimpair_native_batch_index_v1', 'independent_audit': bind(audit),
              'root': str(root), 'items': rows, 'failures_and_unexecuted': failures,
              'rule_updates': bind(holdback_path) if holdback_path.exists() else None,
              'counts': {'qualified_native_executions': len(rows),
                         'unique_physical_robot_dynamics': len({r['dynamic_fingerprint'] for r in rows}),
                         'scene_instances': len({r['scene'] for r in rows}),
                         'mechanism_families': len({r['mechanism'] for r in rows}),
                         'scene_task_programs': len({(r['scene'], r['semantic_key']) for r in rows}),
                         'semantic_task_programs_across_layouts': len({r['semantic_key'] for r in rows}),
                         'executed_trajectory_variants': sum(bool(r['trajectory_variation'] and r['trajectory_variation']['index']) for r in rows),
                         'by_mechanism': dict(Counter(r['mechanism'] for r in rows)),
                         'by_partition': dict(Counter(r['partition'] for r in rows)),
                         'camera_streams': 2*len(rows), 'human_videos': 0, 'formal_pairs': 0}}
    write(output / 'index.json', result)
    gallery(output, result)
    return {'output': str(output), 'counts': result['counts']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--audit', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(build(args.root, args.audit, args.output), ensure_ascii=False))
