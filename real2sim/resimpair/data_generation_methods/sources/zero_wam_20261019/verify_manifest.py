"""Reproduce statistics from the unmodified public HumanGen Robotwin metadata.

Run with PolicyGen/.venv/bin/python verify_manifest.py. No network, PyTorch,
models, video data, simulator, or dataset modification is used.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def inspect(path):
    raw = path.read_bytes()
    document = json.loads(raw)
    samples = document['samples']
    human_uses = Counter(row['human_video_path'] for row in samples)
    human_tasks = defaultdict(set)
    per_task = Counter()
    for row in samples:
        task = row['status_info']['video_rel_path'].split('/')[0]
        human_tasks[row['human_video_path']].add(task)
        per_task[task] += 1
    return {
        'manifest_sha256': hashlib.sha256(raw).hexdigest(),
        'actual_pair_rows': len(samples),
        'unique_robot_video_paths': len({row['status_info']['video_rel_path'] for row in samples}),
        'unique_human_video_paths': len(human_uses),
        'task_roots': len(per_task),
        'human_paths_used_for_multiple_robot_rows': sum(n > 1 for n in human_uses.values()),
        'human_reuse_histogram_uses_to_human_paths': dict(sorted(Counter(human_uses.values()).items())),
        'human_paths_crossing_task_roots': sum(len(tasks) > 1 for tasks in human_tasks.values()),
        'published_score_histogram': dict(sorted(Counter(
            f"semantic={row['semantic_score']},physics={row['physics_score']}" for row in samples).items())),
        'generation_filename_histogram': dict(sorted(Counter(
            Path(row['human_video_path']).name for row in samples).items())),
        'target_duration_seconds_histogram': dict(sorted(Counter(
            row['status_info']['target_video_duration_sec'] for row in samples).items())),
        'header_plan_total_samples': document['stats']['total_samples'],
        'header_existing_pairs': document['stats']['existing_pair_samples'],
        'header_supplemental_pairs': document['stats']['supplemental_samples'],
        'header_qualified_rows': document['stats']['qualified'],
        'reuse_rule_is_explicit': 'reuse a random generated human video from the same task' in document['selection_rule'],
        'actual_pair_counts_by_task_root': dict(sorted(per_task.items())),
        'limits': [
            'A unique path is not an independently decoded or content-hashed human video.',
            'Same task root does not establish fine-grained object, direction, or sequence equivalence.',
            'All surviving scores being 5 does not recover rejection rates or certify a calibrated physical verifier.',
            'Metadata checks do not constitute video quality or action-label validation.'
        ],
    }


if __name__ == '__main__':
    result = inspect(Path(__file__).with_name('ICL_config_robotwin.json'))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
