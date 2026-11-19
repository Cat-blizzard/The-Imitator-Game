"""Pure saved-point diagnostic; no task assignment or candidate promotion."""
from pathlib import Path
import numpy as np

from data_generation_methods.r2h_pipeline.common import read_json, write_json
from data_generation_methods.r2h_pipeline.batch_pipeline import binding, checked_binding

ROOT = Path('/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v3/actor_geometry/hand_assignment_diagnostic')


def coverage(rows):
    n = len(rows)
    return {'frames': n,
            'left_covered_frames': sum(x['left_in_necessary_interval'] for x in rows),
            'right_covered_frames': sum(x['right_in_necessary_interval'] for x in rows),
            'either_covered_frames_optimistic': sum(x['either_in_necessary_interval'] for x in rows),
            'neither_covered_frames': sum(not x['either_in_necessary_interval'] for x in rows),
            'fixed_left_covers_all_proxy_points': all(x['left_in_necessary_interval'] for x in rows),
            'fixed_right_covers_all_proxy_points': all(x['right_in_necessary_interval'] for x in rows),
            'per_frame_either_covers_all_proxy_points_optimistic': all(x['either_in_necessary_interval'] for x in rows),
            'max_left_distance_m': max(x['left_distance_m'] for x in rows),
            'max_right_distance_m': max(x['right_distance_m'] for x in rows),
            'max_per_frame_min_distance_m': max(min(x['left_distance_m'], x['right_distance_m']) for x in rows)}


def run():
    protocol = read_json(ROOT/'protocol.json')
    summary_rows = []
    for camera, bound in protocol['sources'].items():
        checked_binding(bound)
        source = read_json(bound['path'])
        for source_row in source['results']:
            checked_binding(source_row['report'])
            report = read_json(source_row['report']['path'])
            for evidence in report['bindings'].values():
                checked_binding(evidence)
            shoulders = {k: np.asarray(v) for k, v in report['shoulders'].items()}
            frames = {}
            held = set()
            for r in report['event_targets'] + report['held_trajectory_targets']:
                index = r['frame_index_0based']
                if index in frames:
                    assert frames[index]['target_world_m'] == r['target_world_m']
                else:
                    frames[index] = {'target_world_m': r['target_world_m'], 'event_ids': []}
                frames[index]['event_ids'].append(r['event_id'])
                if r['event_id'] == 'measured_held_object_trajectory':
                    held.add(index)
            rows = []
            for index, r in sorted(frames.items()):
                point = np.asarray(r['target_world_m'])
                dl = float(np.linalg.norm(point-shoulders['left']))
                dr = float(np.linalg.norm(point-shoulders['right']))
                left = .04-1e-8 <= dl <= .56+1e-8
                right = .04-1e-8 <= dr <= .56+1e-8
                rows.append({'frame_index_0based': index, **r, 'in_held_interval': index in held,
                             'left_distance_m': dl, 'right_distance_m': dr,
                             'left_in_necessary_interval': left, 'right_in_necessary_interval': right,
                             'either_in_necessary_interval': left or right})
            path = ROOT/camera/(source_row['label']+'.json')
            result = {'schema': 'r2h_saved_proxy_hand_assignment_diagnostic_v1',
                'camera': camera, 'label': source_row['label'], 'source_report': source_row['report'],
                'shoulders': report['shoulders'], 'unique_frame_union': coverage(rows),
                'held_frames_only': coverage([r for r in rows if r['in_held_interval']]),
                'frames': rows, 'unchanged_torso_mesh_collision': report['torso_table']['mesh_collision'],
                'unchanged_framing_blocks': report['projection_blocks'],
                'actual_plan_assignment_unchanged': 'right',
                'hypothetical_fixed_left_is_not_new_plan': True,
                'handover_switching_contact_collision_and_continuous_execution': 'unknown',
                'can_generate_video': False, 'quality_release_approved': False,
                'rig_promoted': False, 'api_calls': 0, 'gpu_runs': 0}
            write_json(path, result)
            summary_rows.append({'camera': camera, 'label': source_row['label'],
                'angle_degrees': source_row['angle_degrees'], 'waist_height_m': source_row['waist_height_m'],
                'diagnostic': binding(path), 'coverage': result['unique_frame_union'],
                'held_frames_only': result['held_frames_only'],
                'torso_collision': result['unchanged_torso_mesh_collision'],
                'framing_blocks': result['unchanged_framing_blocks']})
    result = {'schema': 'r2h_hand_assignment_diagnostic_summary_v1',
        'protocol': binding(ROOT/'protocol.json'), 'implementation': binding(__file__),
        'results': summary_rows, 'can_generate_video': False,
        'rigs_promoted': 0, 'api_calls': 0, 'gpu_runs': 0}
    write_json(ROOT/'summary.json', result)
    for camera in ('opposite', 'native'):
        rows = [x for x in summary_rows if x['camera'] == camera]
        print(camera, {key: sum(x['coverage'][key] for x in rows) for key in (
            'fixed_left_covers_all_proxy_points', 'fixed_right_covers_all_proxy_points',
            'per_frame_either_covers_all_proxy_points_optimistic')})


if __name__ == '__main__':
    run()
