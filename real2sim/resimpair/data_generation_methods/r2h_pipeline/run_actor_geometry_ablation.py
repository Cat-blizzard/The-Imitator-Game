"""Run the predeclared 12-pose CPU grid; never call a paid provider or GPU."""
from __future__ import annotations
import argparse
from pathlib import Path
import time

from . import actor_geometry_candidate as geometry
from .batch_pipeline import binding
from .common import read_json, write_json


def run(package, collision_world, protocol, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new immutable ablation directory")
    output.mkdir(parents=True)
    started = time.monotonic()
    data = geometry.unified.validate_package(package)
    _, table, _ = geometry._load_world(data, collision_world)
    rows = []
    for angle in geometry.LEAN_ANGLES:
        for height in geometry.WAIST_HEIGHTS:
            label = f'lean_{angle:02d}_waist_{round(height*100):+03d}cm'
            directory = output / label
            directory.mkdir()
            rig_path = directory / 'rig.json'
            write_json(rig_path, geometry.nominal_rig(data, table, lean=angle, waist_height=height))
            receipt = geometry.compile_geometry(package, rig_path, collision_world, protocol,
                                                 directory / 'report.json')
            report = read_json(receipt['report']['path'])
            targets = report['event_targets'] + report['held_trajectory_targets']
            rows.append({'label': label, 'angle_degrees': angle, 'waist_height_m': height,
                         **receipt, 'clears_geometry_preflight': report['clears_geometry_preflight'],
                         'maximum_surface_proxy_distance_m': max(x['reach']['distance_m'] for x in targets),
                         'maximum_nearest_surface_distance_m': max(x['closest_surface_distance_m'] for x in targets),
                         'unreachable_surface_proxy_count': sum(x['reach']['status'] == 'point_unreachable' for x in targets),
                         'held_frames': report['held_frame_count'],
                         'torso_table_mesh_collision': report['torso_table']['mesh_collision'],
                         'projection_blocks': report['projection_blocks'],
                         'projection_risks': report['projection']['risks'],
                         'reject_reasons': report['reject_reasons']})
    baseline = next(row for row in rows if row['angle_degrees'] == 0 and row['waist_height_m'] == 0)
    config = {'anchor_model': read_json(Path(package).parent/'initial_request.json')['model']}
    wrapper = geometry.prepare_with_geometry(baseline['report']['path'], output/'baseline_prepared', config)
    # Demonstrate the real wrapper rejects before inspecting an absent budget or credentials.
    try:
        geometry.execute_with_geometry(output/'baseline_prepared/wrapper.json', config,
                                       output/'no_new_api_budget.json')
    except ValueError as exc:
        blocked = {'status': 'blocked_before_provider', 'reason': str(exc), 'new_paid_calls': 0}
    else:
        raise AssertionError('Expected the formal far-upright baseline to be rejected')
    summary = {'schema': 'r2h_actor_geometry_ablation_v1', 'source_package': binding(package),
               'candidate_implementation': binding(geometry.__file__), 'runner': binding(__file__),
               'protocol': binding(protocol), 'grid_size': 12, 'results': rows,
               'baseline_wrapper': binding(output/'baseline_prepared/wrapper.json'),
               'actual_baseline_execute_probe': blocked,
               'cleared_rigs': sum(row['clears_geometry_preflight'] for row in rows),
               'wall_seconds': time.monotonic()-started, 'api_calls': 0, 'gpu_runs': 0,
               'human_or_image_or_video_qualification': False}
    write_json(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('package', 'collision-world', 'protocol', 'output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    result = run(args.package, args.collision_world, args.protocol, args.output)
    print({'grid_size': result['grid_size'], 'cleared_rigs': result['cleared_rigs'],
           'wall_seconds': result['wall_seconds'], 'api_calls': 0})
