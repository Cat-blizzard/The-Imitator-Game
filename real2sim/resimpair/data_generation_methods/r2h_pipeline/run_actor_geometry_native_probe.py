"""Recorded native-camera counterexample; fixed geometry, no API/GPU budget."""
from __future__ import annotations
from pathlib import Path
import time

from . import actor_geometry_candidate as g
from . import actor_geometry_preparation_v2 as prep
from . import sim_state_conditions as conditions
from . import unified_anchor_candidate as unified
from .batch_pipeline import binding, checked_binding
from .common import read_json, write_json

ROOT = Path('/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v3/actor_geometry/native_camera_probe')


def run():
    started = time.monotonic()
    protocol = read_json(ROOT/'protocol.json')
    for b in protocol['code_bindings'].values():
        checked_binding(b)
    original_package = protocol['source_package']['path']
    original = unified.validate_package(original_package)
    original_packet = original['packet']
    receipt_path = original_packet['bindings']['receipt']['path']
    verification_path = original_packet['bindings']['verification']['path']
    packet = conditions.compile_conditions(receipt_path, verification_path, 'native',
                                           ROOT/'conditions', partition='development')
    if packet['status'] != 'prepared_for_condition_image_review':
        write_json(ROOT/'failure.json', {'stage': 'condition_compilation', 'result': packet,
                                         'api_calls': 0, 'gpu_runs': 0})
        return
    assert packet['execution_id'] == original_packet['execution_id']
    visibility = []
    for c in packet['conditions']:
        opposite = next(x for x in original_packet['conditions'] if x['labels'] == c['labels'])
        visibility.append({'labels': c['labels'], 'frame_index_0based': c['frame']['frame_index_0based'],
                           'native_instance_pixels': c['simulator_visibility']['visible_pixels'],
                           'opposite_instance_pixels': opposite['simulator_visibility']['visible_pixels'],
                           'native_frame': c['frame'], 'opposite_frame': opposite['frame'],
                           'source': 'same execution simulator instance segmentation; not human visibility acceptance'})
    write_json(ROOT/'key_event_visibility.json', {
        'native_conditions': binding(ROOT/'conditions/conditions.json'),
        'opposite_conditions': original['package']['inputs']['conditions'],
        'events': visibility,
        'contact_visibility_accepted': False,
        'limitation': 'Native grasp occlusion is preserved; nonzero pixels do not prove a visible grasp/contact.'})
    model = read_json(Path(original_package).parent/'initial_request.json')['model']
    config = {'anchor_model': model}
    # This is a semantic baseline package without a proposed geometry rig, used
    # as the input contract for the opt-in preflight, not a qualified rig output.
    baseline = unified.prepare_unified(ROOT/'conditions/conditions.json',
        original['package']['inputs']['execution_input']['path'],
        original['package']['inputs']['placement']['path'], ROOT/'native_source_package', config)
    package = baseline['package']['path']
    data = unified.validate_package(package)
    assert data['human_source'] == original['human_source']
    world_path = Path(receipt_path).parent.parent/'plans/collision_world.json'
    _, table, _ = g._load_world(data, world_path)
    # ROOT is .../v3/actor_geometry/native_camera_probe.
    grid_protocol = ROOT.parents[1]/'independent_review/actor_geometry_protocol.json'
    zero_budget = ROOT/'zero_budget.json'
    write_json(zero_budget, {'schema': 'r2h_unified_budget_v1',
        'campaign_id': 'harness-v3-native-geometry-probe-zero',
        'ledger_path': str((ROOT/'zero_budget_ledger').resolve()),
        'cumulative_call_limits': {'image': 0, 'video': 0, 'review': 0},
        'max_concurrent': 1, 'timeout_seconds': 600})
    rows = []
    for angle in g.LEAN_ANGLES:
        for height in g.WAIST_HEIGHTS:
            label = f'lean_{angle:02d}_waist_{round(height*100):+03d}cm'
            directory = ROOT/'grid'/label
            directory.mkdir(parents=True)
            rig_path = directory/'rig.json'
            write_json(rig_path, g.nominal_rig(data, table, lean=angle, waist_height=height))
            compiled = g.compile_geometry(package, rig_path, world_path, grid_protocol,
                                          directory/'report.json')
            report = read_json(compiled['report']['path'])
            prepared = prep.prepare(directory/'report.json', directory/'preparation', config)
            rows.append({'label': label, 'angle_degrees': angle, 'waist_height_m': height,
                         **compiled, 'clears_geometry_preflight': report['clears_geometry_preflight'],
                         'maximum_surface_proxy_distance_m': max(x['reach']['distance_m'] for x in
                             report['event_targets'] + report['held_trajectory_targets']),
                         'maximum_nearest_surface_distance_m': max(x['closest_surface_distance_m'] for x in
                             report['event_targets'] + report['held_trajectory_targets']),
                         'torso_table_mesh_collision': report['torso_table']['mesh_collision'],
                         'projection_blocks': report['projection_blocks'],
                         'reject_reasons': report['reject_reasons'],
                         'preparation': binding(directory/'preparation/preparation.json'),
                         'executable_rig_package_created': prepared['executable_unified_package'] is not None})
    # Probe the formal upright baseline and, if any, one qualifying mathematical
    # rig. The latter must still stop at the stable zero cumulative API budget.
    probes = [next(x for x in rows if x['angle_degrees'] == 0 and x['waist_height_m'] == 0)]
    passing = [x for x in rows if x['clears_geometry_preflight']]
    if passing and passing[0] not in probes:
        probes.append(passing[0])
    execution_probes = []
    for row in probes:
        try:
            prep.execute(row['preparation']['path'], config, zero_budget)
        except (ValueError, RuntimeError) as exc:
            execution_probes.append({'label': row['label'], 'status': 'blocked_before_provider',
                                     'reason': str(exc), 'api_calls': 0})
        else:
            raise AssertionError('No API dispatch permitted under this zero-budget probe')
    summary = {'schema': 'r2h_native_camera_geometry_probe_v1', 'protocol': binding(ROOT/'protocol.json'),
        'runner': binding(__file__), 'native_source_package': baseline['package'],
        'source_package': protocol['source_package'], 'execution_id': data['receipt']['execution_id'],
        'camera': 'native', 'native_camera_roles': packet['camera_role_projections']['native']['relative_order'],
        'opposite_camera_roles': original_packet['camera_role_projections']['opposite']['relative_order'],
        'key_event_visibility': binding(ROOT/'key_event_visibility.json'), 'grid_size': 12, 'results': rows,
        'cleared_rigs': len(passing), 'execution_probes': execution_probes,
        'native_grasp_visibility_qualified': False, 'human_anatomy_qualified': False,
        'generated_image_or_video_qualified': False, 'physical_executions_added': 0,
        'api_calls': 0, 'gpu_runs': 0, 'wall_seconds': time.monotonic()-started}
    write_json(ROOT/'summary.json', summary)
    print({'cleared_rigs': len(passing), 'wall_seconds': summary['wall_seconds'], 'api_calls': 0})


if __name__ == '__main__':
    run()
