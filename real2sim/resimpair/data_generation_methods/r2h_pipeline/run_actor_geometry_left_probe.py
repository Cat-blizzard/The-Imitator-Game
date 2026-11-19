"""One fixed g00 native/45deg/+5cm/left development candidate; zero API budget."""
from copy import deepcopy
from pathlib import Path
import time

from data_generation_methods.r2h_pipeline import actor_geometry_candidate as g
from data_generation_methods.r2h_pipeline import actor_geometry_preparation_v2 as prep
from data_generation_methods.r2h_pipeline import unified_anchor_candidate as unified
from data_generation_methods.r2h_pipeline.batch_pipeline import binding, checked_binding
from data_generation_methods.r2h_pipeline.common import read_json, write_json
from data_generation_methods.r2h_pipeline.controlled_experiment import call_counts

ROOT = Path('/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v3/actor_geometry/left_hand_candidate')


def run():
    started = time.monotonic()
    protocol = read_json(ROOT/'protocol.json')
    for value in protocol['code_bindings'].values():
        checked_binding(value)
    checked_binding(protocol['source_package'])
    source = unified.validate_package(protocol['source_package']['path'])
    placement = deepcopy(read_json(source['package']['inputs']['placement']['path']))
    for assignment in placement['event_assignments']:
        assignment['hands'] = ['left']
    write_json(ROOT/'placement_left.json', placement)
    config_path = Path(__file__).parents[1]/'config.echoflow.example.json'
    config = read_json(config_path)
    config['anchor_model'] = read_json(source['package_path'].parent/'initial_request.json')['model']
    write_json(ROOT/'public_config.json', config)
    base = unified.prepare_unified(source['package']['inputs']['conditions']['path'],
        source['package']['inputs']['execution_input']['path'], ROOT/'placement_left.json',
        ROOT/'left_source_package', config)
    data = unified.validate_package(base['package']['path'])
    world_path = Path(data['packet']['bindings']['receipt']['path']).parent.parent/'plans/collision_world.json'
    _, table, _ = g._load_world(data, world_path)
    write_json(ROOT/'rig.json', g.nominal_rig(data, table, lean=45, waist_height=.05))
    compiled = g.compile_geometry(base['package']['path'], ROOT/'rig.json', world_path,
        ROOT.parents[1]/'independent_review/actor_geometry_protocol.json', ROOT/'geometry_report.json')
    report = read_json(ROOT/'geometry_report.json')
    prepared = prep.prepare(ROOT/'geometry_report.json', ROOT/'preparation', config)
    budget = {'schema': 'r2h_unified_budget_v1',
        'campaign_id': 'harness-v3-g00-native-left-pose-zero',
        'ledger_path': str((ROOT/'zero_budget_ledger').resolve()),
        'cumulative_call_limits': {'image': 0, 'video': 0, 'review': 0},
        'max_concurrent': 1, 'timeout_seconds': 600}
    write_json(ROOT/'zero_budget.json', budget)
    try:
        prep.execute(ROOT/'preparation/preparation.json', config, ROOT/'zero_budget.json')
    except (ValueError, RuntimeError) as exc:
        execution_probe = {'exception_type': type(exc).__name__, 'message': str(exc),
            'reached_zero_budget_gate': isinstance(exc, RuntimeError)
                and 'image cumulative campaign budget exhausted; no paid call' in str(exc),
            'new_paid_calls': 0}
    else:
        raise AssertionError('Unexpected dispatch under zero cumulative budget')
    write_json(ROOT/'actual_execute_probe.json', execution_probe)
    if not execution_probe['reached_zero_budget_gate']:
        raise RuntimeError('Positive preparation failed before the required zero-budget boundary; preserve evidence')
    final_package = prepared['executable_unified_package']['path']
    # There is no generated image. Using the original simulation image here is
    # an intentionally unqualified input to demonstrate the unchanged video gate.
    video_gate = unified.video_gate(final_package, data['initial']['frame']['path'])
    write_json(ROOT/'unqualified_reference_video_gate.json', {
        'input_scope': 'simulation initial image, not a generated human image',
        'decision': video_gate})
    summary = {'schema': 'r2h_fixed_left_development_candidate_v1',
        'protocol': binding(ROOT/'protocol.json'), 'runner': binding(__file__),
        'left_source_package': base['package'], 'placement': binding(ROOT/'placement_left.json'),
        'geometry': compiled['report'], 'geometry_status': report['status'],
        'clears_declared_geometry_preflight': report['clears_geometry_preflight'],
        'maximum_surface_proxy_distance_m': max(x['reach']['distance_m'] for x in
            report['event_targets'] + report['held_trajectory_targets']),
        'preparation': binding(ROOT/'preparation/preparation.json'),
        'executable_initial_candidate_package': prepared['executable_unified_package'],
        'execute_probe': binding(ROOT/'actual_execute_probe.json'),
        'actual_campaign_call_counts': call_counts(ROOT/'zero_budget_ledger'),
        'video_gate': binding(ROOT/'unqualified_reference_video_gate.json'),
        'withdrawal_trajectory': 'unknown', 'native_grasp_and_lift_selected_object_pixels': [0, 0],
        'human_anatomy_verified': False, 'generated_human_image_exists': False,
        'video_qualified': False, 'quality_release_approved': False, 'training_admitted': False,
        'api_calls': 0, 'gpu_runs': 0, 'wall_seconds': time.monotonic()-started}
    write_json(ROOT/'summary.json', summary)
    print({'geometry_status': summary['geometry_status'], 'budget_gate_reached': True,
           'can_generate_video': video_gate['can_generate_video'], 'api_calls': 0})


if __name__ == '__main__':
    run()
