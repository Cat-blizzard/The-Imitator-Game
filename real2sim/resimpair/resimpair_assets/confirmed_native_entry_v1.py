"""Fresh-process final-file review and metadata handoff for a v2 native job.

Run this command in a new CPU process after raw_manipulation_v2 finishes. It
reuses the original geometry evaluator and admission rules, does no dynamics,
and emits a new entry/state only when the complete asset passes. It never
publishes a cache or edits the original source's CPU result.
"""
import argparse
import os
from pathlib import Path
import time
import traceback

import numpy as np
import trimesh

from resimpair_grasp.common import binding, read, resolve, write
from resimpair_grasp import evaluator
from . import metric_admission_v1 as admission
from . import native_reload_preflight_v1 as retained_reload
from .raw_manipulation_v2 import _validated_job, accounting

VERSION = 'confirmed_native_entry_v1'


def review(root, prior_entry_path, prior_source_job_path, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('fresh_final_native_review_output_required_no_retry')
    job_path, job = _validated_job(root / 'job.json')
    supervisor = read(root / 'SUPERVISOR.json')
    worker = read(root / 'cpu_result.json')
    if (supervisor['status'] != 'worker_completed' or supervisor['returncode'] != 0
            or supervisor['worker_result'] != binding(root / 'cpu_result.json')
            or worker.get('status') != 'native_geometry_ready' or not worker.get('geometry_qualified')):
        raise ValueError('completed_geometry_qualified_native_only_job_required')
    native = read(resolve(worker['native_result']))
    if native['asset'] != worker['asset']:
        raise ValueError('native_worker_final_descriptor_binding_mismatch')
    descriptor = read(resolve(native['asset']))
    previous_entry_ref = binding(prior_entry_path)
    previous_entry = read(resolve(previous_entry_ref))
    old_state_ref = previous_entry['source_state']
    old_state = read(resolve(old_state_ref))
    original_job_ref = binding(prior_source_job_path)
    original_job = read(resolve(original_job_ref))
    raw = original_job['raw_row']
    original_source = binding(raw['path'])
    source_refs = [previous_entry_ref, old_state_ref, original_job_ref, original_source,
        previous_entry['asset'], old_state['policy'], old_state['raw_size_decision']]
    code_refs = [binding(__file__), binding(retained_reload.__file__), binding(evaluator.__file__),
                 binding(admission.__file__), *job['implementations']]
    compile_accounting = accounting(job['output'])
    helper_pids = [read(resolve(ref))['worker_pid'] for ref in compile_accounting['probe_outcomes']]
    if os.getpid() in [worker['worker_pid'], supervisor['supervisor_pid'], *helper_pids]:
        raise ValueError('fresh_process_independent_of_compile_and_helper_probes_required')
    write(output / 'PROTOCOL.json', {'schema': VERSION, 'job': binding(job_path),
        'native_worker_result': binding(root / 'cpu_result.json'),
        'native_supervisor': binding(root / 'SUPERVISOR.json'), 'asset': native['asset'],
        'original_inputs_and_outcomes': source_refs, 'implementations': code_refs,
        'process_pid': os.getpid(), 'parent_pid': os.getppid(),
        'maximum_full_asset_reload_workers': 1, 'maximum_native_actor_builds': 1,
        'maximum_final_part_loads': len(descriptor['collisions']),
        'strict_convergence_threshold_m': 1e-6, 'original_runtime_threshold_m': 1e-5,
        'original_surface_p95_threshold_m': .003, 'original_mass_threshold_magnitude_kg': 1e-4,
        'part_load_scope': 'all final paths loaded for the first time in this fresh process',
        'simulation_steps': 0, 'new_native_compile_attempts': 0,
        'robot_executions': 0, 'GPU_jobs': 0, 'API_calls': 0, 'existing_results_modified': False})
    began = time.monotonic()
    try:
        reload_result = retained_reload.check([native], output / 'final_reload')
        reload_row = reload_result['rows'][0]
        if 'actual_native_geometry' not in reload_row:
            raise ValueError('fresh_final_native_actor_did_not_produce_geometry')
        actual = read(resolve(reload_row['actual_native_geometry']))['objects'][0]
        part_rows = []
        for number, (ref, shape, passes) in enumerate(zip(
                descriptor['collisions'], actual['shapes'], descriptor['cooking'])):
            mesh = trimesh.load(resolve(ref), force='mesh', process=False)
            vertices = np.asarray(shape['vertices'])
            distances = np.linalg.norm(mesh.vertices[:, None] - vertices[None, :], axis=-1)
            error = float(max(distances.min(0).max(), distances.min(1).max()))
            confirmation = read(resolve(passes[-1]['confirmation']))
            part_rows.append({'part': number, 'collision': ref,
                'declared_vertices': len(mesh.vertices), 'native_vertices': len(vertices),
                'vertex_error_m': error, 'strict_1um_passed': error < 1e-6,
                'original_10um_passed': error < 1e-5,
                'returned_file_was_exact_confirmed_input': ref == passes[-1]['input'] == passes[-1]['output'] == confirmation['input'],
                'export_after_confirmation': passes[-1]['returned_file_exported_after_check'],
                'helper_confirmation': passes[-1]['confirmation'],
                'helper_process_pid': confirmation['worker_pid'], 'fresh_replay_process_pid': os.getpid()})
        write(output / 'FINAL_PART_REPLAY.json', {'asset': native['asset'], 'rows': part_rows,
            'retained_evaluator_result': binding(output / 'final_reload/RESULTS.json'),
            'fresh_worker_pid': os.getpid(), 'native_part_loads': len(actual['shapes']),
            'simulation_steps': 0})
        native_properties = read(resolve(descriptor['native_properties']))
        support = read(resolve(descriptor['native_support']))
        settle = read(resolve(support['cpu_settle']))
        metric_input = read(resolve(job['compile_input']))
        surface = descriptor['surface_comparison']
        state = {**old_state, 'prior_source_state': old_state_ref,
            'native_method_comparison': {'protocol': binding(root / 'PROTOCOL.json'),
                'worker_result': binding(root / 'cpu_result.json'), 'prior_native_entry': previous_entry_ref},
            'scale_policy_reused_without_change': True}
        write(output / 'source_state.json', state)
        entry = {**native, 'role': old_state['role'],
            'source_aliases': original_job.get('source_aliases', []),
            'ingestion_group': original_job['ingestion_group'],
            'source_state': binding(output / 'source_state.json'),
            'native_cook_method': 'stable_cook_confirmed_v2',
            'native_producer': {'job': binding(job_path), 'protocol': binding(root / 'PROTOCOL.json'),
                'worker_result': binding(root / 'cpu_result.json'), 'supervisor': binding(root / 'SUPERVISOR.json')},
            'prior_native_entry': previous_entry_ref, 'new_native_method_comparison': True,
            'fresh_final_reload': binding(output / 'final_reload/RESULTS.json'),
            'new_grasp_required': True, 'previous_grasp_cache_reuse_authorized': False}
        gate = admission.inspect(entry, state)
        entry['semantic_scale_admission'] = gate
        expected_settle_criteria = {'max_final_window_linear_speed_m_s': .005,
            'max_final_window_angular_speed_rad_s': .05, 'minimum_asset_up_dot_world_up': .995,
            'maximum_xy_drift_m': .003, 'maximum_support_height_error_m': .003,
            'contact_observed_required': True}
        checks = {
            'native_compiler_geometry_qualified': native['geometry_qualified'] is True,
            'all_final_parts_loaded': len(part_rows) == len(descriptor['collisions']) == len(actual['shapes']),
            'all_final_parts_pass_strict_1um': all(p['strict_1um_passed'] for p in part_rows),
            'all_final_parts_pass_original_10um': all(p['original_10um_passed'] for p in part_rows),
            'all_returned_files_match_their_confirmations': all(p['returned_file_was_exact_confirmed_input'] and not p['export_after_confirmation'] for p in part_rows),
            'original_evaluator_geometry_and_mass_passed': reload_result['passed'] == 1 and reload_row['passed'],
            'original_surface_3mm_passed': surface['unchanged_absolute_tolerance_m'] == .003 and max(
                surface['visual_to_collision_p95_m'], surface['union_external_p95_m']) <= .003,
            'native_result_surface_matches_descriptor': native['surface'] == surface,
            'original_support_and_settle_passed': support['status'] == 'pass' and settle['status'] == 'pass' and all(settle['checks'].values()),
            'original_settle_rules_preserved': settle['criteria'] == expected_settle_criteria and settle['frames'] == 360 and settle['dt_seconds'] == 1 / 240,
            'new_settle_belongs_to_new_compile': resolve(support['cpu_settle']).is_relative_to(Path(job['output'])),
            'original_admission_passed': gate['passed'],
            'native_entry_property_bindings_match': all(entry.get(k) == descriptor.get(k) for k in ('source_id', 'variant_id', 'native_support', 'native_properties')),
            'source_and_partition_match': raw['id'] == job['source_id'] == descriptor['source_id'] == state['source_id'] == previous_entry['source_id'] and raw['partition'] == state['partition'] == descriptor['partition'],
            'raw_source_bytes_unchanged': original_source['sha256'] == raw['source_sha'] == descriptor['provenance']['original_sha256'],
            'original_size_and_mass_provenance_retained': descriptor['size_provenance'] == metric_input['size_provenance'] and descriptor['mass_provenance'] == metric_input['mass_provenance'],
            'source_scale_preserved': bool(np.allclose(np.linalg.svd(np.asarray(descriptor['source_to_asset'])[:3, :3], compute_uv=False),
                np.linalg.svd(np.asarray(metric_input['source_to_asset'])[:3, :3], compute_uv=False), rtol=1e-7, atol=1e-10)),
            'raw_size_decision_binding_preserved': descriptor['provenance']['raw_size_decision'] == old_state['raw_size_decision'],
            'COM_and_inertia_finite': bool(np.isfinite(native_properties['center_of_mass_m']).all() and np.isfinite(native_properties['inertia_kg_m2']).all()),
            'native_principal_inertia_positive': min(native_properties['principal_inertia_kg_m2']) > 0,
        }
        for ref in source_refs + code_refs:
            resolve(ref)
        checks['all_original_input_outcome_and_implementation_bindings_unchanged'] = True
        acceptance = {'schema': VERSION, 'passed': all(checks.values()), 'checks': checks,
            'asset': native['asset'], 'final_parts': binding(output / 'FINAL_PART_REPLAY.json'),
            'maximum_final_vertex_error_m': max(p['vertex_error_m'] for p in part_rows),
            'surface': surface, 'target_mass_kg': descriptor['mass_kg'],
            'fresh_native_mass_kg': actual['mass'], 'fresh_native_mass_error_kg': abs(actual['mass'] - descriptor['mass_kg']),
            'native_properties': descriptor['native_properties'], 'native_support': descriptor['native_support'],
            'selected_settle': support['cpu_settle'], 'semantic_scale_admission': gate,
            'recorded_settle_steps': compile_accounting['recorded_settle_steps'],
            'new_grasp_or_robot_success_implied': False}
        write(output / 'ACCEPTANCE.json', acceptance)
        if acceptance['passed']:
            entry['final_native_acceptance'] = binding(output / 'ACCEPTANCE.json')
            write(output / 'native_entry.json', entry)
        result = {'schema': VERSION, 'status': 'native_ready_for_new_grasp_cache' if acceptance['passed'] else 'native_final_review_failed',
            'passed': acceptance['passed'], 'protocol': binding(output / 'PROTOCOL.json'),
            'acceptance': binding(output / 'ACCEPTANCE.json'), 'asset': native['asset'],
            'entry': binding(output / 'native_entry.json') if acceptance['passed'] else None,
            'state': binding(output / 'source_state.json'),
            'new_CPU_native_compile_attempts': 1, 'native_compile_accounting': compile_accounting,
            'additional_fresh_full_asset_replay_workers': 1, 'additional_native_actor_builds': 1,
            'additional_final_part_loads': len(actual['shapes']), 'additional_dynamic_steps': 0,
            'total_recorded_dynamic_steps_this_method_comparison': compile_accounting['recorded_settle_steps'],
            'old_source_outcomes_preserved': True, 'metadata_missing': [],
            'robot_executions': 0, 'new_grasp_jobs': 0, 'GPU_jobs': 0, 'API_calls': 0,
            'wall_seconds': time.monotonic() - began,
            'cache_identity_requirement': 'bind the exact new descriptor and collision files; the original metric variant_id alone is insufficient',
            'producer_scope': 'new explicit method-comparison producer, not the original frozen raw queue result',
            'physical_success': 'unmeasured'}
        write(output / 'RESULT.json', result)
        return result
    except Exception as error:
        write(output / 'ERROR.json', {'schema': VERSION, 'status': 'native_final_review_error',
            'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
            'asset': native['asset'], 'new_CPU_native_compile_attempts': 1,
            'native_compile_accounting': compile_accounting, 'new_native_compiles_in_review': 0,
            'simulation_steps': 0, 'robot_executions': 0, 'GPU_jobs': 0, 'API_calls': 0,
            'wall_seconds': time.monotonic() - began})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'prior-entry', 'prior-source-job', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root, args.prior_entry, args.prior_source_job, args.output)
    print({k: result[k] for k in ('status', 'passed', 'asset', 'entry', 'state', 'wall_seconds')})
    raise SystemExit(0 if result['passed'] else 1)
