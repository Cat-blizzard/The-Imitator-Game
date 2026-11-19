"""Reusable saved-state multiview rendering and condition export.

Uses the existing v3 renderer, v4 independent validator and v3 condition
compiler unchanged. Preparation does not call a model. One initial/held/final
screen precedes each full two-camera video; failures remain in the batch.
The current three-keyframe visibility contract supports single-operation
tasks only; multi-operation evidence needs its own versioned policy.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import time

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from resimpair_tabletop.sample_report import native_scalars
from resimpair_pickplace import derived_condition_export_v3 as exporter
from resimpair_pickplace import render_lineage_validator_v4 as validator
from . import deferred_render_v1 as original
from . import scene_domain_v2 as domain
from . import scene_quality_v3 as quality
from . import scene_quality_render_v3 as render

VERSION = 'batch_saved_state_conditions_v1'
CATALOG = ROOT / 'assets/appearance_v2/curated/catalog.json'
RULES = ROOT / 'validation/pickplace_automation_20261029/independent/scene_domain_v2/quality_rules_v4.json'
EXPECTED_VALIDATOR = '0e33739fb66be34d83a32dcaab8f965781004ada97f6b7db1dd12948a2ba57ae'


def keyframes(bundle, arrays):
    operations = bundle['task_intent']['operations']
    if len(operations) != 1:
        raise ValueError('three_keyframe_gate_requires_one_operation_multi_event_policy_not_yet_validated')
    op = operations[0]; obj = bundle['object_ids'].index(op['object']); arm = op['arm']
    held = arrays['grasp'][:, arm, obj].astype(bool)
    lift = arrays['object_states'][:, obj, 2] - arrays['object_states'][0, obj, 2]
    candidates = np.flatnonzero(held & (lift >= .03))
    if not len(candidates): raise ValueError('actual_held_and_lifted_frame_required')
    frame = int(candidates[0]); indices = [0, frame, len(held) - 1]
    if len(set(indices)) != 3: raise ValueError('three_distinct_event_frames_required')
    return indices, {'source': 'native grasp + object_states', 'arm': arm, 'object': op['object'],
        'first_frame_with_grasp_and_lift_at_least_m': .03, 'selected_frame': frame,
        'measured_lift_m': float(lift[frame]), 'measured_grasp': bool(held[frame])}


def prepare(sources, output, prior_budget, *, visual_policy='domain_v2', seed=202610294000):
    output = Path(output).resolve(); sources = [Path(s).resolve() for s in sources]
    if output.exists(): raise ValueError('preserve_existing_render_batch')
    if not sources or len(sources) > 48 or len(sources) != len(set(sources)):
        raise ValueError('one_to_48_distinct_source_executions_required')
    if visual_policy not in ('domain_v2', 'furniture_only', 'combined'):
        raise ValueError('explicit_supported_visual_policy_required')
    if binding(validator.__file__)['sha256'] != EXPECTED_VALIDATOR:
        raise ValueError('frozen_independent_validator_changed')
    prior_budget = Path(prior_budget).resolve(); prior = read(prior_budget)
    historical = prior['historical_actual_render_jobs'] + len(list((prior_budget.parent / 'reservations').glob('*.json')))
    parent_cap = prior['maximum_lineage_render_jobs_including_history']
    if historical > parent_cap: raise ValueError('prior_render_allowance_exceeded')
    began = time.monotonic(); source_rows = []; cases = []; scenes = {}
    for i, path in enumerate(sources):
        source = original.validate_source(path); bundle = source['bundle']; indices, evidence = keyframes(bundle, source['arrays'])
        row = {'id': f'source_{i}', 'source_execution': source['bindings']['execution_bundle.json'],
            'source_artifacts': source['bindings'], 'artifacts': source['bindings'],
            'source_execution_code': bundle['code'], 'dataset_partition': bundle['case']['partition'],
            'split_group': bundle['case']['split_group'], 'source_ids': bundle['case']['source_ids'],
            'requested_task_id': bundle['case']['requested_task_id'], 'operation': source['task']['operations'][0],
            'frame_count': len(source['states']), 'screen_state_indices': indices,
            'screen_state_labels': ['initial', 'actual_held_and_lifted', 'final'],
            'grasp_selection_evidence': evidence, 'experiment_role': 'batch_render_integration',
            'prior_visual_tuning_in_this_round': False}
        source_rows.append(row)
        envelope = domain.build_motion_envelope(source['scene'], path / 'measurements.npz')
        partition = 'development' if row['dataset_partition'] == 'development' else 'heldout_appearance'
        scene = domain.sample_domain(source['scene'], seed + i, catalog_path=CATALOG,
            mode='workspace', camera_mode='bounded', motion_envelope=envelope, asset_partition=partition)
        if visual_policy != 'domain_v2': scene = quality.apply(scene, visual_policy, seed + 10000 + i)
        render.validate_visual_derivative(source['scene'], scene, 'environment_3d_camera')
        for stage in ('screen', 'video'):
            case_id = f'{stage}_{i:03d}'; folder = output / 'cases' / case_id
            write(folder / 'render_scene.json', scene)
            for name in original.COPY_EVIDENCE: shutil.copy2(path / name, folder / name)
            case = {'id': case_id, 'source_id': row['id'], 'stage': stage,
                'variant': 'environment_3d_camera', 'seed': seed + i, 'config': {},
                'quality_condition': 'baseline_v2' if visual_policy == 'domain_v2' else visual_policy,
                'scene': binding(folder / 'render_scene.json'), 'source_execution': row['source_execution'],
                'source_partition': row['dataset_partition'], 'split_group': row['split_group'],
                'appearance_partition': scene['scene_domain'].get('asset_partition', partition),
                'screen_state_indices': indices, 'state_indices': indices if stage == 'screen' else list(range(len(source['states']))),
                'screen_case_id': None if stage == 'screen' else f'screen_{i:03d}', 'predeclared_control': False}
            cases.append(case); scenes[case_id] = scene
    # Metadata key "independent_protocol" is inherited by the old renderer.
    # Rules are independently authored and frozen; this batch selection is root-authored.
    frozen_evaluation = {str(Path(validator.__file__).relative_to(ROOT)): binding(validator.__file__),
        str(RULES.relative_to(ROOT)): binding(RULES)}
    write(output / 'selection_protocol.json', {'schema': VERSION, 'sources': source_rows, 'cases': cases,
        'predeclared_control_video_case_ids': [], 'frozen_original_evaluation_dependencies': frozen_evaluation,
        'selection_author': 'root', 'independent_rule_revision': 'unchanged_v4',
        'new_independent_agent_review': False, 'source_selection': 'explicit_source_paths_before_rendering',
        'source_count': len(sources), 'source_physics_reexecuted': False,
        'visual_policy': visual_policy, 'experimental_material_policy': visual_policy == 'combined'})
    limits = {'schema': VERSION, 'maximum_attempts': 2 * len(sources),
        'allocation': {'screen': len(sources), 'video': len(sources)},
        'historical_actual_render_jobs': historical, 'prior_budget': binding(prior_budget),
        'prior_lineage_limit': parent_cap, 'maximum_lineage_render_jobs_including_history': parent_cap + 2 * len(sources),
        'physical_execution_limit': 0, 'api_limit': 0, 'gpu': 2, 'minimum_free_gpu_mib': 12288,
        'parallel_workers': 1, 'timeout_seconds': {'screen': 120, 'video': 180}, 'resource_wait_seconds': 180,
        'automatic_retries': 0, 'unknown_submission_must_be_recovered': True}
    write(output / 'budget/limits.json', limits)
    old = read(ROOT / 'runs/scene_quality_20261029_v3/protocol.json')
    for ref in old['code'].values(): resolve(ref)
    protocol = {**{k: v for k, v in old.items() if k not in ('sources', 'cases', 'code', 'budget', 'independent_protocol')},
        'schema': render.VERSION, 'batch_adapter': binding(__file__), 'sources': source_rows, 'cases': cases,
        'independent_protocol': binding(output / 'selection_protocol.json'),
        'independent_validator': binding(validator.__file__), 'independent_quality_rules': binding(RULES),
        'budget': binding(output / 'budget/limits.json'), 'code': {**old['code'],
            str(Path(__file__).resolve().relative_to(ROOT)): binding(__file__)},
        'default_promoted': False, 'new_trajectory': False, 'api_calls': 0}
    write(output / 'protocol.json', protocol)
    refs = {'protocol': binding(output / 'protocol.json')}
    for i, ref in enumerate(render.bindings_in(protocol)): resolve(ref); refs[f'dependency_{i}'] = ref
    for scene in scenes.values():
        for ref in render.bindings_in(scene): resolve(ref); refs['resource_' + digest(ref)] = ref
    write(output / 'frozen_inputs.json', refs)
    render.verify(output)
    result = {'schema': VERSION, 'sources': len(sources), 'render_limit': limits['maximum_attempts'],
        'new_physical_executions': 0, 'new_paid_calls': 0, 'visual_policy': visual_policy,
        'prepare_seconds': time.monotonic() - began}
    write(output / 'prepared.json', result); return result


def run(root):
    root = Path(root).resolve(); protocol, limits = render.verify(root)
    if protocol['batch_adapter'] != binding(__file__): raise ValueError('batch_adapter_changed')
    if (root / 'complete.json').exists(): return read(root / 'complete.json')
    began = time.monotonic()
    render.run(root, 'screen', limits['allocation']['screen'])
    rows = []; allowed = []; validations = {}
    for case in protocol['cases']:
        if case['stage'] != 'screen': continue
        folder = root / 'cases' / case['id']; proof = native_scalars(validator.validate(folder)); validations[case['id']] = proof
        write(folder / 'batch_validation.json', proof)
        passed = proof['passed'] and proof['screen_gate_passed']
        rows.append({'case_id': case['id'], 'passed': passed,
            'terminal': binding(folder / 'terminal.json'), 'worker_result': binding(folder / 'worker_result.json'),
            'validation': binding(folder / 'batch_validation.json')})
        if passed: allowed += [c['id'] for c in protocol['cases'] if c['screen_case_id'] == case['id']]
    gate = root / 'screen_gate.json'
    write(gate, {'schema': render.GATE_VERSION, 'protocol': binding(root / 'protocol.json'),
        'screen_receipts': rows, 'allowed_video_case_ids': allowed, 'validator': binding(validator.__file__),
        'rule_changed': False})
    # Explicit zero-attempt rejection records let the retained runner skip
    # disallowed videos, without spending or inventing a render attempt.
    for case in protocol['cases']:
        if case['stage'] == 'video' and case['id'] not in allowed:
            write(root / 'cases' / case['id'] / 'terminal.json', {'stage': 'video',
                'returncode': 'preflight_rejected_visibility', 'render_attempts': 0,
                'screen_gate': binding(gate), 'physical_execution_count': 0, 'api_calls': 0})
    if allowed: render.run(root, 'video', len(allowed), gate=gate)
    outputs = []
    for case in protocol['cases']:
        if case['stage'] != 'video': continue
        folder = root / 'cases' / case['id']
        if case['id'] not in allowed:
            outputs.append({'case_id': case['id'], 'status': 'screen_rejected', 'human_generation_ready': False}); continue
        proof = native_scalars(validator.validate(folder)); write(folder / 'batch_validation.json', proof)
        row = {'case_id': case['id'], 'validation': binding(folder / 'batch_validation.json'),
               'state_media_passed': proof['passed'], 'human_generation_ready': proof['generation_ready']}
        if proof['passed']:
            target = root / 'conditions' / case['id']
            if not target.exists(): exporter.export(folder, target)
            row.update(contract=binding(target / 'contract.json'), status='conditions_prepared',
                robot_video=binding(folder / 'robot_side.mp4'), reference_view_video=binding(folder / 'opposite_high.mp4'))
        else: row['status'] = 'video_validation_failed'
        outputs.append(row)
    report = {'schema': VERSION, 'protocol': binding(root / 'protocol.json'), 'outputs': outputs,
        'source_physical_execution_count': len(protocol['sources']), 'new_physical_executions': 0,
        'render_attempts': len(list((root / 'budget/reservations').glob('*.json'))),
        'new_paid_calls': 0, 'human_videos_generated': 0, 'elapsed_seconds': time.monotonic() - began,
        'paid_submission_requires_separate_budget_and_qualified_single_person_anchors': True,
        'formal_Real_Sim_Human_pairs': 0}
    write(root / 'complete.json', report); render.gallery(root, root / 'gallery.html'); return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--source', type=Path, action='append', required=True)
    prep.add_argument('--root', type=Path, required=True); prep.add_argument('--prior-budget', type=Path, required=True)
    prep.add_argument('--visual-policy', choices=['domain_v2', 'furniture_only', 'combined'], default='domain_v2')
    prep.add_argument('--seed', type=int, default=202610294000)
    launch = sub.add_parser('run'); launch.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    print(prepare(a.source, a.root, a.prior_budget, visual_policy=a.visual_policy, seed=a.seed)
          if a.command == 'prepare' else run(a.root))
