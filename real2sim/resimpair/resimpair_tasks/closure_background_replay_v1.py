"""Bounded native replays with functional furniture, finite light and 720p.

This is a composition adapter over retained implementations. It does not
execute physics, create a baseline RGB comparison, or change physical labels.
Each video's independent screen gate must bind this new production preset.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import shutil
from unittest.mock import patch

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import finite_interior_light_v1 as lighting
from . import visual_furniture_bulk_render_v4 as batch
from . import visual_furniture_room_v3 as furniture
from . import camera_resolution_replay_v1 as resolution

VERSION = 'closure_native_background_production_preset_v1'
render = batch.appearance.render


def candidate_scene(baseline, *, seed, room, partition, motion_envelope, catalog):
    """Compose existing optional visuals, then change only one pixel extent."""
    scene, receipt = furniture.apply_overlay(baseline, seed, room=room,
        partition=partition, motion_envelope=motion_envelope, catalog_path=catalog,
        count=4, view_layout='dual_view')
    if not receipt['appearance_applied']:
        raise ValueError('visual_resources_degraded_no_success_claim')
    scene, diagnosis = lighting.prepare(scene)
    batch.appearance.assert_invariants(baseline, scene)
    scene = resolution.candidate_scene(scene)
    sampling = scene['scene_domain']['camera_sampling']
    sampling.update(status='registered_production_resolution_preset',
                    resolution_policy=VERSION)
    if furniture.domain.physical_scene(scene) != furniture.domain.physical_scene(baseline):
        raise ValueError('candidate_changed_physical_scene')
    expected = deepcopy(baseline['cameras'])
    expected['opposite_high'].update(width=1280, height=720)
    if scene['cameras'] != expected:
        raise ValueError('production_preset_changed_unregistered_camera_values')
    furniture.domain.validate_scene(scene)
    receipt.update(composition_adapter=binding(__file__),
        final_scene_digest=digest(scene), physical_scene_digest=digest(furniture.domain.physical_scene(scene)),
        final_camera_digest=digest(scene['cameras']),
        compound_changes=['functional room furniture and PBR surfaces',
                          'finite interior point key', 'registered opposite_high 1280x720'],
        baseline_RGB_comparison_available=False, quality_improvement_established=False)
    return scene, receipt, diagnosis


def prepare(source_specs, output, *, prior_budget, allocation_id,
            catalog=furniture.CATALOG):
    output, source_specs, catalog = Path(output).resolve(), Path(source_specs).resolve(), Path(catalog).resolve()
    if output.exists():
        raise ValueError('new_output_required_preserve_all_previous_evidence')
    if shutil.disk_usage(output.parent).free < 10 * 2**30:
        raise ValueError('insufficient_disk_preserve_history')
    batch.require_completed_catalog(read(catalog))
    specs = read(source_specs)['sources']
    if len(specs) != 2 or len({s['id'] for s in specs}) != 2:
        raise ValueError('registered_two_sources_required')
    cases, sources, presets, receipts, diagnoses = [], [], [], [], []
    for index, spec in enumerate(specs):
        source = batch.appearance.replay.validate_source(resolve(spec['execution']).parent)
        bridge = source['source'] / 'independent_metric_cli_audit_bridge_v1.json'
        if not bridge.is_file():
            raise ValueError('independent_native_audit_bridge_required')
        baseline, repair = render.baseline_scene(source)
        partition = source['bundle']['case']['partition']
        if spec['partition'] != partition:
            raise ValueError('preserve_native_source_partition')
        envelope = furniture.domain.build_motion_envelope(baseline, source['source'] / 'measurements.npz')
        scene, receipt, diagnosis = candidate_scene(baseline, seed=spec['seed'],
            room=spec['room'], partition=partition, motion_envelope=envelope, catalog=catalog)
        render.validate_visual_derivative(source['scene'], scene, 'environment_3d_camera')
        count = len(source['states'])
        grasp = next(e['frame'] for e in read(source['source'] / 'audit.json')['event_frames'] if e['id'].endswith('.grasp'))
        if not 0 < grasp < count - 1:
            raise ValueError('three_distinct_native_keyframes_required')
        source_id = f'source_{index:02d}'
        row = {'id': source_id, 'source_execution': source['bindings']['execution_bundle.json'],
            'artifacts': source['bindings'], 'source_artifacts': source['bindings'],
            'screen_state_indices': [0, grasp, count - 1],
            'source_execution_code': source['bundle']['code'],
            'frame_count': count, 'fps': float(source['fps']), 'source_spec': spec,
            'native_audit_bridge': binding(bridge)}
        sources.append(row)
        preset = {'schema': VERSION, 'source_id': source_id,
            'source_execution': row['source_execution'],
            'original_camera_configs': deepcopy(baseline['cameras']),
            'production_camera_configs': deepcopy(scene['cameras']),
            'changed_camera': 'opposite_high', 'source_dimensions': [512, 512],
            'candidate_dimensions': [1280, 720],
            'eye_target_vertical_FOV_unchanged': True, 'other_camera_unchanged': True,
            'actual_projection_and_calibration_required': True,
            'same_native_frame_mapping_required': True,
            'same_source_512_RGB_available': False,
            'resolution_improvement_comparison_claimed': False,
            'physical_scene_digest': digest(furniture.domain.physical_scene(scene))}
        presets.append(preset)
        recipe_id = f'{index:02d}_{spec["id"]}'
        for stage in ('screen', 'video'):
            case_id = f'{stage}_{recipe_id}'; folder = output / 'cases' / case_id
            write(folder / 'render_scene.json', scene)
            write(folder / 'source_artifacts.json', source['bindings'])
            write(folder / 'motion_envelope.json', envelope)
            write(folder / 'appearance_receipt.json', receipt)
            write(folder / 'light_ray_diagnosis.json', diagnosis)
            for name in batch.appearance.replay.COPY_EVIDENCE:
                shutil.copy2(source['source'] / name, folder / name)
            repair_ref = None
            if repair is not None:
                write(folder / 'metadata_repair.json', repair)
                repair_ref = binding(folder / 'metadata_repair.json')
            cases.append({'id': case_id, 'source_id': source_id, 'stage': stage,
                'variant': 'environment_3d_camera', 'scene': binding(folder / 'render_scene.json'),
                'source_execution': row['source_execution'],
                'state_indices': row['screen_state_indices'] if stage == 'screen' else None,
                'visualization_only': True, 'R2H_ready': False, 'predeclared_control': False,
                'screen_case_id': f'screen_{recipe_id}' if stage == 'video' else None,
                'metadata_repair': repair_ref, 'recipe_id': recipe_id,
                'room': spec['room'], 'seed': spec['seed']})
            receipts.append(binding(folder / 'appearance_receipt.json'))
            diagnoses.append(binding(folder / 'light_ray_diagnosis.json'))
    write(output / 'registered_camera_preset.json', {'schema': VERSION,
        'scope': 'two registered native sources; production preset, no matched RGB ablation',
        'adapter': binding(__file__), 'sources': presets, 'physics_steps': 0, 'API_calls': 0})
    camera_ref = binding(output / 'registered_camera_preset.json')
    for case in cases:
        case['registered_camera_preset'] = camera_ref
    registration, limits = batch.register_budget(output, prior_budget, allocation_id, screens=2, videos=2)
    selection = {'schema': VERSION, 'author': 'implementation_agent_requires_independent_production_preset_gate',
        'input_sources': binding(source_specs), 'catalog': binding(catalog),
        'sources': sources, 'cases': cases, 'predeclared_control_video_case_ids': [],
        'registered_camera_preset': camera_ref, 'appearance_receipts': receipts,
        'light_ray_diagnoses': diagnoses,
        'frozen_original_evaluation_dependencies': batch.frozen_evaluation_dependencies(),
        'independent_rule_revision': 'registered source-specific production-camera validation required',
        'qualification_scope': 'native state/action preservation plus actual calibrated RGB; visual realism and R2H remain separate',
        'independent_source_executions': 2, 'visual_recipes': 2}
    write(output / 'selection_protocol.json', selection)
    modules = [Path(__file__), Path(batch.__file__), Path(furniture.__file__), Path(furniture.base.__file__),
        Path(lighting.__file__), Path(resolution.__file__), lighting.SHADER,
        Path(batch.appearance.__file__), Path(render.__file__), Path(batch.appearance.replay.__file__),
        Path(furniture.domain.__file__), ROOT / 'resimpair_tasks/scene_quality_v3.py',
        ROOT / 'resimpair_tasks/scene_material_policy_v3.py', ROOT / 'resimpair_tasks/appearance_v1.py',
        ROOT / 'resimpair_pickplace/environment.py', ROOT / 'resimpair_tasks/cudex_shared_gpu_v1.py',
        ROOT / 'corl2026_workshop/experiments/autonomous_pipeline_v4/simulation.py',
        ROOT / 'corl2026_workshop/experiments/autonomous_pipeline_v4/resources.py',
        ROOT / 'PolicyGen/mani_skill/render/shaders.py', ROOT / 'PolicyGen/mani_skill/sensors/camera.py',
        ROOT / 'PolicyGen/mani_skill/envs/scene.py']
    protocol = {'schema': render.VERSION, 'closure_background_adapter': binding(__file__),
        'independent_protocol': binding(output / 'selection_protocol.json'),
        'selection_protocol_author': selection['author'], 'registered_camera_preset': camera_ref,
        'sources': sources, 'cases': cases, 'budget': binding(output / 'budget/limits.json'),
        'budget_registration': registration, 'code': {str(p.relative_to(ROOT)): binding(p) for p in modules},
        'state_roundtrip_tolerance': 1e-6, 'TCP_tolerance_m': 1e-5,
        'physical_execution_count': 0, 'native_execution': False, 'new_trajectory': False,
        'api_calls': 0, 'R2H_ready': False, 'default_promoted': False}
    write(output / 'protocol.json', protocol)
    refs = {'protocol': binding(output / 'protocol.json')}
    for ref in render.bindings_in(protocol):
        resolve(ref); refs['dependency_' + digest(ref)] = ref
    for case in cases:
        for ref in render.bindings_in(read(resolve(case['scene']))):
            resolve(ref); refs['scene_resource_' + digest(ref)] = ref
    write(output / 'frozen_inputs.json', refs)
    batch.validate_prepared_interface(output); render.verify(output)
    result = {'schema': VERSION, 'source_count': 2, 'screen_jobs': 2, 'video_jobs': 2,
        'historical_actual_render_jobs': limits['historical_actual_render_jobs'],
        'lineage_cap': limits['maximum_lineage_render_jobs_including_history'],
        'actual_new_jobs': 0, 'physics_steps': 0, 'API_calls': 0,
        'native_video_frames_per_camera': sum(r['frame_count'] for r in sources),
        'protocol': binding(output / 'protocol.json')}
    write(output / 'prepared.json', result)
    return result


def worker(root, case_id):
    from resimpair_runtime import configure
    configure()
    root = Path(root).resolve(); protocol, _ = render.verify(root)
    if protocol['closure_background_adapter'] != binding(__file__):
        raise ValueError('worker_adapter_binding_mismatch')
    original = render.entity_mapping
    def mapping(env):
        evidence = env.finite_interior_key_runtime_evidence
        env.scene_domain_runtime_evidence['finite_interior_key'] = evidence
        write(root / 'cases' / case_id / 'finite_light_runtime.json', evidence)
        return original(env)
    with lighting.install_for_worker(), patch.object(render, 'entity_mapping', side_effect=mapping):
        return render.worker(root, case_id)


@contextmanager
def dispatch():
    original = render.subprocess
    class LocalSubprocess:
        def __getattr__(self, name): return getattr(original, name)
        def Popen(self, command, *args, **kwargs):
            command = list(command)
            if command[1:4] != ['-m', 'resimpair_tasks.scene_quality_render_v3', 'worker']:
                raise ValueError('unexpected_child_worker_command')
            command[2] = 'resimpair_tasks.closure_background_replay_v1'
            return original.Popen(command, *args, **kwargs)
    with patch.object(render, 'subprocess', LocalSubprocess()):
        yield


def run(root, stage, limit, gate=None):
    protocol, _ = render.verify(root)
    if protocol['closure_background_adapter'] != binding(__file__):
        raise ValueError('supervisor_adapter_binding_mismatch')
    with dispatch():
        return batch.run(root, stage=stage, limit=limit, gate=gate)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare'); q.add_argument('--sources', type=Path, required=True)
    q.add_argument('--root', type=Path, required=True); q.add_argument('--prior-budget', type=Path, required=True)
    q.add_argument('--allocation-id', required=True); q.add_argument('--catalog', type=Path, default=furniture.CATALOG)
    q = sub.add_parser('worker'); q.add_argument('--root', type=Path, required=True); q.add_argument('--case', required=True)
    q = sub.add_parser('run'); q.add_argument('--root', type=Path, required=True)
    q.add_argument('--stage', choices=('screen', 'video'), required=True); q.add_argument('--limit', type=int, required=True)
    q.add_argument('--gate', type=Path)
    a = p.parse_args()
    if a.command == 'prepare': print(prepare(a.sources, a.root, prior_budget=a.prior_budget, allocation_id=a.allocation_id, catalog=a.catalog))
    elif a.command == 'worker': worker(a.root, a.case)
    else: print(run(a.root, a.stage, a.limit, a.gate))
