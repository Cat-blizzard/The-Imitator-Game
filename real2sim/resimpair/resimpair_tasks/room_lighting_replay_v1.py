"""Finite, saved-state comparison for an interior light implementation repair.

Reuse the retained GPU leases, submit-once ledger, pixel/state exports, native
clock and independent video gate. Only the child worker module is redirected.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
from unittest.mock import patch

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import finite_interior_light_v1 as lighting
from . import visual_furniture_bulk_render_v4 as batch
from . import scene_quality_render_v3 as render

VERSION = 'finite_room_light_saved_replay_v1'
DEFAULT_PARENT = ROOT / 'runs/metric_group_functional_rooms_20261030_v1'
DEFAULT_PRIOR = ROOT / 'runs/metric_group_resolution_20261030_v1/budget/limits.json'


def prepare(output, parent=DEFAULT_PARENT, prior_budget=DEFAULT_PRIOR):
    output, parent = Path(output).resolve(), Path(parent).resolve()
    if output.exists():
        raise ValueError('new_finite_light_campaign_directory_required')
    old, _ = render.verify(parent)
    selected = [c for c in old['cases'] if c['variant'] == 'environment_3d'
                and (c['stage'] == 'screen' or c['room'] == 'dining')]
    if (len(selected) != 3 or sum(c['stage'] == 'screen' for c in selected) != 2
            or {c['room'] for c in selected if c['stage'] == 'screen'} != {'office', 'dining'}
            or [c['room'] for c in selected if c['stage'] == 'video'] != ['dining']):
        raise ValueError('exact_office_dining_two_screen_one_dining_video_required')
    cases, controls, diagnostics = [], [], []
    for old_case in selected:
        previous = parent / 'cases' / old_case['id']; folder = output / 'cases' / old_case['id']
        if read(previous / 'terminal.json')['returncode'] != 0:
            raise ValueError('actual_completed_control_required')
        scene = read(resolve(old_case['scene'])); candidate, diagnostic = lighting.prepare(scene)
        write(folder / 'render_scene.json', candidate)
        write(folder / 'light_ray_diagnosis.json', diagnostic)
        for name in render.original.COPY_EVIDENCE:
            shutil.copy2(previous / name, folder / name)
        for name in ['source_artifacts.json', 'motion_envelope.json', 'metadata_repair.json']:
            if (previous / name).exists(): shutil.copy2(previous / name, folder / name)
        case = {**deepcopy(old_case), 'scene': binding(folder / 'render_scene.json'),
                'finite_light_source_scene': old_case['scene']}
        if case.get('metadata_repair'):
            case['metadata_repair'] = binding(folder / 'metadata_repair.json')
        cases.append(case); diagnostics.append(binding(folder / 'light_ray_diagnosis.json'))
        controls.append({'new_case_id': case['id'], 'parent_case': old_case,
            'terminal': binding(previous / 'terminal.json'), 'worker_result': binding(previous / 'worker_result.json'),
            'frame_mapping': binding(previous / 'frame_mapping.json'), 'reused_without_new_render': True,
            'comparison': 'same 512x512 room scene and source states; finite key only'})
    registration, limits = batch.register_budget(output, prior_budget,
        'metric_group_finite_light_20261030_two_screens_one_dining_video', screens=2, videos=1)
    if limits['historical_actual_render_jobs'] != 130 or limits['maximum_lineage_render_jobs_including_history'] != 136:
        raise ValueError('expected_original_130_actual_133cap_plus_fixed_3_amendment')
    selection = deepcopy(read(resolve(old['independent_protocol'])))
    selection.update(schema=VERSION, cases=cases, author='implementation_agent_pending_independent_screen_review',
        parent_actual_protocol=binding(parent / 'protocol.json'),
        existing_controls=controls, light_ray_diagnoses=diagnostics,
        attribution_scope='only infinite directional key replaced by finite interior point key',
        source_states_materials_geometry_FOV_camera_ambient_fill_HDRI_unchanged=True,
        independent_rule_revision='root-appointed reviewer must freeze before screen launch',
        predeclared_control_video_case_ids=[])
    write(output / 'selection_protocol.json', selection)
    protocol = deepcopy(old)
    protocol.update(cases=cases, independent_protocol=binding(output / 'selection_protocol.json'),
        finite_light_adapter=binding(__file__), finite_light_implementation=binding(lighting.__file__),
        parent_actual_protocol=binding(parent / 'protocol.json'),
        budget=binding(output / 'budget/limits.json'), budget_registration=registration)
    for path in [Path(__file__), Path(lighting.__file__), lighting.SHADER,
                 ROOT / '.venv/lib/python3.12/site-packages/sapien/vulkan_shader/default/deferred.frag',
                 ROOT / 'PolicyGen/mani_skill/envs/scene.py']:
        protocol['code'][str(path.relative_to(ROOT))] = binding(path)
    write(output / 'protocol.json', protocol)
    refs = {'protocol': binding(output / 'protocol.json')}
    for ref in render.bindings_in(protocol): resolve(ref); refs['dependency_' + digest(ref)] = ref
    for case in cases:
        for ref in render.bindings_in(read(resolve(case['scene']))):
            resolve(ref); refs['scene_resource_' + digest(ref)] = ref
    write(output / 'frozen_inputs.json', refs)
    batch.validate_prepared_interface(output); render.verify(output)
    value = {'schema': VERSION, 'sources': 2, 'screen_jobs': 2, 'video_jobs': 1,
        'historical_actual_render_jobs': 130, 'new_allowance': 3, 'lineage_cap': 136,
        'old_unspent_allowance_reassigned': 0, 'actual_new_render_jobs': 0,
        'new_physics_steps': 0, 'API_calls': 0, 'default_promoted': False,
        'protocol': binding(output / 'protocol.json'),
        'command': '.venv/bin/python -m resimpair_tasks.room_lighting_replay_v1 run --root ' + str(output) + ' --stage screen --limit 2'}
    write(output / 'prepared.json', value)
    return value


def worker(root, case_id):
    from resimpair_runtime import configure
    configure()
    root = Path(root).resolve(); protocol, _ = render.verify(root)
    if protocol['finite_light_adapter'] != binding(__file__):
        raise ValueError('worker_adapter_binding_mismatch')
    old_mapping = render.entity_mapping
    def mapping(env):
        evidence = env.finite_interior_key_runtime_evidence
        env.scene_domain_runtime_evidence['finite_interior_key'] = evidence
        write(root / 'cases' / case_id / 'finite_light_runtime.json', evidence)
        return old_mapping(env)
    with lighting.install_for_worker(), patch.object(render, 'entity_mapping', side_effect=mapping):
        return render.worker(root, case_id)


def run(root, stage, limit, gate=None):
    root = Path(root).resolve(); protocol, _ = render.verify(root)
    if protocol['finite_light_adapter'] != binding(__file__):
        raise ValueError('supervisor_adapter_binding_mismatch')
    original = render.subprocess.Popen
    def spawn(command, *args, **kwargs):
        command = list(command)
        if command[1:4] != ['-m', 'resimpair_tasks.scene_quality_render_v3', 'worker']:
            raise ValueError('unexpected_child_command_no_generic_subprocess_interception')
        command[2] = 'resimpair_tasks.room_lighting_replay_v1'
        return original(command, *args, **kwargs)
    with patch.object(render.subprocess, 'Popen', side_effect=spawn):
        return batch.run(root, stage=stage, limit=limit, gate=gate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--root', type=Path, required=True)
    p = sub.add_parser('worker'); p.add_argument('--root', type=Path, required=True); p.add_argument('--case', required=True)
    p = sub.add_parser('run'); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['screen', 'video'], required=True); p.add_argument('--limit', type=int, required=True); p.add_argument('--gate', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare': print(prepare(args.root))
    elif args.command == 'worker': worker(args.root, args.case)
    else: print(run(args.root, args.stage, args.limit, args.gate))
