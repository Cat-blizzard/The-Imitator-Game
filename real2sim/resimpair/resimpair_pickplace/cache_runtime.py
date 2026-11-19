"""Generate native trajectories from a precomputed pose folder, without solvers.

The only GPU work in this entry point is simulation/rendering. No cuDex,
GraspGenX, learned weights, training, or paid generation is imported or called.
"""
import argparse
from pathlib import Path
import subprocess

from resimpair_grasp.common import ROOT, binding, read, resolve, write, sha
from .pose_bundle import materialize


def catalog(args):
    from .pose_bundle import inspect, checked_file
    manifest = inspect(args.library); objects = []
    for item in manifest['objects']:
        if args.method not in item['methods']: continue
        d = read(checked_file(args.library, item['descriptor']))
        poses = read(checked_file(args.library, item['methods'][args.method]))['candidates']
        dimensions = [2 * h for h in d['half_size']]
        if not poses and not args.include_empty: continue
        if args.max_dimension_m is not None and max(dimensions) > args.max_dimension_m: continue
        objects.append({'source_id': d['source_id'], 'name': d['name'], 'partition': d['partition'],
            'dimensions_m': dimensions, 'cached_poses': len(poses),
            'width_range_m': [min(p['width_m'] for p in poses), max(p['width_m'] for p in poses)] if poses else None})
    containers = []
    for b in manifest['containers']:
        d = read(checked_file(args.library, b))
        containers.append({'source_id': d['source_id'], 'name': d['name'], 'partition': d['partition'],
            'dimensions_m': d['dimensions_m'], 'inner_floor_radius_m': d['cavity']['flat_floor_radius_m']})
    return {'method': args.method, 'objects': objects, 'containers': containers,
            'pose_generation_calls': 0, 'sizes_are_fixed_cache_geometry': True}


def physical_lineage(root=ROOT):
    """Carry published counts and add new local reservations exactly once."""
    root = Path(root)
    path = root / 'provenance/grasp_compute_lineage.json'
    saved = read(path) if path.exists() else {'physical_reserved': 44, 'carried_reservation_hashes': []}
    count = saved['physical_reserved']; carried = set(saved.get('carried_reservation_hashes', []))
    budgets = set()
    for registry in ['cached_pose_campaign_registry', 'grasp_benchmark_registry']:
        for p in (root / 'runtime' / registry).glob('*.json'):
            value = read(p); budget = Path(value.get('budget', value.get('ledger', ''))).resolve()
            if not (budget / 'limits.json').exists(): raise ValueError('Registered physical ledger missing')
            budgets.add(budget)
    for budget in budgets:
        for marker in (budget / 'reservations').glob('*.json'):
            key = sha(marker)
            if key not in carried:
                count += 1; carried.add(key)
    return count


def initialize(args):
    from .scenes import sample, container_choices
    from .cartesian_reachability import bind_selection
    from .support_surface import check_support
    from .runner import code
    from . import evaluator
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import core, resources
    import trimesh
    root = args.root.resolve(); budget = args.budget.resolve()
    if root.exists(): raise ValueError('Campaign already exists; use run/audit to resume it')
    requests = getattr(args, 'requests', None)
    release_policy = getattr(args, 'release_policy', None) or 'legacy_v1'
    if release_policy not in {'legacy_v1', 'checked_retreat_v1'}: raise ValueError('Unknown release policy')
    if requests is not None:
        from .requested_tasks import request_index
        request_index(read(requests))
    if not 1 <= args.assets <= 24 or not 1 <= args.layouts <= 4 or not 1 <= args.physical_limit <= 96:
        raise ValueError('Bounded asset/layout/physical limits required')
    if args.method not in {'cudex', 'graspgenx', 'hybrid'}: raise ValueError('Unknown cached method')
    snapshot = resources.gpu_snapshot(args.gpu)
    if not (budget / 'limits.json').exists():
        lineage = ROOT / 'provenance/grasp_compute_lineage.json'
        prior = physical_lineage()
        write(budget / 'limits.json', {'physical_attempt_limit': args.physical_limit,
            'api_limit': 0, 'max_concurrent': 1, 'gpu': args.gpu,
            'prior_spent': prior,
            'cumulative_ceiling': prior + args.physical_limit,
            'prior_lineage': binding(lineage) if lineage.exists() else None})
    limits = read(budget / 'limits.json')
    if limits['api_limit'] != 0 or limits.get('gpu', args.gpu) != args.gpu:
        raise ValueError('Budget resource mismatch')
    indices = materialize(args.library, root / 'library')
    if args.method not in indices: raise ValueError('Method absent from pose bundle')
    entries = read(indices[args.method])['entries']
    if requests is not None:
        from .requested_tasks import register
        from . import requested_goal_audit
        write(root / 'experiment.json', {'schema': 'general_pickplace_experiment_v1',
            'budget': str(budget), 'gpu': args.gpu, 'gpu_uuid': snapshot['uuid'],
            'timeout_seconds': 360, 'code': code(), 'legacy_code': core.implementation(),
            'api_limit': 0, 'seed': args.seed, 'cached_method': args.method,
            'release_policy': release_policy,
            'motion_policy': getattr(args, 'motion_policy', 'legacy_v1'),
            'contact_diagnostic': getattr(args, 'contact_diagnostic', False),
            'contact_policy': getattr(args, 'contact_policy', 'native_default'),
            'library': binding(args.library / 'manifest.json'), 'optimizer_imported': False,
            'requests': binding(requests), 'requested_goal_rule': requested_goal_audit.VERSION,
            'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
        write(root / 'rules.json', {'version': requested_goal_audit.VERSION,
            'files': {**evaluator.bindings(), 'requested_goal_evaluator.py': binding(requested_goal_audit.__file__)},
            'frozen_before_attempts': True, 'retained_rules_unchanged': True})
        result = register(read(requests), indices, root / 'library/container_catalog.json', root, args.method, args.seed)
        registry = ROOT / 'runtime/cached_pose_campaign_registry' / (root.name + '.json')
        write(registry, {'root': str(root), 'budget': str(budget), 'experiment': binding(root / 'experiment.json')})
        return result
    entries = [e for e in entries if read(resolve(e['cache']))['candidates']]
    if args.source: entries = [e for e in entries if read(resolve(e['asset']))['source_id'] in args.source]
    entries = entries[:args.assets]
    if not entries: raise ValueError('No matching nonempty native pose cache')
    containers = root / 'library/container_catalog.json'
    clutter = ROOT / 'assets/expanded_library/scene_catalog_v3.json'
    write(root / 'experiment.json', {'schema': 'general_pickplace_experiment_v1',
        'budget': str(budget), 'gpu': args.gpu, 'gpu_uuid': snapshot['uuid'],
        'timeout_seconds': 360, 'code': code(), 'legacy_code': core.implementation(),
        'api_limit': 0, 'seed': args.seed, 'cached_method': args.method,
        'release_policy': release_policy,
        'motion_policy': getattr(args, 'motion_policy', 'legacy_v1'),
        'contact_diagnostic': getattr(args, 'contact_diagnostic', False),
        'contact_policy': getattr(args, 'contact_policy', 'native_default'),
        'library': binding(args.library / 'manifest.json'), 'optimizer_imported': False,
        'baseline_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    write(root / 'rules.json', {'version': evaluator.VERSION, 'files': evaluator.bindings(),
        'frozen_before_attempts': True, 'retained_rules_unchanged': True})
    cases, rejected = [], []
    for ai, entry in enumerate(entries):
        descriptor = read(resolve(entry['asset']))
        choices = container_choices(containers, descriptor['partition'], descriptor['half_size'])
        verified = []
        for b, d in choices:
            mesh = trimesh.load(resolve(d['collision']), force='mesh', process=False)
            if check_support(mesh, d['cavity'])['passed']: verified.append((b, d))
        verified.sort(key=lambda x: (x[1]['dimensions_m'][2], x[1]['id']))
        if not verified and args.tasks != 'table':
            rejected.append({'source_id': descriptor['source_id'], 'reason': 'no_compatible_upward_support_container'})
            continue
        for layout in range(args.layouts):
            chosen = verified[(ai + layout) % len(verified):(ai + layout) % len(verified) + 1] if verified else []
            try:
                scene, tasks = sample(entry, chosen, clutter, args.seed + ai * 100 + layout, layout)
            except ValueError as error:
                rejected.append({'source_id': descriptor['source_id'], 'layout': layout, 'reason': str(error)})
                continue
            shared = root / 'scenes' / (scene['id'] + '.json'); write(shared, scene)
            for task in tasks:
                is_table = task['operations'][0]['target']['support'] == 'table'
                if args.tasks == 'table' and not is_table: continue
                if args.tasks == 'container' and is_table: continue
                folder = root / 'inputs' / task['id']; write(folder / 'role_intent.json', task)
                selected, planned = bind_selection(scene, task, folder)
                case = {'id': task['id'], 'execution_id': task['id'], 'asset': entry['asset'],
                    'source_id': descriptor['source_id'], 'partition': descriptor['partition'],
                    'split_group': descriptor['source_id'], 'layout': layout, 'method': args.method,
                    'shared_scene': binding(shared), 'source_coverage': 'synthetic_only',
                    'eligible': selected is not None, 'arm_selection': binding(folder / 'arm_selection.json')}
                if selected is not None:
                    write(folder / 'scene.json', selected); write(folder / 'task.json', planned)
                    case.update(scene=core.bind(folder / 'scene.json'), task=core.bind(folder / 'task.json'))
                else: case['preflight_reason'] = 'no_single_arm_source_and_destination_grasp'
                cases.append(case)
    write(root / 'cohort.json', {'cases': cases, 'scene_rejections': rejected, 'failures_retained': True,
        'source_coverage': 'synthetic_only', 'runtime_pose_generation': False})
    registry = ROOT / 'runtime/cached_pose_campaign_registry' / (root.name + '.json')
    write(registry, {'root': str(root), 'budget': str(budget), 'experiment': binding(root / 'experiment.json')})
    return {'registered': len(cases), 'eligible': sum(c['eligible'] for c in cases),
            'scene_rejections': len(rejected), 'optimizer_calls': 0, 'api_calls': 0}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); commands = p.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('catalog')
    listing.add_argument('--library', type=Path, default=ROOT / 'assets/precomputed_grasps_v1')
    listing.add_argument('--method', choices=['cudex', 'graspgenx', 'hybrid'], default='cudex')
    listing.add_argument('--max-dimension-m', type=float)
    listing.add_argument('--include-empty', action='store_true')
    init = commands.add_parser('init'); init.add_argument('--library', type=Path, default=ROOT / 'assets/precomputed_grasps_v1')
    init.add_argument('--method', default='cudex'); init.add_argument('--budget', type=Path, required=True)
    init.add_argument('--gpu', type=int, default=2); init.add_argument('--assets', type=int, default=1)
    init.add_argument('--layouts', type=int, default=1); init.add_argument('--seed', type=int, default=20261027)
    init.add_argument('--physical-limit', type=int, default=1); init.add_argument('--source', action='append')
    init.add_argument('--requests', type=Path, help='Metric scenes and target requests; reuse cached poses and the shared controller')
    init.add_argument('--release-policy', choices=['legacy_v1', 'checked_retreat_v1'],
                      help='Default legacy_v1; checked_retreat_v1 is a failed development candidate retained for diagnosis')
    init.add_argument('--contact-diagnostic', action='store_true', help='Read-only native contact trace during release')
    init.add_argument('--motion-policy', choices=['legacy_v1', 'cartesian_transfer_v1', 'transport_fallback_v2'], default='legacy_v1',
                      help='Optional short Cartesian approach/lift/retreat, checked by the retained path guard')
    init.add_argument('--contact-policy', choices=['native_default', 'small_tabletop_v1'], default='native_default',
                      help='Explicit simulator contact-generation profile; does not change rest offsets or acceptance rules')
    init.add_argument('--tasks', choices=['container', 'table', 'all'], default='container')
    for name in ['run', 'audit']:
        child = commands.add_parser(name); child.add_argument('--case', action='append', default=[])
        child.add_argument('--limit', type=int, default=1)
    for name, child in commands.choices.items():
        if name != 'catalog': child.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'catalog':
        import json
        print(json.dumps(catalog(args), ensure_ascii=False, indent=2))
    elif args.command == 'init': print(initialize(args))
    else:
        from . import stability_runner
        if args.command == 'audit' and read(args.root / 'experiment.json').get('requested_goal_rule'):
            from .requested_goal_audit import audit
            print(audit(args))
        else: print(getattr(stability_runner, args.command)(args))
