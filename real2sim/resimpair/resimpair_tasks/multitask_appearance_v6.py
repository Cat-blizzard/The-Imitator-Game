"""Keep optional metric furniture failures inside the appearance layer.

v5's measured office replay is retained unchanged. Its reused v3 furniture
adapter assumes both side walls; some open-workspace presets do not have them.
This wrapper preserves the successfully prepared surfaces, lights and room if
metric furniture is unavailable or incompatible. It does not retry rendering.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from resimpair_grasp.common import binding, digest, read, resolve, write
from . import multitask_appearance_v5 as retained

VERSION = 'multitask_optional_appearance_v6'


def apply_overlay(scene, seed, *, mode='metric_room', **kwargs):
    if mode != 'metric_room':
        result, receipt = retained.apply_overlay(scene, seed, mode=mode, **kwargs)
        return result, {**receipt, 'adapter_version': VERSION, 'adapter': binding(__file__),
                        'effective_mode': mode, 'requested_mode': mode, 'furniture_status': 'not_requested'}
    prepared, receipt = retained.apply_overlay(scene, seed, mode='workspace', **kwargs)
    wrapper = {'adapter_version': VERSION, 'adapter': binding(__file__),
               'requested_mode': mode, 'effective_mode': 'workspace', 'furniture_status': 'not_attempted'}
    if not receipt['appearance_applied']:
        return prepared, {**receipt, **wrapper}
    geometry = prepared['scene_domain']['geometry']
    box_ids = {r['id'] for r in geometry['boxes']}
    missing = sorted({'floor', 'left_wall', 'right_wall'} - box_ids)
    if missing:
        return prepared, {**receipt, **wrapper, 'furniture_status': 'incompatible_open_workspace',
            'furniture_error': {'code': 'required_architecture_absent', 'missing': missing},
            'status': 'prepared_without_optional_furniture_requires_render_validation'}
    try:
        candidate = retained.quality.apply_furniture(prepared,
            catalog_path=kwargs.get('furniture_catalog', retained.quality.FURNITURE))
    except (FileNotFoundError, ValueError, OSError, StopIteration) as exc:
        # Source-specific missing files, fit constraints and the known open-room
        # assumption must not erase an already valid material/background layer.
        if kwargs.get('allow_visual_fallback', True) is False:
            raise
        return prepared, {**receipt, **wrapper, 'furniture_status': 'unavailable_or_incompatible',
            'furniture_error': {'type': type(exc).__name__, 'message': str(exc)},
            'status': 'prepared_without_optional_furniture_requires_render_validation'}
    retained.assert_invariants(scene, candidate)
    retained.domain.validate_scene(candidate)
    return candidate, {**receipt, **wrapper, 'effective_mode': 'metric_room',
        'furniture_status': 'prepared', 'result_scene_digest': digest(candidate),
        'furniture_ids': [r['source_id'] for r in candidate['scene_domain']['geometry'].get('furniture', [])]}


def prepare(source, output, *, seed, mode='metric_room', **kwargs):
    """Use the frozen renderer adapter with the explicitly resolved visual mode."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_optional_appearance_output')
    source_value = retained.replay.validate_source(source)
    baseline, _ = retained.render.baseline_scene(source_value)
    envelope = retained.domain.build_motion_envelope(baseline, source_value['source'] / 'measurements.npz')
    overlay_keys = {'preset', 'catalog', 'material_index', 'furniture_catalog', 'material_partition'}
    overlay_kwargs = {k: v for k, v in kwargs.items() if k in overlay_keys}
    expected_scene, receipt = apply_overlay(baseline, seed, mode=mode, motion_envelope=envelope,
                                           **overlay_kwargs)
    actual = retained.prepare(source, output, seed=seed, mode=receipt['effective_mode'], **kwargs)
    protocol = read(output / 'protocol.json')
    candidate_cases = [c for c in protocol['cases'] if c['variant'] != 'baseline']
    if receipt['appearance_applied']:
        if len(candidate_cases) != 1 or read(resolve(candidate_cases[0]['scene'])) != expected_scene:
            raise RuntimeError('resolved_optional_layer_differs_from_frozen_renderer_scene')
    elif candidate_cases:
        raise RuntimeError('degraded_layer_must_not_claim_candidate_render')
    write(output / 'optional_furniture_receipt.json', {**receipt,
        'renderer_protocol': binding(output / 'protocol.json'), 'renderer_adapter': binding(retained.__file__),
        'source_execution': source_value['bindings']['execution_bundle.json']})
    return {**actual, 'optional_furniture_status': receipt['furniture_status'],
            'requested_mode': mode, 'effective_mode': receipt['effective_mode']}


def run(output):
    receipt = read(Path(output) / 'optional_furniture_receipt.json')
    resolve(receipt['adapter']); resolve(receipt['renderer_adapter']); resolve(receipt['renderer_protocol'])
    return retained.run(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--prior-budget', type=Path, required=True)
    p.add_argument('--allocation-id', required=True); p.add_argument('--mode', choices=retained.MODES, default='metric_room')
    p.add_argument('--preset', choices=retained.domain.PRESET_IDS, default='office_bookcase')
    p.add_argument('--stage', choices=('screen', 'video'), default='screen')
    p.add_argument('--without-baseline', action='store_true')
    p = commands.add_parser('run'); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.source, args.output, seed=args.seed, prior_budget=args.prior_budget,
                      allocation_id=args.allocation_id, mode=args.mode, preset=args.preset,
                      stage=args.stage, include_baseline=not args.without_baseline))
    else:
        print(run(args.output))
