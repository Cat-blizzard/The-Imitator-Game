"""Canonical bindings for old audit consumers, without rerendering any frame.

The raw derivative and its first audit are retained. This stage only changes
path spelling in the legacy fields whose checked() helper requires absolute
canonical paths. Artifact hashes, source identity and derivative counts remain.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import shutil

from resimpair_grasp.common import binding, read, resolve, write
from .deferred_render_v1 import COPY_EVIDENCE, validate_source

VERSION = 'saved_state_render_canonical_binding_v1'


def package(source_directory, output):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.core import bind as canonical_bind
    source_directory, output = Path(source_directory).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('canonical_derivative_output_must_be_new')
    old = read(source_directory / 'execution_bundle.json'); bundle = deepcopy(old)
    if (old.get('saved_state_rerender') is not True or old.get('physical_execution_count') != 0
            or old.get('new_trajectory') is not False):
        raise ValueError('only_explicit_zero_execution_render_derivatives_supported')
    changes = []

    def canonical(ref, field):
        result = canonical_bind(resolve(ref))
        if result['sha256'] != ref['sha256']:
            raise ValueError('canonical_binding_must_preserve_bytes')
        changes.append({'field': field, 'original': ref, 'canonical': result})
        return result

    for field in ('measurements', 'planned_paths', 'event_protocol', 'planning_declaration'):
        bundle[field] = canonical(old[field], field)
    for name, camera in bundle['cameras'].items():
        for field in ('video', 'initial_image', 'final_image'):
            camera[field] = canonical(old['cameras'][name][field], 'cameras.' + name + '.' + field)
    output.mkdir(parents=True)
    for name in COPY_EVIDENCE:
        shutil.copy2(source_directory / name, output / name)
    # These stay byte-identical and bind the source recorded state at each frame.
    for name in ('frame_mapping.json', 'source_artifacts.json'):
        shutil.copy2(source_directory / name, output / name)
    bundle['source_render_bundle'] = binding(source_directory / 'execution_bundle.json')
    bundle['canonical_binding_stage'] = binding(__file__)
    write(output / 'binding_revision.json', {'schema': VERSION,
        'raw_derivative': bundle['source_render_bundle'], 'changes': changes,
        'media_reencoded': False, 'physical_execution_count': 0, 'api_calls': 0,
        'original_failed_audit': binding(source_directory / 'audit.json')})
    bundle['binding_revision'] = binding(output / 'binding_revision.json')
    write(output / 'execution_bundle.json', bundle)
    return binding(output / 'execution_bundle.json')


def audit(output):
    from resimpair_pickplace.requested_goal_audit import evaluate
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import interface
    output = Path(output)
    b = read(output / 'execution_bundle.json')
    source_bundle_path = resolve(b['source_execution'])
    source = validate_source(source_bundle_path.parent)
    native = evaluate(output, source['bundle']['case'])
    try:
        interface.validate_bundle(b, source['arrays'])
        compatibility = {'accepted': True}
    except Exception as exc:
        compatibility = {'accepted': False, 'reason': str(exc).split('\n')[0],
            'boundary': 'new saved-state derivative schema/count0/native_execution=false; original interface unchanged'}
    result = {'schema': VERSION, 'bundle': binding(output / 'execution_bundle.json'),
        'retained_independent_checks_passed': native['passed'],
        'R2H_schema_compatibility': compatibility, 'new_render_jobs': 0,
        'physical_execution_count': 0, 'new_trajectory': False, 'api_calls': 0}
    write(output / 'packaging_audit.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(package(args.source, args.output))
    print(audit(args.output))
