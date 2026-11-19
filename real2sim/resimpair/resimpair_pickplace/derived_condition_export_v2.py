"""Strict independently validated saved-state render to R2H condition adapter.

Preserves the v1 compiler and its outputs, adding the independently authored
lineage/camera/endpoint-image validator before any condition is exported.
No render is relabelled as a new native physics execution.
"""
import argparse
from pathlib import Path

from resimpair_grasp.common import binding, read, write

VERSION = 'independently_validated_render_condition_adapter_v2'


def export(directory, output):
    from . import render_lineage_validator_v2 as validator
    from . import derived_condition_export_v1 as compiler
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_derived_condition_output')
    before = binding(directory / 'execution_bundle.json')
    proof = validator.validate(directory)
    if proof.get('passed') is not True:
        raise ValueError('independent_render_lineage_validation_failed')
    if before != binding(directory / 'execution_bundle.json'):
        raise ValueError('render_bundle_changed_during_validation')
    write(output / 'render_validation.json', proof)
    retained = compiler.export(directory, output / 'retained_compiler_v1')
    original = read(retained['contract'])
    value = {**original, 'input_adapter': VERSION,
             'adapter_validation': binding(output / 'render_validation.json'),
             'retained_condition_compilation': binding(retained['contract']),
             'adapter_code': {'adapter': binding(__file__), 'validator': binding(validator.__file__),
                              'retained_compiler': binding(compiler.__file__)}}
    write(output / 'contract.json', value)
    write(output / 'qualification.json', {'schema': VERSION,
        'render_bundle': before, 'source_native_execution': value['source_native_execution'],
        'render_validation': value['adapter_validation'], 'contract': binding(output / 'contract.json'),
        'original_native_and_requested_goal_acceptance_unchanged': True,
        'new_physical_executions': 0, 'new_physical_trajectories': 0,
        'api_calls': 0, 'human_generation_started': False, 'original_v5_interface_unchanged': True})
    return {'contract': str(output / 'contract.json'), 'api_calls': 0,
            'new_physical_executions': 0, 'human_generation_started': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); print(export(args.execution, args.output))
