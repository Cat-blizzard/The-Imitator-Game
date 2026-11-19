"""Independent CPU audit of real-source unified-anchor bindings.

All mutations are new files; originals and old audit output remain immutable.
This validates protocol behavior, not synthetic human image quality.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from data_generation_methods.r2h_pipeline.batch_pipeline import binding
from data_generation_methods.r2h_pipeline.common import read_json, write_json, file_sha256
from data_generation_methods.r2h_pipeline.unified_anchor_candidate import verify_sources


def audit(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    old = Path('/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1')
    cp = old/'state_conditions/development/conditions.json'
    ip = old/'multiview/development/input.json'
    vp = old/'multiview/validation/input.json'
    packet, source, other = map(read_json, (cp, ip, vp))
    before = {str(p): file_sha256(p) for p in (cp, ip, vp)}
    good = verify_sources(cp, ip)
    cases = []

    def reject(name, mutate_packet=None, mutate_input=None):
        directory = output/name
        directory.mkdir()
        modified_packet, modified_input = deepcopy(packet), deepcopy(source)
        if mutate_packet: mutate_packet(modified_packet)
        if mutate_input: mutate_input(modified_input)
        target_packet, target_input = directory/'conditions.json', directory/'input.json'
        write_json(target_packet, modified_packet)
        write_json(target_input, modified_input)
        try:
            verify_sources(target_packet, target_input)
        except (ValueError, KeyError, TypeError) as exc:
            result = {'case': name, 'blocked': True, 'reason': str(exc),
                      'exception_type': type(exc).__name__}
        else:
            result = {'case': name, 'blocked': False, 'reason': 'Unexpectedly accepted'}
        result['inputs'] = {'conditions': binding(target_packet), 'execution_input': binding(target_input)}
        write_json(directory/'result.json', result)
        cases.append(result)

    def swap_human(x):
        for key in ('human_source_window', 'human_video', 'human_window_id'):
            x['proposal'][key] = deepcopy(other['proposal'][key])
    reject('same_execution_swap_human_source', mutate_input=swap_human)
    reject('foreign_execution_input', mutate_input=lambda x: (x.clear(), x.update(deepcopy(other))))
    reject('change_human_session', mutate_input=lambda x: x['proposal']['human_source_window'].__setitem__('session', 'different-session.mov'))
    reject('change_human_window', mutate_input=lambda x: x['proposal']['human_source_window'].__setitem__('window_frames', [1, 246]))
    reject('change_human_clock', mutate_input=lambda x: x['proposal']['human_source_window'].__setitem__('fps', 60.))
    reject('change_selected_camera_only', mutate_packet=lambda x: x.__setitem__('selected_camera', 'native'))

    def swap_roles(x):
        roles = x['roles']
        roles['source_container'], roles['destination_container'] = roles['destination_container'], roles['source_container']
    reject('swap_entity_source_destination', mutate_packet=swap_roles)

    def reverse_camera_order(x):
        order = x['camera_role_projections'][x['selected_camera']]['relative_order']
        a, b = 'source_relative_to_destination', 'destination_relative_to_source'
        order[a], order[b] = order[b], order[a]
    reject('reverse_screen_role_mapping', mutate_packet=reverse_camera_order)

    def terminal_as_initial(x):
        initial = next(c for c in x['conditions'] if 'initial' in c['labels'])
        terminal = next(c for c in x['conditions'] if 'terminal' in c['labels'])
        initial['frame'] = deepcopy(terminal['frame'])
    reject('terminal_frame_as_initial', mutate_packet=terminal_as_initial)
    reject('foreign_verification', mutate_packet=lambda x: x['bindings'].__setitem__(
        'verification', binding(old/'multiview/validation/verification.json')))
    unchanged = all(file_sha256(p) == sha for p, sha in before.items())
    result = {
        'schema': 'independent_unified_source_mutations_v1',
        'positive_context': good['context'], 'cases': cases,
        'blocked': sum(c['blocked'] for c in cases), 'total': len(cases),
        'old_source_files_unchanged': unchanged,
        'candidate_source': binding(Path(__file__).parents[1]/'r2h_pipeline/unified_anchor_candidate.py'),
        'audit_source': binding(__file__),
        'api_calls': 0, 'gpu_runs': 0,
        'scope': 'Actual-source consistency negative controls, not visual detector accuracy or image-generation success.',
    }
    write_json(output/'audit.json', result)
    if not unchanged or not all(c['blocked'] for c in cases):
        raise ValueError('Independent audit has failed cases; keep artifacts and inspect')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output)
    print({'blocked': result['blocked'], 'total': result['total'], 'api_calls': 0})
