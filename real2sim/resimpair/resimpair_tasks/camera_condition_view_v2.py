"""Compile a camera-scoped generation view without rewriting the task contract.

Catalog product titles describe source assets, not necessarily their scaled,
cropped rendered appearance. Preserve those titles in provenance; present image
instances and geometry to the generator. An out-of-frame *center* is not an
occlusion proof: nonparticipants stay peripheral/extent-unknown, while required
participants without an in-frame anchor block this candidate preparation.
No new images or videos are claimed by this local compiler.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import math

from resimpair_grasp.common import binding, read, write

VERSION = 'camera_scoped_catalog_instances_v2'
INVARIANTS = ('required_events', 'required_event_precedence', 'final_relations',
              'persistent_constraints', 'initial_hands', 'terminal_hands', 'hand_correspondence')


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def compile_view(contract):
    original = contract['generation_information']
    if original.get('task_family') != 'relocate' or original.get('spatial_grounding', {}).get('functional_features'):
        raise ValueError('candidate_scope_is_generic_rigid_pick_and_place_only')
    participants = set(strings({k: original.get(k, []) for k in
        ('required_events', 'final_relations', 'spatial_goal_relations')}))
    view = deepcopy(original)
    scope, provenance, blocked = [], [], []
    for entity in view['inventory']:
        eid = entity['id']; xy = entity.get('initial_image_xy_normalized')
        needed = entity.get('role') == 'target' or eid in participants
        if xy is None:
            status = 'reference_only_projection_unavailable'
            if needed and entity.get('kind') != 'plane':
                blocked.append({'entity': eid, 'code': 'required_participant_has_no_camera_anchor'})
        else:
            if len(xy) != 2 or not all(type(v) in (int, float) and math.isfinite(v) for v in xy):
                raise ValueError('invalid_camera_projection')
            status = ('center_in_frame_visibility_unverified' if all(0 <= v <= 1 for v in xy)
                      else 'center_outside_frame_visible_extent_unknown')
            if needed and status.startswith('center_outside'):
                blocked.append({'entity': eid, 'code': 'required_participant_needs_camera_or_visibility_evidence'})
        provenance.append({'entity': eid, 'catalog_description': entity.get('description'),
            'projected_center': xy, 'camera_scope': status, 'required_participant': needed})
        if entity.get('kind') in {'catalog_mesh', 'stationary_mesh_container'}:
            entity['description'] = ('the SAME rendered instance ' + eid +
                (' anchored at image coordinates ' + str(xy) if xy is not None else '') +
                '; identify its shape, texture, scale and visible extent from the supplied images')
        entity['camera_scope'] = status
        scope.append({'entity': eid, 'scope': status, 'required_participant': needed})
    view['camera_conditioning'] = {
        'schema': VERSION, 'appearance_authority': 'supplied_human_anchor_and_simulation_reference_pixels',
        'catalog_product_names_are_not_appearance_targets': True,
        'fixed_camera': True, 'center_projection_is_not_visibility_proof': True,
        'peripheral_entities': [r['entity'] for r in scope if r['scope'].startswith('center_outside')],
        'peripheral_rule': 'Preserve only the portion already visible in the references, if any. '
                           'Do not introduce, enlarge, complete or move a peripheral item into the shot.',
        'instance_rule': 'One physical instance per entity ID. Never draw an extra canonical product '
                         'to explain a small or unfamiliar existing instance.'}
    for name in INVARIANTS:
        if view.get(name) != original.get(name):
            raise ValueError('task_semantics_changed:' + name)
    return {'schema': VERSION, 'generation_information': view,
        'provenance_only_not_model_input': provenance, 'blockers': blocked,
        'ready_for_candidate_generation': not blocked,
        'task_semantics_and_effector_ownership_unchanged': True,
        'evidence_limits': ['No segmentation or occlusion certification',
            'No claim that video quality improved before a controlled generation comparison'],
        'video_quality': 'unmeasured', 'default_promoted': False}


def prepare(contract_path, output):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.prompts import video_prompt
    contract_path, output = Path(contract_path), Path(output)
    if output.exists():
        raise ValueError('preserve_existing_generation_view')
    contract = read(contract_path)
    candidate = compile_view(contract)
    write(output / 'generation_view.json', candidate)
    proposed = deepcopy(contract)
    proposed['generation_information'] = candidate['generation_information']
    # This prompt is a separately bound request input, not a forged completed
    # contract; native evidence/acceptance still use the original full contract.
    output.mkdir(parents=True, exist_ok=True)
    if candidate['ready_for_candidate_generation']:
        (output / 'video_prompt.txt').write_text(video_prompt(proposed))
    write(output / 'protocol.json', {'schema': VERSION, 'original_contract': binding(contract_path),
        'view': binding(output / 'generation_view.json'), 'implementation': binding(__file__),
        'prompt': binding(output / 'video_prompt.txt') if (output / 'video_prompt.txt').exists() else None,
        'task_contract_modified': False, 'api_calls': 0, 'video_quality': 'unmeasured'})
    return {'ready': candidate['ready_for_candidate_generation'], 'blockers': candidate['blockers'],
            'peripheral_entities': candidate['generation_information']['camera_conditioning']['peripheral_entities'],
            'api_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); print(prepare(a.contract, a.output))
