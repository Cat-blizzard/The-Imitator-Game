"""Blind semantic review contract. Approval permits pairing only, never training.

This module is CPU-only and pure: it neither calls a reviewer nor reads media.
The caller must supply actual decoder/request provenance. Unit fixtures exercise
the gate; they do not establish that any generated video has passed review.
"""
from __future__ import annotations

import copy
import json
import math
import re

MIN_CONFIDENCE = .8
MAX_UNCERTAINTY = .2
UNKNOWN = {'unknown', 'unclear', 'uncertain', 'unresolved', ''}
# These verbs change contact, support, object pose or another task state. An
# additional manipulation can undo an otherwise complete required subsequence.
# It needs review even if the VLM still calls the resulting video the right goal.
MANIPULATION_VERBS = {
    'pick_and_place', 'pick', 'pick_up', 'place', 'put_down', 'grasp', 'release',
    'push', 'pull', 'rotate', 'turn', 'open', 'close', 'insert', 'remove', 'stack',
    'unstack', 'slide', 'move', 'move_object', 'lift', 'drop', 'throw', 'transfer',
    'press', 'wipe', 'stir',
}
DEFAULT_CANDIDATES = [
    {'id': 'choice_a', 'description': 'Remove the block initially in the middle of the three-block stack, place it on the blue pad, and restore the other two blocks as a stable vertical stack.'},
    {'id': 'choice_b', 'description': 'Remove the block initially at the bottom of the three-block stack, place it on the brown pad, and restore the other two blocks as a stable vertical stack.'},
    {'id': 'choice_c', 'description': 'Reverse the complete vertical order of the three blocks; finish with all three blocks in a stable stack on the platform.'},
]
DEFAULT_ROLES = [
    {'role': 'block0', 'description': 'Block at the bottom of the INITIAL stack; keep this identity throughout the video.'},
    {'role': 'block1', 'description': 'Block in the middle of the INITIAL stack; keep this identity throughout the video.'},
    {'role': 'block2', 'description': 'Block at the top of the INITIAL stack; keep this identity throughout the video.'},
    {'role': 'platform', 'description': 'The stack support platform.'},
    {'role': 'brown_pad', 'description': 'The brown parking pad.'},
    {'role': 'blue_pad', 'description': 'The blue parking pad.'},
]


def blind_review_prompt(candidates=None, roles=None):
    """Render only symmetric alternatives/role definitions, never a job/answer.

    Do not pass contract dictionaries here. Canonical goal IDs and source paths
    are deliberately excluded; the private alias map is used only by the gate.
    """
    candidates = copy.deepcopy(DEFAULT_CANDIDATES if candidates is None else candidates)
    roles = copy.deepcopy(DEFAULT_ROLES if roles is None else roles)
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError('Blind review requires at least two unmarked alternatives')
    seen = set()
    for c in candidates:
        if not isinstance(c, dict) or set(c) != {'id', 'description'}:
            raise ValueError('Candidate fields must be exactly id/description')
        if not isinstance(c['id'], str) or not re.fullmatch(r'choice_[A-Za-z0-9]+', c['id']) or c['id'] in seen:
            raise ValueError('Require unique anonymous choice IDs')
        seen.add(c['id'])
        if not isinstance(c['description'], str) or not c['description'].strip():
            raise ValueError('Candidate description is required')
    for role in roles:
        if not isinstance(role, dict) or set(role) != {'role', 'description'}:
            raise ValueError('Role fields must be exactly role/description')
        if not all(isinstance(x, str) and x.strip() for x in role.values()):
            raise ValueError('Role descriptions must be nonempty strings')
    public = json.dumps({'alternatives': candidates, 'object_roles': roles}, ensure_ascii=False)
    if any(token in public for token in ('/mnt/', 'factory:', 'expansion:', 'expected_goal', 'correct_answer')):
        raise ValueError('Private source identifiers/answers must not enter blind prompts')
    return (
        'Independently inspect ONLY the supplied candidate video. No intended or correct goal is supplied. '
        'The following are equally plausible alternatives, not instructions for what the video should show. '
        'Choose an anonymous observed_goal only if the ENTIRE visible outcome supports it; otherwise use unknown. '
        'Track object identities from their INITIAL spatial positions, never rename an object after moving it. '
        'List actual actions, including extra/wrong actions, in events. Use verb pick_and_place only when '
        'grasp, transport, placement and release are observed; destination_role names the actual support/pad/block. '
        'Use other literal verbs or unknown when appropriate, never fill in an unobserved action. '
        'For every event give seconds within this uploaded video, and timestamped visible evidence. '
        'Report full_task only if the initial setup, all necessary operations and stable released final outcome '
        'are visible; a cropped substep is partial, ambiguity is unknown. Set initial_state_visible and '
        'final_state_visible honestly. Record issues with severity blocking/uncertain/minor and whether they '
        'affect a necessary event. Confidence and uncertainty are numbers in [0,1]; uncertainty is higher when '
        'events or identities cannot be established. Do not infer motion, contact, 3D truth or success from a '
        'caption, file name, visual style or intended goal. Return only JSON conforming to the response schema.\n'
        + public
    )


def review_response_schema():
    evidence = {'type': 'object', 'additionalProperties': False,
                'properties': {'timestamp_seconds': {'type': 'number', 'minimum': 0},
                               'description': {'type': 'string', 'minLength': 1}},
                'required': ['timestamp_seconds', 'description']}
    return {'type': 'object', 'additionalProperties': False,
            'properties': {
                'observed_goal': {'type': 'string'},
                'goal_confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                'window_scope': {'type': 'string', 'enum': ['full_task', 'partial', 'unknown']},
                'initial_state_visible': {'type': 'boolean'}, 'final_state_visible': {'type': 'boolean'},
                'events': {'type': 'array', 'maxItems': 128, 'items': {'type': 'object', 'additionalProperties': False,
                    'properties': {'role': {'type': 'string'}, 'verb': {'type': 'string'},
                        'destination_role': {'type': ['string', 'null']},
                        't_start': {'type': 'number', 'minimum': 0}, 't_end': {'type': 'number', 'minimum': 0},
                        'evidence': {'type': 'array', 'minItems': 1, 'maxItems': 16, 'items': evidence},
                        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}},
                    'required': ['role', 'verb', 'destination_role', 't_start', 't_end', 'evidence', 'confidence']}},
                'issues': {'type': 'array', 'maxItems': 32, 'items': {'type': 'object', 'additionalProperties': False,
                    'properties': {'code': {'type': 'string', 'minLength': 1},
                        'severity': {'type': 'string', 'enum': ['blocking', 'uncertain', 'minor']},
                        'description': {'type': 'string', 'minLength': 1},
                        'affects_required_event': {'type': 'boolean'}},
                    'required': ['code', 'severity', 'description', 'affects_required_event']}},
                'uncertainty': {'type': 'number', 'minimum': 0, 'maximum': 1}},
            'required': ['observed_goal', 'goal_confidence', 'window_scope', 'initial_state_visible',
                         'final_state_visible', 'events', 'issues', 'uncertainty']}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _validate(value, schema, path='review'):
    """Small dependency-free validator for this exact JSON schema subset."""
    types = schema.get('type', [])
    types = [types] if isinstance(types, str) else types
    checks = {'object': lambda: isinstance(value, dict), 'array': lambda: isinstance(value, list),
              'number': lambda: _number(value), 'string': lambda: isinstance(value, str),
              'null': lambda: value is None, 'boolean': lambda: isinstance(value, bool)}
    if types and not any(checks[t]() for t in types):
        return [path + ': invalid type']
    errors = []
    if 'enum' in schema and value not in schema['enum']:
        errors.append(path + ': invalid enum')
    if isinstance(value, dict):
        properties = schema.get('properties', {})
        for key in schema.get('required', []):
            if key not in value:
                errors.append(path + '.' + key + ': required')
        if schema.get('additionalProperties') is False:
            errors += [path + '.' + str(k) + ': unexpected' for k in value if k not in properties]
        for key in value.keys() & properties.keys():
            errors.extend(_validate(value[key], properties[key], path + '.' + key))
    elif isinstance(value, list):
        if len(value) < schema.get('minItems', 0):
            errors.append(path + ': too few items')
        if len(value) > schema.get('maxItems', math.inf):
            return errors + [path + ': too many items']
        if 'items' in schema:
            for i, entry in enumerate(value):
                errors.extend(_validate(entry, schema['items'], f'{path}[{i}]'))
    elif isinstance(value, str):
        if len(value.strip()) < schema.get('minLength', 0):
            errors.append(path + ': empty string')
    elif _number(value):
        if value < schema.get('minimum', -math.inf) or value > schema.get('maximum', math.inf):
            errors.append(path + ': outside numeric range')
    return errors


def evaluate_review(job: dict, review: dict, media: dict | None = None) -> dict:
    """Compare a blind observation with a private contract; never call an API.

    ``job['contract']['review_spec']`` contains private alias -> canonical goal
    mapping, allowed_roles/verbs, required_events, event_order, and expected_window_scope.
    ``job['review_request']`` is CALLER provenance, never VLM-authored content:
    origin=actual_blind_vlm, blind=True, expected_goal_provided=False,
    media_sha256 and prompt_sha256. ``media`` is the actual generated-video
    decoder result, not source robot media: sha256, full_decode_passed/decode_ok,
    frames/num_frames, fps, duration_seconds.
    """
    problems, evidence = [], []
    def add(code, message, reject=False):
        problems.append((code, reject))
        evidence.append({'code': code, 'detail': message})
    def result(matches=None):
        codes = list(dict.fromkeys(code for code, _ in problems))
        decision = ('rejected' if any(reject for _, reject in problems)
                    else 'needs_review' if problems else 'accepted_for_pairing')
        return {'schema': 'r2h_review_decision_v1', 'decision': decision,
                'error_codes': codes, 'evidence': evidence, 'matched_events': matches or [],
                'training_admitted': False,
                'source_scope': {'reference_origin': 'synthetic_human_video',
                    'approval_scope': 'task_level_pairing_only', 'full_task_pairing': decision == 'accepted_for_pairing',
                    'human_3d_ground_truth': False, 'human_action_ground_truth': False,
                    'framewise_human_robot_alignment': False,
                    'goal_id': goal_id}}

    goal_id = None
    if not isinstance(job, dict):
        add('JOB_CONTRACT_INVALID', 'job must be a dictionary')
        return result()
    contract = job.get('contract')
    if not isinstance(contract, dict):
        add('JOB_CONTRACT_MISSING', 'Caller must load the private contract before review')
        return result()
    spec = job.get('review_spec', contract.get('review_spec'))
    if not isinstance(spec, dict):
        add('REVIEW_SPEC_MISSING', 'Private review_spec is required')
        return result()
    source = job.get('source', {})
    if not isinstance(source, dict):
        add('JOB_CONTRACT_INVALID', 'Source identity must be a dictionary')
        return result()
    goal_id = source.get('goal_id') if isinstance(source, dict) else None
    goal_id = goal_id or contract.get('goal_id')
    aliases = spec.get('blind_goal_aliases')
    required = spec.get('required_events')
    roles, verbs = spec.get('allowed_roles'), spec.get('allowed_verbs')
    order = spec.get('event_order')
    if (not _text(goal_id) or not isinstance(aliases, dict) or len(aliases) < 2
            or not all(_text(k) and re.fullmatch(r'choice_[A-Za-z0-9]+', k) and _text(v) for k, v in aliases.items())
            or goal_id not in aliases.values() or len(set(aliases.values())) != len(aliases)
            or not isinstance(required, list) or not 1 <= len(required) <= 32
            or not isinstance(roles, list) or not roles or not all(_text(x) for x in roles)
            or not isinstance(verbs, list) or not verbs or not all(_text(x) for x in verbs)
            or not isinstance(order, list)):
        add('REVIEW_SPEC_INVALID', 'Require explicit private goals, roles, verbs, necessary events and order')
        return result()
    if contract.get('goal_id') and contract['goal_id'] != goal_id:
        add('SOURCE_GOAL_CONTRACT_MISMATCH', 'Source goal differs from private contract', True)
    if (spec.get('expected_window_scope') != 'full_task'
            or contract.get('window_scope', 'full_task') != 'full_task'
            or source.get('window_scope', 'full_task') != 'full_task'):
        add('PARTIAL_SOURCE_NOT_FULL_TASK', 'This gate cannot upgrade partial or unknown source scope')
    required_ids = []
    for entry in required:
        if (not isinstance(entry, dict) or not _text(entry.get('id'))
                or entry.get('role') not in roles or entry.get('verb') not in verbs
                or entry.get('destination_role') not in [None, *roles]
                or (entry.get('verb') == 'pick_and_place' and entry.get('destination_role') not in roles)):
            add('REVIEW_SPEC_INVALID', 'Every necessary event must name valid IDs, roles, verbs and destinations')
            return result()
        required_ids.append(entry['id'])
    if len(set(required_ids)) != len(required_ids):
        add('REVIEW_SPEC_INVALID', 'Necessary event IDs must be unique')
        return result()
    if not all(isinstance(edge, list) and len(edge) == 2 and all(x in required_ids for x in edge)
               and edge[0] != edge[1] for edge in order):
        add('REVIEW_SPEC_INVALID', 'Order edges must reference two distinct necessary events')
        return result()
    reach = {(a, b) for a, b in order}
    for mid in required_ids:
        reach |= {(a, b) for a in required_ids for b in required_ids if (a, mid) in reach and (mid, b) in reach}
    if any((x, x) in reach for x in required_ids):
        add('REVIEW_SPEC_INVALID', 'Event order contains a cycle')
        return result()
    forbidden = spec.get('forbidden_events', [])
    if (not isinstance(forbidden, list) or not all(isinstance(x, dict)
            and x.get('role') in roles and x.get('verb') in verbs
            and x.get('destination_role') in [None, '*', *roles] for x in forbidden)):
        add('REVIEW_SPEC_INVALID', 'Forbidden events require explicit role/verb/destination signatures')
        return result()
    validation = _validate(review, review_response_schema())
    if validation:
        add('REVIEW_SCHEMA_INVALID', '; '.join(validation[:12]))
        return result()
    observed = review['observed_goal']
    if observed.strip().lower() in UNKNOWN:
        add('GOAL_UNKNOWN', 'Reviewer did not establish an observed goal')
    elif observed not in aliases:
        add('GOAL_ALIAS_INVALID', 'Only anonymous observed choices may be compared; canonical echoes are not valid', True)
    elif aliases[observed] != goal_id:
        add('GOAL_MISMATCH', 'Blindly observed goal differs from the source task', True)
    if review['goal_confidence'] < MIN_CONFIDENCE:
        add('GOAL_CONFIDENCE_LOW', 'Goal confidence is below the fixed review threshold')
    if review['uncertainty'] > MAX_UNCERTAINTY:
        add('REVIEW_UNCERTAIN', 'Uncertainty exceeds the fixed review threshold')
    if review['window_scope'] == 'partial':
        add('PARTIAL_CLIP_NOT_FULL_TASK', 'Reviewer observed a partial task window', True)
    elif review['window_scope'] != 'full_task':
        add('WINDOW_SCOPE_UNKNOWN', 'Full task coverage is unknown')
    if not review['initial_state_visible'] or not review['final_state_visible']:
        add('TASK_BOUNDARIES_NOT_VISIBLE', 'Initial setup and stable released final state must both be visible')
    for issue in review['issues']:
        if issue['severity'] == 'blocking':
            add('BLOCKING_VISUAL_ISSUE', issue['code'] + ': ' + issue['description'], True)
        elif issue['severity'] == 'uncertain' or issue['affects_required_event']:
            add('VISUAL_ISSUE_REQUIRES_REVIEW', issue['code'] + ': ' + issue['description'])
    duration, tolerance, media_hash = None, 0., None
    if not isinstance(media, dict):
        add('MEDIA_VERIFICATION_MISSING', 'Actual generated-video decode metadata is required')
    else:
        media_hash = media.get('sha256')
        n = media.get('frames', media.get('num_frames'))
        fps = media.get('fps')
        duration = media.get('duration_seconds')
        valid_clock = (isinstance(n, int) and not isinstance(n, bool) and n >= 2
                       and _number(fps) and fps > 0 and _number(duration) and duration > 0)
        if media.get('full_decode_passed', media.get('decode_ok')) is not True or not _sha(media_hash):
            add('MEDIA_NOT_VERIFIED', 'A verified full decode and generated-video SHA256 are required')
        if not valid_clock:
            add('MEDIA_CLOCK_INVALID', 'Require valid generated frame count, FPS and duration')
            duration = None
        else:
            tolerance = 1 / fps + 1e-8
            if abs(n / fps - duration) > 2 / fps + 1e-8:
                add('MEDIA_CLOCK_INCONSISTENT', 'Duration differs from frame-count cadence', True)
    request = job.get('review_request')
    if (not isinstance(request, dict) or request.get('origin') != 'actual_blind_vlm'
            or request.get('blind') is not True or request.get('expected_goal_provided') is not False
            or not _sha(request.get('prompt_sha256'))):
        add('BLIND_REQUEST_PROVENANCE_MISSING', 'Caller must attest and bind the actual blind reviewer request')
    elif not _sha(media_hash) or request.get('media_sha256') != media_hash:
        add('REVIEW_MEDIA_BINDING_MISMATCH', 'Reviewer request does not bind this generated video', True)
    events = review['events']
    seen = set()
    for i, event in enumerate(events):
        role, verb, destination = event['role'], event['verb'], event['destination_role']
        if any(isinstance(x, str) and x.strip().lower() in UNKNOWN for x in (role, verb, destination)):
            add('EVENT_IDENTITY_UNKNOWN', f'Event {i} has an unknown role, verb or destination')
        elif role not in roles or verb not in verbs or destination not in [None, *roles]:
            add('EVENT_ROLE_OR_VERB_INVALID', f'Event {i} names an object/action outside the blinded role catalog', True)
        if verb == 'pick_and_place' and destination is None:
            add('EVENT_DESTINATION_UNKNOWN', f'Event {i} has no established placement destination')
        if any(role == x['role'] and verb == x['verb']
               and x.get('destination_role') in ('*', destination) for x in forbidden):
            add('FORBIDDEN_EVENT_OBSERVED', f'Event {i} violates an explicit non-target invariant', True)
        start, end = event['t_start'], event['t_end']
        if end < start or (verb == 'pick_and_place' and end <= start):
            add('EVENT_TIME_INVALID', f'Event {i} has an invalid action interval', True)
        if duration is not None and end > duration + tolerance:
            add('EVENT_OUTSIDE_MEDIA', f'Event {i} exceeds the actual generated-video duration', True)
        key = (role, verb, destination, start, end)
        if key in seen:
            add('DUPLICATE_OBSERVED_EVENT', f'Event {i} duplicates an already reported event', True)
        seen.add(key)
        for point in event['evidence']:
            if point['description'].strip().lower() in UNKNOWN:
                add('EVENT_EVIDENCE_UNKNOWN', f'Event {i} evidence is unknown')
            t = point['timestamp_seconds']
            if t < start - tolerance or t > end + tolerance or (duration is not None and t > duration + tolerance):
                add('EVENT_EVIDENCE_TIME_INVALID', f'Event {i} evidence timestamp does not support its interval', True)
    choices = {}
    for expected in required:
        signature = (expected['role'], expected['verb'], expected.get('destination_role'))
        choices[expected['id']] = [i for i, observed_event in enumerate(events)
            if (observed_event['role'], observed_event['verb'], observed_event['destination_role']) == signature]
        if not choices[expected['id']]:
            add('REQUIRED_EVENT_MISSING', 'Necessary event not established: ' + expected['id'])
    assignment = {}
    def match(k):
        if k == len(required_ids):
            return True
        name = required_ids[k]
        for i in choices[name]:
            if i in assignment.values():
                continue
            assignment[name] = i
            consistent = all(a not in assignment or b not in assignment
                or events[assignment[a]]['t_end'] <= events[assignment[b]]['t_start'] + tolerance
                for a, b in order)
            if consistent and match(k + 1):
                return True
            assignment.pop(name)
        return False
    matched = bool(all(choices.values())) and match(0)
    if all(choices.values()) and not matched:
        add('EVENT_ORDER_MISMATCH', 'Necessary events cannot be matched one-to-one in the required temporal order', True)
    matches = []
    if matched:
        for name, i in assignment.items():
            event = events[i]
            if event['confidence'] < MIN_CONFIDENCE:
                add('EVENT_CONFIDENCE_LOW', 'Necessary event confidence is low: ' + name)
            matches.append({'required_event_id': name, 'observed_event_index': i,
                            't_start': event['t_start'], 't_end': event['t_end'],
                            'evidence': copy.deepcopy(event['evidence'])})
        explained_events = set(assignment.values())
        for i, event in enumerate(events):
            if i not in explained_events and event['verb'] in MANIPULATION_VERBS:
                add('UNEXPECTED_MANIPULATION',
                    f"Observed event {i} ({event['role']}, {event['verb']}, "
                    f"{event['destination_role']}; {event['t_start']}..{event['t_end']}s) "
                    'is not explained by the specified source program; an extra '
                    'operation may undo its outcome or require a different equivalence review')
    evidence.append({'code': 'REVIEW_SCOPE', 'detail': 'Task-level semantic pairing only; no measured human 3D/action labels or time alignment are created.'})
    return result(matches)
