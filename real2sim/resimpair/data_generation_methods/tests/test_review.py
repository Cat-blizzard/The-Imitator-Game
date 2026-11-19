"""Synthetic JSON fixtures test gate logic, never certify a real generated video."""
from copy import deepcopy
import math

import pytest

from data_generation_methods.r2h_pipeline.review import (
    blind_review_prompt, evaluate_review, review_response_schema,
)


@pytest.fixture
def fixture():
    # The provenance is simulated ONLY inside this unit fixture. No invocation,
    # generated media, production decision or pairing export is written.
    aliases = {'choice_a': 'factory:middle_blue', 'choice_b': 'factory:bottom_brown',
               'choice_c': 'factory:reverse_stack'}
    spec = {'schema': 'r2h_review_spec_v1', 'blind_goal_aliases': aliases,
            'allowed_roles': ['block0', 'block1', 'block2', 'blue_pad', 'brown_pad', 'platform'],
            'allowed_verbs': ['pick_and_place', 'release'], 'expected_window_scope': 'full_task',
            'required_events': [
                {'id': 'r1', 'role': 'block2', 'verb': 'pick_and_place', 'destination_role': 'brown_pad'},
                {'id': 'r2', 'role': 'block1', 'verb': 'pick_and_place', 'destination_role': 'blue_pad'},
                {'id': 'r3', 'role': 'block2', 'verb': 'pick_and_place', 'destination_role': 'block0'}],
            'event_order': [['r1', 'r2'], ['r2', 'r3']]}
    job = {'source': {'goal_id': aliases['choice_a']},
           'contract': {'goal_id': aliases['choice_a'], 'review_spec': spec},
           'review_request': {'origin': 'actual_blind_vlm', 'blind': True,
                              'expected_goal_provided': False, 'media_sha256': 'a' * 64,
                              'prompt_sha256': 'b' * 64}}
    events = []
    for i, e in enumerate(spec['required_events']):
        start = .5 + 2 * i
        events.append({'role': e['role'], 'verb': e['verb'], 'destination_role': e['destination_role'],
                       't_start': start, 't_end': start + 1,
                       'evidence': [{'timestamp_seconds': start + .5,
                                     'description': 'The stated block is visibly grasped, placed and released on that support.'}],
                       'confidence': .95})
    review = {'observed_goal': 'choice_a', 'goal_confidence': .95, 'window_scope': 'full_task',
              'initial_state_visible': True, 'final_state_visible': True,
              'events': events, 'issues': [], 'uncertainty': .05}
    media = {'sha256': 'a' * 64, 'full_decode_passed': True,
             'frames': 192, 'fps': 24., 'duration_seconds': 8.}
    return job, review, media


def assert_blocked(result, code):
    assert result['decision'] in ('rejected', 'needs_review')
    assert code in result['error_codes']
    assert result['training_admitted'] is False
    assert result['source_scope']['full_task_pairing'] is False


def test_positive_fixture_only_allows_pairing(fixture):
    job, review, media = fixture
    snapshots = deepcopy(fixture)
    result = evaluate_review(job, review, media)
    assert result['decision'] == 'accepted_for_pairing'
    assert result['error_codes'] == []
    assert len(result['matched_events']) == 3
    assert result['training_admitted'] is False
    assert result['source_scope']['human_3d_ground_truth'] is False
    assert result['source_scope']['human_action_ground_truth'] is False
    assert result['source_scope']['framewise_human_robot_alignment'] is False
    assert fixture == snapshots


@pytest.mark.parametrize('value,code', [
    ('choice_b', 'GOAL_MISMATCH'), ('unknown', 'GOAL_UNKNOWN'),
    ('factory:middle_blue', 'GOAL_ALIAS_INVALID'), ('made_up', 'GOAL_ALIAS_INVALID')])
def test_observed_goal_is_blind_alias_not_expected_echo(fixture, value, code):
    job, review, media = fixture
    review['observed_goal'] = value
    assert_blocked(evaluate_review(job, review, media), code)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'bool_time', 'nan', 'bad_issue', 'empty_evidence'])
def test_schema_is_strict_and_finite(fixture, mutation):
    job, review, media = fixture
    if mutation == 'missing':
        del review['uncertainty']
    elif mutation == 'extra':
        review['training_admitted'] = True
    elif mutation == 'bool_time':
        review['events'][0]['t_start'] = True
    elif mutation == 'nan':
        review['events'][0]['confidence'] = math.nan
    elif mutation == 'bad_issue':
        review['issues'] = [{'code': 'x', 'severity': 'fine', 'description': 'x', 'affects_required_event': False}]
    else:
        review['events'][0]['evidence'] = []
    assert_blocked(evaluate_review(job, review, media), 'REVIEW_SCHEMA_INVALID')


@pytest.mark.parametrize('mutation,code', [
    ('reverse_interval', 'EVENT_TIME_INVALID'), ('past_video', 'EVENT_OUTSIDE_MEDIA'),
    ('evidence_wrong_time', 'EVENT_EVIDENCE_TIME_INVALID'), ('duplicate', 'DUPLICATE_OBSERVED_EVENT'),
    ('wrong_role', 'EVENT_ROLE_OR_VERB_INVALID'), ('unknown_role', 'EVENT_IDENTITY_UNKNOWN'),
    ('missing_event', 'REQUIRED_EVENT_MISSING'), ('wrong_object_destination', 'REQUIRED_EVENT_MISSING'),
    ('wrong_order', 'EVENT_ORDER_MISMATCH'), ('low_confidence', 'EVENT_CONFIDENCE_LOW')])
def test_event_identity_order_and_clock(fixture, mutation, code):
    job, review, media = fixture
    if mutation == 'reverse_interval':
        review['events'][0]['t_end'] = .1
    elif mutation == 'past_video':
        review['events'][2]['t_end'] = 9.
    elif mutation == 'evidence_wrong_time':
        review['events'][0]['evidence'][0]['timestamp_seconds'] = 7.
    elif mutation == 'duplicate':
        review['events'].append(deepcopy(review['events'][0]))
    elif mutation == 'wrong_role':
        review['events'][0]['role'] = 'red_hypothetical_object'
    elif mutation == 'unknown_role':
        review['events'][0]['role'] = 'unknown'
    elif mutation == 'missing_event':
        review['events'].pop(1)
    elif mutation == 'wrong_object_destination':
        review['events'][1]['destination_role'] = 'brown_pad'
    elif mutation == 'wrong_order':
        for key in ('t_start', 't_end', 'evidence'):
            review['events'][0][key], review['events'][1][key] = review['events'][1][key], review['events'][0][key]
    else:
        review['events'][0]['confidence'] = .79
    assert_blocked(evaluate_review(job, review, media), code)


def test_one_event_cannot_satisfy_two_necessary_actions(fixture):
    job, review, media = fixture
    spec = job['contract']['review_spec']
    spec['required_events'].append({**spec['required_events'][0], 'id': 'r4'})
    assert_blocked(evaluate_review(job, review, media), 'EVENT_ORDER_MISMATCH')


def test_explicit_non_target_invariant_is_checked(fixture):
    job, review, media = fixture
    job['contract']['review_spec']['forbidden_events'] = [
        {'role': 'block0', 'verb': 'pick_and_place', 'destination_role': '*'}]
    extra = deepcopy(review['events'][0])
    extra['role'] = 'block0'
    review['events'].append(extra)
    assert_blocked(evaluate_review(job, review, media), 'FORBIDDEN_EVENT_OBSERVED')


def test_partial_contract_cannot_hide_behind_full_review_spec(fixture):
    job, review, media = fixture
    job['contract']['window_scope'] = 'partial'
    assert_blocked(evaluate_review(job, review, media), 'PARTIAL_SOURCE_NOT_FULL_TASK')


def test_malformed_source_identity_fails_closed(fixture):
    job, review, media = fixture
    job['source'] = None
    assert_blocked(evaluate_review(job, review, media), 'JOB_CONTRACT_INVALID')


def test_null_placement_destination_is_not_established(fixture):
    job, review, media = fixture
    review['events'][0]['destination_role'] = None
    assert_blocked(evaluate_review(job, review, media), 'EVENT_DESTINATION_UNKNOWN')


@pytest.mark.parametrize('mutation,code', [
    ('partial', 'PARTIAL_CLIP_NOT_FULL_TASK'), ('unknown_scope', 'WINDOW_SCOPE_UNKNOWN'),
    ('missing_final', 'TASK_BOUNDARIES_NOT_VISIBLE'), ('uncertain', 'REVIEW_UNCERTAIN'),
    ('blocking', 'BLOCKING_VISUAL_ISSUE'), ('uncertain_issue', 'VISUAL_ISSUE_REQUIRES_REVIEW')])
def test_coverage_and_individual_issues_fail_closed(fixture, mutation, code):
    job, review, media = fixture
    if mutation == 'partial': review['window_scope'] = 'partial'
    elif mutation == 'unknown_scope': review['window_scope'] = 'unknown'
    elif mutation == 'missing_final': review['final_state_visible'] = False
    elif mutation == 'uncertain': review['uncertainty'] = .8
    else:
        review['issues'] = [{'code': 'missing_hand_contact', 'severity': 'blocking' if mutation == 'blocking' else 'uncertain',
                             'description': 'Grasp and release cannot be confirmed.', 'affects_required_event': True}]
    assert_blocked(evaluate_review(job, review, media), code)


@pytest.mark.parametrize('mutation,code', [
    ('missing_media', 'MEDIA_VERIFICATION_MISSING'), ('no_decode', 'MEDIA_NOT_VERIFIED'),
    ('bad_clock', 'MEDIA_CLOCK_INCONSISTENT'), ('bad_hash', 'REVIEW_MEDIA_BINDING_MISMATCH'),
    ('missing_request', 'BLIND_REQUEST_PROVENANCE_MISSING'), ('mock', 'BLIND_REQUEST_PROVENANCE_MISSING'),
    ('expected_in_prompt', 'BLIND_REQUEST_PROVENANCE_MISSING')])
def test_actual_review_invocation_must_bind_generated_media(fixture, mutation, code):
    job, review, media = fixture
    if mutation == 'missing_media': media = None
    elif mutation == 'no_decode': media['full_decode_passed'] = False
    elif mutation == 'bad_clock': media['duration_seconds'] = 9.
    elif mutation == 'bad_hash': job['review_request']['media_sha256'] = 'c' * 64
    elif mutation == 'missing_request': del job['review_request']
    elif mutation == 'mock': job['review_request']['origin'] = 'mock'
    else: job['review_request']['expected_goal_provided'] = True
    assert_blocked(evaluate_review(job, review, media), code)


@pytest.mark.parametrize('mutation', ['cycle', 'duplicate_id', 'no_alias_map', 'no_scope'])
def test_private_contract_cannot_be_guessed_or_relaxed(fixture, mutation):
    job, review, media = fixture
    spec = job['contract']['review_spec']
    if mutation == 'cycle': spec['event_order'].append(['r3', 'r1'])
    elif mutation == 'duplicate_id': spec['required_events'][1]['id'] = 'r1'
    elif mutation == 'no_alias_map': del spec['blind_goal_aliases']
    else: del spec['expected_window_scope']
    assert evaluate_review(job, review, media)['decision'] != 'accepted_for_pairing'


def test_blind_prompt_is_same_for_every_private_goal(fixture):
    job, _, _ = fixture
    prompt = blind_review_prompt()
    assert 'choice_a' in prompt and 'choice_b' in prompt and 'choice_c' in prompt
    assert 'factory:' not in prompt and 'expected_goal' not in prompt and '/mnt/' not in prompt
    assert job['source']['goal_id'] not in prompt
    with pytest.raises(ValueError):
        blind_review_prompt([{'id': 'choice_a', 'description': 'one only'}])
    with pytest.raises(ValueError):
        blind_review_prompt([{'id': 'choice_a', 'description': 'factory:private'},
                             {'id': 'choice_b', 'description': 'Other'}])
    assert review_response_schema() is not review_response_schema()


@pytest.mark.parametrize('verb,destination', [
    ('pick_and_place', 'brown_pad'), ('release', None), ('rotate', 'blue_pad')])
def test_extra_manipulation_after_required_program_cannot_silently_pass(fixture, verb, destination):
    job, review, media = fixture
    if verb not in job['contract']['review_spec']['allowed_verbs']:
        job['contract']['review_spec']['allowed_verbs'].append(verb)
    # The reported goal is still choice_a, but moving/releasing the selected
    # object after the required program can undo its supposedly final outcome.
    review['events'].append({'role': 'block1', 'verb': verb,
                             'destination_role': destination, 't_start': 6., 't_end': 7.,
                             'evidence': [{'timestamp_seconds': 6.5,
                                           'description': 'The target is manipulated again after restoration.'}],
                             'confidence': .95})
    result = evaluate_review(job, review, media)
    assert_blocked(result, 'UNEXPECTED_MANIPULATION')
    assert result['decision'] == 'needs_review'
    assert len(result['matched_events']) == 3


def test_extra_manipulation_before_required_program_also_requires_review(fixture):
    job, review, media = fixture
    review['events'].insert(0, {'role': 'block1', 'verb': 'pick_and_place',
                               'destination_role': 'brown_pad', 't_start': .05, 't_end': .25,
                               'evidence': [{'timestamp_seconds': .15,
                                             'description': 'Extra target placement before the specified program.'}],
                               'confidence': .95})
    result = evaluate_review(job, review, media)
    assert_blocked(result, 'UNEXPECTED_MANIPULATION')
    assert result['decision'] == 'needs_review'
