"""Offline checks for hidden-content preparation; no API, renderer or training."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from data_generation_methods.r2h_pipeline import hidden_content_conditions as candidate
from data_generation_methods.r2h_pipeline.common import write_json

BASE = Path('/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v3/hidden_conditions')
PROPOSAL = Path('/mnt/sdg/datasets/real_sim_human_pipeline/automation_20261020d/scene_proposals/cabinet_new_v2/proposal.json')
VERIFICATION = Path('/mnt/sdf/datasets/hypericl_human_robot_task_library_20261019_v1/generation/pilot_review_CPU_v1/robot/cabinet_contact_v7_891914000/raw_check.json')


def test_instance_visibility_is_not_recognizability():
    event = candidate.visibility_events([0, 0, 9, 17, 18, 19], np.arange(6)/20)
    assert event['first_any_instance_pixels_frame'] == 2
    assert event['sustained_candidate_frame'] == 3
    assert event['first_rgb_recognizable_frame'] == 'unknown'
    assert event['first_observed_transition_interval_frames'] == [1, 2]


def test_sparse_observations_cannot_claim_exact_first_or_consecutive():
    event = candidate.visibility_events([0, 20, 20, 20], [0., .5, 1., 1.5], frame_indices=[0, 10, 20, 30])
    assert event['first_observed_transition_interval_frames'] == [0, 10]
    assert not event['first_any_is_exact_recorded_frame']
    assert event['sustained_candidate_frame'] is None


@pytest.mark.parametrize('counts,times', [([0, -1, 5], [0, 1, 2]), ([0, 1, 2], [0, 2, 1])])
def test_invalid_visibility_rejected(counts, times):
    with pytest.raises(ValueError):
        candidate.visibility_events(counts, times)


def test_triangle_surface_geometry():
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    np.testing.assert_allclose(candidate.vertical_surface_hits(mesh, [0, .1]), [-1, 1])
    assert len(candidate.vertical_surface_hits(mesh, [5, 5])) == 0


@pytest.fixture
def actual_review():
    path = BASE/'source_rgb_review_v2.json'
    if not path.exists():
        pytest.skip('Archived local cabinet evidence unavailable')
    return json.loads(path.read_text())


@pytest.fixture
def actual_packet():
    path = BASE/'development/hidden_conditions.json'
    if not path.exists():
        pytest.skip('Archived local cabinet evidence unavailable')
    return json.loads(path.read_text())


def test_actual_three_stage_request_is_prepare_only(actual_packet, monkeypatch):
    import data_generation_methods.r2h_pipeline.providers as providers
    def forbidden_transport(*args, **kwargs):
        raise AssertionError('Transport must never be invoked during preparation')
    monkeypatch.setattr(providers, '_urllib_transport', forbidden_transport)
    spec = candidate._request_spec(actual_packet, actual_packet['source_frames'])
    assert [x['stage_id'] for x in spec['stages']] == ['closed_initial', 'revealed_keyframe', 'open_terminal']
    assert spec['stages'][0]['arguments']['reference_images'] == []
    assert not spec['downstream_video_path']['first_frame_only_carries_hidden_RGB_identity']
    assert not spec['generation_allowed']
    assert 'approved_single_operator_body_plan_missing' in spec['blocking_reasons']
    preflight = candidate.payload_preflight(spec)
    assert preflight['transport_calls'] == 0
    assert not preflight['generation_success']
    assert [len(x['image_parts']) for x in preflight['stages']] == [1, 2, 2]
    for stage, result in zip(spec['stages'], preflight['stages']):
        assert result['image_parts'][0]['sha256'] == stage['arguments']['image_path']['sha256']


@pytest.mark.parametrize('mutation', ['slot', 'UID', 'frame', 'color_identity_pass', 'missing_text', 'duplicate', 'old_schema'])
def test_review_binding_and_identity_overclaim_rejected(actual_review, actual_packet, tmp_path, mutation):
    review = copy.deepcopy(actual_review)
    if mutation == 'slot':
        review['physical_slot_id'] = 'wrong_shelf'
    elif mutation == 'UID':
        review['asset_uid'] = 'pear'
    elif mutation == 'frame':
        review['observations'][-1]['frame']['frame_index_0based'] = 0
    elif mutation == 'color_identity_pass':
        review['observations'][-1]['generated_identity_pass'] = True
    elif mutation == 'missing_text':
        review['observations'][-1].pop('observation')
    elif mutation == 'duplicate':
        review['observations'].append(copy.deepcopy(review['observations'][-1]))
    else:
        review['schema'] = 'r2h_hidden_content_rgb_observations_v1'
        review['reviewer'] = 'VLM reviewer string'
    path = tmp_path/'review.json'
    write_json(path, review)
    qa = dict(np.load(actual_packet['bindings']['visibility']['path'], allow_pickle=False))
    with pytest.raises(ValueError):
        candidate._review(path, video=actual_packet['bindings']['video'], receipt=actual_packet['bindings']['receipt'],
            asset={'uid': actual_packet['object']['asset_uid']}, object_index=0, slot_id=actual_packet['slot']['id'],
            frames=actual_packet['source_frames'], qa=qa)


def test_no_rgb_review_keeps_reveal_unknown(actual_packet):
    review = candidate._review(None, video=None, receipt=None, asset=None, object_index=0,
                               slot_id=None, frames=None, qa=None)
    assert review['first_observed_confirmed_reference'] is None
    packet = copy.deepcopy(actual_packet)
    packet['rgb_observations'] = review
    spec = candidate._request_spec(packet, packet['source_frames'])
    assert len(spec['stages']) == 1
    assert 'source_revealed_RGB_reference_unconfirmed' in spec['blocking_reasons']


def test_actual_preparation_preserves_physics_scope_and_first_observed_claim(tmp_path):
    if not PROPOSAL.exists():
        pytest.skip('Archived cabinet source unavailable')
    packet = candidate.compile_conditions(PROPOSAL, VERIFICATION, tmp_path/'actual',
        rgb_review_path=BASE/'source_rgb_review_v2.json')
    assert packet['visibility_events']['first_any_instance_pixels_frame'] == 276
    assert packet['visibility_events']['sustained_candidate_frame'] == 278
    assert packet['legacy_scene_proposals_visibility_candidate']['first_sustained_reveal_frame'] == 276
    assert packet['rgb_observations']['first_observed_confirmed_reference'] == 330
    assert packet['visibility_events']['first_rgb_recognizable_frame'] == 'unknown'
    assert not packet['physical_qualification']['apple_strict_qualification_inherited']
    assert packet['conditions'][0]['source_visible_object_size_px'] is None
    assert not packet['generation_allowed']
    assert packet['physical_execution_UUID'] == 'unknown_not_exported'
    assert packet['schema'] == 'r2h_hidden_cabinet_content_conditions_v2'
    provenance = packet['geometry_provenance']['recorded_QA_cooked_hull']
    assert not provenance['cooked_vertices_exported']
    assert not provenance['raw_OBJ_equivalence_verified']
    comparison = packet['conditions'][0]['raw_OBJ_vs_recorded_QA_comparison']
    assert comparison['status'] == 'exact_equality_failed'
    assert comparison['difference_cause'] == 'unknown'
    assert comparison['acceptance_tolerance_applied'] is None
    assert comparison['maximum_absolute_difference_normalized'] == pytest.approx(0.0005332811105892565)
    assert 'projected_collision_hull_box_normalized' not in packet['conditions'][0]
    assert packet['conditions'][0]['ratio_geometry_sources']['numerator'] == 'recorded_QA_cooked_hull_box_normalized'
    with pytest.raises(FileExistsError):
        candidate.compile_conditions(PROPOSAL, VERIFICATION, tmp_path/'actual')


def test_actual_wrong_selected_object_is_rejected(tmp_path):
    if not PROPOSAL.exists():
        pytest.skip('Archived cabinet source unavailable')
    with pytest.raises(ValueError, match='small-cube asset mapping'):
        candidate.compile_conditions(PROPOSAL, VERIFICATION, tmp_path/'wrong', object_index=1)


def test_vlm_yes_cannot_promote_source_recognizability(actual_review, actual_packet, tmp_path):
    review = copy.deepcopy(actual_review)
    review['reviewer'] = {'id': 'VLM all-positive answer', 'kind': 'vlm'}
    entry = copy.deepcopy(review['observations'][3])
    assert entry['frame']['frame_index_0based'] == 278
    entry['object_extent_visible'] = 'yes'
    entry['small_cuboid_shape_supported'] = 'yes'
    review['observations'] = [entry]
    path = tmp_path/'vlm.json'
    write_json(path, review)
    qa = dict(np.load(actual_packet['bindings']['visibility']['path'], allow_pickle=False))
    result = candidate._review(path, video=actual_packet['bindings']['video'], receipt=actual_packet['bindings']['receipt'],
        asset={'uid': actual_packet['object']['asset_uid']}, object_index=0, slot_id=actual_packet['slot']['id'],
        frames=actual_packet['source_frames'], qa=qa)
    assert result['status'] == 'unknown_vlm_only'
    assert result['first_observed_confirmed_reference'] is None


def test_no_sustained_visibility_stops_preparation_explicitly(monkeypatch, tmp_path):
    if not PROPOSAL.exists():
        pytest.skip('Archived cabinet source unavailable')
    original = candidate._load
    def transient_pixels(*args, **kwargs):
        data = original(*args, **kwargs)
        # In-memory edge-case injection only; no original array/media changed.
        data[2]['object_pixels_512'][:, 0] = np.minimum(data[2]['object_pixels_512'][:, 0], 9)
        return data
    monkeypatch.setattr(candidate, '_load', transient_pixels)
    with pytest.raises(ValueError, match='No sustained instance visibility candidate'):
        candidate.compile_conditions(PROPOSAL, VERIFICATION, tmp_path/'no_sustained')
