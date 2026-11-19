"""Compact failure receipts preserve actual native manifold values."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mani_skill.examples.motionplanning.dual.sync.observation import ObservedEnvironment


class Entity:
    def __init__(self, name):
        self.name = name


def observer(points):
    robot, payload = Entity("link6"), Entity("apple_out")
    contact = NS(bodies=[NS(entity=robot), NS(entity=payload)], points=points)
    contacts = [contact]
    target = NS(pose=NS(p=np.array([[0., 0., .1]])))
    capture = ObservedEnvironment.__new__(ObservedEnvironment)
    capture.unwrapped = NS(scene=NS(px=NS(get_contacts=lambda: contacts)))
    capture.original_hook = lambda: None
    capture.substeps = 0
    capture.checker_provider = lambda: None
    capture.task = NS(targets=(target, target),
                     initial_supports=(NS(name="table"), NS(name="table")),
                     destination_supports=(NS(name="basket_apple"), NS(name="basket_banana")))
    capture.physical_released = [True, True]
    capture.reference_frozen = [True, True]
    capture.reference_z = [0., 0.]
    capture.references = None
    capture.phase = "retreat"
    capture.srdf = {0: set(), 1: set()}
    capture.pending_errors = []
    capture._retreat_contacts = None
    capture._retreat_initial_contacts = None
    capture._retreat_separated_contacts = set()
    capture._retreat_remaining_contacts = set()
    capture._last_own_finger_contacts = set()
    capture._last_own_finger_contact_phase = None
    capture.identities = {robot: {"kind": "robot", "robot_index": 0, "link_name": "link6",
                                 "is_grasp_finger": False, "actor": "link6"},
                          payload: {"kind": "payload", "target_index": 0, "actor": "apple_out"}}
    return capture


def test_forbidden_events_keep_original_native_points_and_substep():
    point = NS(position=np.array([.1, -.2, .03]), normal=np.array([0., 1., 0.]),
               separation=-.0002, impulse=np.array([0., .003, 0.]))
    capture = observer([point])
    capture.record_substep()
    assert {row["reason"] for row in capture.pending_errors} == {
        "forbidden_payload_contact", "forbidden_robot_contact"}
    encoded = json.loads(json.dumps(capture.pending_errors, allow_nan=False))
    for row in encoded:
        assert row["physical_substep_index"] == 1
        assert row["points"] == [{"position_m": [.1, -.2, .03], "normal": [0., 1., 0.],
                                  "separation_m": -.0002, "impulse_Ns": [0., .003, 0.]}]
    np.testing.assert_array_equal(point.impulse, [0., .003, 0.])


@pytest.mark.parametrize("points", [[], [NS(position=np.zeros(3), normal=np.array([0., 1., 0.]),
    separation=.001, impulse=np.zeros(3))]])
def test_empty_or_proximity_only_manifold_does_not_create_a_failure(points):
    capture = observer(points)
    capture.record_substep()
    assert capture.substeps == 1 and capture.pending_errors == []
