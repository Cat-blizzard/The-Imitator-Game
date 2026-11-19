"""Opt-in measured release preconditions, matching retained place predicates."""
import numpy as np


def readiness(object_world, target_position, held):
    transform = np.asarray(object_world, dtype=float)
    error = float(np.linalg.norm(transform[:3, 3]-np.asarray(target_position, dtype=float)))
    upright = float(transform[2, 2])
    return {'held': bool(held), 'position_error_m': error, 'upright_cosine': upright,
            'position_tolerance_m': .015, 'minimum_upright_cosine': .95,
            'ready_to_release': bool(held and error < .015 and upright > .95)}
