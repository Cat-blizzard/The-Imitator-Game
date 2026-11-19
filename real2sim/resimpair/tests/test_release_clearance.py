import numpy as np
import trimesh

from resimpair_pickplace.release_clearance import finger_sweep_collisions


def finger_at(x, y, z):
    mesh = trimesh.creation.box(extents=[.01, .01, .03])
    return [('native_finger', mesh.vertices + [x, y, z])]


def test_withdrawal_sweep_detects_overhanging_geometry():
    # Finger is initially below the obstacle and collision-free. Its upward
    # path still catches the ledge: an endpoint-only opening test misses this.
    target = trimesh.creation.box(extents=[.04, .04, .02])
    object_world = np.eye(4); object_world[2, 3] = .12
    result = finger_sweep_collisions(target, object_world, np.eye(4), .04,
                                    world_fingers=finger_at(0., 0., 0.))
    assert not result['passed']


def test_wider_opening_clears_the_whole_withdrawal():
    target = trimesh.creation.box(extents=[.04, .04, .02])
    object_world = np.eye(4); object_world[2, 3] = .12
    result = finger_sweep_collisions(target, object_world, np.eye(4), .08,
                                    world_fingers=finger_at(0., .05, 0.))
    assert result['passed'] and result['minimum_geometric_gap_m'] > .02
