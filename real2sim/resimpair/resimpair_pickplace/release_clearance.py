"""Geometry-driven aperture selection and collision-free released-object exit.

Fixed '+8 mm' opening can release a narrow grasp cross-section yet hook a wider
part during withdrawal. Check the swept fingers, the cavity, and the full path
before opening. There are no asset IDs or category branches in this policy.
"""
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from resimpair_grasp.common import write
from resimpair_tabletop.geometry import panda_meshes, bvh
from .stability_motion import StabilityMotion


def finger_sweep_collisions(mesh, object_world, tcp_world, width, distance=.18, world_fingers=None):
    from mplib.pymp.collision_detection import fcl
    target = mesh.copy(); target.apply_transform(object_world)
    target_geometry = bvh(target); target_object = fcl.CollisionObject(target_geometry)
    hits = []; gaps = []
    if world_fingers is None:
        world_fingers = [(name, part.vertices @ tcp_world[:3, :3].T + tcp_world[:3, 3])
                         for name, _, part in panda_meshes(width)[0]
                         if name in {'panda_leftfinger', 'panda_rightfinger'}]
    for name, points in world_fingers:
        swept = trimesh.convex.convex_hull(np.vstack([points, points + [0, 0, distance]]))
        geometry = bvh(swept); obj = fcl.CollisionObject(geometry)
        gaps.append(float(fcl.distance(obj, target_object, fcl.DistanceRequest()).min_distance))
        equations = ConvexHull(swept.vertices).equations
        contains_target = np.any(np.all(target.vertices @ equations[:, :3].T + equations[:, 3] <= 1e-9, axis=1))
        if fcl.collide(obj, target_object, fcl.CollisionRequest()).is_collision() or contains_target:
            hits.append(name)
    return {'passed': not hits, 'colliding_fingers': hits, 'width_m': float(width),
            'minimum_geometric_gap_m': min(gaps),
            'withdrawal_m': distance, 'scope': 'straight fixed-orientation finger sweep; full robot path checked separately'}


class ClearReleaseMotion(StabilityMotion):
    require_released_target_clearance = False

    def independent_table_collisions(self, full, tcp):
        pairs = super().independent_table_collisions(full, tcp)
        if self.check_stage == 'retreat' and self.require_released_target_clearance:
            from mplib.pymp.collision_detection import fcl
            for name, _, collision in self.physical_robot_objects:
                if name in {'panda_leftfinger', 'panda_rightfinger'}:
                    if fcl.collide(collision, self.selected_collision, fcl.CollisionRequest()).is_collision():
                        pairs.append({'link1': name, 'link2': 'released_target'})
        return pairs


def prepare(guard, current, minimum_width, retreat):
    """Bounded planning only; return a clear aperture or reject before release."""
    import sapien
    from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array
    tcp = array(guard.agent.tcp.pose.to_transformation_matrix()).reshape(4, 4)
    target = array(guard.selected_actor.pose.to_transformation_matrix()).reshape(4, 4)
    mesh = trimesh.load(guard.selected_mesh_path, force='mesh', process=False)
    from data_generation_methods.human_to_robot_pipeline.strict_motion import pose7_to_matrix
    native_fingers = [(name, index, trimesh.load(guard.directory / ('robot_' + name + '.obj'), force='mesh', process=False))
                      for name, index, _ in guard.physical_robot_objects
                      if name in {'panda_leftfinger', 'panda_rightfinger'}]
    widths = sorted({float(minimum_width), *[float(w) for w in np.arange(.08, minimum_width, -.005)]}, reverse=True)
    decisions = []; result = None
    for width in widths:
        future = current.copy(); future[7:] = width / 2
        model = guard.planner.pinocchio_model; model.compute_forward_kinematics(future)
        world_fingers = []
        for name, index, part in native_fingers:
            T = guard.root @ pose7_to_matrix(model.get_link_pose(index))
            world_fingers.append((name, part.vertices @ T[:3, :3].T + T[:3, 3]))
        sweep = finger_sweep_collisions(mesh, target, tcp, width, world_fingers=world_fingers)
        row = {'width_m': width, 'released_object_sweep': sweep}; decisions.append(row)
        if not sweep['passed']: continue
        opening = guard.opening_sweep(width); row['environment_opening_sweep'] = opening
        if not opening['passed']: continue
        attached = guard.use_attach; guard.use_attach = False
        guard.require_released_target_clearance = True
        try:
            path = guard.plan(sapien.Pose(retreat), 'retreat', future)
            row['future_path_guard'] = path['guard_evidence']; row['passed'] = True
            result = (width, opening)
        except RuntimeError as error:
            row['passed'] = False; row['reason'] = str(error)
        finally: guard.use_attach = attached
        if result is not None: break
        # Opening is allowed to start in contact. Strict released-target checks
        # are enabled for the actual withdrawal, after the aperture is chosen.
        guard.require_released_target_clearance = False
    write(guard.rec.directory / (guard.rec.stage.split(':')[0] + '_clear_release_preparation.json'),
          {'policy': 'checked_retreat_v1', 'selected_width_m': None if result is None else result[0],
           'minimum_legacy_width_m': float(minimum_width), 'decisions': decisions,
           'physical_retries': 0, 'planning_only': True})
    if result is None: raise RuntimeError('no_collision_free_release_aperture_and_retreat')
    return result
