"""Native open-hand swept volume for the controller's WORLD-Z approach.

The old tabletop proposals have local +Z exactly downward. cuDex may produce
tilted grasps: sweeping along local -Z would check a different path. This
candidate checker preserves orientation and sweeps along scene +Z instead.
"""
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from resimpair_tabletop.geometry import panda_meshes, bvh


def swept_vertices(vertices, T, approach):
    world = np.asarray(vertices) @ T[:3, :3].T + T[:3, 3]
    return np.vstack([world, world + [0, 0, approach]])


class VerticalClearance:
    def __init__(self, mesh, approach=.17):
        from mplib.pymp.collision_detection import fcl
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        self.fcl = fcl
        self.geometry = bvh(mesh)
        self.obj = fcl.CollisionObject(self.geometry)
        self.parts = [m for _, _, m in panda_meshes()[0]]
        self.vertices = np.vstack([m.vertices for m in self.parts])
        self.mesh_ray = RayMeshIntersector(mesh)
        self.closed_mesh = mesh.is_watertight
        self.approach = approach
        # One vertex per disconnected component catches containment even when
        # triangle surfaces do not intersect. No collision proxy enlarges holes.
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        edges = mesh.edges_unique
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                           shape=(len(mesh.vertices), len(mesh.vertices)))
        _, labels = connected_components(graph, directed=False)
        self.component_points = mesh.vertices[np.unique(labels, return_index=True)[1]]

    def check(self, T, table_z, margin):
        T = np.asarray(T, float)
        vertices = self.vertices @ T[:3, :3].T + T[:3, 3]
        clearance = float(vertices[:, 2].min() - table_z)
        if clearance < margin:
            return False, clearance, 'gripper_table'
        for mesh in self.parts:
            swept = trimesh.convex.convex_hull(swept_vertices(mesh.vertices, T, self.approach))
            geometry = bvh(swept)
            part = self.fcl.CollisionObject(geometry)
            if self.fcl.collide(part, self.obj, self.fcl.CollisionRequest()).is_collision():
                return False, clearance, 'open_hand_or_world_z_approach_collision'
            equations = ConvexHull(swept.vertices).equations
            if np.any(np.all(self.component_points @ equations[:, :3].T + equations[:, 3] <= 1e-9, axis=1)):
                return False, clearance, 'object_inside_swept_hand'
            if self.closed_mesh and self.mesh_ray.contains_points(swept.vertices[:1]).any():
                return False, clearance, 'hand_inside_object'
        return True, clearance, None
