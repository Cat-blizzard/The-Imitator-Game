"""Rendering the towel together with the robot on a host without NVIDIA graphics capability.

Two renderers:
  SapienMirror   SAPIEN rasteriser on the CPU (lavapipe Vulkan device 'pci:0'). ManiSkill's GPU-sim renderer needs a
                 CUDA-visible Vulkan device, so for route A with GPU physics we mirror each rendered env into a CPU
                 SAPIEN scene: the robot is the same URDF loaded kinematically (qpos copied from the GPU sim), the towel
                 is a render mesh rebuilt from Newton's particle positions every rendered step (RenderShapeTriangleMesh;
                 SAPIEN 3.0.3 has no CPU-side vertex update, RenderCudaMeshComponent needs CUDA-Vulkan interop), plus a
                 table box. -> "cloth mesh shown in SAPIEN renders by updating a render mesh each step".
  NewtonCamera   Newton 1.7 SensorCamera (Warp CUDA ray casting, no RT cores, no Vulkan) over the cloth model, into
                 which the robot's visual meshes are added as kinematic visual-only bodies (poses copied each step).
                 Used by route B, and usable by route A as a GPU alternative.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

from .colliders import Collider
from .panda_ik import PANDA_URDF, _rpy, _T

PANDA_VISUAL_LINKS = ["panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5",
                      "panda_link6", "panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger"]


def look_at_wxyz(eye, target, up=(0, 0, 1)):
    """SAPIEN camera convention (x forward, y left, z up) pose as (p, q_wxyz)."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    l = np.cross(up, f); l /= np.linalg.norm(l)
    u = np.cross(f, l)
    R = np.stack([f, l, u], 1)
    return eye, _mat_to_wxyz(R)


def _mat_to_wxyz(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


def urdf_visual_meshes(urdf: str = PANDA_URDF, links=PANDA_VISUAL_LINKS, max_faces: int = 6000):
    """{link: [(vertices (V,3) float32, faces (F,3) int32, T_local 4x4)]} from the URDF <visual> meshes (trimesh)."""
    import trimesh
    root = ET.parse(urdf).getroot()
    base = os.path.dirname(urdf)
    out = {}
    for ln in root.findall("link"):
        name = ln.get("name")
        if name not in links:
            continue
        lst = []
        for vis in ln.findall("visual"):
            m = vis.find("geometry/mesh")
            if m is None:
                continue
            o = vis.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else [0, 0, 0])]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else [0, 0, 0])]
            mesh = trimesh.load(os.path.join(base, m.get("filename")), force="mesh")
            if len(mesh.faces) > max_faces:
                try:
                    mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
                except Exception:
                    pass
            lst.append((np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32), _T(_rpy(*rpy), xyz)))
        out[name] = lst
    return out


def panda_render_links(colliders: list[Collider], urdf: str = PANDA_URDF):
    """Collider list extended with visual-only links, every link carrying its visual meshes (for NewtonCamera)."""
    vis = urdf_visual_meshes(urdf)
    by = {c.link: c for c in colliders}
    out = []
    for c in colliders:
        out.append(Collider(c.link, c.boxes, visual=vis.get(c.link, [])))
    for ln in PANDA_VISUAL_LINKS:
        if ln not in by:
            out.append(Collider(ln, [], visual=vis.get(ln, [])))
    return out


# ---------------------------------------------------------------------------------------------- SAPIEN (CPU)
class SapienMirror:
    """CPU SAPIEN scenes mirroring robot qpos + towel vertices of `num` envs, rendered by lavapipe."""

    def __init__(self, num: int, tri: np.ndarray, width: int = 128, height: int = 128, fovy: float = np.pi / 2,
                 eye=(0.35, -0.45, 0.45), target=(-0.05, 0.0, 0.0), urdf: str = PANDA_URDF, base=(-0.615, 0.0, 0.0),
                 device: str = "pci:0"):
        import sapien
        self.sapien = sapien
        self.dev = sapien.Device(device)
        self.num, self.W, self.H = num, width, height
        self.tri = np.asarray(tri, np.uint32)
        self.tri2 = np.vstack([self.tri, self.tri[:, ::-1]])           # two-sided towel
        self.scenes, self.robots, self.cams, self.cloth_ents, self.cloth_rbs = [], [], [], [], []
        self.mat = sapien.render.RenderMaterial(base_color=[0.35, 0.55, 0.85, 1.0], roughness=0.8)
        tmat = sapien.render.RenderMaterial(base_color=[0.55, 0.42, 0.30, 1.0], roughness=0.9)
        for i in range(num):
            sc = sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(self.dev)])
            sc.set_ambient_light([0.35, 0.35, 0.35])
            sc.add_directional_light([0.3, 0.6, -1.0], [0.9, 0.9, 0.9], shadow=True)
            loader = sc.create_urdf_loader(); loader.fix_root_link = True
            rob = loader.load(urdf); rob.set_root_pose(sapien.Pose(list(base)))
            ab = sc.create_actor_builder()
            ab.add_box_visual(sapien.Pose([0.0, 0.0, -0.02]), half_size=[0.6, 0.6, 0.02], material=tmat)
            ab.build_static(name="table")
            ent = sapien.Entity(); sc.add_entity(ent)
            cam = sc.add_camera("cam", width, height, fovy, 0.01, 10.0)
            p, q = look_at_wxyz(eye, target)
            cam.entity.set_pose(sapien.Pose(p, q))
            self.scenes.append(sc); self.robots.append(rob); self.cams.append(cam); self.cloth_ents.append(ent)
            self.cloth_rbs.append(None)

    def _cloth_body(self, v):
        sapien = self.sapien
        v = np.asarray(v, np.float32)
        V2 = np.vstack([v, v])
        tri = self.tri2.copy(); tri[len(self.tri):] += 0     # second copy shares vertices (normals flipped below)
        n = np.zeros_like(v)
        a, b, c = v[self.tri[:, 0]], v[self.tri[:, 1]], v[self.tri[:, 2]]
        fn = np.cross(b - a, c - a)
        for k in range(3):
            np.add.at(n, self.tri[:, k], fn)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        N2 = np.vstack([n, -n])
        T2 = np.vstack([self.tri, self.tri[:, ::-1] + len(v)]).astype(np.uint32)
        uv = np.zeros((len(V2), 2), np.float32)
        rb = sapien.render.RenderBodyComponent()
        rb.attach(sapien.render.RenderShapeTriangleMesh(V2, T2, N2.astype(np.float32), uv, self.mat))
        return rb

    def render(self, qpos: np.ndarray, verts: np.ndarray) -> np.ndarray:
        """qpos (num, 9), verts (num, V, 3) -> uint8 (num, H, W, 3)."""
        out = np.zeros((self.num, self.H, self.W, 3), np.uint8)
        for i in range(self.num):
            self.robots[i].set_qpos(np.asarray(qpos[i], np.float32))
            ent = self.cloth_ents[i]
            if self.cloth_rbs[i] is not None:
                ent.remove_component(self.cloth_rbs[i])
            rb = self._cloth_body(verts[i]); ent.add_component(rb); self.cloth_rbs[i] = rb
            self.scenes[i].update_render()
            self.cams[i].take_picture()
            rgba = self.cams[i].get_picture("Color")
            out[i] = np.clip(rgba[..., :3] * 255, 0, 255).astype(np.uint8)
        return out


# ---------------------------------------------------------------------------------------------- Newton (CUDA)
class NewtonCamera:
    """Batched Warp ray-cast camera over a NewtonClothWorlds model (one view per world)."""

    def __init__(self, cloth, width: int = 128, height: int = 128, fovy: float = np.pi / 2,
                 eye=(0.35, -0.45, 0.45), target=(-0.05, 0.0, 0.0), shadows: bool = True):
        import warp as wp
        from newton.sensors import SensorCamera
        self.wp, self.cloth, self.W, self.H = wp, cloth, width, height
        m = cloth.model
        self.cam = SensorCamera(m)
        self.cam.create_default_light(enable_shadows=shadows)
        try:
            self.cam.assign_checkerboard_material(shape_indices=np.array([cloth.ground_shape], dtype=np.uint32))
        except Exception:
            pass
        cfg = self.cam.default_render_config
        cfg.enable_shadows = shadows
        cfg.enable_backface_culling = False
        self.rays = SensorCamera.compute_camera_rays_pinhole(width, height, camera_fov=float(fovy), device=m.device)
        n = cloth.n
        self.tr = wp.array([self._look_at(eye, target)] * n, dtype=wp.transformf, device=m.device)
        self.img = self.cam.allocate_image_color(n, width, height)
        self.wi = wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device=m.device)
        self.SensorCamera = SensorCamera

    def _look_at(self, eye, target, up=(0, 0, 1)):
        wp = self.wp
        eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
        f = target - eye; f /= np.linalg.norm(f)
        r = np.cross(f, up); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        R = np.stack([r, u, -f], axis=1)                # camera +X right, +Y up, looks along -Z
        q = _mat_to_wxyz(R)
        return wp.transformf(wp.vec3f(*eye), wp.quatf(q[1], q[2], q[3], q[0]))

    def render_all(self, env=None):
        """uint8 (N, H, W, 3) torch tensor on the GPU (one view per world)."""
        m, st = self.cloth.model, self.cloth.s0
        m.bvh_refit_shapes(st)
        m.bvh_refit_particles(st)
        self.cam.update(st, self.tr, self.rays, color_image=self.img, world_indices=self.wi)
        rgba = self.SensorCamera.Utils.to_rgba_from_color(self.img)
        return self.wp.to_torch(rgba)[..., :3]
