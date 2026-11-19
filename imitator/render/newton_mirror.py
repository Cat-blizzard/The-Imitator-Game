"""Newton (Warp ray-cast) camera for rigid imsim scenes: a GPU renderer that needs neither RT cores nor a Vulkan device.

NewtonRigidMirror builds, once per env, a visual-only Newton model with one kinematic body per robot link (the link's URDF
visual meshes, from the derived URDFs), one per dynamic / kinematic actor of the task (cube / block: box, target square:
thin box) and a ground plane at the table top; render_all() copies the current SAPIEN poses of all envs (GPU buffers, no
host copy on PhysX GPU) into Newton's body_q and ray-casts one RGB view per env (newton.sensors.SensorCamera).
NewtonRigidCameraWrapper(env) adds it to any rigid imsim env: env.render() returns (N, H, W, 3) uint8 (torch, cuda) and,
with add_to_obs=True, obs['sensor_data']['newton_camera']['rgb'] (obs_mode 'state_dict').
"""
from __future__ import annotations

import numpy as np
import torch

import gymnasium as gym


class NewtonRigidMirror:
    def __init__(self, u, width: int = 128, height: int = 128, fovy: float = np.pi / 3,
                 eye=(1.15, 0.0, 0.85), target=(0.25, 0.0, 0.05), max_faces: int = 3000, device: str = "cuda:0"):
        import newton
        import warp as wp
        from newton.sensors import SensorCamera
        from ..soft.render_sync import _mat_to_wxyz
        from .meshes import robot_meshes
        self.wp, self.u, self.W, self.H = wp, u, width, height
        self.device = wp.get_device(device)
        self.N = u.num_envs
        tmpl = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        vcfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False, density=0.0,
                                               is_visible=True)
        self.sources = []                                   # SAPIEN Actor / Link per body, in body order
        for k, ag in enumerate(u.agent.agents):
            vis = robot_meshes(ag.record.urdf_path, [l.name for l in ag.robot.get_links()], max_faces=max_faces)
            for l in ag.robot.get_links():
                meshes = vis.get(l.name) or []
                if not meshes:
                    continue
                b = tmpl.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()), mass=0.0,
                                  is_kinematic=True, label=f"{k}:{l.name}")
                for V, F, T in meshes:
                    T = np.asarray(T)
                    q = _mat_to_wxyz(T[:3, :3])
                    mesh = newton.Mesh(np.asarray(V, np.float32), np.asarray(F, np.int32).reshape(-1), compute_inertia=False)
                    tmpl.add_shape_mesh(b, xform=wp.transform(wp.vec3(*T[:3, 3]), wp.quat(q[1], q[2], q[3], q[0])), mesh=mesh,
                                        cfg=vcfg, color=(0.85, 0.85, 0.85) if k == 0 else (0.7, 0.75, 0.85))
                self.sources.append(l)
        boxes, seen = [], []
        for name, color in (("block", (0.15, 0.35, 0.85)), ("cube", (0.85, 0.15, 0.12)), ("target", (0.1, 0.75, 0.2))):
            a = getattr(u, name, None)
            if a is None or any(a is s for s in seen):
                continue
            seen.append(a)
            if name == "target":
                half = (u.dual.target_half, u.dual.target_half, 0.0008)
            else:
                h = np.asarray(u.block_half if name == "block" else u.cube_half, float).reshape(-1)
                half = tuple(np.repeat(h, 3) if h.size == 1 else h)
            boxes.append((a, half, color))
        for a, half, color in boxes:
            b = tmpl.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()), mass=0.0, is_kinematic=True)
            tmpl.add_shape_box(b, hx=float(half[0]), hy=float(half[1]), hz=float(half[2]), cfg=vcfg, color=color)
            self.sources.append(a)
        self.K = len(self.sources)
        scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        scene.replicate(tmpl, world_count=self.N)
        self.ground = scene.add_ground_plane(height=0.0)
        with wp.ScopedDevice(self.device):
            self.model = scene.finalize()
        self.state = self.model.state()
        self.cam = SensorCamera(self.model)
        self.cam.create_default_light(enable_shadows=True)
        try:
            self.cam.assign_checkerboard_material(shape_indices=np.array([self.ground], dtype=np.uint32))
        except Exception:
            pass
        cfg = self.cam.default_render_config
        cfg.enable_shadows = True
        cfg.enable_backface_culling = False
        self.rays = SensorCamera.compute_camera_rays_pinhole(width, height, camera_fov=float(fovy), device=self.model.device)
        f = np.asarray(target, float) - np.asarray(eye, float); f /= np.linalg.norm(f)
        r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
        up = np.cross(r, f)
        q = _mat_to_wxyz(np.stack([r, up, -f], 1))
        self.tr = wp.array([wp.transformf(wp.vec3f(*eye), wp.quatf(q[1], q[2], q[3], q[0]))] * self.N, dtype=wp.transformf,
                           device=self.model.device)
        self.img = self.cam.allocate_image_color(self.N, width, height)
        self.wi = wp.array(np.arange(self.N, dtype=np.int32), dtype=wp.int32, device=self.model.device)
        self.SensorCamera = SensorCamera

    def render_all(self):
        from ..soft.cloth import _set_body_poses
        wp = self.wp
        P = torch.stack([s.pose.raw_pose for s in self.sources], 1).to("cuda:0").float().reshape(-1, 7).contiguous()
        src = wp.from_torch(P, dtype=wp.float32)
        wp.launch(_set_body_poses, dim=self.N * self.K, inputs=[src, src, 1.0, 1.0, 0.0],
                  outputs=[self.state.body_q, self.state.body_qd], device=self.device)
        self.model.bvh_refit_shapes(self.state)
        self.cam.update(self.state, self.tr, self.rays, color_image=self.img, world_indices=self.wi)
        rgba = self.SensorCamera.Utils.to_rgba_from_color(self.img)
        return wp.to_torch(rgba)[..., :3]


class NewtonRigidCameraWrapper(gym.Wrapper):
    def __init__(self, env, add_to_obs: bool = False, **cam_kw):
        super().__init__(env)
        self.add_to_obs, self.cam_kw = add_to_obs, cam_kw
        self.mirror = None

    def _cam(self):
        if self.mirror is None:
            self.mirror = NewtonRigidMirror(self.env.unwrapped, **self.cam_kw)
        return self.mirror

    def _add(self, obs):
        if self.add_to_obs and isinstance(obs, dict):
            obs.setdefault("sensor_data", {})["newton_camera"] = {"rgb": self._cam().render_all()}
        return obs

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return self._add(obs), info

    def step(self, action):
        obs, r, te, tr, info = self.env.step(action)
        return self._add(obs), r, te, tr, info

    def render(self):
        return self._cam().render_all()
