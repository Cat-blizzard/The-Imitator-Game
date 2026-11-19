"""New task IDs; do not modify any Imitator Game task or scoring predicate."""
import numpy as np
import torch
import sapien
from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2quat
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.structs.pose import Pose
from .registry import SCENES, PROGRAMS
from .assets import inspect


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


@register_env('HyperICLTaskExpansion-v1', max_episode_steps=3000)
class TaskExpansionEnv(BaseEnv):
    SUPPORTED_ROBOTS = [('panda_wristcam', 'panda_wristcam')]
    agent: MultiAgent

    def __init__(self, *args, scene_name='grocery_fruit', program='sort_direct', **kwargs):
        self.spec = SCENES[scene_name]
        self.scene_name = scene_name
        self.program = program
        self.robot_init_qpos_noise = 0.
        self.asset_specs = [inspect(s) for s in self.spec.objects]
        self.inner_half = np.array([.115, .085])
        self.floor_top = .012
        self.wall_height = .042
        self.goal_centers = np.array([[-.015, -.120, 0.], [-.015, .120, 0.]])
        self.objects = []
        self.bins = []
        super().__init__(*args, robot_uids=('panda_wristcam', 'panda_wristcam'),
                         reconfiguration_freq=0, **kwargs)

    @property
    def left_agent(self):
        return self.agent.agents[0]

    @property
    def right_agent(self):
        return self.agent.agents[1]

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-.4, 0., .65], target=[0., 0., .10])
        return [CameraConfig('zed2i', pose, 224, 224, np.pi * 1.1 / 2, .01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([.65, -.5, .65], [0., 0., .1])
        return CameraConfig('render_camera', pose, 512, 512, 1, .01, 100)

    def _load_agent(self, options):
        super()._load_agent(options, [sapien.Pose([0, -1, 0]), sapien.Pose([0, 1, 0])])

    def _load_scene(self, options):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0.)
        self.table_scene.build()
        self.objects = []
        for i, spec in enumerate(self.asset_specs):
            s = spec['uniform_scale']
            actor = create_actor(self.scene, sapien.Pose([0, 0, .15]), spec['category'],
                scale=(s, s, s), replace_scale=True, convex=True,
                model_id=spec['model_id'], mass=.08, name=f'object_{i}')
            self.objects.append(actor.actor)
        self.bins = []
        for j, color in enumerate(([.65, .30, .16, 1.], [.12, .32, .62, 1.])):
            builder = self.scene.create_actor_builder()
            # A physically hollow tray: five separate box colliders, not one solid hull.
            shapes = [(np.array([.125, .095, .006]), np.array([0., 0., .006]))]
            shapes += [(np.array([.005, .095, .021]), np.array([x, 0., .021])) for x in (-.12, .12)]
            shapes += [(np.array([.115, .005, .021]), np.array([0., y, .021])) for y in (-.09, .09)]
            for half, center in shapes:
                pose = sapien.Pose(center)
                builder.add_box_collision(pose=pose, half_size=half)
                builder.add_box_visual(pose=pose, half_size=half, material=color)
            builder.initial_pose = sapien.Pose(self.goal_centers[j])
            self.bins.append(builder.build_static(f'container_{j}'))

    def _initialize_episode(self, env_idx, options):
        self.program = options.get('program', self.program)
        if self.program not in PROGRAMS:
            raise ValueError(self.program)
        self.table_scene.initialize(env_idx)
        rng = np.random.default_rng(int(options.get('layout_seed', 0)))
        # Draw all randomness independently of goal/program; same layout can have many goals.
        jitter = rng.uniform(-.025, .025, (2, 2))
        self.goal_centers = np.array([[-.015, -.120, 0.], [-.015, .120, 0.]])
        self.goal_centers[:, :2] += rng.uniform(-.018, .018, (2, 2))
        for j, actor in enumerate(self.bins):
            actor.set_pose(sapien.Pose(self.goal_centers[j]))
        for i, (actor, spec) in enumerate(zip(self.objects, self.asset_specs)):
            angle = float(rng.uniform(-.28, .28))
            rotation = Rotation.from_euler('z', angle) * Rotation.from_euler('x', spec['rotation_x_deg'], degrees=True)
            qxyzw = rotation.as_quat()
            q = np.r_[qxyzw[3], qxyzw[:3]]
            # Recalculate floor height for the yaw; x rotation already included in inspected bounds.
            bounds = np.asarray(spec['bounds_rotated_m'])
            p = np.array([-.115, (-.32 if i == 0 else .32), -bounds[0, 2] + .005])
            p[:2] += jitter[i]
            actor.set_pose(Pose.create_from_pq(p, q))

    def _get_obs_extra(self, info):
        # No privileged object states, task identity, or goal ID in policy observations.
        return {}

    def evaluate(self):
        # Full physical qualification is done after release and settling in collector.
        return {'success': torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)}

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)


def make_env(gpu, scene, program):
    import gymnasium as gym
    return gym.make('HyperICLTaskExpansion-v1', scene_name=scene, program=program,
                    sim_backend='physx_cpu', render_backend=gpu['render_backend'],
                    control_mode='pd_joint_pos', obs_mode='rgb', reward_mode='none',
                    render_mode='rgb_array', max_episode_steps=3000,
                    sensor_configs=dict(shader_pack='rt-fast'),
                    human_render_camera_configs=dict(shader_pack='rt-fast'))
