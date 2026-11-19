"""Opt-in, hash-bound scene additions; never changes task roles or collision gates."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .planning_contract import check_binding


def validate_background(background):
    if not isinstance(background, dict) or set(background) - {'wall', 'table', 'hdri'}:
        raise ValueError('Only explicit wall/table/hdri channels are supported')
    for kind, item in background.items():
        if not isinstance(item, dict) or not all(item.get(k) for k in ('path', 'sha256', 'asset_id', 'license', 'source')):
            raise ValueError('Background must bind provenance and exact file: '+kind)
        check_binding(item)
    return background


def remove_visual_actor(env, name):
    actor = env.scene.actors.get(name)
    if actor is None:
        return
    if any(body.get_collision_shapes() for body in actor._bodies):
        raise ValueError('Refuse to replace any collidable actor: '+name)
    if name in env.scene.state_dict_registry.actors:
        env.scene.remove_from_state_dict_registry(actor)
    for entity in actor._objs:
        env.scene.sub_scenes[0].remove_entity(entity)
    env.scene.actors.pop(name)


def apply_background(env, background):
    """Called during _load_scene before lights; textures alter RGB only."""
    if not background:
        return
    validate_background(background)
    import sapien
    for kind in ('wall', 'table'):
        if kind not in background:
            continue
        item = background[kind]
        material = sapien.render.RenderMaterial()
        material.set_base_color([1, 1, 1, 1])
        material.set_base_color_texture(sapien.render.RenderTexture2D(item['path']))
        material.set_roughness(.9 if kind == 'wall' else .75)
        if kind == 'wall':
            remove_visual_actor(env, 'appearance_only_wall')
            placements = [('candidate_wall_native', [.7498, 0, -.3], [.00005, 3, 1.5]),
                          ('candidate_wall_opposite', [-1.05, 0, -.3], [.00005, 3, 1.5])]
            env.randomization['wall_material'] = item
        else:
            remove_visual_actor(env, 'appearance_only_table_surface')
            placements = [('candidate_table_surface', [0, 0, .0001], [.65, .65, .00005])]
            env.randomization['diffuse_surface'] = item
        for name, position, half in placements:
            builder = env.scene.create_actor_builder()
            builder.add_box_visual(half_size=half, material=material)
            builder.initial_pose = sapien.Pose(position)
            builder.build_static(name)
    if 'hdri' in background:
        env.randomization['hdri'] = background['hdri']
    env.randomization['candidate_background'] = background
    env.randomization['candidate_background_collision_unchanged'] = True


def register_views(parent_id, candidate_id, background):
    """Keep native cameras; add one same-step opposite view for any supported task."""
    from mani_skill.utils.registration import REGISTERED_ENVS, register_env
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    from .multiview_recording import CAMERA_IDS, OPPOSITE_CAMERA
    validate_background(background)
    parent = REGISTERED_ENVS[parent_id].cls

    @register_env(candidate_id, max_episode_steps=3000)
    class CandidateScene(parent):
        def _load_scene(self, options):
            super()._load_scene(options)
            apply_background(self, background)

        @property
        def _default_sensor_configs(self):
            configs = list(super()._default_sensor_configs)
            if [c.uid for c in configs] != list(CAMERA_IDS[:2]):
                raise ValueError('Original camera identities changed')
            c = OPPOSITE_CAMERA
            configs.append(CameraConfig('opposite', sapien_utils.look_at(c['eye'], c['target']),
                c['width'], c['height'], c['fov_radians'], c['near'], c['far']))
            return configs
    return candidate_id
