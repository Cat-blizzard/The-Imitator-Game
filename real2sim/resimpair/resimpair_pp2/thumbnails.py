"""Fast visual-only thumbnails of asset descriptors (one SAPIEN scene, one camera, actors swapped).

Used for semantic labelling; never a physics or task statement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from resimpair_grasp.common import ROOT, binding, read, resolve, write


def render(descriptors, output, size=320):
    from resimpair_runtime import configure
    configure()
    import sapien
    from PIL import Image
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    scene = sapien.Scene()
    scene.set_ambient_light([.55, .55, .55])
    scene.add_directional_light([.4, .3, -1.], [1.4, 1.4, 1.4])
    scene.add_directional_light([-.5, -.4, -.6], [.5, .5, .5])
    ground = scene.create_actor_builder()
    ground.add_box_visual(half_size=[2, 2, .005], material=sapien.render.RenderMaterial(base_color=[.82, .82, .80, 1]))
    ground.build_static().set_pose(sapien.Pose([0, 0, -.005]))
    camera = scene.add_camera('thumb', size, size, .75, .01, 10.)
    rows = []
    for key, ref in descriptors:
        path = output / f'{key}.png'
        if path.exists():
            rows.append({'key': key, 'image': str(path)}); continue
        d = read(resolve(ref)); dims = np.asarray(d['dimensions_m'])
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(str(resolve(d['visual'])))
        actor = builder.build_kinematic()
        actor.set_pose(sapien.Pose([0, 0, dims[2] / 2]))
        radius = float(np.linalg.norm(dims)) / 2
        eye = np.array([1., -.8, .75]); eye = eye / np.linalg.norm(eye) * max(radius * 2.6, .12)
        target = np.array([0, 0, dims[2] / 2]); eye = eye + target
        forward = (target - eye) / np.linalg.norm(target - eye)
        left = np.cross([0, 0, 1], forward); left /= np.linalg.norm(left); up = np.cross(forward, left)
        pose = np.eye(4); pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = forward, left, up, eye
        camera.set_entity_pose(sapien.Pose(pose))
        scene.update_render(); camera.take_picture()
        rgb = (np.clip(camera.get_picture('Color')[..., :3], 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(path)
        scene.remove_actor(actor)
        rows.append({'key': key, 'image': str(path)})
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', type=Path, required=True, help='JSON list of [key, descriptor binding]')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); print(len(render(read(a.list), a.output)))
