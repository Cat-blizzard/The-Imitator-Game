"""Local mesh qualification and explicit metric scaling, without moving user assets."""
from pathlib import Path
import json
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from .common import sha, write, DEFAULT_ROOT
from .registry import SCENES

ASSETS = Path.home() / '.maniskill/data/robotwin/objects'


def inspect(spec):
    root = ASSETS / spec.category
    def mesh_file(kind):
        candidates = [folder / f'base{spec.model_id}.{ext}'
                      for folder in (root / kind, root) for ext in ('glb', 'obj')]
        return next((p for p in candidates if p.is_file()), candidates[0])
    collision = mesh_file('collision')
    visual = mesh_file('visual')
    annotation = root / f'model_data{spec.model_id}.json'
    for p in (collision, visual, annotation):
        if not p.is_file():
            raise FileNotFoundError(p)
    mesh = trimesh.load(collision, force='mesh')
    rotation = Rotation.from_euler('x', spec.rotation_x_deg, degrees=True).as_matrix()
    vertices = np.asarray(mesh.vertices) @ rotation.T
    extents = np.ptp(vertices, axis=0)
    scale = spec.largest_extent_m / float(extents.max())
    bounds = np.stack([vertices.min(0), vertices.max(0)]) * scale
    horizontal_width = float(np.sort(extents[:2])[0] * scale)
    if horizontal_width > .072:
        # Panda's physical aperture is 8 cm; leave clearance for a top grasp.
        scale *= .072 / horizontal_width
        bounds *= .072 / horizontal_width
    data = json.loads(annotation.read_text())
    return dict(category=spec.category, model_id=spec.model_id, name=spec.name,
                uniform_scale=scale, bounds_rotated_m=bounds.tolist(),
                rotation_x_deg=spec.rotation_x_deg,
                annotated_contact_frames=len(data.get('contact_points_pose', [])),
                grasp_method='Collision-mesh oriented bounding box; physical lift qualification required.',
                files={str(p): sha(p) for p in (collision, visual, annotation)},
                upstream='https://github.com/RoboTwin-Platform/RoboTwin',
                license_note='Repository code is MIT; per-asset upstream licenses are not inferred from the code license.')


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = p.parse_args()
    rows = {scene: [inspect(s) for s in cfg.objects] for scene, cfg in SCENES.items()}
    write(args.root / 'asset_inventory.json', rows)
    print({k: [r['name'] for r in v] for k, v in rows.items()}, flush=True)


if __name__ == '__main__':
    main()
