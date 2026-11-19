"""Ray-traced lighting for scene-domain rooms (visual-only).

The scene-domain room is a closed box, so its directional key and fill lights never reach the
table under ray tracing; only the rasterizer ignores the occluding walls. Interior emitters
replace them: an area light in the room's own primary source (window opening or ceiling
fixture) keeps the domain's colour temperature and intensity, a broad dim ceiling panel stands
in for the light a real room inter-reflects (it also gives metals something to mirror), and a
panel facing the rear wall lights it and its shelves, which otherwise fall to black behind the
robots. An area light only emits into its front half-space, so that panel never reaches the
table or the robots and the exposure stays set by the table.
Exposure is fixed per camera from the first frame, so a video never flickers.

Some downloaded meshes shade with normals pointing into the solid (typically: no NORMAL attribute,
inside-out winding, so the loader derives inward normals). The rasterizer shades them anyway, but a
ray tracer offsets shadow rays into the solid and renders them black, so `repair_winding` flips
those closed components in the render shapes only. Inside-out winding with outward normals renders
correctly and is left alone.
"""
from __future__ import annotations

import numpy as np

SHADER = 'rt-med'
TONE_MAPPERS = {'gamma': 0, 'srgb': 1, 'aces': 2}
AUDIT = 'runs/pp2_rt_audit_20261102_v12/audit.json'  # resimpair_pp2.rt_audit: which assets get repaired
TARGET_MEDIAN = .45  # median luminance of the approved rasterized keyframes (36 cells, 11/02 grid)


def intensity(lighting):
    from resimpair_tasks.scene_domain_v2 import TEMPERATURE_RGB
    return float(max(lighting['key_rgb']) / max(TEMPERATURE_RGB[lighting['temperature_k']]))


def _facing(direction):
    """Pose quaternion whose +x (the emitting side of a SAPIEN area light) points along `direction`."""
    from scipy.spatial.transform import Rotation
    x = np.asarray(direction, float); x /= np.linalg.norm(x)
    up = np.array([0., 0., 1.]) if abs(x[2]) < .9 else np.array([0., 1., 0.])
    y = np.cross(up, x); y /= np.linalg.norm(y)
    return Rotation.from_matrix(np.stack([x, y, np.cross(x, y)], 1)).as_quat(scalar_first=True)


def emitters(scene, key_radiance=8., fill_ratio=.35, wash_ratio=8.):
    """Area lights as plain rows, so the render record shows exactly what lit the frame."""
    domain = scene['scene_domain']; geometry = domain['geometry']; lighting = domain['lighting']
    color = np.asarray(lighting['key_rgb'], float) / max(lighting['key_rgb'])
    source = geometry['primary_lights'][0]; position = np.asarray(source['position_m'], float)
    ceiling = next(b for b in geometry['boxes'] if b['id'] == 'ceiling')
    ceiling_z = ceiling['center_m'][2] - ceiling['size_m'][2] / 2
    if source['kind'] == 'window':
        key = {'role': 'window', 'position_m': (position + [.02, 0, 0]).tolist(), 'direction': [1., 0., 0.],
               'half_size_m': [.62, .45]}
    else:
        key = {'role': 'ceiling_fixture', 'position_m': (position - [0, 0, .05]).tolist(), 'direction': [0., 0., -1.],
               'half_size_m': [.30, .14]}
    key['rgb'] = (color * key_radiance * intensity(lighting)).tolist()
    area = 4 * key['half_size_m'][0] * key['half_size_m'][1]
    fill = {'role': 'ceiling_fill', 'position_m': [-.12, 0., ceiling_z - .12], 'direction': [0., 0., -1.],
            'half_size_m': [1.0, .8]}
    fill['rgb'] = (color * key_radiance * intensity(lighting) * fill_ratio * area / (4 * 1.0 * .8)).tolist()
    behind_sweep = geometry['protected_bounds_m'][0][0] - .02  # in front of the room's furniture, behind every robot pose
    wash = {'role': 'rear_wall_washer', 'position_m': [behind_sweep, 0., .9], 'direction': [-1., 0., 0.],
            'half_size_m': [1.4, .5]}
    wash['rgb'] = (color * key_radiance * intensity(lighting) * wash_ratio * area / (4 * 1.4 * .5)).tolist()
    return [key, fill, wash]


def install(env, rows):
    import sapien
    for sub in env.scene.sub_scenes:
        for row in rows:
            pose = sapien.Pose(row['position_m'], _facing(row['direction']))
            sub.add_area_light_for_ray_tracing(pose, row['rgb'], *row['half_size_m'])


def expose(env, tone='aces', target=TARGET_MEDIAN, rounds=5):
    """Per-camera exposure so the first frame's median luminance hits `target`; returns the exposures."""
    from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array
    cameras = {name: sensor.camera._render_cameras for name, sensor in env._sensors.items()}
    exposure = {name: 1. for name in cameras}
    for name, components in cameras.items():
        for c in components:
            c.set_property('toneMapper', TONE_MAPPERS[tone]); c.set_property('exposure', 1.)
    for _ in range(rounds):
        data = env.get_obs()['sensor_data']
        for name, components in cameras.items():
            luminance = array(data[name]['rgb']).astype(float).mean(-1) / 255.
            median = max(float(np.median(luminance)), 1e-3)
            exposure[name] *= float(np.clip((target / median) ** 2.2, .25, 4.))
            for c in components:
                c.set_property('exposure', exposure[name])
    return exposure


def inverted_faces(vertices, triangles, normals, max_open=.01):
    """Per face: its connected component is closed (<1% boundary edges) and its shading normals point inward,
    i.e. the normal flux through the surface about the component centre is negative.

    Open pieces are left alone: their enclosed volume has no meaning, and a mesh split per material
    is often open in every piece."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    v = np.asarray(vertices, np.float64); t = np.asarray(triangles, np.int64)
    _, canon = np.unique(np.round(v, 6), axis=0, return_inverse=True)
    f = canon.reshape(-1)[t]; n = int(f.max()) + 1
    rows = np.concatenate([f[:, 0], f[:, 1]]); cols = np.concatenate([f[:, 1], f[:, 2]])
    _, label = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)
    label = label[f[:, 0]]; count = label.max() + 1
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, edge_id, uses = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    boundary = (uses[edge_id.reshape(-1)] == 1).astype(float)
    open_share = np.bincount(np.tile(label, 3), boundary, count) / np.bincount(np.tile(label, 3), None, count)
    p = v[t]; area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) / 2
    centre = np.stack([np.bincount(label, p[:, :, k].mean(1) * area, count) for k in range(3)], 1)
    centre /= np.maximum(np.bincount(label, area, count), 1e-30)[:, None]
    shading = np.asarray(normals, np.float64)[t].mean(1)
    shading /= np.maximum(np.linalg.norm(shading, axis=1, keepdims=True), 1e-12)
    volume = area * np.einsum('ij,ij->i', p.mean(1) - centre[label], shading) / 3
    part_area = np.bincount(label, area, count); part_volume = np.bincount(label, volume, count)
    bad = (open_share < max_open) & (part_volume < -.002 * np.maximum(part_area, 1e-30) ** 1.5)
    return bad[label], area


def repair_entity(entity, mode='closed'):
    """Swap the entity's render body for one whose inside-out closed parts ('closed') or all faces ('all')
    are flipped; returns the flipped area share."""
    import sapien
    body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
    if body is None:
        return 0.
    shapes, flipped, total = [], 0., 0.
    for shape in body.render_shapes:
        if not isinstance(shape, sapien.render.RenderShapeTriangleMesh):
            shapes.append(None); continue
        parts = []
        for part in shape.parts:
            bad, area = inverted_faces(part.get_vertices(), part.get_triangles(), part.get_vertex_normal())
            if mode == 'all':
                bad = np.ones_like(bad)
            total += area.sum(); flipped += area[bad].sum(); parts.append((part, bad))
        shapes.append(parts)
    if not flipped:
        return 0.
    fresh = sapien.render.RenderBodyComponent(); fresh.name = body.name
    for shape, parts in zip(body.render_shapes, shapes):
        if parts is None:
            fresh.attach(shape.clone()); continue
        for part, bad in parts:
            triangles = part.get_triangles().copy(); normals = part.get_vertex_normal().copy()
            normals[np.unique(triangles[bad])] *= -1; triangles[bad] = triangles[bad][:, ::-1]
            uv = part.get_vertex_uv()
            uv = uv if uv is not None and len(uv) == len(normals) else np.zeros((len(normals), 2), np.float32)
            rebuilt = sapien.render.RenderShapeTriangleMesh(part.get_vertices().astype(np.float32), triangles.astype(np.uint32),
                                                            normals.astype(np.float32), uv.astype(np.float32), part.material)
            rebuilt.set_local_pose(shape.local_pose); rebuilt.set_scale(shape.scale)
            fresh.attach(rebuilt)
    entity.remove_component(body); entity.add_component(fresh)
    return float(flipped / total)


def repair_winding(env, policy):
    """Repair the objects, distractors and receptacles the ray-tracing audit chose (source_id -> mode).

    Render components only: collision shapes and entity ids (hence segmentation) are untouched."""
    specs = {o['id']: o for o in env.scene_spec['objects'] + env.scene_spec['supports']}
    rows = []
    for name, actor in [*zip(env.object_ids, env.objects), *((s['id'], a) for s, a in zip(env.scene_spec['supports'], env.support_actors))]:
        mode = policy.get(specs.get(name, {}).get('pp2', {}).get('source_id'))
        for obj in actor._objs if mode else ():
            entity = getattr(obj, 'entity', obj)
            rows.append({'object': name, 'mode': mode, 'flipped_area_fraction': repair_entity(entity, mode)})
    return rows


def audit_policy(path=None):
    from resimpair_grasp.common import ROOT, read
    path = path or ROOT / AUDIT
    return {r['source_id']: r['decision'][len('repair_'):] for r in read(path)['rows'] if r['decision'].startswith('repair_')} \
        if path.exists() else {}


def setup(env, scene, first_state, tone='aces'):
    """Repair, light and expose one ray-traced env at the first saved state; returns the render record."""
    import torch
    from resimpair_grasp.common import ROOT, binding
    repaired = repair_winding(env, audit_policy()); rows = emitters(scene); install(env, rows)
    env.set_state(torch.as_tensor(first_state, device='cpu'))
    target = TARGET_MEDIAN * intensity(scene['scene_domain']['lighting']) ** .5
    return {'shader': SHADER, 'tone_mapper': tone, 'emitters': rows, 'exposure_target_median': target,
            'exposure': expose(env, tone, target), 'winding_repaired': repaired,
            'audit': binding(ROOT / AUDIT) if (ROOT / AUDIT).exists() else None}
