"""Opt-in, repository-local visual variation, independent of task physics.

Surface texture/roughness, panorama or wall, and light direction/white balance
are sampled from one seed. No collision, mass, friction, camera or task field is
changed. Missing or modified resources fail before constructing an environment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import shutil

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, sha, write

VERSION = 'resimpair_visual_appearance_v1'
CATALOG_VERSION = 'resimpair_local_appearance_catalog_v1'
DEFAULT_CATALOG = ROOT / 'assets/appearance_v1/catalog.json'
SURFACE_ROLES = {
    'angli_veneer': ['table'], 'brown_leather': ['table'],
    'fabric_leather_02': ['table'], 'bi_stretch': ['table'],
    'scuba_suede': ['table'], 'crepe_georgette': ['table'],
    'blue_metal_plate': ['table', 'wall'], 'anti_skid_tiles': ['table'],
    'brick_wall_04': ['wall'], 'bamboo_wall': ['wall'],
    'bamboo_wall_02': ['wall'], 'bamboo_wall_03': ['wall'],
    'broken_wall': ['wall'], 'container_side': ['wall'],
    'corrugated_iron': ['wall'], 'box_profile_metal_sheet': ['wall'],
}
INDOOR_HDRIS = ['abandoned_factory_canteen_01', 'abandoned_factory_canteen_02',
               'abandoned_workshop', 'abandoned_workshop_02',
               'aerodynamics_workshop', 'abandoned_tiled_room']
# Deliberately an artistic white-balance palette, not calibrated photometry.
TEMPERATURE_PALETTE = {3500: [1., .76, .54], 4500: [1., .86, .70],
                       6500: [1., .97, .95], 8000: [.83, .90, 1.]}


def prepare_catalog(polyhaven_root, prior_background_root, output):
    """Copy already downloaded CC0 sources, verifying their retained receipts."""
    source, prior, output = map(Path, (polyhaven_root, prior_background_root, output))
    if output.exists():
        raise ValueError('appearance_catalog_output_must_be_new')
    output.mkdir(parents=True)
    entries = []

    def copy_file(path, expected, directory):
        path = Path(path)
        if sha(path) != expected:
            raise ValueError('original_appearance_source_hash_mismatch:' + str(path))
        target = directory / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        return binding(target)

    def receipt_entry(asset_id, receipt, receipt_path, roles, files):
        if receipt['license'] != 'CC0-1.0':
            raise ValueError('this_importer_requires_existing_CC0_receipt')
        directory = output / asset_id
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(receipt_path, directory / 'original_receipt.json')
        entry = {'id': 'polyhaven:' + asset_id, 'roles': roles,
                 'license': receipt['license'], 'source': receipt['source'],
                 'attribution': receipt.get('attribution', 'Powered by Poly Haven'),
                 'receipt': binding(directory / 'original_receipt.json'),
                 'resources': {}, 'usage': 'visual_only',
                 'geometry_or_contact_material_changed': False}
        for channel, path, expected in files:
            entry['resources'][channel] = copy_file(path, expected, directory)
        entries.append(entry)

    for asset_id, roles in SURFACE_ROLES.items():
        folder = source / 'textures' / asset_id
        receipt_path = folder / 'ASSET_RECEIPT.json'; receipt = read(receipt_path)
        hashes = {f['path']: f['sha256'] for f in receipt['files']}
        channels = {'Diffuse': 'base_color', 'Rough': 'roughness', 'nor_gl': 'normal'}
        files = [(channels[c['name']], c['path'], hashes[c['path']])
                 for c in receipt['channels'] if c['name'] in channels]
        if 'base_color' not in [f[0] for f in files]:
            raise ValueError('surface_diffuse_texture_required')
        receipt_entry(asset_id, receipt, receipt_path, roles, files)
    for asset_id in INDOOR_HDRIS:
        folder = source / 'hdris' / asset_id
        receipt_path = folder / 'ASSET_RECEIPT.json'; receipt = read(receipt_path)
        files = [('environment', f['path'], f['sha256']) for f in receipt['files']
                 if Path(f['path']).suffix.lower() == '.hdr']
        if len(files) != 1:
            raise ValueError('exactly_one_HDRI_required')
        receipt_entry(asset_id, receipt, receipt_path, ['environment'], files)
    for asset_id in ['small_empty_room_3', 'small_empty_room_4', 'poly_haven_studio']:
        folder = prior / 'downloaded' / asset_id
        receipt_path = folder / 'receipt.json'; receipt = read(receipt_path)
        receipt_entry(asset_id, receipt, receipt_path, ['environment'],
                      [('environment', receipt['path'], receipt['sha256'])])
    write(output / 'catalog.json', {'schema': CATALOG_VERSION, 'entries': entries,
        'source_mode': 'byte_identical_copy_of_previously_downloaded_local_assets',
        'api_calls': 0, 'network_downloads': 0, 'assets_are_visual_only': True,
        'role_selection': 'surface roles chosen once for tabletop/wall visual use; not object affordances'})
    return {'catalog': binding(output / 'catalog.json'), 'resources': len(entries),
            'surfaces': len(SURFACE_ROLES), 'environments': len(entries) - len(SURFACE_ROLES)}


def validate_catalog(path):
    path = Path(path); value = read(path)
    if value.get('schema') != CATALOG_VERSION:
        raise ValueError('unsupported_appearance_catalog_schema')
    ids = set()
    for entry in value['entries']:
        if entry['id'] in ids or not set(entry['roles']) <= {'table', 'wall', 'environment'}:
            raise ValueError('invalid_or_duplicate_appearance_resource')
        ids.add(entry['id'])
        resolve(entry['receipt'])
        for resource in entry['resources'].values():
            resolve(resource)
    return value


def sample(scene, seed, *, catalog=DEFAULT_CATALOG, config=None):
    """Return the same task/physics scene with a separate visual specification."""
    config = deepcopy(config or {})
    allowed = {'table_ids', 'wall_ids', 'environment_ids', 'background_mode',
               'key_intensity', 'fill_intensity', 'ambient_intensity', 'temperature_k'}
    if set(config) - allowed or type(seed) is not int or seed < 0:
        raise ValueError('invalid_appearance_configuration_or_seed')
    catalog_value = validate_catalog(catalog)
    rng = np.random.default_rng(seed)

    def choose(role):
        selected = config.get(role + '_ids')
        if selected is not None and (not isinstance(selected, list) or not selected):
            raise ValueError('appearance_filter_must_be_nonempty_list')
        choices = [e for e in catalog_value['entries'] if role in e['roles']
                   and (selected is None or e['id'] in selected)]
        if not choices or (selected is not None and set(selected) - {e['id'] for e in choices}):
            raise ValueError('unknown_or_incompatible_appearance_resource:' + role)
        return deepcopy(choices[int(rng.integers(len(choices)))])

    def interval(key, default, allowed_range):
        values = np.asarray(config.get(key, default), dtype=float)
        if (values.shape != (2,) or not np.isfinite(values).all()
                or not allowed_range[0] <= values[0] <= values[1] <= allowed_range[1]):
            raise ValueError('invalid_light_range:' + key)
        return float(rng.uniform(*values))

    mode = config.get('background_mode', ['wall', 'environment'][int(rng.integers(2))])
    if mode not in {'wall', 'environment'}:
        raise ValueError('unsupported_background_mode')
    table, wall, environment = choose('table'), choose('wall'), choose('environment')
    temperatures = config.get('temperature_k', list(TEMPERATURE_PALETTE))
    if not isinstance(temperatures, list) or not temperatures or set(temperatures) - set(TEMPERATURE_PALETTE):
        raise ValueError('unsupported_visual_temperature_palette')
    temperature = int(temperatures[int(rng.integers(len(temperatures)))])
    key = interval('key_intensity', [.7, 1.5], [0, 5])
    fill = interval('fill_intensity', [.15, .45], [0, 2])
    ambient = interval('ambient_intensity', [.15, .32], [0, 1])
    azimuth = float(rng.uniform(-math.pi, math.pi)); elevation = float(rng.uniform(.55, 1.2))
    direction = [math.cos(azimuth) * math.cos(elevation),
                 math.sin(azimuth) * math.cos(elevation), -math.sin(elevation)]
    fill_direction = [-direction[0], -direction[1], -.8]
    fill_direction = (np.asarray(fill_direction) / np.linalg.norm(fill_direction)).tolist()
    result = deepcopy(scene)
    physical = {k: v for k, v in scene.items() if k != 'appearance'}
    result['appearance'] = {'schema': VERSION, 'render_revision': 2, 'seed': seed, 'catalog': binding(catalog),
        'table_rgb': [1., 1., 1.], 'background_rgb': [1., 1., 1.],
        'ambient_rgb': [ambient] * 3,
        'key_light_rgb': (np.asarray(TEMPERATURE_PALETTE[temperature]) * key).tolist(),
        'key_direction': direction, 'fill_direction': fill_direction,
        'fill_rgb': [fill] * 3, 'temperature_k': temperature,
        'temperature_model': 'coarse_artistic_white_balance_palette_not_photometric_calibration',
        'table_surface': table, 'wall_surface': wall, 'environment': environment,
        'background_mode': mode, 'roughness_multiplier': float(rng.uniform(.65, 1.)),
        'metallic': 0., 'physical_scene_digest': digest(physical),
        'task_or_camera_changed': False, 'independent_physical_trajectory': False}
    return result


def validate_scene(scene, *, check_scene_binding=True):
    appearance = scene['appearance']
    if appearance.get('schema') != VERSION:
        raise ValueError('unsupported_visual_appearance_schema')
    resolve(appearance['catalog'])
    if check_scene_binding and appearance['physical_scene_digest'] != digest({k: v for k, v in scene.items() if k != 'appearance'}):
        raise ValueError('appearance_bound_scene_physics_or_camera_changed')
    for role in ('table_surface', 'wall_surface', 'environment'):
        for resource in appearance[role]['resources'].values():
            resolve(resource)
    return appearance


def load_lighting(env):
    a = validate_scene(env.scene_spec)
    env.scene.set_ambient_light(a['ambient_rgb'])
    env.scene.add_directional_light(a['key_direction'], a['key_light_rgb'],
                                    shadow=True, shadow_scale=5, shadow_map_size=2048)
    env.scene.add_directional_light(a['fill_direction'], a['fill_rgb'])
    if env.scene.can_render() and a['background_mode'] == 'environment':
        for subscene in env.scene.sub_scenes:
            subscene.set_environment_map(str(resolve(a['environment']['resources']['environment'])))


def build_visuals(env):
    import sapien
    # ContainerEnv temporarily removes supports while invoking the shared
    # parent's _load_scene. Lighting checks the complete scene separately.
    a = validate_scene(env.scene_spec, check_scene_binding=False)
    can_render = env.scene.can_render()
    env.appearance_runtime_evidence = {'render_revision': a.get('render_revision', 1),
        'visual_only_changes': [], 'collision_geometry_changed': False}

    def hide_visual(entity):
        physical = [c for c in entity.components if isinstance(c, sapien.physx.PhysxRigidBaseComponent)]

        def collision_snapshot():
            rows = []
            for component in physical:
                for shape in component.get_collision_shapes():
                    row = {'type': type(shape).__name__,
                           'local_pose': shape.local_pose.to_transformation_matrix().tolist()}
                    for attr in ('half_size', 'radius', 'half_length', 'vertices', 'scale'):
                        if hasattr(shape, attr):
                            row[attr] = np.asarray(getattr(shape, attr)).tolist()
                    rows.append(row)
            return {'entity_name': entity.name, 'entity_pose': entity.pose.to_transformation_matrix().tolist(),
                    'physical_components': [type(c).__name__ for c in physical], 'shapes': rows}

        before = collision_snapshot()
        render_body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
        if render_body is not None:
            render_body.visibility = 0
        after = collision_snapshot()
        same_components = all(c in entity.components for c in physical)
        if digest(before) != digest(after) or not same_components:
            raise RuntimeError('visual_hiding_must_preserve_physical_components')
        env.appearance_runtime_evidence['visual_only_changes'].append({
            'entity': entity.name, 'collision_before': digest(before), 'collision_after': digest(after),
            'physical_components_preserved': same_components, 'collision_shapes': len(before['shapes']),
            'render_body_present': render_body is not None, 'render_visibility': 0})

    def material(entry):
        if not can_render:
            return [1., 1., 1., 1.]
        mat = sapien.render.RenderMaterial(base_color=[1., 1., 1., 1.],
            metallic=a['metallic'], roughness=a['roughness_multiplier'])
        for channel, attribute in [('base_color', 'base_color_texture'),
                                    ('roughness', 'roughness_texture'), ('normal', 'normal_texture')]:
            if channel in entry['resources']:
                texture = sapien.render.RenderTexture2D(str(resolve(entry['resources'][channel])),
                            mipmap_levels=5, srgb=channel == 'base_color')
                setattr(mat, attribute, texture)
        return mat

    builder = env.scene.create_actor_builder()
    builder.add_box_visual(half_size=[.58, .94, .0002], material=material(a['table_surface']))
    builder.initial_pose = sapien.Pose([-.12, 0., .0003])
    builder.build_static('table_surface_visual')
    builder = env.scene.create_actor_builder()
    if a['background_mode'] == 'wall':
        builder.add_box_visual(half_size=[.003, 1.2, .8], material=material(a['wall_surface']))
    elif can_render:
        hide_visual(env.table_scene.wall.actor)
        if a.get('render_revision', 1) >= 2:
            # The 100 m stock ground covers the panorama in downward-facing
            # cameras. Hide its visual component only; retain the actor,
            # rigid body, collision shapes, pose and state registry.
            for obj in env.table_scene.ground._objs:
                hide_visual(obj)
    # Preserve the actor identity even for a panorama: state layout is stable.
    builder.initial_pose = sapien.Pose([-1.0, 0., .3])
    builder.build_static('background_visual')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare-catalog')
    prepare.add_argument('--polyhaven-root', type=Path, required=True)
    prepare.add_argument('--prior-background-root', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    s = commands.add_parser('sample'); s.add_argument('--scene', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True); s.add_argument('--seed', type=int, required=True)
    s.add_argument('--catalog', type=Path, default=DEFAULT_CATALOG)
    s.add_argument('--config', type=Path)
    args = p.parse_args()
    if args.command == 'prepare-catalog':
        print(prepare_catalog(args.polyhaven_root, args.prior_background_root, args.output))
    else:
        write(args.output, sample(read(args.scene), args.seed, catalog=args.catalog,
                                config=read(args.config) if args.config else None))
