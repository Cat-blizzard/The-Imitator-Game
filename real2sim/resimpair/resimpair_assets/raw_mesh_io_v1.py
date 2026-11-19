"""Original rigid mesh loading and material-preserving metric visual export."""
from pathlib import Path
from collections import Counter
import json
import struct

import numpy as np
import trimesh

from resimpair_grasp.common import binding, resolve
from .raw_metric_containers_v1 import raw_dependencies, texture_summary
from .metric_visual_precision_v3 import bake_visual_world


def check_embedded_dependencies(path):
    path = Path(path)
    if path.suffix.lower() == '.glb':
        with path.open('rb') as stream:
            header = stream.read(12)
            if len(header) != 12 or struct.unpack('<III',header)[:2] != (0x46546C67,2):
                raise ValueError('invalid_GLB_header')
            chunk = stream.read(8)
            if len(chunk) != 8: raise ValueError('invalid_GLB_JSON_chunk')
            length,kind = struct.unpack('<II',chunk)
            if kind != 0x4E4F534A or length > 32 * 2**20:
                raise ValueError('invalid_or_excessive_GLB_JSON_chunk')
            model = json.loads(stream.read(length))
        for entry in model.get('images',[]) + model.get('buffers',[]):
            uri = entry.get('uri')
            if uri and not uri.startswith('data:'):
                raise ValueError('GLB_external_resource_requires_explicit_dependency_adapter')
    elif path.suffix.lower() == '.ply':
        with path.open('rb') as stream:
            header = stream.read(65536).split(b'end_header',1)[0].lower()
        if b'texturefile' in header or b'texture_file' in header:
            raise ValueError('PLY_external_texture_requires_explicit_dependency_adapter')


def load_raw_scene(row):
    path = Path(row['path']).resolve()
    if not path.is_file() or path.stat().st_size > 256 * 2**20:
        raise ValueError('raw_mesh_missing_or_exceeds_256MiB')
    if binding(path)['sha256'] != row['source_sha']:
        raise ValueError('raw_source_bytes_changed')
    if path.suffix.lower() == '.obj':
        deps = raw_dependencies(row)
        resolver = {key: resolve(ref).read_bytes() for key,ref in deps['resolver'].items()}
        scene = trimesh.load(path, force='scene', process=False, resolver=resolver)
    elif path.suffix.lower() in {'.glb', '.ply', '.stl'}:
        check_embedded_dependencies(path)
        deps = {'source': binding(path), 'resolver': {}, 'raw_files_modified': False}
        scene = trimesh.load(path, force='scene', process=False)
    else:
        raise ValueError('raw_format_requires_explicit_dependency_adapter')
    scene = bake_visual_world(scene)
    mesh = scene.to_geometry()
    if not isinstance(mesh,trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('raw_mesh_has_no_triangle_geometry')
    if len(mesh.faces) > 300000:
        raise ValueError('raw_mesh_face_budget_300000')
    if not np.isfinite(mesh.vertices).all() or np.min(mesh.extents) <= 1e-10:
        raise ValueError('raw_mesh_nonfinite_or_degenerate')
    return scene, mesh, {'dependencies': deps, 'raw_bounds_units': mesh.bounds.tolist(),
        'raw_materials': texture_summary(scene), 'faces': len(mesh.faces),
        'source_scene_node_transforms_baked': True, 'geometry_rescaled_in_loader': False,
        'implementation': binding(__file__)}


def export_visual(scene, path, expected_mesh):
    path = Path(path)
    if path.exists(): raise ValueError('preserve_existing_visual_export')
    path.parent.mkdir(parents=True, exist_ok=True)
    baked = bake_visual_world(scene)
    original = texture_summary(baked)
    baked.export(path)
    loaded = trimesh.load(path, force='scene', process=False)
    mesh = loaded.to_geometry(); after = texture_summary(loaded)
    if not np.allclose(mesh.bounds, expected_mesh.bounds, rtol=1e-6, atol=1e-7):
        raise ValueError('visual_export_changed_scale_or_origin')
    textures = lambda rows: Counter(x['texture']['pixels_sha256'] for x in rows if x['texture'])
    if textures(original) != textures(after):
        raise ValueError('visual_export_lost_source_texture_pixels')
    if sum(x['has_UV'] for x in original) != sum(x['has_UV'] for x in after):
        raise ValueError('visual_export_lost_UV_presence')
    return {'visual': binding(path), 'bounds_preserved': True, 'source': original, 'reloaded': after,
        'texture_pixels_preserved': True, 'UV_presence_preserved': True,
        'material_check_scope': 'base_color_texture_pixels_and_UV_presence; other_PBR_channels_not_independently_qualified',
        'render_equivalence': 'not_evaluated', 'implementation': binding(__file__)}
