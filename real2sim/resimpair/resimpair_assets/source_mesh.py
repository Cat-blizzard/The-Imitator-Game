"""Load declared local mesh resources, including GSO's separate texture folder.

No raw files are edited. Missing declared textures are failures, not gray fallbacks.
"""
from pathlib import Path
import json
import shlex
import trimesh
from trimesh.resolvers import FilePathResolver
from resimpair_grasp.common import binding, resolve


def _root(row):
    p=Path(row['primary_path']).resolve()
    return p.parent.parent if row['source']=='google_scanned_objects_original_fuel' else p.parent


def _find(row,name):
    p=Path(row['primary_path']).resolve();root=_root(row)
    choices=[p.parent/name]
    if row['source']=='google_scanned_objects_original_fuel':
        choices.append(root/'materials/textures'/Path(name).name)
    for candidate in choices:
        candidate=candidate.resolve()
        if candidate.is_relative_to(root) and candidate.is_file():return candidate
    raise ValueError('Declared mesh resource missing: '+name)


def resources(row):
    p=Path(row['primary_path']);found={};missing=[];textures=[]
    def add(name,texture=False):
        try:
            path=_find(row,name);found[str(path)]=binding(path)
            if texture:textures.append(str(path))
            return path
        except ValueError:missing.append(name)
    if p.suffix.lower()=='.obj':
        for line in p.read_text(errors='replace').splitlines():
            if line.strip().startswith('mtllib '):
                for name in shlex.split(line)[1:]:
                    mtl=add(name)
                    if mtl:
                        for item in mtl.read_text(errors='replace').splitlines():
                            tokens=shlex.split(item)
                            if tokens and tokens[0].lower() in {'map_kd','map_ka','map_bump','bump','map_d'}:
                                add(tokens[-1],True)
    elif p.suffix.lower()=='.gltf':
        obj=json.loads(p.read_text())
        for kind in ['buffers','images']:
            for item in obj.get(kind,[]):
                uri=item.get('uri','')
                if uri and not uri.startswith('data:'):add(uri,kind=='images')
    return {'schema':'local_mesh_resources_v1','files':list(found.values()),
            'declared_texture_count':len(set(textures)),'missing':missing,
            'source_adapter':'gso_material_directory_v1' if row['source']=='google_scanned_objects_original_fuel' else 'relative_resources_v1'}


def load(row,manifest=None):
    manifest=manifest or resources(row)
    if manifest['missing']:raise ValueError('Missing mesh resources: '+repr(manifest['missing']))
    for b in manifest['files']:resolve(b)
    class LocalResolver(FilePathResolver):
        def get(self,name):return _find(row,name).read_bytes()
    scene=trimesh.load(row['primary_path'],force='scene',process=False,resolver=LocalResolver(row['primary_path']))
    textured=sum(any(getattr(getattr(g.visual,'material',None),name,None) is not None
                     for name in ['image','baseColorTexture']) for g in scene.geometry.values())
    if manifest['declared_texture_count'] and not textured:
        raise ValueError('Declared texture was not loaded')
    return scene,{**manifest,'textured_geometry_count':textured,'semantic_identity_confirmed':False}
