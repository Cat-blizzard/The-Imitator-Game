"""Offline loading of explicit rigid RoboCasa MJCF object meshes.

This deliberately narrow adapter returns the existing (scene, mesh, receipt)
interface. No simulator, renderer, remote asset lookup, geometric substitute or
size normalization is used. The shared raw loader and size policy are untouched.

Mesh transform semantics: MuJoCo XMLreference asset/mesh and mjCMesh::
ApplyTransformations in google-deepmind/mujoco/src/user/user_mesh.cc:
subtract refpos, inverse-rotate by refquat, apply per-axis scale, then apply
the referencing geom and parent-body transforms.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh

from resimpair_grasp.common import binding, resolve
from .metric_visual_precision_v3 import bake_visual_world
from .raw_metric_containers_v1 import texture_summary

VERSION = "raw_robocasa_rigid_xml_v1"
GEOM_FIELDS = {"name", "class", "type", "mesh", "material", "group", "contype", "conaffinity",
               "rgba", "size", "pos", "quat", "density", "mass", "friction", "solimp", "solref",
               "condim", "margin", "gap"}
TAG_FIELDS = {
    "mujoco": {"model"}, "asset": set(), "worldbody": set(),
    "body": {"name", "pos", "quat", "childclass"}, "default": {"class"}, "geom": GEOM_FIELDS,
    "mesh": {"name", "file", "scale", "refpos", "refquat"},
    "texture": {"name", "file", "type"},
    "material": {"name", "texture", "specular", "shininess", "reflectance", "rgba", "texrepeat", "texuniform"},
    "compiler": {"meshdir", "texturedir", "assetdir", "angle", "coordinate", "strippath"},
    "site": {"name", "rgba", "size", "pos", "quat", "type"},
}


def vector(text, count, label):
    try:
        value = np.array([float(x) for x in text.split()], dtype=float)
    except (ValueError, AttributeError) as error:
        raise ValueError("invalid_MJCF_numeric_" + label) from error
    if value.shape != (count,) or not np.isfinite(value).all():
        raise ValueError("invalid_MJCF_numeric_" + label)
    return value


def quaternion(text):
    value = vector(text, 4, "quaternion_wxyz")
    if np.linalg.norm(value) < 1e-12:
        raise ValueError("zero_MJCF_quaternion")
    return Rotation.from_quat(value[[1, 2, 3, 0]]).as_matrix()


def pose(attributes):
    transform = np.eye(4)
    transform[:3, :3] = quaternion(attributes.get("quat", "1 0 0 0"))
    transform[:3, 3] = vector(attributes.get("pos", "0 0 0"), 3, "position")
    return transform


def mesh_reference_transform(attributes):
    scale = vector(attributes.get("scale", "1 1 1"), 3, "mesh_scale")
    if np.any(scale <= 0):
        raise ValueError("nonpositive_mesh_scale_requires_separate_adapter")
    rotation = quaternion(attributes.get("refquat", "1 0 0 0")).T
    reference = vector(attributes.get("refpos", "0 0 0"), 3, "mesh_refpos")
    transform = np.eye(4)
    transform[:3, :3] = np.diag(scale) @ rotation
    transform[:3, 3] = -transform[:3, :3] @ reference
    return transform


def _read_xml(row):
    path = Path(row["path"]).resolve()
    if row.get("source") != "robocasa" or path.suffix.lower() != ".xml":
        raise ValueError("explicit_RoboCasa_XML_source_required")
    if not path.is_file() or path.stat().st_size > 4 * 2**20:
        raise ValueError("XML_missing_or_exceeds_4MiB")
    if binding(path)["sha256"] != row["source_sha"]:
        raise ValueError("raw_source_bytes_changed")
    data = path.read_bytes()
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("XML_DTD_or_entity_not_supported")
    root = ET.fromstring(data)
    if root.tag != "mujoco":
        raise ValueError("MJCF_root_required")
    if any(e.tag in {"joint", "freejoint", "actuator", "tendon", "equality", "flex", "skin", "include"}
           for e in root.iter()):
        raise ValueError("articulated_or_external_component_requires_separate_adapter")
    if len(list(root.iter())) > 20000:
        raise ValueError("bounded_XML_element_count_required")
    for element in root.iter():
        if element.tag not in TAG_FIELDS or set(element.attrib) - TAG_FIELDS[element.tag]:
            raise ValueError("unsupported_MJCF_element_or_attribute:" + element.tag)
    if (len(root.findall("asset")) != 1 or len(root.findall("worldbody")) != 1
            or len(root.findall("default")) > 1 or len(root.findall("compiler")) > 1):
        raise ValueError("single_asset_worldbody_default_compiler_required")
    if (any(e.tag not in {"asset", "worldbody", "default", "compiler"} for e in root)
            or any(e.tag not in {"mesh", "material", "texture"} for e in root.find("asset"))):
        raise ValueError("unknown_MJCF_asset_or_root_component")
    world = root.find("worldbody")
    if len(world.findall("body")) != 1 or any(e.tag != "body" for e in world):
        raise ValueError("one_connected_rigid_root_body_required")
    return path, root


def _defaults(root):
    classes = {}
    def visit(element, inherited, main=False):
        name = element.get("class", "main" if main else None)
        if not name or name in classes:
            raise ValueError("unique_named_MJCF_default_class_required")
        current = deepcopy(inherited)
        for child in element:
            if child.tag == "geom":
                current.update(child.attrib)
            elif child.tag != "default":
                raise ValueError("non_geom_default_requires_separate_adapter")
        classes[name] = current
        for child in element.findall("default"):
            visit(child, current)
    default = root.find("default")
    if default is not None:
        visit(default, {}, main=True)
    else:
        classes["main"] = {}
    return classes


def _declarations(root, kind):
    result = {}
    for element in root.find("asset"):
        if element.tag != kind:
            continue
        name = element.get("name")
        if not name or name in result:
            raise ValueError("unique_named_asset_declaration_required:" + kind)
        result[name] = dict(element.attrib)
    return result


def _dependency(path, compiler, kind, name):
    if not isinstance(name, str) or not name or "://" in name:
        raise ValueError("local_declared_MJCF_dependency_required")
    relative = Path(name)
    if relative.is_absolute():
        # Never guess that an unrelated local basename replaces a stale author
        # workstation path. Unused declarations are recorded, not resolved.
        raise ValueError("referenced_absolute_dependency_requires_explicit_relocation")
    subdir = compiler.get("meshdir" if kind == "mesh" else "texturedir", compiler.get("assetdir", ""))
    target = (path.parent / subdir / relative).resolve()
    if not target.is_relative_to(path.parent) or not target.is_file():
        raise ValueError("missing_or_outside_object_package_dependency:" + name)
    if target.stat().st_size > 256 * 2**20:
        raise ValueError("raw_dependency_exceeds_256MiB")
    return target


def inspect_raw_xml(row):
    """Validate declarations and bind referenced files, without loading meshes."""
    path, root = _read_xml(row)
    compiler = dict(root.find("compiler").attrib) if root.find("compiler") is not None else {}
    if compiler.get("coordinate", "local") != "local" or compiler.get("strippath", "false") != "false":
        raise ValueError("global_coordinates_or_stripped_paths_require_separate_adapter")
    classes = _defaults(root)
    meshes, materials, textures = (_declarations(root, kind) for kind in ("mesh", "material", "texture"))
    geoms, helpers, used_meshes, used_materials, used_textures = [], [], set(), set(), set()
    dependencies = {}
    def visit(body, parent_transform, inherited_class="main"):
        current = parent_transform @ pose(body.attrib)
        active_class = body.get("childclass", inherited_class)
        if active_class not in classes:
            raise ValueError("unknown_MJCF_childclass")
        for child in body:
            if child.tag == "body":
                visit(child, current, active_class); continue
            if child.tag == "site":
                helpers.append({"kind": "site", "attributes": dict(child.attrib), "body_transform": current.tolist()}); continue
            if child.tag != "geom":
                raise ValueError("unknown_rigid_body_component")
            name = child.get("class", active_class)
            if name not in classes:
                raise ValueError("unknown_MJCF_geom_class")
            attrs = {**classes[name], **child.attrib}
            rgba = vector(attrs.get("rgba", "0.5 0.5 0.5 1"), 4, "geom_rgba")
            if np.any(rgba < 0) or np.any(rgba > 1):
                raise ValueError("geom_rgba_outside_unit_interval")
            group = int(attrs.get("group", 0))
            noncolliding = int(attrs.get("contype", 1)) == 0 and int(attrs.get("conaffinity", 1)) == 0
            if noncolliding and rgba[3] == 0:
                if attrs.get("type", "sphere") not in {"box", "cylinder", "sphere", "mesh"}:
                    raise ValueError("unknown_invisible_region_geometry")
                helpers.append({"kind": "transparent_noncollision_region", "attributes": attrs,
                                "transform": (current @ pose(attrs)).tolist()}); continue
            if attrs.get("type") != "mesh" or "mesh" not in attrs:
                raise ValueError("visible_or_colliding_primitive_requires_separate_adapter")
            if group == 1 and noncolliding:
                kind = "visual"
            elif group == 0 and not noncolliding:
                kind = "collision"
            else:
                raise ValueError("unrecognized_RoboCasa_visual_collision_group_semantics")
            mesh_name = attrs["mesh"]
            if mesh_name not in meshes:
                raise ValueError("unknown_referenced_mesh")
            mesh = meshes[mesh_name]
            asset_path = _dependency(path, compiler, "mesh", mesh.get("file"))
            if asset_path.suffix.lower() not in {".obj", ".stl", ".ply"}:
                raise ValueError("declared_mesh_format_requires_separate_adapter")
            dependencies[str(asset_path)] = binding(asset_path); used_meshes.add(mesh_name)
            item = {"kind": kind, "geom_attributes": attrs, "mesh_declaration": mesh,
                    "mesh_file": dependencies[str(asset_path)],
                    "mesh_to_world": (current @ pose(attrs) @ mesh_reference_transform(mesh)).tolist()}
            material_name = attrs.get("material")
            if kind == "visual" and material_name:
                if material_name not in materials:
                    raise ValueError("unknown_referenced_material")
                material = materials[material_name]; used_materials.add(material_name)
                if material.get("texuniform", "false") != "false":
                    raise ValueError("spatial_texture_mapping_requires_separate_adapter")
                repeat = vector(material.get("texrepeat", "1 1"), 2, "texrepeat")
                if np.any(repeat <= 0):
                    raise ValueError("positive_texture_repeat_required")
                item["material_declaration"] = material
                texture_name = material.get("texture")
                if texture_name:
                    if texture_name not in textures or textures[texture_name].get("type") != "2d":
                        raise ValueError("declared_2d_texture_required")
                    texture = textures[texture_name]
                    texture_path = _dependency(path, compiler, "texture", texture.get("file"))
                    dependencies[str(texture_path)] = binding(texture_path); used_textures.add(texture_name)
                    item["texture_declaration"] = texture; item["texture_file"] = dependencies[str(texture_path)]
            geoms.append(item)
    visit(root.find("worldbody/body"), np.eye(4))
    if not any(g["kind"] == "visual" for g in geoms):
        raise ValueError("no_explicit_RoboCasa_visual_mesh")
    return {"schema": VERSION, "source": binding(path), "source_id": row["id"],
        "compiler": compiler, "geoms": geoms, "nonphysical_helpers": helpers,
        "dependencies": {"source": binding(path), "files": list(dependencies.values()), "raw_files_modified": False},
        "unused_declarations": {"mesh": [v for k, v in meshes.items() if k not in used_meshes],
            "material": [v for k, v in materials.items() if k not in used_materials],
            "texture": [v for k, v in textures.items() if k not in used_textures]},
        "rigid_definition": "one connected body tree with no joint/freejoint/constraint/skin/include",
        "authored_dimensions_basis": "preserved XML mesh scale and reference/geom/body transforms; no real-world metric measurement asserted",
        "geometry_normalized_or_rescaled_by_policy": False, "robot_physics_started": False,
        "collision_geometry_status": "original referenced mesh definitions and transforms bound; not independently physically qualified",
        "implementation": binding(__file__)}


def load_raw_scene(row):
    receipt = inspect_raw_xml(row)
    scene = trimesh.Scene()
    count = 0
    for index, item in enumerate(receipt["geoms"]):
        if item["kind"] != "visual":
            continue
        mesh_path = resolve(item["mesh_file"])
        # MJCF material overrides OBJ/MTL appearances. OBJ positions, faces,
        # per-corner UVs and normals are retained; no guessed MTL is substituted.
        raw = trimesh.load(mesh_path, force="scene", process=False, skip_materials=True, resolver={})
        raw = bake_visual_world(raw)
        material = item.get("material_declaration", {})
        color = vector(item["geom_attributes"].get("rgba", material.get("rgba", "1 1 1 1")), 4, "visual_rgba")
        if np.any(color < 0) or np.any(color > 1):
            raise ValueError("material_rgba_outside_unit_interval")
        texture = None
        if item.get("texture_file"):
            with Image.open(resolve(item["texture_file"])) as image:
                if image.width * image.height > 64 * 2**20:
                    raise ValueError("bounded_texture_pixel_count_required")
                texture = image.copy()
        for part, geometry in raw.geometry.items():
            if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
                raise ValueError("visual_component_without_triangle_geometry")
            geometry = geometry.copy()
            uv = getattr(geometry.visual, "uv", None)
            if texture is not None and (uv is None or len(uv) != len(geometry.vertices)):
                raise ValueError("referenced_texture_without_complete_mesh_UV")
            if uv is not None:
                uv = np.asarray(uv, float) * vector(material.get("texrepeat", "1 1"), 2, "texrepeat")
                if not np.isfinite(uv).all():
                    raise ValueError("nonfinite_mesh_UV")
            appearance = trimesh.visual.material.PBRMaterial(name=material.get("name", "MJCF_RGBA"),
                baseColorTexture=texture, baseColorFactor=np.round(color * 255).astype(np.uint8))
            geometry.visual = trimesh.visual.TextureVisuals(uv=uv, material=appearance)
            geometry.apply_transform(np.asarray(item["mesh_to_world"], float))
            scene.add_geometry(geometry, node_name=f"visual_{index}_{part}", geom_name=f"visual_{index}_{part}")
            count += len(geometry.faces)
            if count > 300000:
                raise ValueError("raw_mesh_face_budget_300000")
    mesh = scene.to_geometry()
    if not np.isfinite(mesh.vertices).all() or np.min(mesh.extents) <= 1e-10:
        raise ValueError("raw_mesh_nonfinite_or_degenerate")
    receipt.update(raw_bounds_units=mesh.bounds.tolist(), raw_materials=texture_summary(scene), faces=len(mesh.faces),
        source_scene_node_transforms_baked=True, geometry_rescaled_in_loader=False,
        source_declared_mesh_scales_applied=True, rendered_material_scope="base color, texture pixels, UV and declared repetition; MJCF specular/shininess/reflectance preserved in receipt only",
        render_equivalence="not_evaluated", geometry_declaration_counts_by_kind=dict(Counter(g["kind"] for g in receipt["geoms"])),
        visual_mesh_load_calls=sum(g["kind"] == "visual" for g in receipt["geoms"]),
        collision_mesh_load_calls=0)
    return scene, mesh, receipt
