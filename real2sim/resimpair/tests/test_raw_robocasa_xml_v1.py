from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest

from resimpair_grasp.common import binding
from resimpair_assets import raw_robocasa_xml_v1 as loader
from resimpair_assets.raw_mesh_io_v1 import export_visual


def fixture(tmp_path, texture=True):
    # A tiny explicit test mesh, not an added production asset.
    (tmp_path / "visual.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
        "vt 0 0\nvt 1 0\nvt 0 1\nvt 1 1\n"
        "f 1/1 3/3 2/2\nf 1/1 2/2 4/4\nf 1/1 4/4 3/3\nf 2/2 3/3 4/4\n")
    pixels = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], np.uint8)
    Image.fromarray(pixels).save(tmp_path / "color.png")
    appearance = ('<texture name="tex" type="2d" file="color.png"/>'
                  '<material name="mat" texture="tex"/>') if texture else '<material name="mat" rgba="0.2 0.4 0.6 1"/>'
    text = ('<mujoco model="test"><asset>'
            '<mesh name="v" file="visual.obj" scale="0.1 0.2 0.3"/>' + appearance + '</asset>'
            '<default><default class="region"><geom group="1" contype="0" conaffinity="0" rgba="0 1 0 0"/></default></default>'
            '<worldbody><body pos="1 2 3"><body name="object" pos="0.1 0.2 0.3">'
            '<geom type="mesh" mesh="v" group="1" contype="0" conaffinity="0" material="mat"/>'
            '<geom class="region" type="box" size="10 10 10"/>'
            '</body></body></worldbody></mujoco>')
    (tmp_path / "model.xml").write_text(text)
    return row(tmp_path / "model.xml"), pixels


def row(path):
    return {"id": "robocasa:test", "source": "robocasa", "path": str(path),
            "source_sha": binding(path)["sha256"], "partition": "development"}


def change(raw, modify):
    path = Path(raw["path"]); root = ET.parse(path).getroot(); modify(root)
    ET.ElementTree(root).write(path)
    return row(path)


def test_mesh_reference_transform_matches_documented_inverse_rotation_then_scale():
    transform = loader.mesh_reference_transform({"refpos": ".1 .2 .3",
        "refquat": "0.7071067811865476 0 0 0.7071067811865476", "scale": "2 3 4"})
    actual = transform @ np.array([1, 2, 3, 1.])
    assert np.allclose(actual, [3.6, -2.7, 10.8, 1.])


def test_body_transforms_sizes_uv_texture_and_helpers_are_preserved(tmp_path):
    raw, pixels = fixture(tmp_path)
    scene, mesh, receipt = loader.load_raw_scene(raw)
    assert np.allclose(mesh.bounds, [[1.1, 2.2, 3.3], [1.2, 2.4, 3.6]])
    assert len(mesh.faces) == 4
    geometry = next(iter(scene.geometry.values()))
    assert np.array_equal(np.asarray(geometry.visual.material.baseColorTexture), pixels)
    assert len(geometry.visual.uv) == len(geometry.vertices)
    assert len(receipt["nonphysical_helpers"]) == 1
    assert receipt["geometry_normalized_or_rescaled_by_policy"] is False
    check = export_visual(scene, tmp_path / "export.glb", mesh)
    assert check["bounds_preserved"] and check["texture_pixels_preserved"] and check["UV_presence_preserved"]


def test_geom_quaternion_composes_with_body_and_reference_quaternion(tmp_path):
    raw, _ = fixture(tmp_path)
    def edit(root):
        root.find("asset/mesh").set("refquat", "0.7071067811865476 0 0 0.7071067811865476")
        root.find("worldbody/body/body/geom").set("quat", "0.7071067811865476 0 0 0.7071067811865476")
    raw = change(raw, edit)
    _, mesh, _ = loader.load_raw_scene(raw)
    # Nonuniform scale occurs BETWEEN inverse reference and forward geom
    # rotations, so the x/y scale axes swap rather than cancelling directly.
    assert np.allclose(mesh.bounds, [[1.1, 2.2, 3.3], [1.3, 2.3, 3.6]])


def test_rgba_without_texture_uses_original_material(tmp_path):
    raw, _ = fixture(tmp_path, texture=False)
    scene, _, receipt = loader.load_raw_scene(raw)
    geometry = next(iter(scene.geometry.values()))
    assert np.array_equal(geometry.visual.material.baseColorFactor, [51, 102, 153, 255])
    assert not receipt["unused_declarations"]["material"]


def test_unused_stale_texture_is_recorded_but_never_substituted(tmp_path):
    raw, _ = fixture(tmp_path)
    raw = change(raw, lambda root: ET.SubElement(root.find("asset"), "texture",
        {"name": "unused", "type": "2d", "file": "/author/workstation/missing.png"}))
    _, _, receipt = loader.load_raw_scene(raw)
    assert receipt["unused_declarations"]["texture"][0]["name"] == "unused"
    assert all("workstation" not in ref["path"] for ref in receipt["dependencies"]["files"])


@pytest.mark.parametrize("filename", ["missing.png", "/author/workstation/color.png", "../elsewhere.png"])
def test_missing_or_unknown_referenced_texture_rejects_without_gray_fallback(tmp_path, filename):
    raw, _ = fixture(tmp_path)
    raw = change(raw, lambda root: root.find("asset/texture").set("file", filename))
    with pytest.raises(ValueError, match="dependency"):
        loader.load_raw_scene(raw)


@pytest.mark.parametrize("tag", ["joint", "freejoint", "include", "skin"])
def test_articulation_or_external_components_require_another_adapter(tmp_path, tag):
    raw, _ = fixture(tmp_path)
    raw = change(raw, lambda root: ET.SubElement(root.find("worldbody/body/body"), tag))
    with pytest.raises(ValueError, match="separate_adapter"):
        loader.load_raw_scene(raw)


def test_unknown_transform_and_visible_primitive_do_not_get_ignored(tmp_path):
    raw, _ = fixture(tmp_path)
    raw = change(raw, lambda root: root.find("worldbody/body").set("euler", "0 0 90"))
    with pytest.raises(ValueError, match="unsupported_MJCF"):
        loader.load_raw_scene(raw)
    raw = change(raw, lambda root: root.find("worldbody/body").attrib.pop("euler"))
    raw = change(raw, lambda root: ET.SubElement(root.find("worldbody/body/body"), "geom", {"type": "box", "size": "1 1 1"}))
    with pytest.raises(ValueError, match="primitive_requires_separate_adapter"):
        loader.load_raw_scene(raw)


def test_changed_source_and_zero_or_negative_scale_fail(tmp_path):
    raw, _ = fixture(tmp_path)
    Path(raw["path"]).write_text(Path(raw["path"]).read_text() + "\n")
    with pytest.raises(ValueError, match="source_bytes_changed"):
        loader.load_raw_scene(raw)
    for scale in ["1 0 1", "1 -1 1"]:
        raw = change(raw, lambda root: root.find("asset/mesh").set("scale", scale))
        with pytest.raises(ValueError, match="nonpositive_mesh_scale"):
            loader.load_raw_scene(raw)
