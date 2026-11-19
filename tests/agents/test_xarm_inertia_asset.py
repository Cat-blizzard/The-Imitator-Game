"""CPU-only asset transformation, cache isolation and concurrent publication."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from mani_skill.agents.robots.xarm6 import inertia_asset


def _source_asset(tmp_path: Path) -> Path:
    directory = tmp_path / "robots" / "xarm6"
    directory.mkdir(parents=True)
    products = [("1.095E-05", "1.635E-06", "0.000793"),
                ("-0.00430651", "-0.00067797", "0.00457245"),
                ("0.0014185", "-0.00092094", "-0.00169178"),
                ("5.028E-05", "1.374E-05", "0.00045338"),
                ("0.000492428", "-0.00039147", "-1.235E-04"),
                ("0", "0.0", "-0.0")]
    links = []
    for index, (xy, xz, yz) in enumerate(products, 1):
        links.append(f'''<link name="link{index}">
  <!-- link {index} retained -->
  <visual><geometry><mesh filename="visual/link.stl"/></geometry></visual>
  <inertial><origin xyz="0.01 0.02 0.03" rpy="0 0 0"/><mass value="{index}.25"/>
  <inertia ixx="0.02" ixy="{xy}" ixz="{xz}" iyy="0.03" iyz="{yz}" izz="0.04"/>
  </inertial></link>''')
    source = directory / "xarm6_robotiq.urdf"
    source.write_text('''<?xml version="1.0" encoding="utf-8"?>
<!-- document comment retained -->
<robot name="fixture">
<link name="link_base"><inertial><mass value="2.7"/>
<inertia ixx="0.04" ixy="-3.5E-06" ixz="1.25E-05" iyy="0.04" iyz="1.67E-06" izz="0.02"/>
</inertial></link>
''' + "\n".join(links) + '''
<link name="finger"><inertial><mass value="0.05"/>
<inertia ixx="0.001" ixy="0.0001" ixz="0.0002" iyy="0.002" iyz="0.0003" izz="0.003"/>
</inertial></link>
<joint name="finger_joint" type="revolute"><parent link="link6"/><child link="finger"/>
<axis xyz="0 1 0"/><limit lower="0" upper="0.81" effort="0.1" velocity="1"/></joint>
</robot>
<!-- trailing comment retained -->
''', encoding="utf-8")
    source.with_suffix(".srdf").write_bytes(b'<robot><disable_collisions link1="link1" link2="link2" reason="Adjacent"/></robot>')
    for relative, data in {"visual/link.stl": b"unchanged mesh",
                           "collision/link.obj.convex.stl": b"convex companion",
                           "meshes_robotiq/visual/finger.dae": b"relative texture reference",
                           "meshes_robotiq/visual/texture.png": b"texture",
                           "LICENSE": b"source license"}.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return source


def _identify_fixture(monkeypatch: pytest.MonkeyPatch, source: Path) -> str:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(inertia_asset, "KNOWN_SOURCE_SHA256", digest)
    return digest


def _files(directory: Path) -> dict[str, bytes]:
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def test_only_arm_products_change_and_all_resources_remain(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source_asset(tmp_path)
    digest = _identify_fixture(monkeypatch, source)
    original = _files(source.parent)
    prepared = inertia_asset.prepare_xarm6_robotiq_inertia(source)
    assert prepared.parent != source.parent
    assert prepared.parent.parent == source.parent.parent
    assert inertia_asset.INERTIA_VERSION in prepared.parent.name
    assert digest[:12] in prepared.parent.name
    assert _files(source.parent) == original
    prepared_files = _files(prepared.parent)
    for relative, data in original.items():
        if relative != source.name:
            assert prepared_files[relative] == data
    parser = lambda: ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    before = ET.fromstring(source.read_bytes(), parser=parser())
    after = ET.fromstring(prepared.read_bytes(), parser=parser())
    differences = []
    for index in range(1, 7):
        original_inertia = before.find(f"link[@name='link{index}']/inertial/inertia")
        fixed_inertia = after.find(f"link[@name='link{index}']/inertial/inertia")
        for name in ("ixy", "ixz", "iyz"):
            if float(original_inertia.get(name)) == 0.0:
                assert fixed_inertia.get(name) == original_inertia.get(name)
            else:
                assert float(fixed_inertia.get(name)) == -float(original_inertia.get(name))
                differences.append((index, name))
                fixed_inertia.set(name, original_inertia.get(name))
    assert len(differences) == 15
    # Restoring exactly those fields restores the entire XML element tree.
    assert ET.tostring(before) == ET.tostring(after)
    assert b"<!-- document comment retained -->" in prepared.read_bytes()
    assert b"<!-- trailing comment retained -->" in prepared.read_bytes()
    manifest = json.loads((prepared.parent / "source.json").read_text())
    assert manifest["source_sha256"] == digest
    assert manifest["transformation_version"] == inertia_asset.INERTIA_VERSION
    assert manifest["source_url"] == inertia_asset.SOURCE_URL


def test_completed_cache_is_reused_without_copying_or_double_negation(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source_asset(tmp_path)
    _identify_fixture(monkeypatch, source)
    prepared = inertia_asset.prepare_xarm6_robotiq_inertia(source)
    original_result = _files(prepared.parent)
    monkeypatch.setattr(inertia_asset.shutil, "copytree", lambda *args, **kwargs:
                        pytest.fail("Completed cache must not be copied again"))
    assert inertia_asset.prepare_xarm6_robotiq_inertia(source) == prepared
    assert inertia_asset.prepare_xarm6_robotiq_inertia(prepared) == prepared
    assert _files(prepared.parent) == original_result


def test_unknown_source_returns_unchanged_without_parsing_or_preparation(tmp_path: Path) -> None:
    source = tmp_path / "custom.urdf"
    source.write_bytes(b"Unknown version: not even parsed as XML")
    assert inertia_asset.prepare_xarm6_robotiq_inertia(source) == source
    assert list(tmp_path.iterdir()) == [source]


def test_missing_srdf_does_not_publish_or_modify_source(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source_asset(tmp_path)
    _identify_fixture(monkeypatch, source)
    source.with_suffix(".srdf").unlink()
    original = _files(source.parent)
    with pytest.raises(FileNotFoundError):
        inertia_asset.prepare_xarm6_robotiq_inertia(source)
    assert _files(source.parent) == original
    assert list(source.parent.parent.iterdir()) == [source.parent]


def test_failed_copy_cleans_only_its_staging_directory(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source_asset(tmp_path)
    _identify_fixture(monkeypatch, source)
    unrelated = source.parent.parent / "unrelated"
    unrelated.mkdir()
    marker = unrelated / "keep"
    marker.write_text("preserved")
    original = _files(source.parent)

    def interrupted_copy(src: Path, dst: Path, **kwargs: object) -> None:
        (dst / "partial").write_text("partial")
        raise OSError("fixture interrupted copy")

    monkeypatch.setattr(inertia_asset.shutil, "copytree", interrupted_copy)
    with pytest.raises(OSError, match="interrupted copy"):
        inertia_asset.prepare_xarm6_robotiq_inertia(source)
    assert _files(source.parent) == original
    assert set(source.parent.parent.iterdir()) == {source.parent, unrelated}
    assert marker.read_text() == "preserved"


def test_existing_incomplete_cache_is_not_overwritten(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source_asset(tmp_path)
    digest = _identify_fixture(monkeypatch, source)
    directory = source.parent.parent / f"xarm6-inertia-{inertia_asset.INERTIA_VERSION}-{digest[:12]}"
    directory.mkdir()
    (directory / "keep").write_text("incomplete but preserved")
    with pytest.raises(FileExistsError, match="left unchanged"):
        inertia_asset.prepare_xarm6_robotiq_inertia(source)
    assert (directory / "keep").read_text() == "incomplete but preserved"


def test_concurrent_first_preparation_publishes_one_complete_asset(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source_asset(tmp_path)
    digest = _identify_fixture(monkeypatch, source)
    original = _files(source.parent)
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    # Direct module loading keeps child processes CPU-only and avoids importing
    # the simulation package. All three copies finish before any is published.
    code = r'''
import importlib.util, pathlib, sys, time
module_path, source_path, digest, barrier_path, index = sys.argv[1:]
spec = importlib.util.spec_from_file_location("inertia_asset_cpu", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.KNOWN_SOURCE_SHA256 = digest
source = pathlib.Path(source_path)
barrier = pathlib.Path(barrier_path)
copytree = module.shutil.copytree
def synchronized_copy(src, dst, *args, **kwargs):
    result = copytree(src, dst, *args, **kwargs)
    if pathlib.Path(src) == source.parent:
        (barrier / index).write_text("copied")
        deadline = time.monotonic() + 15
        while len(list(barrier.iterdir())) != 3:
            if time.monotonic() >= deadline:
                raise TimeoutError("three-process preparation barrier")
            time.sleep(.01)
    return result
module.shutil.copytree = synchronized_copy
print(module.prepare_xarm6_robotiq_inertia(source))
'''
    processes = [subprocess.Popen([sys.executable, "-c", code, inertia_asset.__file__,
                                  str(source), digest, str(barrier), str(index)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for index in range(3)]
    try:
        results = [process.communicate(timeout=25) for process in processes]
        for process, (_, error) in zip(processes, results):
            assert process.returncode == 0, error
        paths = [Path(output.strip()) for output, _ in results]
        assert len(set(paths)) == 1
        assert paths[0].is_file() and paths[0].with_suffix(".srdf").is_file()
        assert _files(source.parent) == original
        assert set(source.parent.parent.iterdir()) == {source.parent, paths[0].parent}
        assert inertia_asset.prepare_xarm6_robotiq_inertia(source) == paths[0]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
