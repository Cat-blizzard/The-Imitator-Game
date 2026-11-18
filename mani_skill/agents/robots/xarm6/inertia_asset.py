"""Prepare an isolated copy of the identified legacy xArm6 inertia asset.

The vendor corrected CAD products of inertia to URDF tensor entries by
negating the three off-diagonal terms. Apply that convention only to the
byte-identified ManiSkill-XArm6 v0.1.1 Robotiq description. Custom and future
descriptions retain their original loading behavior.
"""
from __future__ import annotations

import errno
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from xml.dom import minidom


KNOWN_SOURCE_SHA256 = "1a1147c0a49a5c0764f7345836e3e4fb0cbe26a43e9cd48e27966fcdc46ff242"
INERTIA_VERSION = "urdf-products-v1"
SOURCE_URL = "https://github.com/haosulab/ManiSkill-XArm6/blob/v0.1.1/xarm6_robotiq.urdf"
VENDOR_FIX_URL = "https://github.com/xArm-Developer/xarm_ros/commit/04dc79616ef55433bd589daee01c2d0db713f2f0"
_MANIFEST = "source.json"


def _correct_inertia(source: bytes) -> bytes:
    """Keep the XML document, including comments, and negate arm products only."""
    with minidom.parseString(source) as document:
        links = {node.getAttribute("name"): node
                 for node in document.getElementsByTagName("link")}
        for index in range(1, 7):
            inertia = links[f"link{index}"].getElementsByTagName("inertia")[0]
            for name in ("ixy", "ixz", "iyz"):
                value = inertia.getAttribute(name)
                if float(value) != 0.0:
                    inertia.setAttribute(name, value[1:] if value.startswith("-")
                                         else "-" + value.removeprefix("+"))
        return document.toxml(encoding="utf-8")


def _cache_ready(directory: Path, source_name: str, source_sha: str) -> bool:
    """Inspect only the small completion manifest and required descriptions."""
    try:
        manifest = json.loads((directory / _MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(manifest, dict)
            and manifest.get("source_sha256") == source_sha
            and manifest.get("transformation_version") == INERTIA_VERSION
            and manifest.get("urdf") == source_name
            and (directory / source_name).is_file()
            and (directory / Path(source_name).with_suffix(".srdf")).is_file())


def prepare_xarm6_robotiq_inertia(source_urdf: Path) -> Path:
    """Return a corrected, atomically prepared URDF or the unknown source itself.

    The original tree is read-only. A versioned sibling directory contains a
    complete copy, including same-stem SRDF, mesh companions and textures.
    Concurrent callers publish their completed staging directory by rename;
    a caller losing that race reuses the winner's completed asset.
    """
    source_urdf = Path(source_urdf)
    source = source_urdf.read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    if source_sha != KNOWN_SOURCE_SHA256:
        return source_urdf

    directory = source_urdf.parent.parent / f"xarm6-inertia-{INERTIA_VERSION}-{source_sha[:12]}"
    if _cache_ready(directory, source_urdf.name, source_sha):
        return directory / source_urdf.name
    if directory.exists():
        # Another process may have published after the first cache check.
        if _cache_ready(directory, source_urdf.name, source_sha):
            return directory / source_urdf.name
        raise FileExistsError(f"Incomplete xArm inertia cache is left unchanged: {directory}")
    if not source_urdf.with_suffix(".srdf").is_file():
        raise FileNotFoundError(source_urdf.with_suffix(".srdf"))

    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        shutil.copytree(source_urdf.parent, staging, dirs_exist_ok=True)
        (staging / source_urdf.name).write_bytes(_correct_inertia(source))
        (staging / _MANIFEST).write_text(json.dumps({
            "source_url": SOURCE_URL,
            "vendor_fix_url": VENDOR_FIX_URL,
            "source_sha256": source_sha,
            "transformation_version": INERTIA_VERSION,
            "urdf": source_urdf.name,
            "transformation": "negate link1..6 ixy, ixz, iyz; zero strings preserved",
        }, indent=2) + "\n", encoding="utf-8")
        try:
            staging.rename(directory)
        except OSError as error:
            if (error.errno not in (errno.EEXIST, errno.ENOTEMPTY)
                    or not _cache_ready(directory, source_urdf.name, source_sha)):
                raise
        return directory / source_urdf.name
    finally:
        # Only this invocation's unpublished staging directory may be removed.
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    synthetic = ("<robot><!-- retained -->" + "".join(
        f'<link name="link{i}"><inertial><inertia ixy="1E-5" ixz="-2E-5" iyz="0"/>'
        "</inertial></link>" for i in range(1, 7)) + "</robot>").encode()
    result = _correct_inertia(synthetic)
    with minidom.parseString(result) as document:
        entries = document.getElementsByTagName("inertia")
        assert len(entries) == 6
        assert all(entry.getAttribute("ixy") == "-1E-5"
                   and entry.getAttribute("ixz") == "2E-5"
                   and entry.getAttribute("iyz") == "0" for entry in entries)
    assert b"<!-- retained -->" in result
    print("xArm inertia document: 6 links, 3 products each; zero values and comments retained")
