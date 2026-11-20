"""Correct the identified legacy xArm CAD products in derived URDFs only.

The source is ManiSkill-XArm6 v0.1.1. The manufacturer's correction negates
ixy/ixz/iyz of links 1..6; mass, COM and diagonal inertia remain unchanged.
Custom source bytes are left alone, and already corrected tensors are stable.
"""
from __future__ import annotations

from hashlib import sha256
import xml.etree.ElementTree as ET

KNOWN_SOURCE_SHA256 = "1a1147c0a49a5c0764f7345836e3e4fb0cbe26a43e9cd48e27966fcdc46ff242"
VERSION = "urdf-products-v1"
VENDOR_FIX = "https://github.com/xArm-Developer/xarm_ros/commit/04dc79616ef55433bd589daee01c2d0db713f2f0"


def correct_known_xarm_products(source: bytes, derived: ET.Element) -> bool:
    """Apply the known source's tensor convention to a derived robot tree.

    Input/output inertia values use kg*m^2. Return whether this identified
    source is covered; a mismatch inside an identified tensor is an error.
    """
    if sha256(source).hexdigest() != KNOWN_SOURCE_SHA256:
        return False
    original = ET.fromstring(source)
    updates: list[tuple[ET.Element, str, str]] = []
    for i in range(1, 7):
        query = f"link[@name='link{i}']/inertial/inertia"
        before, after = original.find(query), derived.find(query)
        if before is None or after is None:
            raise ValueError(f"Identified xArm model is missing link{i} inertia")
        for key in ("ixy", "ixz", "iyz"):
            value = before.attrib[key]
            expected = float(value)
            current = float(after.attrib[key])
            if current not in (expected, -expected):
                raise ValueError(f"Derived link{i} {key} differs from identified source")
            if expected != 0:
                fixed = value[1:] if value.startswith("-") else "-" + value.removeprefix("+")
                updates.append((after, key, fixed))
    for element, key, fixed in updates:
        element.set(key, fixed)
    return True


def self_test() -> None:
    """Unknown/custom descriptions remain byte-equivalent as XML trees."""
    root = ET.fromstring('<robot><link name="custom"/></robot>')
    before = ET.tostring(root)
    assert not correct_known_xarm_products(before, root)
    assert ET.tostring(root) == before
    print("inertia: unknown-source preservation passed")


if __name__ == "__main__":
    self_test()
