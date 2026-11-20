"""Known-source physics repair, shared by simulation and regenerated planners."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from imitator.embodiments import inertia

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "imitator/assets/embodiments/_src/xarm6/xarm6_robotiq.urdf"


def test_only_arm_products_change_and_correction_is_idempotent():
    source = SOURCE.read_bytes()
    original = ET.fromstring(source)
    fixed = deepcopy(original)
    assert inertia.correct_known_xarm_products(source, fixed)
    expected = deepcopy(original)
    for i in range(1, 7):
        a = expected.find(f"link[@name='link{i}']/inertial/inertia")
        for key in ("ixy", "ixz", "iyz"):
            old = a.attrib[key]
            if float(old):
                a.set(key, old[1:] if old.startswith("-") else "-" + old.removeprefix("+"))
        v = fixed.find(f"link[@name='link{i}']/inertial/inertia").attrib
        tensor = np.array([[float(v['ixx']), float(v['ixy']), float(v['ixz'])],
                           [float(v['ixy']), float(v['iyy']), float(v['iyz'])],
                           [float(v['ixz']), float(v['iyz']), float(v['izz'])]])
        principal = np.linalg.eigvalsh(tensor)
        assert np.all(principal > 0)
        assert principal[-1] <= principal[:2].sum() + 1e-12
    assert ET.tostring(fixed) == ET.tostring(expected)
    once = ET.tostring(fixed)
    assert inertia.correct_known_xarm_products(source, fixed)
    assert ET.tostring(fixed) == once
    assert SOURCE.read_bytes() == source


@pytest.mark.parametrize("filename", ["xarm6_robotiq.urdf", "xarm6_robotiq_mplib.urdf"])
def test_checked_in_simulation_and_planning_tensors_are_corrected(filename):
    path = ROOT / "imitator/assets/embodiments/xarm6_robotiq" / filename
    root = ET.parse(path).getroot()
    before = ET.tostring(root)
    assert inertia.correct_known_xarm_products(SOURCE.read_bytes(), root)
    assert ET.tostring(root) == before


def test_custom_sources_are_untouched():
    original = SOURCE.read_bytes() + b"\n<!-- custom asset -->"
    root = ET.fromstring(original)
    before = ET.tostring(root)
    assert not inertia.correct_known_xarm_products(original, root)
    assert ET.tostring(root) == before


def test_known_source_with_changed_tensor_fails_before_mutating_any_link():
    original = SOURCE.read_bytes()
    root = ET.fromstring(original)
    root.find("link[@name='link6']/inertial/inertia").set("ixy", "99")
    before = ET.tostring(root)
    with pytest.raises(ValueError, match="differs"):
        inertia.correct_known_xarm_products(original, root)
    assert ET.tostring(root) == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
