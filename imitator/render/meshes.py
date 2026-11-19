"""Robot meshes for the Newton cameras: URDF visual meshes, falling back per link to the convex hull of the link's
collision geometry when a visual mesh cannot be loaded (e.g. PiPER's COLLADA .dae files need pycollada, which is not
installed)."""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

from ..soft.panda_ik import _T, _rpy


def robot_meshes(urdf: str, links, max_faces: int = 3000) -> dict:
    """{link: [(vertices float32 (V,3), faces int32 (F,3), T_local 4x4)]}"""
    import trimesh
    from ..embodiments.urdf_tools import load_urdf
    root = ET.parse(urdf).getroot()
    base = os.path.dirname(urdf)
    want = set(links)
    out, model = {}, None
    for ln in root.findall("link"):
        name = ln.get("name")
        if name not in want:
            continue
        lst, failed = [], False
        for vis in ln.findall("visual"):
            m = vis.find("geometry/mesh")
            if m is None:
                continue
            o = vis.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else [0, 0, 0])]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else [0, 0, 0])]
            sc = [float(v) for v in m.get("scale", "1 1 1").split()]
            try:
                mesh = trimesh.load(os.path.join(base, m.get("filename")), force="mesh")
                if len(mesh.faces) > max_faces:
                    try:
                        mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
                    except Exception:
                        pass
                V = np.asarray(mesh.vertices, np.float32) * np.asarray(sc, np.float32)
                lst.append((V, np.asarray(mesh.faces, np.int32), _T(_rpy(*rpy), xyz)))
            except Exception:
                failed = True
        if failed or not lst:
            model = model or load_urdf(urdf)
            P = model.collision_points(name)
            if len(P) >= 4:
                try:
                    h = trimesh.Trimesh(P).convex_hull
                    lst = [(np.asarray(h.vertices, np.float32), np.asarray(h.faces, np.int32), np.eye(4))]
                except Exception:
                    lst = []
        if lst:
            out[name] = lst
    return out
