"""Prepare portable external robot assets from a local RoboTwin asset checkout.

No third-party meshes are added to the ManiSkill source tree. The source files
are left untouched. Output URDFs use relative mesh paths and one canonical TCP.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


ROBOTS = {
    "piper": ("piper", "piper.urdf", "piper.srdf", "link6", "0 0 0.12", "0 0 0"),
    "arx_x5": ("ARX-X5", "X5A.urdf", None, "link6", "0.12 0 0", "0 1.5707963267948966 0"),
    "ur5_wsg": ("ur5-wsg", "ur5_wsg_gripper.urdf", "ur5.srdf", "wsg_50_base_link", "0 0 0.11", "0 0 -1.5707963267948966"),
}


def _resolve_resource(source: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        # Match a package suffix to the local embodiment directory.
        suffix = filename.removeprefix("package://").split("/", 1)[-1]
        candidate = source / suffix
        if not candidate.exists() and "/meshes/" in filename:
            candidate = source / "meshes" / filename.split("/meshes/", 1)[1]
    elif filename.startswith("file://"):
        candidate = Path(filename.removeprefix("file://"))
    else:
        candidate = source / filename
    candidate = candidate.resolve()
    candidate.relative_to(source.resolve())
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def prepare_robot(robot: str, robotwin_root: Path, output_root: Path) -> Path:
    folder, urdf, srdf, tcp_parent, tcp_xyz, tcp_rpy = ROBOTS[robot]
    source = robotwin_root / "assets" / "embodiments" / folder
    output = output_root / robot
    marker = output / "source.json"
    if output.exists():
        # Preparing an already valid output is deliberately non-destructive.
        if marker.is_file() and (output / "robot.urdf").is_file():
            return output
        raise FileExistsError(f"Destination exists without a prepared asset manifest: {output}")
    tree = ET.parse(source / urdf)
    root = tree.getroot()
    resources = []
    for mesh in root.findall(".//mesh"):
        resource = _resolve_resource(source, mesh.attrib["filename"])
        relative = resource.relative_to(source.resolve())
        mesh.set("filename", relative.as_posix())
        resources.append(relative)
    if robot == "ur5_wsg":
        # q_left was negative outward, q_right positive outward in the source.
        # q'_left=-q_left preserves physical geometry while allowing one scalar
        # parallel-gripper command to drive both independent finger joints.
        joint = root.find("joint[@name='base_joint_gripper_left']")
        axis = joint.find("axis")
        axis.set("xyz", " ".join(str(-float(x)) for x in axis.get("xyz").split()))
        limit = joint.find("limit")
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        limit.set("lower", str(-upper))
        limit.set("upper", str(-lower))
    ET.SubElement(root, "link", {"name": "grasp_tcp"})
    joint = ET.SubElement(root, "joint", {"name": "grasp_tcp_joint", "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": tcp_parent})
    ET.SubElement(joint, "child", {"link": "grasp_tcp"})
    ET.SubElement(joint, "origin", {"xyz": tcp_xyz, "rpy": tcp_rpy})
    output.mkdir(parents=True)
    # Copy the mesh tree so DAE/GLB texture dependencies remain portable too.
    shutil.copytree(source / "meshes", output / "meshes")
    for relative in resources:
        if not (output / relative).exists():
            (output / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, output / relative)
    # MPLib 0.1.x expects precomputed convex companions when convex=True.
    # SAPIEN also uses convex collision meshes for dynamic robot links.
    import trimesh
    collision_files = {mesh.get("filename") for mesh in root.findall("./link/collision/geometry/mesh")}
    for filename in collision_files:
        mesh_path = output / filename
        convex_path = Path(str(mesh_path) + ".convex.stl")
        mesh = trimesh.load(mesh_path, force="mesh")
        mesh.convex_hull.export(convex_path)
    ET.indent(tree, space="  ")
    tree.write(output / "robot.urdf", encoding="utf-8", xml_declaration=True)
    if srdf:
        # Some source SRDFs retain pairs for a different gripper. SAPIEN
        # resolves every pair strictly; only actual URDF links can be retained.
        semantic = ET.parse(source / srdf)
        link_names = {link.get("name") for link in root.findall("link")}
        for pair in list(semantic.getroot().findall("disable_collisions")):
            unknown = pair.get("link1") not in link_names or pair.get("link2") not in link_names
            # The imported Piper's zero pose produces real link2/link4 contact;
            # the source SRDF incorrectly declares this pair "Never" colliding.
            piper_contact = robot == "piper" and {pair.get("link1"), pair.get("link2")} == {"link2", "link4"}
            if unknown or piper_contact:
                semantic.getroot().remove(pair)
        ET.indent(semantic, space="  ")
        semantic.write(output / "robot.srdf", encoding="utf-8", xml_declaration=True)
    else:
        semantic_root = ET.Element("robot", {"name": root.get("name", robot)})
        for joint in root.findall("joint"):
            ET.SubElement(semantic_root, "disable_collisions", {
                "link1": joint.find("parent").get("link"),
                "link2": joint.find("child").get("link"),
                "reason": "Adjacent",
            })
        semantic = ET.ElementTree(semantic_root)
        ET.indent(semantic, space="  ")
        semantic.write(output / "robot.srdf", encoding="utf-8", xml_declaration=True)
    for license_name in ("LICENSE", "LICENSE.txt"):
        if (robotwin_root / license_name).is_file():
            shutil.copy2(robotwin_root / license_name, output / license_name)
    marker.write_text(json.dumps({
        "robot": robot,
        "source": "https://github.com/RoboTwin-Platform/RoboTwin",
        "source_urdf": f"assets/embodiments/{folder}/{urdf}",
        "transformations": ["relative resource paths", "fixed canonical grasp_tcp",
                            "convex collision mesh companions", "SRDF actual-link filtering"]
        + (["left finger q coordinate sign inversion (geometry preserved)"] if robot == "ur5_wsg" else []),
        "tcp": {"parent": tcp_parent, "xyz": tcp_xyz, "rpy": tcp_rpy,
                "convention": "+z approach, +y gripper axis; fixed midpoint between fingers"},
    }, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--robots", choices=list(ROBOTS), nargs="+", default=list(ROBOTS))
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("MS_ASSET_DIR", Path.home() / ".maniskill")) / "data" / "robots")
    args = parser.parse_args()
    for robot in args.robots:
        print(f"{robot}: {prepare_robot(robot, args.robotwin_root, args.output_root)}")


if __name__ == "__main__":
    main()
