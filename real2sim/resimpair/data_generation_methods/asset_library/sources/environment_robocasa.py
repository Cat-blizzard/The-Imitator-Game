"""Download pinned official RoboCasa assets and inspect archives/geometry on CPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from pathlib import Path, PurePosixPath
import stat
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

from environment_common import Guard, digest, download, write


ROOT = Path("/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1/environments")


def safe_members(archive):
    infos = archive.infolist()
    for info in infos:
        path = PurePosixPath(info.filename)
        if (path.is_absolute() or ".." in path.parts or "\\" in info.filename or
                stat.S_ISLNK(info.external_attr >> 16)):
            raise ValueError("Unsafe archive member")
    return infos


def inspect_and_extract(path, root, guard, reserved_download_bytes=0):
    # Match the official shared assets/{fixtures,textures,objects} layout.
    base = root / "assets" / "objects" if path.stem in {"objaverse", "aigen_objs"} else root / "assets"
    target = base / path.stem
    done = root / "extraction_receipts" / (path.stem + ".json")
    if done.exists():
        return json.loads(done.read_text())
    with zipfile.ZipFile(path) as archive:
        infos = safe_members(archive)
        total = sum(info.file_size for info in infos)
        manifest = {"archive": str(path), "archive_sha256": digest(path), "members": len(infos),
                    "uncompressed_bytes": total, "available_assigned_bytes": guard.limit - guard.used,
                    "original_member_paths_preserved": True, "training_or_simulation_qualified": False}
        write(root / "archive_inventory" / (path.stem + ".json"), manifest)
        write(root / "archive_inventory" / (path.stem + "_members.json"), [
            {"name": info.filename, "bytes": info.file_size, "CRC32": f"{info.CRC:08x}"} for info in infos])
        if guard.used + reserved_download_bytes + total > guard.limit:
            bad = archive.testzip()
            if bad:
                raise ValueError("Archive CRC mismatch")
            manifest.update(status="archive_CRC_valid_extraction_deferred_budget", CRC_checked=True)
            return manifest
        if any(base.joinpath(*PurePosixPath(info.filename).parts).exists() for info in infos if not info.is_dir()):
            # A partial extraction is not a complete asset collection.
            raise ValueError("Partial extraction exists without completion receipt")
        base.mkdir(parents=True, exist_ok=True)
        for info in infos:
            destination = base.joinpath(*PurePosixPath(info.filename).parts)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("xb") as output:
                while True:
                    block = source.read(4 << 20)
                    if not block:
                        break
                    guard.charge(len(block))
                    output.write(block)
        manifest.update(status="fully_extracted_CRC_valid", CRC_checked=True,
                        extracted_root=str(target), all_archive_members_extracted=True)
        write(done, manifest)
        return manifest


def catalog(root):
    records, failures = [], []
    mesh_extensions = {".obj", ".stl", ".glb", ".gltf", ".ply"}
    for path in sorted((root / "assets").rglob("*.xml")):
        # Do not turn every mesh file/material include into another asset count.
        try:
            tree = ET.parse(path).getroot()
            if tree.tag != "mujoco":
                continue
            meshes, missing = [], []
            compiler = tree.find("compiler")
            meshdir = compiler.get("meshdir", "") if compiler is not None else ""
            texturedir = compiler.get("texturedir", "") if compiler is not None else ""
            for element in tree.iter():
                file = element.get("file")
                if not file:
                    continue
                subdir = meshdir if element.tag == "mesh" else texturedir if element.tag == "texture" else ""
                candidate = path.parent / subdir / file
                candidates = [candidate, path.parent / file]
                # Official MJCF may use a package-root-relative dependency.
                existing = next((p for p in candidates if p.is_file()), None)
                if existing is None:
                    missing.append({"tag": element.tag, "file": file, "not_resolved_locally": True})
                elif element.tag == "mesh":
                    meshes.append(existing)
            joint_types = [element.get("type", "hinge") for element in tree.iter("joint")]
            source_ids = sorted(set(re.findall(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])", str(path))))
            record = {"asset_id": "robocasa:" + str(path.relative_to(root / "assets")),
                      "source": "RoboCasa official assets", "license": "CC-BY-4.0",
                      "attribution": "RoboCasa authors; preserve upstream per-object provenance where supplied",
                      "model_xml": str(path), "model_xml_sha256": digest(path), "source_object_ids": source_ids,
                      "mesh_paths": [str(p) for p in meshes], "mesh_count": len(meshes),
                      "declared_joint_types": joint_types, "articulation_declared": bool(joint_types),
                      "dependency_missing": missing, "geometry_readability": "pending_mesh_loader",
                      "physics_qualified": False, "simulation_qualified": False, "training_admitted": False}
            if meshes and not missing:
                try:
                    import trimesh
                    mesh = trimesh.load(meshes[0], force="scene", process=False)
                    record["geometry_readability"] = "first_mesh_CPU_readable" if len(mesh.geometry) else "empty"
                    record["geometry_bounds_raw_units"] = mesh.bounds.tolist() if mesh.bounds is not None else None
                    record["raw_units_not_recalibrated"] = True
                except Exception as error:
                    record["geometry_readability"] = "first_mesh_loader_failed"
                    record["geometry_error"] = type(error).__name__
            records.append(record)
        except Exception as error:
            failures.append({"path": str(path), "error_type": type(error).__name__})
    write(root / "asset_catalog.json", {"records": records, "failures": failures,
          "count_scope": "MJCF XML definitions incl fixture variants/components; not independently qualified mechanisms",
          "no_simulator_loaded": True})
    return {"model_definitions": len(records), "joint_declaring_models": sum(r["articulation_declared"] for r in records),
            "first_mesh_CPU_readable": sum(r["geometry_readability"] == "first_mesh_CPU_readable" for r in records),
            "models_with_unresolved_dependencies": sum(bool(r["dependency_missing"]) for r in records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--budget-gib", type=float, default=90.)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        raise ValueError("RoboCasa max2 workers; remaining1 assigned Poly Haven")
    root = args.root / "robocasa"
    guard = Guard(root, int(args.budget_gib * 2**30))
    metadata = args.root / "metadata"
    info = json.loads((metadata / "robocasa_repo_info.json").read_text())
    files = json.loads((metadata / "robocasa_assets.json").read_text())
    revision = info["sha"]
    if info.get("gated") or info.get("private") or info.get("cardData", {}).get("license") != "cc-by-4.0":
        raise ValueError("Official source no longer matches public CC-BY4 license evidence")
    order = ["fixtures.zip", "objaverse.zip", "aigen_objs.zip", "textures.zip", "generative_textures.zip"]
    by_name = {file["path"]: file for file in files}
    items = [{"name": name, "size": by_name[name]["size"], "sha256": by_name[name]["lfs"]["oid"],
              "url": f"https://huggingface.co/datasets/robocasa/robocasa-assets/resolve/{revision}/{name}"} for name in order]
    write(root / "download_plan.json", {"revision": revision, "files": items,
          "license": "CC-BY-4.0", "source": "https://huggingface.co/datasets/robocasa/robocasa-assets",
          "raw_download_bytes": sum(item["size"] for item in items), "assigned_budget_bytes": guard.limit,
          "disk_reserve_bytes": guard.reserve, "workers": args.workers})
    started = time.time()
    outcomes, failures = [], []
    pending_sizes = {item["name"]: max(0, item["size"] - (root / "archives" / item["name"]).stat().st_size)
                     if (root / "archives" / item["name"]).exists() else item["size"] for item in items}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, item, root / "archives" / item["name"], guard): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                receipt = future.result()
                pending_sizes.pop(item["name"], None)
                extracted = inspect_and_extract(Path(receipt["path"]), root, guard, sum(pending_sizes.values()))
                outcomes.append({"name": item["name"], "download": receipt, "archive": extracted})
            except Exception as error:
                failures.append({"name": item["name"], "error_type": type(error).__name__, "error": str(error)})
            write(root / "state.json", {"status": "downloading_and_CPU_checking", "completed": outcomes,
                  "failed": failures, "bytes_written": guard.used, "elapsed_seconds": time.time() - started})
            print(json.dumps({"completed": item["name"], "success": not failures or failures[-1]["name"] != item["name"],
                              "bytes_written": guard.used}), flush=True)
    # Early completions conservatively reserve still-active downloads at full
    # size. Retry deferred whole-package extraction after transfers finish.
    for result in outcomes:
        if not result["archive"].get("all_archive_members_extracted"):
            result["archive"] = inspect_and_extract(Path(result["download"]["path"]), root, guard)
    counts = catalog(root)
    report = {"downloaded_packages": len(outcomes), "failures": failures, "packages": outcomes,
              **counts, "all_five_packages_downloaded": len(outcomes) == 5,
              "all_five_packages_fully_extracted": len(outcomes) == 5 and all(
                  r["archive"].get("all_archive_members_extracted") for r in outcomes),
              "actual_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
              "assigned_budget_bytes": guard.limit, "elapsed_seconds": time.time() - started,
              "simulation_started": False, "training_started": False, "physics_qualified": False}
    write(root / "REPORT.json", report)
    print(json.dumps({k:v for k,v in report.items() if k != "packages"}), flush=True)


if __name__ == "__main__":
    main()
