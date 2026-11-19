"""Fetch only the official v1.0 assets subtree and merge without overwriting ZIP assets."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

from environment_common import Guard, digest, download, write

ROOT = Path("/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1/environments")
REVISION = "8f3c96ec8d1bfcd8126cad2bca887da98d30e997"
PREFIX = "robocasa/models/assets/"


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def main():
    root = ROOT / "robocasa"
    staging = root / "coreassets_source"
    guard = Guard(staging, 20 * 2**20)
    tree = json.loads((ROOT / "metadata/robocasa_v1_0_tree.json").read_text())
    if tree.get("sha") != REVISION or tree.get("truncated"):
        raise ValueError("Expected exact official v1.0 tree")
    files = [row for row in tree["tree"] if row["type"] == "blob" and row["path"].startswith(PREFIX)]
    results, failures = [], []
    def fetch(row):
        relative = Path(row["path"].removeprefix(PREFIX))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe official-tree path")
        path = staging / relative
        receipt = download({"url": f"https://raw.githubusercontent.com/robocasa/robocasa/{REVISION}/" + row["path"],
                            "size": row["size"]}, path, guard)
        if git_blob_sha1(path) != row["sha"]:
            raise ValueError("Official Git blob SHA1 differs")
        return {"relative_path": str(relative), "source": receipt, "git_blob_sha1": row["sha"],
                "Git_blob_verified": True, "commit": REVISION}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(fetch, row): row for row in files}
        for future in as_completed(futures):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                failures.append({"source_path": row["path"], "error_type": type(error).__name__, "error": str(error)})
            write(root / "coreassets_download_state.json", {"completed": len(results), "total": len(files),
                  "failures": failures, "commit": REVISION, "simulation_started": False})
    if failures:
        write(root / "COREASSETS_REPORT.json", {"passed": False, "failures": failures, "files": results})
        return
    # Wait for the ZIP exporter/catalog to close its output before amending it.
    deadline = time.time() + 1800
    while not (root / "REPORT.json").exists():
        if time.time() > deadline:
            raise TimeoutError("Wait for bounded ZIP export before coreassets merge")
        time.sleep(5)
    copied, reused, conflicts = [], [], []
    for record in sorted(results, key=lambda row: row["relative_path"]):
        source = Path(record["source"]["path"])
        destination = root / "assets" / record["relative_path"]
        if destination.exists():
            if digest(destination) != record["source"]["sha256"]:
                conflicts.append({"path": str(destination), "upstream_core": record,
                                  "ZIP_asset_sha256": digest(destination), "original_preserved": True})
            else:
                reused.append(str(destination))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        guard.charge(source.stat().st_size)
        shutil.copyfile(source, destination)
        if digest(destination) != record["source"]["sha256"]:
            raise ValueError("Coreassets copy hash differs")
        copied.append(str(destination))
    catalog_path = root / "asset_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    records = catalog["records"]
    known = {record["asset_id"] for record in records}
    for record in records:
        path = Path(record["model_xml"])
        record["asset_kind"] = "object_MJCF_definition" if "objects" in path.relative_to(root / "assets").parts else "fixture_template_or_component"
        record["source_UID_status"] = "original_UID_in_path" if record.get("source_object_ids") else "original_UID_not_identified_do_not_invent_mapping"
    for file in copied:
        path = Path(file)
        if path.suffix != ".xml":
            continue
        tree = ET.parse(path).getroot()
        if tree.tag != "mujoco":
            continue
        key = "robocasa:" + str(path.relative_to(root / "assets"))
        if key in known:
            continue
        dependency_missing, meshes = [], []
        compiler = tree.find("compiler")
        for element in tree.iter():
            filename = element.get("file")
            if not filename:
                continue
            subdir = compiler.get("meshdir", "") if compiler is not None and element.tag == "mesh" else compiler.get("texturedir", "") if compiler is not None and element.tag == "texture" else ""
            candidates = [path.parent / subdir / filename, path.parent / filename]
            found = next((p for p in candidates if p.is_file()), None)
            if found is None:
                dependency_missing.append({"tag": element.tag, "file": filename})
            elif element.tag == "mesh":
                meshes.append(str(found))
        worldbody = tree.find("worldbody")
        joints = [] if worldbody is None else [joint.get("type", "hinge") for joint in worldbody.iter("joint")]
        record = {"asset_id": key, "source": "RoboCasa official v1.0 coreassets", "license": "CC-BY-4.0",
                  "upstream_code_license": "MIT; see preserved LICENSE", "upstream_commit": REVISION,
                  "model_xml": str(path), "model_xml_sha256": digest(path), "asset_kind": "fixture_template_or_component",
                  "mesh_paths": meshes, "mesh_count": len(meshes), "dependency_missing": dependency_missing,
                  "geometry_readability": "MJCF_template_XML_readable_requires_runtime_parameterization",
                  "declared_joint_types": joints, "articulation_declared": bool(joints),
                  "source_object_ids": [], "source_UID_status": "not_an_Objaverse_object",
                  "primitive_geom_count": sum(element.get("type", "sphere") != "mesh" for element in tree.iter("geom")),
                  "physics_qualified": False, "simulation_qualified": False, "training_admitted": False}
        records.append(record);known.add(key)
    catalog["coreassets_supplement"] = {"commit": REVISION, "copied": len(copied), "reused_identical": len(reused),
                                        "conflicts_original_preserved": conflicts}
    catalog["count_scope"] = "Object MJCF definitions and fixture templates/components explicitly separated by asset_kind"
    write(catalog_path, catalog)
    report = {"passed": not conflicts, "official_commit": REVISION, "downloaded_files": len(results),
              "copied_new_files": len(copied), "reused_identical_ZIP_files": len(reused),
              "conflicts_original_preserved": conflicts, "definition_counts": dict(Counter(r["asset_kind"] for r in records)),
              "all_source_files_Git_blob_and_SHA256_verified": True, "files": results,
              "simulation_or_training_started": False, "qualified_mechanisms": 0}
    write(root / "COREASSETS_REPORT.json", report)
    print(json.dumps({k:v for k,v in report.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()
