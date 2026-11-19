"""Relocate upstream author-absolute MJCF paths without editing original files."""
import copy
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from environment_common import digest, write

ROOT = Path("/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1/environments/robocasa")


def semantic_tree(element):
    attributes = {key: value for key, value in element.attrib.items()
                  if key != "file" and not (element.tag == "compiler" and key in {"meshdir", "texturedir"})}
    return [element.tag, attributes, (element.text or "").strip(), [semantic_tree(child) for child in element]]


def resolve_file(original_xml, element, compiler, assets):
    filename = element.get("file")
    subdir = compiler.get("meshdir", "") if compiler is not None and element.tag == "mesh" else compiler.get("texturedir", "") if compiler is not None and element.tag == "texture" else ""
    candidates = [original_xml.parent / subdir / filename, original_xml.parent / filename]
    method = "existing_original_relative_dependency"
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve().is_relative_to(assets.resolve()):
            return candidate.resolve(), method
    marker = "/models/assets/"
    if filename.startswith("/") and marker in filename:
        suffix = filename.rsplit(marker, 1)[1]
        candidate = assets / suffix
        if candidate.is_file() and candidate.resolve().is_relative_to(assets.resolve()):
            return candidate.resolve(), "exact_upstream_models_assets_suffix_rebase"
    raise ValueError("No exact official-package dependency resolution: " + filename)


def main():
    started = time.time()
    catalog_path = ROOT / "asset_catalog.json"
    source_catalog = json.loads(catalog_path.read_text())
    updates, failures, sha_cache = {}, [], {}
    for row in source_catalog["records"]:
        if not row.get("dependency_missing"):
            continue
        source = Path(row["model_xml"])
        original_sha = digest(source)
        try:
            tree = ET.parse(source)
            root = tree.getroot()
            original_semantics = semantic_tree(root)
            compiler = root.find("compiler")
            destination = ROOT / "portable_xml" / source.relative_to(ROOT / "assets")
            destination.parent.mkdir(parents=True, exist_ok=True)
            relocations = []
            for element in root.iter():
                filename = element.get("file")
                if not filename:
                    continue
                resolved, method = resolve_file(source, element, compiler, ROOT / "assets")
                relative = os.path.relpath(resolved, destination.parent)
                element.set("file", relative)
                evidence = {"tag": element.tag, "original_file": filename,
                            "portable_file": relative, "resolved_path": str(resolved), "method": method}
                if method == "exact_upstream_models_assets_suffix_rebase":
                    if str(resolved) not in sha_cache:
                        sha_cache[str(resolved)] = digest(resolved)
                        from PIL import Image
                        with Image.open(resolved) as image:
                            image.load()
                    evidence["official_dependency_sha256"] = sha_cache[str(resolved)]
                relocations.append(evidence)
            if compiler is not None:
                compiler.attrib.pop("meshdir", None)
                compiler.attrib.pop("texturedir", None)
            if semantic_tree(root) != original_semantics:
                raise ValueError("Non-path MJCF semantics changed")
            tree.write(destination, encoding="utf-8", xml_declaration=True)
            for element in ET.parse(destination).getroot().iter():
                filename = element.get("file")
                if filename and not (destination.parent / filename).is_file():
                    raise ValueError("Portable dependency remains unreadable")
            if digest(source) != original_sha:
                raise ValueError("Original upstream XML modified")
            mesh_path = next(Path(item["resolved_path"]) for item in relocations if item["tag"] == "mesh")
            import trimesh
            scene = trimesh.load(mesh_path, force="scene", process=False)
            import numpy as np
            if not scene.geometry or scene.bounds is None or not np.isfinite(scene.bounds).all():
                raise ValueError("Resolved first mesh unreadable or nonfinite")
            update = {"source_original_model_xml": str(source), "source_original_model_xml_sha256": original_sha,
                      "source_dependency_missing_literal": row["dependency_missing"],
                      "portable_model_xml": str(destination), "portable_model_xml_sha256": digest(destination),
                      "file_relocations": relocations,
                      "portable_dependency_missing": [], "portable_geometry_readability": "first_mesh_CPU_readable",
                      "dependency_resolution_scope": "exact_official_package_path_rebase_only",
                      "non_path_MJCF_attributes_and_tree_preserved": True, "original_XML_unchanged": True,
                      "geometry_bounds_raw_units": scene.bounds.tolist(), "physics_qualified": False,
                      "simulation_qualified": False, "training_admitted": False}
            updates[row["asset_id"]] = update
        except Exception as error:
            failures.append({"asset_id": row["asset_id"], "source_xml": str(source),
                             "error_type": type(error).__name__, "error": str(error)})
        if len(updates) % 100 == 0:
            write(ROOT / "PORTABLE_XML_STATE.json", {"resolved": len(updates), "failed": len(failures),
                  "elapsed_seconds": time.time() - started, "source_XML_changed": False})
    # Coreassets appends fixture templates; merge into its latest completed
    # catalog rather than replacing it with this earlier object-only snapshot.
    deadline = time.time() + 900
    while (ROOT / "coreassets_download_state.json").exists() and not (ROOT / "COREASSETS_REPORT.json").exists():
        if time.time() > deadline:
            raise TimeoutError("Coreassets catalog merge did not reach terminal status")
        time.sleep(5)
    latest = json.loads(catalog_path.read_text())
    for row in latest["records"]:
        if row["asset_id"] in updates:
            row.update(updates[row["asset_id"]])
    latest["portable_XML_relocation"] = {"resolved_objects": len(updates), "failures": failures,
                                         "original_XMLs_unchanged": True}
    write(catalog_path, latest)
    objects = [row for row in latest["records"] if row.get("asset_kind") == "object_MJCF_definition" or
               "/objects/" in row.get("model_xml", "")]
    report = {"passed": not failures, "source_literal_missing_objects": len(updates) + len(failures),
              "portable_objects_resolved": len(updates), "failures": failures,
              "relocated_official_dependencies_sha256": sha_cache,
              "object_MJCF_definitions": len(objects),
              "object_source_first_mesh_CPU_readable": sum(row["geometry_readability"] == "first_mesh_CPU_readable" for row in objects),
              "object_native_or_portable_first_mesh_CPU_readable": sum(
                  row["geometry_readability"] == "first_mesh_CPU_readable" or row.get("portable_geometry_readability") == "first_mesh_CPU_readable" for row in objects),
              "object_source_literal_missing_dependencies": sum(bool(row["dependency_missing"]) for row in objects),
              "object_remaining_effective_missing_dependencies": sum(
                  bool(row["dependency_missing"]) and "portable_dependency_missing" not in row for row in objects),
              "all_catalog_MJCF_definitions": len(latest["records"]),
              "original_XMLs_unchanged": True, "simulation_or_training_started": False,
              "elapsed_seconds": time.time() - started}
    write(ROOT / "PORTABLE_XML_REPORT.json", report)
    # Preserve the original validation report as evidence of the source format.
    original_report = json.loads((ROOT / "REPORT.json").read_text())
    snapshot = ROOT / "REPORT_original_literal_XML_validation.json"
    if not snapshot.exists():
        write(snapshot, original_report)
    original_report.update(final_portable_XML_report=str(ROOT / "PORTABLE_XML_REPORT.json"),
                           source_literal_unresolved_objects=original_report["models_with_unresolved_dependencies"],
                           model_definitions=len(latest["records"]),
                           first_mesh_CPU_readable=sum(row["geometry_readability"] == "first_mesh_CPU_readable" for row in latest["records"]),
                           models_with_unresolved_dependencies=sum(bool(row["dependency_missing"]) for row in latest["records"]),
                           portable_path_resolved_object_count=len(updates),
                           source_model_xml_and_literal_validation_fields_preserved=True,
                           coreassets_templates_separate=True)
    write(ROOT / "REPORT.json", original_report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
