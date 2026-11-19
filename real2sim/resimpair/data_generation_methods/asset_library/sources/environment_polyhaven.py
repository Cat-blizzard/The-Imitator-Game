"""Single-download-worker CC0 household geometry/PBR/HDRI expansion via official API."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path, PurePosixPath
import time
from urllib.parse import urlsplit

from environment_common import Guard, digest, download, fetch_metadata, write

ROOT = Path("/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1/environments")


def old_ids():
    result = set()
    base = Path("/mnt/sdf/datasets/open_assets")
    for root in [base / "task_factory_20261016_v1" / "polyhaven",
                 base / "task_factory_backgrounds_20261017_v2" / "metadata"]:
        if root.exists():
            result.update(path.name for path in root.iterdir() if path.is_dir())
    p = base / "task_factory_20261016_v1" / "polyhaven_downloads.json"
    if p.exists():
        result.update(row["asset"] for row in json.loads(p.read_text()).get("assets", []))
    p = base / "task_factory_backgrounds_20261017_v2" / "download_and_decode_audit_v1.json"
    if p.exists():
        x = json.loads(p.read_text())
        result.update(x.get("train_asset_ids", []))
        result.update(x.get("heldout_asset_ids", []))
    return result


def image_check(path, hdr=False):
    import numpy as np
    if hdr:
        import cv2
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or not np.isfinite(image).all():
            raise ValueError("HDRI decode or finite check failed")
        return {"shape": list(image.shape), "dtype": str(image.dtype), "finite": True}
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        shape = [image.height, image.width]
    return {"shape": shape, "full_decode_passed": True}


def geometry_check(path):
    import numpy as np
    import trimesh
    source = json.loads(path.read_text())
    dependencies = []
    for block in source.get("buffers", []) + source.get("images", []):
        uri = block.get("uri")
        if not uri or uri.startswith("data:"):
            continue
        uri_path = PurePosixPath(uri)
        if uri_path.is_absolute() or ".." in uri_path.parts:
            raise ValueError("Nonlocal GLTF dependency")
        dependency = path.parent.joinpath(*uri_path.parts)
        if not dependency.is_file():
            raise ValueError("Missing GLTF dependency")
        dependencies.append(str(dependency))
    scene = trimesh.load(path, force="scene", process=False)
    if not scene.geometry:
        raise ValueError("Empty geometry")
    for mesh in scene.geometry.values():
        if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
            raise ValueError("Empty or nonfinite mesh")
    if scene.bounds is None or not np.isfinite(scene.bounds).all() or max(scene.extents) <= 0:
        raise ValueError("Invalid geometry bounds")
    return {"status": "all_GLTF_dependencies_present_all_meshes_CPU_readable", "mesh_count": len(scene.geometry),
            "vertices": sum(len(mesh.vertices) for mesh in scene.geometry.values()),
            "faces": sum(len(mesh.faces) for mesh in scene.geometry.values()),
            "bounds_raw_units": scene.bounds.tolist(), "dependent_paths": dependencies,
            "scene_transforms_preserved": True, "physical_units_not_recalibrated": True}


def choose_texture(files, channel):
    levels = files.get(channel, {})
    for size in ("1k", "2k"):
        formats = levels.get(size, {})
        for fmt in ("jpg", "png"):
            if fmt in formats:
                return formats[fmt], size
    raise ValueError("Required PBR channel unavailable at1k/2k")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--budget-gib", type=float, default=9.9)
    parser.add_argument("--models", type=int, default=150)
    parser.add_argument("--textures", type=int, default=30)
    parser.add_argument("--hdris", type=int, default=15)
    args = parser.parse_args()
    root = args.root / "polyhaven"
    guard = Guard(root, int(args.budget_gib * 2**30))
    old = old_ids()
    metadata = args.root / "metadata"
    models = dict(fetch_metadata("https://api.polyhaven.com/assets?t=models&c=furniture", metadata / "polyhaven_models_furniture.json"))
    for category in ("decorative", "appliances", "containers"):
        models.update(fetch_metadata("https://api.polyhaven.com/assets?t=models&c=" + category,
                                     metadata / ("polyhaven_models_" + category + ".json")))
    def score_model(pair):
        key, value = pair
        text = " ".join([key, value.get("name", ""), *value.get("categories", []), *value.get("tags", [])]).lower()
        words = ["cabinet", "shelf", "drawer", "table", "chair", "stool", "lamp", "basket", "book", "vase", "cup", "bowl", "pot", "kitchen", "sofa", "bed", "furniture"]
        return (-sum(word in text for word in words), key)
    selected_models = [(key, value) for key, value in sorted(models.items(), key=score_model) if key not in old][:args.models]
    textures = json.loads((metadata / "polyhaven_textures.json").read_text())
    texture_groups = {"wood": [], "floor": [], "tile": [], "fabric": [], "metal": [], "stone": []}
    for key, value in textures.items():
        if key in old:
            continue
        text = " ".join([key, *value.get("categories", []), *value.get("tags", [])]).lower()
        for group in texture_groups:
            if group in text or group == "stone" and "marble" in text:
                texture_groups[group].append((key, value))
                break
    selected_textures = []
    for offset in range(args.textures):
        for group, values in texture_groups.items():
            if offset < len(values) and len(selected_textures) < args.textures:
                selected_textures.append((*sorted(values)[offset], group))
    hdris = json.loads((metadata / "polyhaven_hdris.json").read_text())
    indoor = [(key, value) for key, value in hdris.items() if key not in old and any(
        word in " ".join([key, *value.get("categories", []), *value.get("tags", [])]).lower()
        for word in ("indoor", "interior", "kitchen", "room", "living", "studio", "hall", "apartment"))]
    selected_hdris = sorted(indoor)[:args.hdris]
    write(root / "selection.json", {"Powered by Poly Haven": True, "license": "CC0-1.0",
          "source": "https://polyhaven.com/license", "old_ids_excluded": sorted(old),
          "selected_models": [key for key, _ in selected_models],
          "selected_textures": [key for key, _, _ in selected_textures],
          "selected_HDRIs": [key for key, _ in selected_hdris],
          "download_workers": 1, "api_metadata_interval_seconds": .6, "budget_bytes": guard.limit})
    records, failures = [], []
    started = time.time()
    jobs = [(key, info, "model", None) for key, info in selected_models] + [
        (key, info, "texture", group) for key, info, group in selected_textures] + [
        (key, info, "HDRI", None) for key, info in selected_hdris]
    for key, info, kind, group in jobs:
        try:
            files = fetch_metadata("https://api.polyhaven.com/files/" + key, root / "metadata" / key / "files.json")
            target = root / ("models" if kind == "model" else "textures" if kind == "texture" else "hdris") / key
            pieces, main_path, channels = [], None, []
            if kind == "model":
                model = files.get("gltf", {}).get("1k", {}).get("gltf")
                if not model:
                    raise ValueError("No1kGLTF_with_dependency_manifest")
                main_path = target / Path(urlsplit(model["url"]).path).name
                pieces = [(model, main_path)]
                for name, spec in model.get("include", {}).items():
                    relative = PurePosixPath(name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("Unsafe model dependency path")
                    pieces.append((spec, target.joinpath(*relative.parts)))
            elif kind == "texture":
                for channel in ("Diffuse", "Rough", "nor_gl"):
                    spec, size = choose_texture(files, channel)
                    path = target / Path(urlsplit(spec["url"]).path).name
                    pieces.append((spec, path));channels.append({"name": channel, "size": size, "path": str(path)})
            else:
                spec = files.get("hdri", {}).get("1k", {}).get("hdr")
                if not spec:
                    raise ValueError("No1kHDR")
                pieces = [(spec, target / Path(urlsplit(spec["url"]).path).name)]
            remaining = sum(spec["size"] for spec, path in pieces if not path.exists())
            if guard.used + remaining > guard.limit:
                raise RuntimeError("assigned_asset_budget_exceeded")
            receipts = [download(spec, path, guard) for spec, path in pieces]
            if kind == "model":
                readability = geometry_check(main_path)
                imagechecks = [image_check(path) for _, path in pieces if path.suffix.lower() in {".jpg", ".png"}]
            else:
                readability = None
                imagechecks = [image_check(path, hdr=kind == "HDRI") for _, path in pieces]
            record = {"asset_id": "polyhaven:" + key, "source_asset_id": key, "name": info.get("name"),
                      "kind": kind, "category": group or info.get("categories"), "tags": info.get("tags", []),
                      "authors": info.get("authors"), "source": "https://polyhaven.com/a/" + key,
                      "license": "CC0-1.0", "attribution": "Powered by Poly Haven",
                      "main_path": str(main_path) if main_path else None, "files": receipts,
                      "geometry_readability": readability, "image_checks": imagechecks, "channels": channels,
                      "intended_use": "furniture_or_household_geometry_candidate" if kind == "model" else "visual_background_only",
                      "articulation_defined": False, "physics_qualified": False,
                      "simulation_qualified": False, "training_admitted": False}
            records.append(record)
            write(target / "ASSET_RECEIPT.json", record)
        except Exception as error:
            failures.append({"asset_id": key, "kind": kind, "error_type": type(error).__name__, "error": str(error)})
            if isinstance(error, RuntimeError):
                break
        write(root / "asset_catalog.json", {"records": records, "failures": failures, "Powered by Poly Haven": True})
        write(root / "state.json", {"completed": len(records), "failed": len(failures), "last_asset": key,
              "bytes_written": guard.used, "elapsed_seconds": time.time() - started})
        print(json.dumps({"asset": key, "completed": len(records), "failed": len(failures), "bytes_written": guard.used}), flush=True)
    write(root / "asset_catalog.json", {"records": records, "failures": failures, "Powered by Poly Haven": True})
    write(root / "REPORT.json", {"counts": dict(Counter(row["kind"] for row in records)), "completed": len(records),
          "failures": failures, "bytes_written": guard.used, "assigned_budget_bytes": guard.limit,
          "elapsed_seconds": time.time() - started, "old_assets_not_redownloaded": sorted(old),
          "simulation_or_training_started": False, "physics_qualified": False})


if __name__ == "__main__":
    main()
