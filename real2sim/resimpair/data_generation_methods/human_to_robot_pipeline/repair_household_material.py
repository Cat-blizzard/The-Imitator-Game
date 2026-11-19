"""Explicit material-only candidate repair, preserving GLB geometry binary bytes."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import trimesh
import numpy as np

from .household_assets import render_preview
from .planning_contract import bind,save_new,check_binding
from ..r2h_pipeline.common import json_sha256


def split_glb(path):
    data=Path(path).read_bytes()
    magic,version,length=struct.unpack_from("<III",data)
    if magic!=0x46546C67 or version!=2 or length!=len(data):
        raise ValueError("Invalid GLB2 container")
    chunks=[];offset=12
    while offset<len(data):
        size,kind=struct.unpack_from("<II",data,offset)
        start=offset+8
        chunks.append((kind,data[start:start+size]))
        offset=start+size
    if offset!=len(data) or chunks[0][0]!=0x4E4F534A:
        raise ValueError("Invalid GLB chunks")
    return json.loads(chunks[0][1]),chunks[1:]


def matte_material_candidate(source,destination):
    original,chunks=split_glb(source)
    result=deepcopy(original)
    if not result.get("materials"):raise ValueError("No source materials")
    for material in result["materials"]:
        # Deliberately simplified paperboard appearance, not recovered albedo.
        material["pbrMetallicRoughness"]={
            "baseColorFactor":[.90,.88,.80,1.0],
            "metallicFactor":0.0,"roughnessFactor":.85,
        }
        for key in ("normalTexture","occlusionTexture","emissiveTexture","emissiveFactor","extensions"):
            material.pop(key,None)
        material["alphaMode"]="OPAQUE"
        material.pop("alphaCutoff",None)
        material["name"]=material.get("name","material")+"_explicit_matte_paperboard"
    original_geometry={k:v for k,v in original.items() if k!="materials"}
    result_geometry={k:v for k,v in result.items() if k!="materials"}
    if original_geometry!=result_geometry:raise RuntimeError("Non-material GLB JSON changed")
    encoded=json.dumps(result,separators=(",",":")).encode()
    encoded+=b" "*((-len(encoded))%4)
    out_chunks=[(0x4E4F534A,encoded),*chunks]
    body=b"".join(struct.pack("<II",len(payload),kind)+payload for kind,payload in out_chunks)
    Path(destination).write_bytes(struct.pack("<III",0x46546C67,2,12+len(body))+body)
    reread,binary=split_glb(destination)
    if binary!=chunks:raise RuntimeError("Geometry/image BIN chunk changed")
    before=trimesh.load(source,force="scene",process=False)
    after=trimesh.load(destination,force="scene",process=False)
    np.testing.assert_array_equal(before.bounds,after.bounds)
    for name,mesh in before.geometry.items():
        np.testing.assert_array_equal(mesh.vertices,after.geometry[name].vertices)
        np.testing.assert_array_equal(mesh.faces,after.geometry[name].faces)
        np.testing.assert_array_equal(mesh.visual.uv,after.geometry[name].visual.uv)
    return {
        "source":bind(source),"candidate":bind(destination),
        "original_materials":original["materials"],"candidate_materials":reread["materials"],
        "nonmaterial_json_sha256":json_sha256(original_geometry),
        "binary_chunks_sha256":[hashlib.sha256(payload).hexdigest() for _,payload in chunks],
        "binary_chunks_byte_identical":True,
        "mesh_vertices_faces_UV_node_transforms_unchanged":True,
        "appearance_is_explicit_design_choice_not_recovered_ground_truth":True,
    }


def prepare(asset_spec,proposal_sources,output):
    asset_spec=Path(asset_spec);output=Path(output)
    original=json.loads(asset_spec.read_text())
    for path,sha in original["files"].items():check_binding({"path":path,"sha256":sha})
    output.mkdir(parents=True,exist_ok=False)
    visual=output/"visual.glb"
    patch=matte_material_candidate(original["visual"],visual)
    asset=deepcopy(original)
    asset["parent_asset_spec"]=bind(asset_spec)
    asset["uid"]=original["uid"]+"_matte_paperboard_v1"
    asset["visual"]=str(visual)
    asset["files"]={str(visual):bind(visual)["sha256"],
                    original["collision"]:original["files"][original["collision"]]}
    asset["material_patch"]=patch
    asset["semantic_scope"]="plain closed paperboard carton; original milk-print design intentionally removed"
    asset["render_qualified"]=False
    asset["visual_semantics_reviewed"]=False
    asset["training_admitted"]=False
    preview=output/"preview.jpg"
    render_preview(trimesh.load(visual,force="scene",process=False),preview,"matte closed carton candidate")
    asset["preview"]=bind(preview)
    save_new(output/"asset.json",asset)
    proposals=[]
    for path in map(Path,proposal_sources):
        source=json.loads(path.read_text())
        for old in source["cases"]:
            if not any(d["asset_uid"]==original["uid"] for d in old["added_distractors"]):
                continue
            # Prefer one exact original matched case and the geometry-repaired
            # four-clutter case; don't manufacture redundant repeats.
            if "_nativeclear_v3" in old["case_id"] and "clutter4" not in old["case_id"]:
                continue
            if "_nativeclear_v3" not in old["case_id"] and "clutter2" not in old["case_id"]:
                continue
            new=deepcopy(old)
            new["case_id"]=old["case_id"]+"_matte_v1"
            for distractor in new["added_distractors"]:
                if distractor["asset_uid"]==original["uid"]:
                    distractor["asset_uid"]=asset["uid"]
                    distractor["visual"]=asset["visual"]
                    distractor["asset_files"]=asset["files"]
                    distractor["asset_spec"]=bind(output/"asset.json")
            new["material_only_repair"]={
                "parent_case_id":old["case_id"],"parent_proposals":bind(path),
                "geometry_collision_layout_camera_yaw_seed_unchanged":True,
                "asset_patch":bind(output/"asset.json"),
                "requires_actual_renderer_review":True,
            }
            proposals.append(new)
    save_new(output/"proposals.json",{
        "schema":"material_only_household_qualification_candidates_v1",
        "cases":proposals,"simulation_executed":False,"training_admitted":False})
    save_new(output/"render_qa_quarantine.json",{
        "source_asset":bind(asset_spec),"source_visual":bind(original["visual"]),
        "finding":"Original print has low semantic readability at the current simulated view and resolution.",
        "source_base_texture_inspected":True,
        "source_texture_design":"Original atlas intentionally contains large black/white geometric print.",
        "geometry_or_normal_corruption_proven":False,
        "cpu_preview_defect":"Vertex color sampling does not reproduce UV texture detail or PBR illumination.",
        "current_view_training_admission":"pending readability review; do not infer a general asset rendering defect",
        "matte_variant_is_optional_engineering_choice":True,
        "renderer_bug_fix_claimed":False,
        "candidate_material_render_qualified":False,
        "same_collision_file":bind(original["collision"]),
        "scope":"Appearance quarantine only; old physical success evidence remains intact.",
    })
    return {"case_ids":[c["case_id"] for c in proposals],"new_api_calls":0,"new_simulator_runs":0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset",type=Path,required=True)
    p.add_argument("--proposals",type=Path,action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(prepare(args.asset,args.proposals,args.output)))


if __name__=="__main__":main()
