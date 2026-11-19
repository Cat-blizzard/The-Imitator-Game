"""Local household distractor adapters; metric design sizes are not measurements.

Original assets remain unchanged. Visual and conservative static collision
geometry share one baked coordinate transform. No simulator or download calls.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from ..r2h_pipeline.common import file_sha256, json_sha256, write_json

CATALOG=Path("/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1/objaverse/catalog.jsonl")
DEFAULT_SELECTION=[
    ("mug","5f7ff57d7f5546d7b16b5ce5bbe1735d",.10,"upright_y"),
    ("soda_can","e4fbfc1835ad42519d30983ae6c0ca6d",.11,"upright_y"),
    ("water_bottle","93d9991078204bc4921269c3c8fa4576",.16,"upright_y"),
    ("closed_jam_jar","8cc4336395c64a95a8f986a575f0b5f3",.10,"upright_y"),
    ("tissue_box","6f8040c86eb84e7bba2cd928c0755029",.14,"upright_y"),
    ("soap_dispenser","33a99ad861ba429c804327db5299c407",.14,"upright_y"),
    ("spatula","19e30338a2524d7ea38ea6f1226bd6f2",.18,"flat_shortest_axis"),
    ("remote_control","5db7c77a0e46426f89251401815dbbbb",.14,"flat_shortest_axis"),
    ("stapler","36c8998a6cd64895a75a14232bb6f434",.11,"flat_shortest_axis"),
    ("screwdriver","73cb248ad76d45ee9b638ec4bfea4128",.16,"flat_shortest_axis"),
    ("kettle","04342898e09c4b82a037b2830fd8ddd6",.16,"upright_y"),
    ("milk_carton","0db687e8e18b49f5a6458ad5a75f057b",.15,"upright_y"),
]


def bind(path):
    return {"path":str(Path(path).resolve()),"sha256":file_sha256(path)}


def metric_transform(vertices, longest_dimension_m, orientation):
    vertices=np.asarray(vertices,float)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():
        raise ValueError("Nonfinite or malformed mesh vertices")
    if not .04<=longest_dimension_m<=.30:
        raise ValueError("Explicit tabletop design size outside supported range")
    rotation=Rotation.from_euler("x",90,degrees=True).as_matrix()
    oriented=vertices@rotation.T
    if orientation=="flat_shortest_axis":
        axis=int(np.ptp(oriented,axis=0).argmin())
        if axis!=2:
            # Proper rotation, never axis reflection.
            further=Rotation.from_euler("y" if axis==0 else "x",90,degrees=True).as_matrix()
            rotation=further@rotation
            oriented=vertices@rotation.T
    elif orientation!="upright_y":
        raise ValueError("Unknown orientation policy")
    extent=np.ptp(oriented,axis=0)
    if np.min(extent)<=1e-10:
        raise ValueError("Degenerate zero-thickness object")
    scale=float(longest_dimension_m/extent.max())
    scaled=oriented*scale
    low,high=scaled.min(0),scaled.max(0)
    translation=np.array([-(low[0]+high[0])/2,-(low[1]+high[1])/2,-low[2]])
    transform=np.eye(4)
    transform[:3,:3]=rotation*scale
    transform[:3,3]=translation
    return transform,{"uniform_scale_source_units_to_design_m":scale,
                      "orientation_policy":orientation,
                      "source_up_assumption":"glTF +Y, artist orientation not independently measured",
                      "rotation_source_to_adapter":rotation.tolist(),
                      "translation_adapter_m":translation.tolist(),
                      "target_longest_dimension_m":longest_dimension_m,
                      "is_measured_real_world_scale":False}


def conservative_collision(mesh,max_hull_vertices=192):
    hull=mesh.convex_hull
    kind="whole_object_convex_hull"
    if len(hull.vertices)>max_hull_vertices:
        # An enclosing box is conservative, unlike dropping extreme hull points.
        bounds=mesh.bounds
        hull=trimesh.creation.box(extents=np.ptp(bounds,axis=0),
                                 transform=trimesh.transformations.translation_matrix(bounds.mean(0)))
        kind="conservative_AABB_due_to_convex_vertex_budget"
    if not hull.is_watertight or not hull.is_convex or hull.volume<=1e-12:
        raise ValueError("Invalid conservative static collision mesh")
    return hull,kind


def render_preview(scene,path,title):
    """Rough vertex-color geometry preview, not UV texture or PBR rendering."""
    geometry=list(scene.dump())
    vertices=np.vstack([g.vertices for g in geometry])
    center=vertices.mean(0)
    canvas=Image.new("RGB",(960,360),"white")
    for view,(az,el) in enumerate([(35,25),(150,25),(0,75)]):
        azi,ele=np.radians([az,el])
        eye=np.array([np.cos(ele)*np.cos(azi),np.cos(ele)*np.sin(azi),np.sin(ele)])
        right=np.cross([0,0,1],eye);right/=np.linalg.norm(right)
        up=np.cross(eye,right)
        basis=np.stack([right,up,eye])
        allproj=(vertices-center)@basis.T
        scale=260/max(np.ptp(allproj[:,:2],axis=0).max(),1e-9)
        projected_center=(allproj[:,:2].min(0)+allproj[:,:2].max(0))/2
        triangles=[]
        for mesh in geometry:
            proj=(mesh.vertices-center)@basis.T
            xy=(proj[:,:2]-projected_center)*scale+[160,170]
            xy[:,1]=340-xy[:,1]
            try:
                visual=mesh.visual.to_color() if hasattr(mesh.visual,"to_color") else mesh.visual
                if len(visual.vertex_colors)==len(mesh.vertices):
                    colors=np.asarray(visual.vertex_colors)[mesh.faces][:,:,:3].mean(axis=1)
                else:
                    colors=np.asarray(visual.face_colors)[:,:3]
            except Exception:
                colors=np.tile([170,180,190],(len(mesh.faces),1))
            light=np.clip(.45+.55*np.abs(mesh.face_normals@np.array([.3,-.4,.866])),.3,1)
            for i,face in enumerate(mesh.faces):
                triangles.append((float(proj[face,2].mean()),xy[face],
                                  np.clip(colors[i]*light[i],0,255).astype(np.uint8)))
        image=np.full((340,320,3),248,dtype=np.uint8)
        for _,poly,color in sorted(triangles,key=lambda x:x[0]):
            cv2.fillConvexPoly(image,np.rint(poly).astype(np.int32),tuple(int(x) for x in color),
                              lineType=cv2.LINE_AA)
        canvas.paste(Image.fromarray(image),(view*320,20))
    ImageDraw.Draw(canvas).text((10,3),title+" | rough vertex-color geometry; NOT UV/PBR renderer",fill="black")
    canvas.save(path)


def prepare_asset(record,category,longest,orientation,output):
    source=Path(record["geometry_path"])
    if file_sha256(source)!=record["sha256"]:
        raise ValueError("Source SHA differs from indexed local asset")
    if record.get("license") not in {"CC-BY-4.0","CC-BY-SA-4.0","CC0-1.0"}:
        raise ValueError("Missing supported explicit license record")
    if any(x in category.lower() for x in ["apple","bowl"]):
        raise ValueError("Distractor creates undeclared task target")
    loaded=trimesh.load(source,force="scene",process=False)
    pieces=list(loaded.dump())
    if not pieces or sum(len(g.faces) for g in pieces)>150000:
        raise ValueError("Missing or excessive mesh geometry")
    for g in pieces:
        if not np.isfinite(g.vertices).all() or not np.isfinite(g.face_normals).all():
            raise ValueError("Nonfinite geometry/normals")
    source_mesh=trimesh.util.concatenate(pieces)
    transform,transform_info=metric_transform(source_mesh.vertices,longest,orientation)
    loaded.apply_transform(transform)
    mesh=trimesh.util.concatenate(list(loaded.dump()))
    collision,collision_kind=conservative_collision(mesh)
    folder=Path(output)/category
    folder.mkdir(parents=True,exist_ok=False)
    visual=folder/"visual.glb";physical=folder/"collision.obj"
    visual.write_bytes(loaded.export(file_type="glb"))
    collision.export(physical)
    reloaded=trimesh.load(physical,force="mesh",process=False)
    np.testing.assert_allclose(reloaded.bounds,mesh.bounds,atol=1e-7,rtol=0)
    visual_check=trimesh.load(visual,force="scene",process=False)
    np.testing.assert_allclose(visual_check.bounds,mesh.bounds,atol=1e-6,rtol=0)
    preview=folder/"preview.jpg"
    render_preview(loaded,preview,category)
    record_snapshot=folder/"source_record.json";write_json(record_snapshot,record)
    info={
        "schema":"static_household_asset_candidate_v1",
        "uid":f"household_e_{category}_{record['uid']}",
        "name":category,"category":category,
        "source":record["source"],"source_uid":record["uid"],"source_name":record["name"],
        "source_url":record.get("source_url"),"license":record["license"],
        "license_url":record.get("license_url"),"attribution":record.get("author"),
        "source_file":bind(source),"source_record":bind(record_snapshot),
        "visual":str(visual),"collision":str(physical),
        "files":{str(visual):file_sha256(visual),str(physical):file_sha256(physical)},
        "bounds_rotated_m":mesh.bounds.tolist(),
        "extents_m":mesh.extents.tolist(),
        "coordinate_frame":"baked metric design frame; +Z up; horizontal bbox centre at0, bottom z0",
        "source_to_adapter_4x4":transform.tolist(),
        "transform":transform_info,
        "preview":bind(preview),
        "preview_kind":"rough_vertex_color_geometry_not_actual_UV_or_PBR_rendering",
        "preflight":{
            "finite_geometry":True,"source_mesh_count":len(pieces),
            "source_faces":sum(len(g.faces) for g in pieces),
            "source_all_parts_watertight":all(g.is_watertight for g in pieces),
            "collision_watertight":bool(collision.is_watertight),
            "collision_convex":bool(collision.is_convex),
            "collision_vertices":len(collision.vertices),
            "collision_faces":len(collision.faces),
            "collision_volume_m3":float(collision.volume),
            "collision_AABB_volume_fraction":float(collision.volume/np.prod(mesh.extents)),
            "visual_collision_bounds_match_after_reload":True,
        },
        "collision_policy":collision_kind,
        "intended_use":"static untouched distractor; no handle/cavity interaction",
        "cavities_handles_may_be_filled_by_conservative_collision":True,
        "required_runtime_integration":["renderer","physics_collision","planner_world_obstacles"],
        "runtime_scale_must_be_one":True,"physics_qualified":False,
        "visual_semantics_reviewed":False,"training_admitted":False,
    }
    write_json(folder/"asset.json",info)
    return info


def prepare_catalog(output,source_catalog=CATALOG,selection=DEFAULT_SELECTION):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    wanted={x[1] for x in selection}
    records={}
    for line in Path(source_catalog).open():
        row=json.loads(line)
        if row.get("uid") in wanted:records[row["uid"]]=row
    if set(records)!=wanted:raise ValueError("Some selected local asset records are missing")
    write_json(output/"selection.json",{"selected":[list(x) for x in selection],
        "source_catalog_locator":str(source_catalog),
        "selected_record_content_sha256":json_sha256(records),
        "selection_policy":"Existing local distinct household categories; no apple/bowl distractors; small meshes.",
        "source_catalog_is_mutable_download_log":True,
        "new_downloads":0,"simulation_executed":False})
    assets=[];failures=[]
    for category,uid,size,orientation in selection:
        try:assets.append(prepare_asset(records[uid],category,size,orientation,output))
        except Exception as exc:
            failures.append({"category":category,"uid":uid,"error":str(exc)})
    result={"schema":"household_static_distractor_catalog_v1","assets":assets,
            "failed_candidates":failures,"source_categories_requested":len(selection),
            "new_downloads":0,"simulation_executed":False,"training_admitted":False,
            "limitations":["Design sizes are engineering choices, not recovered metric GT.",
                           "Static convex collision can overapproximate handles/cavities.",
                           "CPU preflight and preview do not establish planner/physics qualification."]}
    write_json(output/"catalog.json",result)
    gallery=Image.new("RGB",(1280,((len(assets)+3)//4)*160),"white")
    for i,a in enumerate(assets):
        image=Image.open(a["preview"]["path"]).crop((0,0,320,360)).resize((142,160))
        x=(i%4)*320;y=(i//4)*160
        gallery.paste(image,(x,y))
        draw=ImageDraw.Draw(gallery)
        draw.text((x+145,y+30),a["name"],fill="black")
        draw.text((x+145,y+55),"dims cm "+",".join(f"{v*100:.1f}" for v in a["extents_m"]),fill="black")
        draw.text((x+145,y+80),"static convex collision",fill="black")
    gallery.save(output/"asset_gallery.jpg")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--source-catalog",type=Path,default=CATALOG)
    a=p.parse_args()
    r=prepare_catalog(a.output,a.source_catalog)
    print(json.dumps({"assets":len(r["assets"]),"failed":r["failed_candidates"],
                      "output":str(a.output),"simulation_executed":False}))


if __name__=="__main__":main()
