"""Generic CPU local boundary discontinuity proposals, never task/physics GT.

Decode every adjacent frame, compensate robust background motion and global
colour offsets, then compare high-residual tiles with nearby temporal/spatial
activity. No category, colour, manual ROI, target location or frame ID enters
the detector. Thresholds are exploratory, not held-out calibrated.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from .common import file_sha256,write_json


@dataclass(frozen=True)
class Config:
    max_side: int = 512
    tile_size: int = 32
    tile_stride: int = 16
    top_pixel_fraction: float = .125
    context_transitions: int = 12
    spatial_context_tile_radius: int = 2
    boundary_seconds: float = .5
    minimum_context_transitions: int = 4
    robust_noise_floor: float = 2.
    minimum_tile_energy: float = 12.
    minimum_excess_energy: float = 10.
    minimum_energy_ratio: float = 3.
    robust_z_threshold: float = 6.
    pixel_change_threshold: float = 12.
    minimum_component_pixels: int = 8
    max_features: int = 400
    minimum_alignment_inliers: int = 12
    minimum_alignment_inlier_fraction: float = .40
    max_alignment_corner_displacement_fraction: float = .08


def pixel_hash(frame):
    h=hashlib.sha256()
    h.update(str(frame.shape).encode())
    h.update(frame.tobytes())
    return h.hexdigest()


def aligned_residual(previous,current,config=Config()):
    """Partial-affine RANSAC compensates camera motion; failed fits stay explicit."""
    height,width=previous.shape[:2]
    gray0=cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY)
    gray1=cv2.cvtColor(current,cv2.COLOR_BGR2GRAY)
    points=cv2.goodFeaturesToTrack(gray0,maxCorners=config.max_features,
                                  qualityLevel=.01,minDistance=7,blockSize=5)
    transform=np.array([[1.,0.,0.],[0.,1.,0.]],np.float32)
    evidence={"status":"identity_fallback","features":0,"inliers":0,"reason":"insufficient_tracks"}
    if points is not None and len(points)>=config.minimum_alignment_inliers:
        forward,status,_=cv2.calcOpticalFlowPyrLK(gray0,gray1,points,None,
            winSize=(21,21),maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
        backward=backstatus=None
        if forward is not None:
            backward,backstatus,_=cv2.calcOpticalFlowPyrLK(gray1,gray0,forward,None,
                winSize=(21,21),maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
        if forward is not None and backward is not None:
            valid=(status.ravel()>0)&(backstatus.ravel()>0)
            valid&=np.linalg.norm(backward-points,axis=2).ravel()<1.5
            p=points.reshape(-1,2)[valid];q=forward.reshape(-1,2)[valid]
            evidence["features"]=len(p)
            if len(p)>=config.minimum_alignment_inliers:
                fit,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,
                    ransacReprojThreshold=1.5,maxIters=1000,confidence=.99)
                if fit is not None and np.isfinite(fit).all():
                    n=int(inliers.sum())
                    corners=np.array([[0,0,1],[width,0,1],[0,height,1],[width,height,1]],float)
                    displacement=np.max(np.linalg.norm(corners@fit.T-corners[:,:2],axis=1))
                    scale=np.sqrt(np.linalg.det(fit[:,:2]))
                    evidence.update(inliers=n,inlier_fraction=n/len(p),
                                    max_corner_displacement_px=float(displacement),
                                    estimated_scale=float(scale))
                    if (n>=config.minimum_alignment_inliers
                        and n/len(p)>=config.minimum_alignment_inlier_fraction
                        and displacement<config.max_alignment_corner_displacement_fraction*np.hypot(width,height)
                        and .94<scale<1.06):
                        transform=fit.astype(np.float32)
                        evidence.update(status="robust_affine_background_proxy",reason=None)
                    else:evidence["reason"]="fit_rejected_large_or_unsupported_global_motion"
    warped=cv2.warpAffine(previous,transform,(width,height),flags=cv2.INTER_LINEAR)
    valid=cv2.warpAffine(np.ones((height,width),np.uint8),transform,(width,height),
                         flags=cv2.INTER_NEAREST)>0
    valid=cv2.erode(valid.astype(np.uint8),np.ones((5,5),np.uint8))>0
    delta=current.astype(np.float32)-warped.astype(np.float32)
    offset=np.median(delta[valid],axis=0) if valid.any() else np.zeros(3)
    residual=np.max(np.abs(delta-offset),axis=2)
    residual[~valid]=0.
    evidence.update(transform_previous_to_current=transform.tolist(),
                    removed_median_BgR_offset=offset.tolist(),
                    valid_pixel_fraction=float(valid.mean()),
                    no_semantic_background_segmentation=True)
    return residual,evidence


def tile_energy(residual,config=Config()):
    size=config.tile_size
    if min(residual.shape)<size:raise ValueError("Frame smaller than tile size")
    # Pad with zeros so tiny edge changes are not silently excluded.
    h,w=residual.shape
    rows=max(1,int(np.ceil((h-size)/config.tile_stride))+1)
    cols=max(1,int(np.ceil((w-size)/config.tile_stride))+1)
    padded=np.pad(residual,((0,(rows-1)*config.tile_stride+size-h),
                            (0,(cols-1)*config.tile_stride+size-w)))
    tiles=np.lib.stride_tricks.sliding_window_view(padded,(size,size))[
        ::config.tile_stride,::config.tile_stride].reshape(rows,cols,size*size)
    k=max(1,int(np.ceil(size*size*config.top_pixel_fraction)))
    return np.partition(tiles,-k,axis=2)[:,:,-k:].mean(axis=2)


def temporal_scores(energies,config=Config()):
    """Spatial context prevents smooth motion entering a new tile being an outlier."""
    energies=np.asarray(energies,float)
    if energies.ndim!=3:raise ValueError("Need transition x tile rows x tile columns")
    radius=config.spatial_context_tile_radius
    kernel=np.ones((2*radius+1,2*radius+1),np.uint8)
    spatial=np.stack([cv2.dilate(e.astype(np.float32),kernel) for e in energies])
    output=[]
    for i,current in enumerate(energies):
        indices=[j for j in range(max(0,i-config.context_transitions),
                                  min(len(energies),i+config.context_transitions+1))
                 if abs(j-i)>1]
        if len(indices)<config.minimum_context_transitions:
            output.append(None);continue
        context=spatial[indices]
        baseline=np.median(context,axis=0)
        mad=np.median(np.abs(context-baseline),axis=0)
        scale=1.4826*mad+config.robust_noise_floor
        excess=current-baseline
        z=excess/scale
        ratio=current/(baseline+config.robust_noise_floor)
        eligible=(current>=config.minimum_tile_energy)&(excess>=config.minimum_excess_energy)
        eligible&=(ratio>=config.minimum_energy_ratio)&(z>=config.robust_z_threshold)
        output.append({"baseline":baseline,"mad":mad,"robust_z":z,"energy_ratio":ratio,
                       "candidate_tiles":eligible,"context_indices":indices})
    return output


def regions(residual,score,config=Config()):
    support=np.zeros(residual.shape,np.uint8)
    for row,col in np.argwhere(score["candidate_tiles"]):
        y=row*config.tile_stride;x=col*config.tile_stride
        support[y:y+config.tile_size,x:x+config.tile_size]=1
    changed=((residual>=config.pixel_change_threshold)&(support>0)).astype(np.uint8)
    changed=cv2.morphologyEx(changed,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(changed,connectivity=8)
    rois=[]
    for i in range(1,n):
        x,y,w,h,pixels=map(int,stats[i])
        if pixels<config.minimum_component_pixels:continue
        values=residual[labels==i]
        rows=np.arange(max(0,(y-config.tile_size+1)//config.tile_stride),
                       min(score["robust_z"].shape[0],(y+h-1)//config.tile_stride+1))
        cols=np.arange(max(0,(x-config.tile_size+1)//config.tile_stride),
                       min(score["robust_z"].shape[1],(x+w-1)//config.tile_stride+1))
        z=float(score["robust_z"][np.ix_(rows,cols)].max())
        rois.append({"processing_bbox_xyxy":[x,y,x+w,y+h],"changed_pixels":pixels,
                     "peak_robust_z":z,"mean_residual":float(values.mean()),
                     "max_residual":float(values.max()),
                     "kind":"local_pixel_discontinuity_review_proposal"})
    return sorted(rois,key=lambda r:r["peak_robust_z"],reverse=True)


def probe_pts(video):
    raw=subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","frame=best_effort_timestamp_time","-of","json",str(video)],
        check=True,capture_output=True,text=True)
    pts=[float(f["best_effort_timestamp_time"]) for f in json.loads(raw.stdout)["frames"]]
    if len(pts)<8 or any(b<=a for a,b in zip(pts,pts[1:])):
        raise ValueError("Need at least eight monotonically timed frames")
    return np.asarray(pts)


def export_evidence(video,events,output,dimensions):
    """Lossless decoded frames + an annotated review sheet; no generated pixels."""
    needed={f for event in events for f in (event["previous_frame"],event["frame_index"])}
    if not needed:return []
    cap=cv2.VideoCapture(str(video));frames={};index=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if index in needed:frames[index]=frame
        index+=1
    cap.release()
    evidence=[]
    for index in sorted(needed):
        path=output/f"frame_{index:06d}.png";cv2.imwrite(str(path),frames[index])
        evidence.append({"frame_index":index,"path":str(path),"sha256":file_sha256(path),
                         "decoded_BgR_hash":pixel_hash(frames[index])})
    panels=[]
    for event in sorted(events,key=lambda e:e["peak_robust_z"],reverse=True)[:8]:
        left=frames[event["previous_frame"]].copy();right=frames[event["frame_index"]].copy()
        for roi in event["regions"]:
            x0,y0,x1,y1=roi["native_bbox_xyxy"]
            cv2.rectangle(right,(x0,y0),(x1,y1),(0,255,255),2)
        width,height=dimensions
        ph=max(1,round(height*320/width))
        panel=np.concatenate([cv2.resize(left,(320,ph)),cv2.resize(right,(320,ph))],axis=1)
        header=np.full((30,640,3),255,np.uint8)
        cv2.putText(header,f"frames {event['previous_frame']} -> {event['frame_index']}; "
                    f"local z={event['peak_robust_z']:.1f}; review only",
                    (8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1,cv2.LINE_AA)
        panels.append(np.concatenate([header,panel],axis=0))
    cv2.imwrite(str(output/"candidate_sheet.jpg"),np.concatenate(panels,axis=0))
    return evidence


def diagnose(video,output,config=Config()):
    video=Path(video).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    pts=probe_pts(video);median_dt=float(np.median(np.diff(pts)))
    boundary={i for i in range(1,len(pts))
              if pts[i]-pts[0]<=config.boundary_seconds
              or pts[-1]-pts[i]<=config.boundary_seconds}
    cap=cv2.VideoCapture(str(video))
    previous=None;energies=[];rows=[];maps={};native_dims=None;proc_dims=None
    hashes={};index=0
    while True:
        ok,native=cap.read()
        if not ok:break
        if index>=len(pts):raise ValueError("Decode/PTS count mismatch")
        if native_dims is None:
            h,w=native.shape[:2];native_dims=(w,h)
            scale=min(1.,config.max_side/max(w,h))
            proc_dims=(round(w*scale),round(h*scale))
        frame=cv2.resize(native,proc_dims,interpolation=cv2.INTER_AREA)
        hashes[index]=pixel_hash(native)
        if previous is not None:
            residual,alignment=aligned_residual(previous,frame,config)
            dt=float(pts[index]-pts[index-1])
            normalized=residual*(median_dt/dt)
            energies.append(tile_energy(normalized,config))
            rows.append({"previous_frame":index-1,"frame_index":index,
                         "pts_seconds":float(pts[index]),"previous_pts_seconds":float(pts[index-1]),
                         "dt_seconds":dt,"boundary":index in boundary,
                         "global_mean_residual":float(normalized.mean()),
                         "alignment":alignment})
            if index in boundary:maps[index]=normalized
        previous=frame;index+=1
    cap.release()
    if index!=len(pts):raise ValueError("Decode/PTS count mismatch")
    scores=temporal_scores(energies,config)
    events=[]
    for row,energy,score in zip(rows,energies,scores):
        row["max_tile_energy"]=float(energy.max())
        row["score_status"]="insufficient_temporal_context" if score is None else "computed"
        if score is None:continue
        row["peak_robust_z"]=float(score["robust_z"].max())
        row["max_energy_ratio"]=float(score["energy_ratio"].max())
        row["candidate_tile_count"]=int(score["candidate_tiles"].sum())
        row["context_frame_indices"]=[i+1 for i in score["context_indices"]]
        if not row["boundary"]:continue
        rois=regions(maps[row["frame_index"]],score,config)
        for roi in rois:
            x0,y0,x1,y1=roi["processing_bbox_xyxy"]
            w,h=proc_dims;nw,nh=native_dims
            roi["normalized_xyxy"]=[x0/w,y0/h,x1/w,y1/h]
            roi["native_bbox_xyxy"]=[int(np.floor(x0*nw/w)),int(np.floor(y0*nh/h)),
                                      int(np.ceil(x1*nw/w)),int(np.ceil(y1*nh/h))]
        row["region_count"]=len(rois)
        if rois:
            events.append({**row,"regions":rois,
                "previous_decoded_BgR_hash":hashes[row["previous_frame"]],
                "current_decoded_BgR_hash":hashes[row["frame_index"]],
                "task_failure":None,"physical_impossibility":None})
    evidence=export_evidence(video,events,output,native_dims)
    report={
        "schema":"generic_local_boundary_discontinuity_v1",
        "source":{"path":str(video),"sha256":file_sha256(video),"frames":index,
                  "native_dimensions_wh":list(native_dims),"processing_dimensions_wh":list(proc_dims),
                  "PTS_source":"ffprobe best_effort_timestamp_time, verified against continuous decode"},
        "config":asdict(config),"config_development":"posthoc development aware of one reported end-frame issue; "
            "single frozen parameter set for all six clips; no held-out calibration",
        "boundary_transition_indices":sorted(boundary),
        "all_adjacent_transitions_evaluated":len(rows),
        "candidate_transition_count":len(events),"candidates":events,
        "transitions":rows,"lossless_frame_evidence":evidence,
        "elapsed_cpu_seconds":time.perf_counter()-started,
        "diagnostic_status":"review_candidates" if events else "none_detected_not_a_pass",
        "task_outcome":"unknown","training_admitted":False,
        "automatic_approval":False,
        "limitations":[
            "Local residual and robust z are heuristic scores, not calibrated failure probabilities.",
            "Normal abrupt motion, occlusion boundaries, lighting or camera-fit errors may be false alerts.",
            "Small low-contrast changes, moving backgrounds and sustained wrong states may be missed.",
            "The affine alignment is a background-motion proxy, without semantic segmentation.",
            "Only start/end review candidates are exported; interior scores are retained but not semantic QA.",
            "No claim of object identity, action intention, contact, task failure or physical impossibility.",
        ],
    }
    write_json(output/"report.json",report)
    np.savez_compressed(output/"tile_energies.npz",energies=np.asarray(energies),
                        frame_indices=np.arange(1,index),pts_seconds=pts[1:])
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    cv2.setNumThreads(2)
    report=diagnose(args.video,args.output)
    print(json.dumps({"source":report["source"]["path"],
                      "candidate_frames":[e["frame_index"] for e in report["candidates"]],
                      "seconds":report["elapsed_cpu_seconds"],"training_admitted":False}))


if __name__=="__main__":main()
