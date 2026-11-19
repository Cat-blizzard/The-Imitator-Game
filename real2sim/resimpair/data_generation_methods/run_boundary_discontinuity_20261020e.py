"""One frozen-parameter CPU batch for the six existing generated candidates."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import json
from pathlib import Path
import time
import cv2

from data_generation_methods.r2h_pipeline import boundary_discontinuity as module
from data_generation_methods.r2h_pipeline.common import file_sha256,write_json


ROOT=Path("/mnt/sdg/datasets/real_sim_human_pipeline/controlled_20261020e")


def worker(case,video,output):
    cv2.setNumThreads(2)
    result=module.diagnose(video,output)
    return {"case":case,"video":str(video),"video_sha256":file_sha256(video),
            "frames":result["source"]["frames"],
            "candidate_frames":[r["frame_index"] for r in result["candidates"]],
            "report":str(output/"report.json"),"seconds":result["elapsed_cpu_seconds"]}


def main():
    output=ROOT/"boundary_diagnostics"
    output.mkdir(exist_ok=False)
    source=Path(module.__file__)
    (output/"detector_source.py").write_bytes(source.read_bytes())
    cases=[(f"{task}_{arm}",ROOT/task/arm/"attempt_01/candidate.mp4")
           for task in ("apple","book","cabinet") for arm in ("A","B")]
    write_json(output/"development_protocol.json",{
        "config":asdict(module.Config()),"detector_source_sha256":file_sha256(source),
        "known_before_development":"Operator reported one apple/B terminal-frame back-jump and apple/A control.",
        "not_blind_or_held_out":True,
        "threshold_policy":"One parameter set frozen before reading real-video score outputs; do not retune to these six.",
        "input_policy":"Only RGB video and its native timestamps; no task text, ROI, colour or reported frame index.",
        "planned_cases":[{"case":c,"video":str(v),"exists":v.exists()} for c,v in cases],
        "review_labels_are_not_detector_inputs":True,"new_api_calls":0,"new_gpu_jobs":0,
    })
    rows=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(worker,c,v,output/c):c for c,v in cases if v.exists()}
        for future in as_completed(jobs):
            case=jobs[future]
            try:row=future.result()
            except Exception as exc:row={"case":case,"error":repr(exc)}
            rows.append(row)
            write_json(output/f"batch_progress_{len(rows):04d}.json",
                       {"completed":rows,"seconds":time.perf_counter()-started})
            print(json.dumps(row),flush=True)
    write_json(output/"batch_summary.json",{"cases":sorted(rows,key=lambda r:r["case"]),
        "seconds":time.perf_counter()-started,"automatic_approval":False,
        "task_success_not_inferred":True,"new_api_calls":0,"new_gpu_jobs":0})


if __name__=="__main__":main()
