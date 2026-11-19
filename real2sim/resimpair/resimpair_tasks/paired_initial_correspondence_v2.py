"""One bounded paired-image diagnosis; retains the original full-video review.

Uses the human_examples_v2 frozen transport and same cumulative ledger. This is
an explicitly versioned local comparison, never a default-rule promotion.
"""
from contextlib import ExitStack
import argparse
import json
from pathlib import Path
import time

from resimpair_tasks import human_examples_v2 as parent
from corl2026_workshop.experiments.autonomous_pipeline_v4.core import bind, checked, read, require, write, write_bytes

VERSION="paired_initial_correspondence_v2"
CHECKS={
    "initial_object_correspondence":"The visible scene objects in generated frame0 correspond to the human anchor and simulation initial view, excluding the intended robot-to-one-person replacement.",
    "no_added_visible_instances":"Generated frame0 has no additional visible scene-object instances absent from both initial reference views. Naming a plausible real-world object does not justify adding one.",
    "no_missing_visible_instances":"Visible non-robot objects from the initial reference views are not missing in generated frame0; distinguish occlusion and uncertainty from deletion.",
    "relative_size_correspondence":"Object sizes relative to the table and other objects remain compatible at generated frame0; distinguish camera framing differences from object rescaling.",
    "initial_layout_correspondence":"Initial visible object placement and object-object spatial relations remain compatible with the human anchor and simulation initial view."
}


def obj(properties):
    return {"type":"object","additionalProperties":False,"properties":properties,"required":list(properties)}


def schema():
    evidence=obj({"image_label":{"type":"string","enum":["A","B","C","D"]},
        "bbox_xyxy_normalized":{"type":"array","items":{"type":"number","minimum":0,"maximum":1},"minItems":4,"maxItems":4},
        "observation":{"type":"string","minLength":1}})
    return obj({"checks":{"type":"array","minItems":len(CHECKS),"maxItems":len(CHECKS),"items":obj({
        "check_id":{"type":"string","enum":list(CHECKS)},"value":{"type":"string","enum":["yes","no","unknown"]},
        "evidence":{"type":"array","items":evidence},"reason":{"type":"string","minLength":1}})},
        "object_correspondences":{"type":"array","items":obj({"description":{"type":"string","minLength":1},
            "status":{"type":"string","enum":["matched","added","missing","rescaled","layout_changed","unknown"]},
            "evidence":{"type":"array","items":evidence}})},
        "limitations":{"type":"array","minItems":1,"items":{"type":"string","minLength":1}}})


def validate_observation(value):
    from jsonschema import validate
    validate(value,schema())
    require([r['check_id'] for r in value['checks']] and {r['check_id'] for r in value['checks']}==set(CHECKS),"exact_checks_required")
    for row in value['checks']:
        labels={e['image_label'] for e in row['evidence']}
        require(row['value']=='unknown' or ('C' in labels and labels & {'A','B'}),"paired_reference_and_frame0_evidence_required")
    for row in value['checks']+value['object_correspondences']:
        for e in row['evidence']:
            x0,y0,x1,y1=e['bbox_xyxy_normalized']
            require(x0<x1 and y0<y1,"ordered_nonempty_pixel_box_required")
    return {"status":"fail" if any(r['value']=='no' for r in value['checks']) else
            "unknown" if any(r['value']=='unknown' for r in value['checks']) else "pass",
            "failed_checks":[r['check_id'] for r in value['checks'] if r['value']=='no'],
            "unknown_checks":[r['check_id'] for r in value['checks'] if r['value']=='unknown'],
            "scope":"Four-image initial correspondence diagnosis, not a full-video acceptance.","default_pipeline_promoted":False}


def folder(root): return Path(root).resolve()/"diagnostics"/VERSION


def prepare(root):
    root=Path(root).resolve(); module,index,containment,h=parent.validate(root); d=folder(root)
    require(not d.exists(),"diagnostic_already_prepared_never_reset")
    row=index['records'][0]; contract=read(checked(row['contract'])); v=root/"runs"/row['key']/"motion_reference"/parent.VERSION
    prior=read(v/'review.json'); packet=read(v/'review_frames/packet.json')
    generated0=next(f['image'] for f in packet['frames'] if f['frame_index']==0)
    generated24=next(f['image'] for f in packet['frames'] if f['frame_index']==24)
    images=[{'label':'A: REFERENCE simulation initial view; robot removal/replacement is allowed.','image':contract['initial_image']},
            {'label':'B: REFERENCE approved human initial anchor; intended starting object layout.','image':bind(Path(row['anchor_directory'])/'human.png')},
            {'label':'C: CANDIDATE generated video frame0, before the new task events.','image':generated0},
            {'label':'D: CANDIDATE generated video frame24, continuity/context only; task motion may already have started.','image':generated24}]
    prompt=("Compare the actual pixels in the four separately labelled images. This is an initial-scene correspondence check, not a task-success review. "
        "A and B are references; C and D are generated candidates. Ignore the intended replacement of robots by one human, and natural differences in human anatomy. "
        "Compare every visible scene object, including non-targets and foreground/background clutter. Do not rely on category expectations or imagine objects outside the reference view. "
        "D only helps show instance persistence; do not penalize legitimate manipulation after frame0. Evaluate these fixed checks: "+json.dumps(CHECKS)+
        " For each check answer yes when correspondence is supported, no when visibly contradicted, unknown if unresolved. "
        "Cite paired pixel evidence from C and A or B, using normalized [x0,y0,x1,y1] boxes in the individual images. "
        "For an added or missing object, also cite its expected/reference region where the instance is absent, explaining the absence. "
        "Return object_correspondences with useful visual descriptions and status, and explicit limitations. "
        "Do not infer 3D poses, force, mass, or hidden identities from RGB. Return only the schema JSON.")
    request={"schema":VERSION,"implementation":module('core').implementation(),"model_config":bind(root/'config.json'),
        "prompt":prompt,"response_schema":schema(),"images":images,
        "wire_schema_adapter":module('review_transport').WIRE_SCHEMA_ADAPTER,
        "paired_native_contract":row['contract']}
    write(d/'request.json',request);write(d/'response_schema.json',schema())
    write_bytes(d/'frozen_driver.py',Path(__file__).read_bytes())
    write(d/'protocol.json',{"schema":VERSION,"request":bind(d/'request.json'),"driver":bind(d/'frozen_driver.py'),
        "parent_protocol":bind(root/'protocol.json'),"budget":bind(root/'budget.json'),"old_review":bind(v/'review.json'),
        "old_review_request":bind(v/'review_request.json'),"same_video":prior['candidate'],"same_anchor":images[1]['image'],
        "new_review_limit":1,"review_total_limit":6,"same_ledger":str(root/'campaign_ledger'),
        "original_video_review_unchanged":True,"default_pipeline_promoted":False,"diagnostic_only":True,
        "independent_pixel_observations_not_in_prompt":True,"new_unknown_stops":True,"automatic_retry":False})
    from jsonschema import Draft202012Validator
    Draft202012Validator.check_schema(schema())
    return {"status":"prepared","protocol":bind(d/'protocol.json'),"api_calls":0}


def run(root):
    root=Path(root).resolve();d=folder(root);module,index,c,h=parent.validate(root);protocol=read(d/'protocol.json')
    parent.pilot.check_bindings(protocol)
    require(protocol['driver']['sha256']==bind(__file__)['sha256'],"execute_frozen_diagnostic")
    result=read(root/'results.json')
    require(result['status']=='terminal' and len(result['rows'])==3
            and not any('unknown' in r['status'] for r in result['rows']),"complete_original_fixed3_without_unknown_first")
    module('core').activate()
    with ExitStack() as stack:
        for lock in [parent.ROOT/'runtime/.paid_containment.lock',root/'.paid_batch.lock']+[Path(p)/'.execution.lock' for p in h['ledgers']]:
            stack.enter_context(parent.exclusive(lock))
        if (d/'result.json').exists():return read(d/'result.json')
        budget=parent.guarded_budget(module,root,c)
        require(budget.usage()['new_started']['review']<=5,"one_remaining_review_required")
        start=time.time()
        try:
            observation,evidence=module('models').reviewed_json(root,budget,d/'request.json')
            gate=validate_observation(observation)
        except Exception as exc:
            write(d/'failure.json',{'exception_type':type(exc).__name__,'code':getattr(exc,'code','operation_failed'),
                'automatic_retry':False,'elapsed_seconds':time.time()-start})
            raise
        value={"schema":VERSION,"observations":observation,"gate":gate,"model_evidence":evidence,
            "elapsed_seconds":time.time()-start,"old_review":protocol['old_review'],"same_video":protocol['same_video'],
            "budget":protocol['budget'],"original_review_unchanged":True,"promoted":False,
            "comparison_limit":"Same failure case, narrower scope plus supplied references. A local diagnostic, not causal attribution or general validation."}
        write(d/'result.json',value)
        after=module('billing').snapshot(root,d/'credit_after.json')
        before=read(root/'billing/campaign_before.json')
        module('billing').token_logs(root,d/'provider_final_logs.json',before['started_unix'],after['finished_unix']+1)
        write(d/'usage_after.json',{"budget":bind(root/'budget.json'),**budget.usage(),
            "credit_after":bind(d/'credit_after.json'),"provider_logs":bind(d/'provider_final_logs.json'),
            "old_unknown_preserved":bind(root/'containment.json')})
        return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();print(json.dumps((prepare if a.command=='prepare' else run)(a.root),ensure_ascii=False,indent=2))
