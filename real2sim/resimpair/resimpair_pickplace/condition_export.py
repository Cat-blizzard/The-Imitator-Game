"""Export only independently accepted new scenes to the existing R2H compiler."""
import argparse
import json
from pathlib import Path
from .evaluator import evaluate
from resimpair_grasp.common import read,write,binding


def export(directory):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.generic_compiler import compile_bundle
    report=evaluate(directory)
    if not report['passed']:raise ValueError('Native and container acceptance required before R2H conditioning')
    case=read(directory/'execution_bundle.json')['case'];requested_acceptance=None
    if 'source_request' in case:
        from .requested_goal_audit import evaluate as evaluate_requested
        if not evaluate_requested(directory,case)['passed']:
            raise ValueError('Original requested metric goal acceptance required before R2H conditioning')
        requested_acceptance=binding(directory/'requested_goal_audit.json')
    mapping={'correspondence':'synthetic_only','source_status':'missing','formal_pair_qualified':False,
             'scope':'new role/asset/layout composition; no real-source equivalence inferred'}
    value=compile_bundle(directory/'execution_bundle.json',mapping)
    path=directory/'compiled/contract.json';write(path,value)
    write(directory/'compiled/general_pickplace_provenance.json',{'container_acceptance':binding(directory/'container_audit.json'),
        'requested_goal_acceptance':requested_acceptance,
        'contract':binding(path),'human_generation_started':False,'api_calls':0,
        'planning_agent_participation':'automatic shared skill; no agent supplies per-task motion waypoints',
        'legacy_bundle_field_note':'motion_planning_agent_participation is an inherited v4 label; this derivative states actual participation'})
    return {'contract':str(path),'api_calls':0,'human_generation_started':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.execution),indent=2))
