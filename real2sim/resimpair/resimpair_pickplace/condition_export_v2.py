"""Versioned metadata bridge to the retained R2H compiler; no new semantics.

The new task registrar omitted the legacy semantic_key field. Recover it from
the immutable pre-execution task, never edit the original execution bundle.
"""
import argparse
import json
from pathlib import Path
from .evaluator import evaluate
from resimpair_grasp.common import read,write,binding,digest


def export(directory):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.generic_compiler import compile_bundle
    from corl2026_workshop.experiments.autonomous_pipeline_v4.core import checked,bind
    report=evaluate(directory)
    if not report['passed']:raise ValueError('Native and container acceptance required before R2H conditioning')
    original=directory/'execution_bundle.json';bundle=read(original)
    requested_acceptance=None
    if 'source_request' in bundle['case']:
        from .requested_goal_audit import evaluate as evaluate_requested
        if not evaluate_requested(directory,bundle['case'])['passed']:
            raise ValueError('Original requested metric goal acceptance required before R2H conditioning')
        requested_acceptance=binding(directory/'requested_goal_audit.json')
    intent=read(checked(bundle['case']['task']))
    if intent!=bundle['task_intent']:raise ValueError('Execution intent differs from pre-execution input')
    key=digest(intent)
    if bundle['case'].get('semantic_key',key)!=key:raise ValueError('Inconsistent semantic key')
    implementation=binding(__file__)
    output=directory/'compiled'/('bridge_'+implementation['sha256'][:12])
    original_declaration=read(checked(bundle['planning_declaration']))
    if original_declaration['case']!=bundle['case'] or original_declaration['event_protocol']!=bundle['event_protocol']:
        raise ValueError('Original pre-execution declaration does not match original bundle')
    if original_declaration['scope']!='before_physical_execution':raise ValueError('Original protocol was not predeclared')
    new_case={**bundle['case'],'semantic_key':key}
    declaration={**original_declaration,'case':new_case,
        'original_planning_declaration':bundle['planning_declaration'],
        'metadata_only_derivative':True,'protocol_content_and_original_declaration_time_unchanged':True,
        'bridge_implementation':implementation}
    declaration_path=output/'planning_declaration.json';write(declaration_path,declaration)
    derived={**bundle,'case':new_case,'planning_declaration':bind(declaration_path),
        'motion_planning_agent_participation':False,'original_execution_bundle':binding(original),
        'metadata_bridge':{'implementation':binding(__file__),'changes':['semantic_key = SHA256 of existing pre-execution intent',
            'planning participation = automatic shared controller; no per-task agent waypoint input'],
            'physics_actions_states_and_criteria_unchanged':True}}
    bridge=output/'execution_bundle_v5.json';write(bridge,derived)
    mapping={'correspondence':'synthetic_only','source_status':'missing','formal_pair_qualified':False,
             'scope':'new role/asset/layout composition; no real-source equivalence inferred'}
    value=compile_bundle(bridge,mapping)
    from resimpair_tabletop.sample_report import native_scalars
    path=output/'contract.json';write(path,native_scalars(value))
    write(output/'general_pickplace_provenance.json',{'container_acceptance':binding(directory/'container_audit.json'),
        'requested_goal_acceptance':requested_acceptance,
        'contract':binding(path),'bridge':binding(bridge),'human_generation_started':False,'api_calls':0})
    return {'contract':str(path),'api_calls':0,'human_generation_started':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.execution),indent=2))
