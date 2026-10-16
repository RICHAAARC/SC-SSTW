"""Read saved P48 gradients and the single CPU solve receipt; no solver/model rerun.

The adopted fixed-target CPU attempt did not resolve a candidate. This delivery
therefore performs only the continuation report stage and spends zero GPU calls.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time
from main.tube_state import local_joint_p48_secondary_v1 as method
from main.tube_state import local_joint_p48_joint_control_v1 as original
from main.tube_state.local_joint_p0_input_vjp_v1 import finite_tensor,LATENT_SHAPE
from runtime.wan.local_joint_p0_input_vjp_v1 import tensor,write_json
from runtime.wan.local_joint_p48_journal_v1 import check_json
from runtime.wan.local_joint_state_payload_provider_v1 import PUBLIC_LATENT_SUPPORT

FUTURE_PHYSICAL_BUDGET=dict(transformer_prepare=1,vae_load=1,dit_ordinary_forward=2,native_update=2,
    ordinary_decode=1,dit_gradient_forward=0,dit_input_vjp=0,gradient_decode=0,decoder_vjp=0,
    original_DiT_blocks=60,DiT_recompute=0,vae_original_chunks=46,vae_recompute=0)


def run(source_run, output, decision_path, *, shape=LATENT_SHAPE,support=PUBLIC_LATENT_SUPPORT):
    import torch
    source_run,output,decision_path=map(Path,(source_run,output,decision_path))
    if output.resolve()==source_run.resolve():raise ValueError('continuation output must be separate from original run')
    previous=None
    if (output/'result.json').exists():previous=json.loads((output/'result.json').read_text())
    result=dict(status='READING_SAVED_EVIDENCE',scientific_pass=False,source_run=str(source_run),
        calls=[],resources=[],failures=copy.deepcopy(previous.get('failures',[])) if previous else [],
        actual_model_execution='NOT_STARTED',automatic_retry=False,primary_solve_count=0,secondary_solve_count=0,
        cpu_validation_solve_count=1,created_at_unix=time.time(),endpoints={n:dict(status='MISSING',reason='not_read') for n in original.VIEWS},
        objectives={n:dict(status='MISSING',reason='not_read') for n in original.OBJECTIVES},
        physical_budget_if_qualified_candidate=copy.deepcopy(FUTURE_PHYSICAL_BUDGET),actual_new_stage_calls=0,
        resume_policy='Re-read saved inputs and the same CPU decision only; no solver, model, gradient or physical retry.',
        ceiling=original.CEILING)
    write_json(output/'result.json',result)
    try:
        old=json.loads((source_run/'result.json').read_text())
        saved_solution=check_json(source_run/'solution.json','solution')
        result['original_result']=copy.deepcopy(old)
        result['original_solution']=copy.deepcopy(saved_solution)
        result['fixed_denominators']=copy.deepcopy(old['fixed_denominators'])
        for n in ('ZERO','ANCHOR'):
            values=check_json(source_run/n/'values.json','values')
            result['endpoints'][n]=dict(status=old['endpoints'][n]['status'],values=values,evidence_location=str(source_run/n))
        result['endpoints']['CANDIDATE']=dict(status='MISSING',reason='fixed secondary numerical result unresolved')
        anchor=finite_tensor(tensor(source_run/'anchor.pt'),shape,'saved anchor',torch.float32)
        gradients=[]
        for n in original.OBJECTIVES:
            g=finite_tensor(tensor(source_run/'objectives'/n/'control_cotangent.pt'),shape,'saved '+n+' control cotangent',torch.float32)
            gradients.append(g)
            result['objectives'][n]=dict(status='SAVED_GRADIENT_READ',evidence_location=str(source_run/'objectives'/n/'control_cotangent.pt'))
        # Persist the actual one-shot receipt before its diagnostic readback. Later
        # Run all uses this saved result, never selects a new packaged solution.
        decision_file=output/'cpu_solution.json'
        if not decision_file.exists():write_json(decision_file,json.loads(decision_path.read_text()))
        decision=json.loads(decision_file.read_text())
        candidate,checked=method.checked_candidate(anchor,gradients,saved_solution,decision,support)
        result['solution']=checked
        write_json(output/'checked_solution.json',checked)
        if candidate is None:
            result['status']='NO_RESOLVED_CANDIDATE'
            result['endpoints']['CANDIDATE']['reason']=checked.get('reason','fixed secondary unresolved')
        else:
            # This frozen failed-attempt delivery has no physical execution branch.
            # A changed future decision cannot silently cause model work here.
            result['status']='QUALIFIED_CPU_WITNESS_NOT_EXECUTED'
            result['endpoints']['CANDIDATE']['reason']='this delivery contains no physical execution branch'
        result['comparison']=dict(status='CANDIDATE_NOT_MEASURED',scientific_pass=False,
            candidate_losses=None,candidate_half_energy_goals=[None]*8)
        result['original_call_count']=len(old['calls'])
        result['combined_physical_call_count']=len(old['calls'])
    except Exception as exc:
        result['status']='SAVED_EVIDENCE_ERROR'
        result['failures'].append(dict(stage='saved_evidence',reason=repr(exc),time_unix=time.time()))
    finally:
        result['finished_at_unix']=time.time()
        write_json(output/'result.json',result);write_json(output/'report.json',result)
    return result


def finalize_interrupted(output,reason):
    p=Path(output)/'result.json'
    if not p.exists():return None
    r=json.loads(p.read_text())
    if r['status']=='READING_SAVED_EVIDENCE':
        r['status']='READ_INTERRUPTED'
        r['failures'].append(dict(stage='saved_evidence',reason=reason,time_unix=time.time()))
        write_json(p,r);write_json(Path(output)/'report.json',r)
    return r
