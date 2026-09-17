"""Four saved-prefix no_grad zeros, read-only original evidence and no media."""
from __future__ import annotations
import argparse
import gc
import json
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from main.tube_state import projection_margin as carrier, velocity_coefficients as method
from runtime.wan.generation import prepare_generation
from runtime.wan.velocity_direction import tail
from runtime.wan.zero_restore import restore_snapshot,validate_snapshot
from runtime.wan.flow_step import measures
from runtime.wan.io import dump
from .velocity_direction_run import endpoint_metrics

ORIGINAL_RUN='velocity_calibration_20260917T011448482341Z'
ORIGINAL_SOURCE='5db00fa60e1a3cad88b6736e1a402381afad3623'
CASES=('dev_p0_s0','dev_p0_s1','dev_p1_s0','dev_p1_s1')
FLOW_NAMES=tuple(f'FLOW_r{i:02d}_{s}' for i in range(3) for s in ('A','B'))
PLAN={'transformer':12,'scheduler_step':6,'shadow_step':6,'backward':0,
      'prefix_transformer':0,'response_probe_step':0,'vae_decode':0,'vae_encode':0,'mp4_save':0}


def load(path):return json.loads(Path(path).read_text())


def check_paths(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if source.name!=ORIGINAL_RUN:raise ValueError('only the explicitly fixed original run is supported')
    if output==source or source in output.parents or output in source.parents:
        raise ValueError('new output must be separate from the read-only original run')
    return source,output


def environment():
    import diffusers,transformers
    return {'python':platform.python_version(),'torch':str(torch.__version__),'diffusers':diffusers.__version__,
        'transformers':transformers.__version__,'cuda':torch.version.cuda,
        'device':torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}


def preflight(source,case_id):
    manifest=load(source/'manifest.json')
    if tuple(case['id'] for case in manifest['development'])!=CASES or manifest['strengths']!=[.1,.3,1.]:
        raise ValueError('original fixed roster or saved strength list differs')
    case=next(case for case in manifest['development'] if case['id']==case_id)
    directory=source/'dev'/case_id/'terminal';config=load(directory/'config.json');record=load(directory/'result.json')
    if record['source_commit']!=ORIGINAL_SOURCE or record['source_dirty'] is not False:
        raise ValueError('original terminal must identify the clean published source5db00fa')
    expected=dict(manifest['base_config']['generation'],prompt=case['prompt'],seed=case['seed'])
    if any(value!=config['generation'].get(key) for key,value in expected.items() if key!='role'):
        raise ValueError('case conditioning/generation differs from fixed original manifest')
    if config['model']!=manifest['base_config']['model'] or config['key_utf8']!=manifest['base_config']['key_utf8']:
        raise ValueError('original model/key differs from fixed manifest')
    current=environment()
    for key in ('torch','diffusers'):
        if current[key]!=record['environment'][key]:
            raise ValueError(f'original {key}={record["environment"][key]}, current={current[key]}; restore original software before continuation, no prefix fallback')
    # This is the user's own saved scheduler object, not an arbitrary external pickle.
    payload=torch.load(directory/'pre_intervention_state.pt',map_location='cpu',weights_only=False)
    validate_snapshot(payload,record['scheduler'],carrier.SHAPE)
    with np.load(directory/'codebook.npz',allow_pickle=False) as archive:book={key:archive[key].copy() for key in archive.files}
    if book['directions'].shape!=(1760,1024) or book['codes'].shape!=(2,1760) or book['directions'].dtype!=np.float32:
        raise ValueError('saved codebook has incompatible geometry/dtype')
    if not np.isfinite(book['directions']).all() or not np.isfinite(book['codes']).all():raise ValueError('saved codebook nonfinite')
    return directory,config,record,payload,book,current


def optional_json(path):
    try:return {'status':'PRESERVED_ORIGINAL','path':str(path),'value':load(path)}
    except (OSError,ValueError) as exc:return {'status':'MISSING_OR_UNREADABLE_ORIGINAL','path':str(path),'error':repr(exc)}


def compare_terminals(zero,directory,book,record):
    directions=torch.from_numpy(book['directions']);codes=torch.from_numpy(book['codes'])
    baseline=endpoint_metrics(zero,directions,codes)
    report={'Z0_metrics':baseline,'gradient_zeros':{},'flow':{},
        'interpretation':'conditioning was reconstructed; Z0-Zg cannot isolate gradient-path causality or establish a bug; no Z0 media baseline exists'}
    for name in ('ZERO_A','ZERO_B',*FLOW_NAMES):
        group=report['gradient_zeros'] if name.startswith('ZERO') else report['flow']
        row={'status':'NOT_READ','path':str(directory/(name+'_terminal.pt')),
             'original_record':record.get('conditions',{}).get(name,{'status':'MISSING'})}
        group[name]=row
        try:
            value=torch.load(directory/(name+'_terminal.pt'),map_location='cpu',weights_only=True).float()
            if tuple(value.shape)!=carrier.SHAPE or not torch.isfinite(value).all():raise ValueError('saved terminal invalid')
            # Subtract tensors BEFORE RMS. Never subtract their scalar RMS values.
            row['tensor_difference_rms']=measures(zero-value if name.startswith('ZERO') else value-zero)
            row['difference_definition']='Z0-Zg' if name.startswith('ZERO') else 'Zrho-Z0'
            current=endpoint_metrics(value,directions,codes);m=0 if name.endswith('A') else 1
            signed=np.asarray(book['codes'][m],dtype=np.float64)
            gain=float(np.mean(signed*(np.asarray(current['projection'])-np.asarray(baseline['projection']))))
            margin=current['signed_projection_mean_by_message'][m]-current['signed_projection_mean_by_message'][1-m]
            base_margin=baseline['signed_projection_mean_by_message'][m]-baseline['signed_projection_mean_by_message'][1-m]
            loss0=baseline['loss_by_message'][m];loss=current['loss_by_message'][m]
            row.update(status='MEASURED',loss=loss,Z0_loss=loss0,loss_decrease_vs_Z0=loss0-loss,
                relative_loss_decrease_vs_Z0=(loss0-loss)/loss0 if loss0 else None,
                correct_code_signed_projection_gain_vs_Z0=gain,
                recomputed_absolute_competition_margin=margin,Z0_competition_margin=base_margin,
                competition_margin_gain_vs_Z0=margin-base_margin)
        except Exception as exc:row.update(status='MISSING_OR_FAILED',error=repr(exc),traceback=traceback.format_exc())
    return report


def run_case(source,output,case_id):
    source,output=check_paths(source,output)
    if case_id not in CASES:raise ValueError('case outside fixed roster')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    result={'status':'RUNNING','case':case_id,'original_run':str(source),'original_source':ORIGINAL_SOURCE,
        'new_output':str(output),'fixed_calls':PLAN,'actual_calls':{k+'_'+s:0 for k in PLAN for s in ('attempted','completed')},
        'failures':[],'steps':[],'restoration':None,'comparison':None,
        'original_media':optional_json(source/'dev'/case_id/'media/result.json'),
        'evidence_ceiling':'one restored no_grad zero; no path-only/codec attribution, selection update, threshold or scientific PASS'}
    def save():result['elapsed_seconds']=time.monotonic()-started;dump(output/'result.json',result)
    def count(kind,done):
        key=kind+('_completed' if done else '_attempted')
        result['actual_calls'][key]=result['actual_calls'].get(key,0)+1;save()
    pipe=prefix=snapshot=prompt=negative=initial=zero=coefficients=directions=None
    save()
    try:
        directory,config,record,payload,book,current=preflight(source,case_id)
        result['environment']=current;result['original_environment']=record['environment']
        dump(output/'original_config.json',config)
        result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        result['source_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
        with torch.no_grad():
            pipe,initial,prompt,negative,dtype=prepare_generation(config,load_vae=False)
            initial=None  # Never continue the newly generated initial noise.
            if str(dtype)!=record['precision']['transformer_input'] or prompt.dtype!=torch.bfloat16 or negative.dtype!=torch.bfloat16:
                raise ValueError('loaded model conditioning precision differs from original BF16 path')
            if pipe.transformer.training or any(p.requires_grad for p in pipe.transformer.parameters()):raise ValueError('expected frozen eval transformer')
            device=next(pipe.transformer.parameters()).device
            prefix,snapshot,restoration=restore_snapshot(payload,pipe.scheduler,device,record['scheduler'],carrier.SHAPE)
            result['restoration']=restoration
            result['conditioning']={'model':config['model'],'resolved_transformer_commit':getattr(pipe.transformer.config,'_commit_hash',None),
                'prompt_shape':list(prompt.shape),'negative_shape':list(negative.shape),'dtype':str(prompt.dtype),
                'original_embeddings_saved':False,'original_resolved_model_revision_saved':config['model'].get('revision') is not None,
                'note':'same recorded prompts/model identifier/software; original weight/tokenizer/embedding identity not provable from saved artifacts'}
            payload=None;save()
            directions=torch.from_numpy(book['directions']).to(device)
            coefficients=torch.zeros(method.COEFFICIENT_SHAPE,device=device,dtype=torch.float32)
            def record_step(row,arrays):result['steps'].append(row);save()
            zero=tail(pipe,snapshot,prefix,prompt,negative,dtype,config['generation']['guidance_scale'],
                coefficients,directions,record.get('R'),count,record_step,use_checkpoint=True,responses=record['responses'])
            zero=zero.detach().cpu().float()
            torch.save(zero,output/'Z0_terminal.pt')
        result['comparison']=compare_terminals(zero,directory,book,record)
        result['status']='MEASURED_REQUIRES_REVIEW' if all(row['status']=='MEASURED' for group in ('gradient_zeros','flow') for row in result['comparison'][group].values()) else 'MEASURED_WITH_MISSING_OR_FAILED_ORIGINALS'
    except Exception as exc:
        result['status']='FAILED_RESTORATION_OR_CONTINUATION'
        result['failures'].append({'error':repr(exc),'traceback':traceback.format_exc(),
            'minimal_alternative':'restore the named missing original artifact/software/conditioning input; no automatic prefix, gradient, media or baseline rerun'})
    finally:
        result['resources']={'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None}
        pipe=prefix=snapshot=prompt=negative=initial=zero=coefficients=directions=None
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        save()
    return result


def run_all(source,output):
    source,output=check_paths(source,output);output.mkdir(parents=True,exist_ok=False)
    result={'status':'RUNNING','case_denominator':4,'cases':{name:{'status':'NOT_RUN'} for name in CASES},
        'fixed_total_calls':{key:4*value for key,value in PLAN.items()},
        'original_selection':optional_json(source/'selection.json'),
        'original_terminal_selection':optional_json(source/'terminal_selection.json'),
        'original_run':str(source),'original_data_modified':False,'selection_recomputed':False,'media_executed':False}
    dump(output/'result.json',result)
    for case_id in CASES:
        log=output/(case_id+'.log')
        print(case_id,'one saved-prefix no_grad zero; log:',log,flush=True)
        try:
            with log.open('w') as stream:
                process=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.zero_path_run',
                    '--input-run',str(source),'--output',str(output/case_id),'--case-id',case_id],stdout=stream,stderr=subprocess.STDOUT,check=False)
            row=load(output/case_id/'result.json');row['exit_code']=process.returncode
            result['cases'][case_id]=row
        except Exception as exc:result['cases'][case_id]={'status':'FAILED_LAUNCH_OR_RESULT','error':repr(exc),'traceback':traceback.format_exc()}
        dump(output/'result.json',result)
    result['actual_calls']={key:sum(row.get('actual_calls',{}).get(key,0) for row in result['cases'].values()) for key in (kind+'_'+state for kind in PLAN for state in ('attempted','completed'))}
    result['status']='MEASURED_REQUIRES_REVIEW' if all(row['status']=='MEASURED_REQUIRES_REVIEW' for row in result['cases'].values()) else 'WITH_RETAINED_FAILURES_OR_MISSING'
    dump(output/'result.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case-id',choices=CASES)
    args=parser.parse_args()
    result=run_case(args.input_run,args.output,args.case_id) if args.case_id else run_all(args.input_run,args.output)
    print(result['status'],flush=True)
    if result['status']!='MEASURED_REQUIRES_REVIEW':raise SystemExit(1)


if __name__=='__main__':main()
