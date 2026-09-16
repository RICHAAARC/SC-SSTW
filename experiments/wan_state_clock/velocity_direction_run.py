"""One fixed symmetric direction diagnostic through the real terminal trajectory."""
from __future__ import annotations
import argparse
import gc
import json
import platform
import resource
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from main.tube_state import state_clock, projection_margin as carrier
from main.tube_state import velocity_coefficients as method
from runtime.wan.generation import prepare_generation
from runtime.wan.flow_generation import continue_steps
from runtime.wan.flow_step import measures
from runtime.wan.velocity_direction import tail, detached_scheduler, response_coefficients, BLOCK_CALL_KINDS
from runtime.wan.io import dump

CONDITIONS = ('ZERO_A','ZERO_B','PLUS_A','MINUS_A','PLUS_B','MINUS_B')


def endpoint_metrics(z, directions, codes):
    p=method.projections(z,directions).detach().cpu()
    signed=codes.detach().cpu().double()*p[None]
    losses=torch.relu(1.-signed).square().mean(dim=1)
    means=signed.mean(dim=1)
    clipped=signed.clamp(-1,1).mean(dim=1)
    return {'projection':p.tolist(),'loss_by_message':losses.tolist(),
        'signed_projection_mean_by_message':means.tolist(),
        'clipped_matched_score_by_message':clipped.tolist(),
        'message0_minus_message1_unclipped_margin':float(means[0]-means[1]),
        'message0_minus_message1_clipped_margin':float(clipped[0]-clipped[1]),
        'meaning':'actual terminal normalized-latent diagnostics, not MP4 or blind receiver evidence'}


def run(config,output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    fixed={'transformer':160,'scheduler_step':80,'shadow_step':36,'response_probe_step':6,
           'backward':2,'vae_decode':0,'vae_encode':0,'mp4_save':0}
    kinds=tuple(fixed)+('transformer_replay',)+BLOCK_CALL_KINDS
    result={'status':'RUNNING','conditions':{c:{'status':'NOT_RUN','steps':[],
        'calls':{k+'_'+s:0 for k in kinds for s in ('attempted','completed')},
        'elapsed_seconds':None,'resources':None,'endpoint':None,'loss':None} for c in CONDITIONS},
        'diagnostic_denominator':6,'formal_science_denominator':0,'fixed_calls':fixed,
        'call_plan':{'prefix_transformer':88,'six_tail_transformer':72,'two_zero_backwards':2,
                     'checkpoint_replay':'up to 20 outer pure-Transformer replays, possibly early-stopped; counted separately',
                     'block_checkpoint_calls':'gradient-bearing calls only: block original forward, outer reconstruction, and block replay are separate units; no solver replay',
                     'prefix_scheduler':44,'six_tail_scheduler':36,'budget_shadow':36,'scalar_response_probe':6},
        'actual_calls':{k+'_'+s:0 for k in kinds for s in ('attempted','completed')},
        'failures':[],'directions':{},'responses':None,'R':None,'zero_repeat_floor':None,
        'evidence_ceiling':'one symmetric finite step per message; BF16 AD is not the derivative of a discrete sampler',
        'amplitude_policy':'one budget-derived epsilon per message, same +/- coefficients, no clipping/search/retry',
        'artifact_paths':dict(config.get('artifact_paths',{}),result_directory=str(output.resolve()),
                              effective_config=str((output/'config.json').resolve()))}
    active='prefix'
    cuda_peaks={'allocated_bytes':0,'reserved_bytes':0}
    def save():
        result['elapsed_seconds']=time.monotonic()-started
        dump(output/'result.json',result)
    def count(kind,completed):
        key=kind+('_completed' if completed else '_attempted')
        result['actual_calls'][key]+=1
        if active in result['conditions']:
            calls=result['conditions'][active]['calls'];calls[key]=calls.get(key,0)+1
        if kind not in BLOCK_CALL_KINDS: save()
        if kind in ('backward','transformer_replay') or (kind=='transformer' and completed and result['actual_calls'][key]%12==0):
            print(active,kind,'completed' if completed else 'attempted',result['actual_calls'][key],flush=True)
    def failure(stage,exc):
        result['failures'].append({'stage':stage,'error':repr(exc),'traceback':''.join(traceback.format_exception(type(exc),exc,exc.__traceback__)),'classification':'ENGINEERING_OR_DIRECTION_UNAVAILABLE'})
        save()
    def resources():
        value={'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        if torch.cuda.is_available():
            value['cuda_current_allocated_bytes']=torch.cuda.memory_allocated()
            value['cuda_current_reserved_bytes']=torch.cuda.memory_reserved()
            for label,fn in (('allocated_bytes',torch.cuda.max_memory_allocated),('reserved_bytes',torch.cuda.max_memory_reserved)):
                value['cuda_stage_peak_'+label]=fn();cuda_peaks[label]=max(cuda_peaks[label],fn())
        return value
    def release():
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    dump(output/'config.json',config)
    save()
    pipe=snapshot=prefix=prompt=negative=directions=codes=None
    gradients={}
    def condition(name,coefficients,*,backward=False):
        nonlocal active
        active=name
        row=result['conditions'][name];row['status']='RUNNING'
        tick=time.monotonic()
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        m=0 if name.endswith('A') else 1
        endpoint=loss=gradient=None
        row['resource_snapshots']=[]
        def sample_resources(stage):
            row['resource_snapshots'].append(dict(stage=stage,**resources()))
            save()
        try:
            def record(step,arrays):
                row['steps'].append(step)
                torch.save(arrays,output/f'{name}_step{step["index"]}.pt')
                save()
            context=torch.enable_grad() if backward else torch.no_grad()
            with context:
                endpoint=tail(pipe,snapshot,prefix,prompt,negative,dtype,config['generation']['guidance_scale'],
                              coefficients,directions,result['R'],count,record,use_checkpoint=True,responses=result['responses'])
                loss=method.terminal_loss(endpoint,directions,codes[m])
                if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite actual terminal loss')
                cpu=endpoint.detach().cpu().float()
                torch.save(cpu,output/f'{name}_terminal.pt')
                torch.save(coefficients.detach().cpu(),output/f'{name}_coefficients.pt')
                row['actual_coefficient_l2']=float(torch.linalg.vector_norm(coefficients.detach().double()))
                if not backward:
                    requested=result['directions'][name[-1]]['epsilon']
                    row['requested_probe_epsilon']=requested
                    row['actual_coefficient_relative_norm_error']=(row['actual_coefficient_l2']-requested)/requested
                row['endpoint']=endpoint_metrics(cpu,directions.cpu(),codes.cpu())
                row['loss']=float(loss.detach())
                row['status']='FORWARD_COMPLETE'
                sample_resources('forward_complete')
                if backward:
                    sample_resources('before_backward')
                    count('backward',False)
                    gradient=torch.autograd.grad(loss,coefficients)[0]
                    count('backward',True)
                    sample_resources('after_backward')
                    if not bool(torch.isfinite(gradient).all()): raise FloatingPointError('nonfinite coefficient gradient')
                    torch.save(gradient.detach().cpu(),output/f'{name}_gradient.pt')
                    answer=gradient.detach().cpu()
                else:
                    answer=None
            row['status']='COMPLETE'
            return answer
        except Exception as exc:
            row['status']='FAILED_AFTER_FORWARD' if row.get('endpoint') else 'FAILED_FORWARD'
            sample_resources('failure_with_traceback_live')
            failure(name,exc)
            return None
        finally:
            endpoint=loss=gradient=None
            row['elapsed_seconds']=time.monotonic()-tick
            row['resources']=resources()
            release()
            sample_resources('condition_finally_after_graph_release')
    def after_condition(name):
        # Called after condition() and its exception frame have actually exited.
        result['conditions'][name]['resource_snapshots'].append(
            dict(stage='after_condition_return',**resources()))
        save()
    try:
        if config['generation']['steps']!=50: raise ValueError('fixed experiment requires 50 steps')
        source=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True)
        result['source_commit']=source.stdout.strip() if source.returncode==0 else None
        result['source_snapshot']=config.get('source_snapshot','local worktree')
        result['source_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()) if source.returncode==0 else None
        import diffusers
        result['environment']={'python':platform.python_version(),'torch':torch.__version__,'diffusers':diffusers.__version__,
            'cuda':torch.version.cuda,'device':torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        book=state_clock.codebook(config['key_utf8'].encode())
        np.savez(output/'codebook.npz',**book)
        pipe,prefix,prompt,negative,dtype=prepare_generation(config,load_vae=False)
        result['precision']={'transformer_input':str(dtype),'CFG':'original model-output arithmetic',
            'scheduler_input':'float32 after completed CFG','coefficient':'float32','projection_accumulation':'float64'}
        result['checkpoint']={'scope':'outer pure Transformer plus official inner Wan blocks','use_reentrant':False,
            'block_count':len(getattr(pipe.transformer,'blocks',())),
            'block_activation':'grad enabled and input requires_grad; scoped official enable_gradient_checkpointing, valid in eval with frozen parameters',
            'boundary_lifetime':'outer discards block saved inputs across tail calls; replay reconstructs current invocation boundaries, consumed by per-block backward',
            'scheduler_replayed':False,'no_VAE_loaded':pipe.vae is None,
            'memory_claim':'CPU equivalence and boundary lifetime only; current user GPU peak/feasibility remains unverified'}
        directions=torch.as_tensor(book['directions'],device=prefix.device)
        codes=torch.as_tensor(book['codes'],device=prefix.device)
        prefix=continue_steps(pipe,pipe.scheduler,prefix,prompt,negative,dtype,
                              config['generation']['guidance_scale'],0,44,count)
        snapshot=detached_scheduler(pipe.scheduler)
        torch.save({'latent':prefix.cpu(),'scheduler':detached_scheduler(snapshot)},output/'pre_intervention_state.pt')
        # Embeddings are newly obtained by the same prompt/seed configuration;
        # an old snapshot alone would not provide them.
        result['scheduler']={'class':type(snapshot).__name__,'config':dict(snapshot.config),'sigmas':snapshot.sigmas.tolist()}
        result['responses']=response_coefficients(snapshot,count)
        result['prefix_resources']=resources();save()
        for name in ('ZERO_A','ZERO_B'):
            a=torch.zeros(method.COEFFICIENT_SHAPE,device=prefix.device,requires_grad=True)
            gradients[name[-1]]=condition(name,a,backward=True)
            del a
            after_condition(name)
            if name=='ZERO_A' and (output/'ZERO_A_terminal.pt').exists():
                off=torch.load(output/'ZERO_A_terminal.pt',map_location='cpu',weights_only=True).numpy()
                radii=[None,None]
                for message in (0,1):
                    try:
                        marked,record=carrier.write(off,book,message)
                        value=measures(torch.from_numpy(marked-off))['support_rms']
                        dump(output/f'terminal_reference_{message}.json',record)
                        torch.save(torch.from_numpy(marked),output/f'terminal_reference_{message}.pt')
                        radii[message]=value
                    except Exception as exc:
                        failure('terminal_reference_'+str(message),exc)
                result['R']=max(radii) if all(v is not None for v in radii) else None
                result['terminal_reference_radii']=radii
                result['OFF_definition']='ZERO_A forward terminal; same-run existing terminal A/B writes fix shared R'
                del off
                save()
            release()
        for suffix in ('A','B'):
            active='direction_'+suffix
            q=None
            try:
                if gradients[suffix] is None: raise ValueError('own zero-control gradient unavailable')
                if result['R'] is None: raise ValueError('ZERO_A/OFF-derived radius unavailable')
                q,info=method.direction_and_amplitude(gradients[suffix],directions.cpu(),
                    [r['sigma'] for r in result['responses']],[r['h'] for r in result['responses']],result['R'])
                result['directions'][suffix]=info
                torch.save(q,output/f'{suffix}_q.pt')
                save()
            except Exception as exc:
                failure(active,exc)
            for sign,label in ((1,'PLUS'),(-1,'MINUS')):
                name=label+'_'+suffix
                if q is None:
                    result['conditions'][name].update(status='MISSING_DIRECTION',reason='own direction or shared radius unavailable')
                    save();continue
                a=(info['epsilon']*q*sign).to(prefix.device)
                condition(name,a)
                del a
                after_condition(name)
                release()
            q=None
    except Exception as exc:
        failure('setup_or_prefix',exc)
    finally:
        result['final_resources']=resources()
        pipe=snapshot=prefix=prompt=negative=directions=codes=None
        gradients.clear();release()
    # Reporting only; none of these observations select another epsilon or run.
    for name,row in result['conditions'].items():
        p=output/f'{name}_terminal.pt';zero=output/f'ZERO_{name[-1]}_terminal.pt';off=output/'ZERO_A_terminal.pt'
        if p.exists() and off.exists():
            value=torch.load(p,weights_only=True,map_location='cpu')
            reference=torch.load(off,weights_only=True,map_location='cpu')
            row['final_difference_vs_OFF']=measures(value-reference)
            del value,reference
        if row.get('endpoint') and result['conditions']['ZERO_'+name[-1]].get('endpoint'):
            m=0 if name.endswith('A') else 1
            base=result['conditions']['ZERO_'+name[-1]]['endpoint'];current=row['endpoint']
            delta=np.asarray(current['projection'])-np.asarray(base['projection'])
            sign=book['codes'][m]
            row['correct_code_signed_projection_gain']=float(np.mean(sign*delta))
            row['correct_code_signed_projection_gain_by_block']=(sign*delta).tolist()
            row['correct_minus_competitor_margin']=current['signed_projection_mean_by_message'][m]-current['signed_projection_mean_by_message'][1-m]
            row['competitor_margin_gain_vs_own_zero']=row['correct_minus_competitor_margin']-(base['signed_projection_mean_by_message'][m]-base['signed_projection_mean_by_message'][1-m])
    if all(result['conditions'][c].get('endpoint') for c in ('ZERO_A','ZERO_B')):
        a,b=(result['conditions'][c]['endpoint'] for c in ('ZERO_A','ZERO_B'))
        result['zero_repeat_floor']={'terminal_difference':result['conditions']['ZERO_B'].get('final_difference_vs_OFF'),
            'loss_difference_by_message':(np.asarray(b['loss_by_message'])-np.asarray(a['loss_by_message'])).tolist(),
            'projection_difference_rms':float(np.sqrt(np.mean((np.asarray(b['projection'])-np.asarray(a['projection']))**2))),
            'meaning':'two same-control forward observations only, not a distributional noise threshold'}
    result['direction_comparisons']={}
    for suffix in ('A','B'):
        info=result['directions'].get(suffix)
        plus,minus=(result['conditions'][s+'_'+suffix] for s in ('PLUS','MINUS'))
        comparison={'status':'MISSING_EVIDENCE'}
        if info and plus.get('endpoint') and minus.get('endpoint'):
            ap=torch.load(output/f'PLUS_{suffix}_coefficients.pt',weights_only=True,map_location='cpu').double()
            am=torch.load(output/f'MINUS_{suffix}_coefficients.pt',weights_only=True,map_location='cpu').double()
            q=torch.load(output/f'{suffix}_q.pt',weights_only=True,map_location='cpu').double()
            central=(plus['loss']-minus['loss'])/(2*info['epsilon'])
            ad=info['AD_directional_derivative']
            comparison={'status':'MEASURED_REQUIRES_INTERPRETATION','central_difference':central,'AD_derivative':ad,
                'central_minus_AD':central-ad,'central_to_AD_ratio':central/ad if ad else None,
                'sign_consistent':bool(np.sign(central)==np.sign(ad)),
                'plus_loss_minus_own_zero':plus['loss']-result['conditions']['ZERO_'+suffix]['loss'],
                'minus_loss_minus_own_zero':minus['loss']-result['conditions']['ZERO_'+suffix]['loss'],
                'actual_pair_antisymmetry_l2':float(torch.linalg.vector_norm(ap+am)),
                'plus_coefficient_rounding_l2':float(torch.linalg.vector_norm(ap-info['epsilon']*q)),
                'actual_plus_q_dot':float((ap*q).sum()),'actual_minus_q_dot':float((am*q).sum()),
                'scientific_pass':None,'claim':'finite BF16 direction evidence only; no curvature, MP4, or payload claim'}
        result['direction_comparisons'][suffix]=comparison
    result['over_budget_conditions']=[name for name,row in result['conditions'].items()
        if any(s['step_within_budget'] is False or s['sum_within_budget'] is False for s in row['steps'])]
    result['response_check_failed_conditions']=[name for name,row in result['conditions'].items()
        if any(s.get('response_valid') is False for s in row['steps'])]
    result['status']='EXECUTED_REQUIRES_DIRECTION_REVIEW' if all(r['status']=='COMPLETE' for r in result['conditions'].values()) and not result['failures'] else 'EXECUTED_WITH_RETAINED_FAILURES'
    result['cuda_peak_across_stages']=cuda_peaks
    save()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(json.loads(args.config.read_text()),args.output)
    print(json.dumps({k:result.get(k) for k in ('status','actual_calls','R','directions','direction_comparisons','zero_repeat_floor','over_budget_conditions','failures')},indent=2))
    if result['failures']: raise SystemExit(1)


if __name__=='__main__': main()
