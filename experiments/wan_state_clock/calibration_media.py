"""Saved-terminal media stage; native codec/VAE/full blind reader, no generation."""
from __future__ import annotations
import gc
import resource
import traceback
from pathlib import Path
import numpy as np
import torch
from main.tube_state import state_clock
from runtime.wan.generation import load_frozen_vae
from runtime.wan.io import dump, encode_rgb, read_mp4
from runtime.wan.vae import decode_normalized_latent, reencode_rgb24_readback, _clear_cache
from .flow_run import quality


def run(config,terminals,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    book=state_clock.codebook(config['key_utf8'].encode())
    result={'status':'RUNNING','videos':{name:{'status':'NOT_RUN','terminal_path':str(path),
        'observations':{str(g):{'status':'NOT_RUN'} for g in range(4)}} for name,path in terminals.items()},
        'fixed_calls':{'vae_decode':len(terminals),'mp4_save':len(terminals),'vae_encode':4*len(terminals)},
        'actual_calls':{},'failures':[],'formal_science_denominator':0,
        'quality_rule':{'relative_limit':1.5,'meaning':'engineering tolerance, not validated perception'},
        'claim':'five-mode ranking is not FPR; truth used only after read returns'}
    for kind in result['fixed_calls']:
        result['actual_calls'].update({kind+'_attempted':0,kind+'_completed':0})
    def save():dump(output/'result.json',result)
    def count(kind,done):result['actual_calls'][kind+('_completed' if done else '_attempted')]+=1;save()
    def fail(stage,exc):
        result['failures'].append({'stage':stage,'error':repr(exc),'traceback':traceback.format_exc()});save()
    def release():
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    dump(output/'config.json',config);save()
    vae=None
    try:
        vae=load_frozen_vae(config)
        vae.to('cuda')
        for name,terminal_path in terminals.items():
            item=result['videos'][name];path=output/'received_videos'/f'{name}.mp4'
            rgb=z=encoded=reference=None;obs={};media_success=False
            if not Path(terminal_path).is_file():
                item['status']='MISSING_TERMINAL';save();continue
            try:
                z=torch.load(terminal_path,map_location='cpu',weights_only=True)
                count('vae_decode',False)
                rgb=decode_normalized_latent(vae,z.to(next(vae.parameters()).device)).cpu()
                count('vae_decode',True)
                count('mp4_save',False);encode_rgb(rgb,path,8,18);count('mp4_save',True)
                media_success=True
                item.update(status='VIDEO_PERSISTED',path=str(path))
            except Exception as exc:
                item['status']='FAILED_MEDIA';fail(name+'/media',exc)
            finally:
                z=rgb=None;_clear_cache(vae);release()
            try:
                rgb=read_mp4(path)
                if len(rgb)!=181:raise ValueError('normal MP4 requires 181 frames')
                if name!='OFF':
                    reference=read_mp4(output/'received_videos/OFF.mp4')
                    item['saved_quality_vs_off']=quality(reference,rgb);reference=None
                for g in range(4):
                    row=item['observations'][str(g)]
                    try:
                        groups,tail=divmod(len(rgb)-g-1,4)
                        count('vae_encode',False)
                        encoded=reencode_rgb24_readback(vae,rgb[g:g+1+4*groups]).cpu().float()
                        count('vae_encode',True)
                        if tuple(encoded.shape)!=(1,16,1+groups,40,64):raise ValueError('receiver latent geometry mismatch')
                        dest=output/'receiver_observations'/name;dest.mkdir(parents=True,exist_ok=True)
                        torch.save(encoded,dest/f'g{g}.pt');obs[g]=encoded.numpy()
                        row.update(status='COMPLETE',frames_used=1+4*groups,tail_discarded=tail)
                    except Exception as exc:
                        row.update(status='FAILED',error=repr(exc));fail(name+f'/g{g}',exc)
                    finally:
                        encoded=None;_clear_cache(vae);release();save()
                # The unchanged reader gets observations and public book only.
                detection=state_clock.read(obs,book)
                dump(output/'detections'/f'{name}.json',detection)
                item['rankings']=detection['rankings']
                if name!='OFF':
                    truth=0 if name.endswith('A') else 1
                    item['reporting_only']=state_clock.report(detection,truth,0)
                    item['five_methods_unique_correct']=(len(detection['rankings'])==5 and
                        all(row['message_unique'] and row['best']['message']==truth for row in detection['rankings'].values())) if len(obs)==4 else None
                item['status']='COMPLETE' if media_success and len(obs)==4 else 'PARTIAL_OR_FAILED'
            except Exception as exc:
                item['status']='FAILED_READBACK';fail(name+'/readback',exc)
            finally:
                rgb=encoded=reference=obs=None;release();save()
    except Exception as exc:
        fail('media_setup',exc)
    finally:
        if vae is not None:_clear_cache(vae)
        vae=None;release()
    result['resources']={'cpu_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None}
    result['status']='COMPLETE' if all(row['status']=='COMPLETE' for row in result['videos'].values()) and not result['failures'] else 'WITH_RETAINED_FAILURES'
    save();return result
