"""Fixed known-position pre-VAE frame repair diagnostic for C2-T1 interleaved."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
from runtime.c2a.generation import load_frozen_vae
from runtime.c2t1.run import dump,read_mp4,receiver_observation
from runtime.c2t1.run_interleaved import FIT,HOLDOUT,PAYLOAD,allocate,interleaved_sequence,score
from runtime.c2t1.run import mapped_frame
from runtime.c2t1.run_local_paths import path_time

def repair(r: Any) -> Any:
    import torch
    return torch.cat((r[:138],r[137:138],r[138:]),dim=0)

def fixed_allocation(n:int,kind:str) -> tuple[list[int|None],list[int],list[float|None]]:
    return allocate(n,0,1.,0,(n-1)//4,kind,138 if kind=="delete" else None)

def detail(qs:list[list[float]],sel:list[int|None],seq:list[str]) -> dict[str,Any]:
    import numpy as np
    base={"qualified_observation_selection":sel,"fit_indices":list(FIT),"holdout_indices":list(HOLDOUT),"payload_indices":list(PAYLOAD)}
    if any(sel[i] is None or sel[i]>=len(qs) for i in FIT): return base|{"status":"PUBLIC_FIT_MISSING"}
    p={seq[i]:np.asarray(qs[sel[i]],float) for i in FIT}; d2=sum(float(np.sum((p[x]-p[y])**2)) for ix,x in enumerate(("Z","A","B","C","D")) for y in ("Z","A","B","C","D")[ix+1:])/10
    rows=[]
    for i in HOLDOUT+PAYLOAD:
        j=sel[i]
        if j is None or j>=len(qs): rows.append({"position":i,"role":"holdout" if i in HOLDOUT else "payload","status":"MISSING"}); continue
        value=np.asarray(qs[j],float); nearest=min(p,key=lambda state:float(np.sum((value-p[state])**2)))
        rows.append({"position":i,"role":"holdout" if i in HOLDOUT else "payload","selected_group":j,"target":seq[i],"nearest_prototype":nearest,"absolute_squared_distance":float(np.sum((value-p[seq[i]])**2)),"normalized_loss":None if d2<=0 else float(np.sum((value-p[seq[i]])**2)/d2)})
    pairs=[]
    for ix,x in enumerate(("Z","A","B","C","D")):
        for y in ("Z","A","B","C","D")[ix+1:]: pairs.append({"pair":[x,y],"absolute_squared_distance":float(np.sum((p[x]-p[y])**2))})
    return base|{"status":"DETAIL","prototype_q":{k:v.tolist() for k,v in p.items()},"prototype_pairwise_absolute_squared_distances":pairs,"prototype_pairwise_minimum_squared_distance":min(x["absolute_squared_distance"] for x in pairs),"prototype_absolute_pairwise_mean_squared_distance":d2,"rows":rows}

def qualified_counts(n:int,kind:str)->list[int]:
    counts=[]; complete=(n-1)//4
    for i in range(15):
        total=0; lo,hi=1+12*i,13+12*i
        for j in range(complete):
            ts=[path_time(mapped_frame(1.,1+4*j+h,0),kind,138 if kind=="delete" else None) for h in range(4)]
            total+=int(all(lo<=t<hi for t in ts))
        counts.append(total)
    return counts

def run(config:dict[str,Any],output:Path)->dict[str,Any]:
    import torch
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); result={"status":"SETUP_RUNNING","science_denominator":0,"fixed_counts":{"generation_invocations":0,"transformer_calls":0,"vae_decodes":0,"vae_encodes":2},"actual_calls":{"vae_encode_attempted":0,"vae_encode_completed":0},"failures":[]}; dump(output/"config.json",config); dump(output/"result.json",result)
    source=Path(config["source_run"]); sequences={m:interleaved_sequence(config["payload_key_utf8"],m) for m in (0,1)}
    try:
        observations={}
        for m in (0,1):
            for label,n,kind in (("normal",181,"none"),("delete",180,"delete")):
                p=source/"receiver_observations"/f"MESSAGE_{m}_{'NORMAL' if label=='normal' else 'DELETE138'}"/"g0"/"observation.json"; row=json.loads(p.read_text())
                if row.get("received_frames")!=n: raise ValueError(f"unexpected source observation {p}")
                observations[f"message{m}_{label}"]={"source_observation":str(p),"q":row.get("q",[]),"allocation":fixed_allocation(n,kind)}
        dump(output/"source_observations.json",observations)
        vae=load_frozen_vae(config)
        for m in (0,1):
            source_mp4=source/"received_videos"/f"MESSAGE_{m}_DELETE138.mp4"; received=read_mp4(source_mp4)
            fixed=repair(received)
            if int(received.shape[0])!=180 or int(fixed.shape[0])!=181: raise RuntimeError("repair length mismatch")
            if not bool(torch.equal(fixed[:138],received[:138])) or not bool(torch.equal(fixed[139:],received[138:])): raise RuntimeError("repair prefix/suffix changed")
            target=output/"repaired_inputs"/f"MESSAGE_{m}_repaired_rgb.pt"; target.parent.mkdir(exist_ok=True); torch.save(fixed.detach().cpu(),target)
            dest=output/"receiver_observations"/f"MESSAGE_{m}_REPAIRED"/"g0"; dest.mkdir(parents=True); result["actual_calls"]["vae_encode_attempted"]+=1
            try:
                row=receiver_observation(vae,fixed,0,dest); result["actual_calls"]["vae_encode_completed"]+=int(row["status"]=="COMPLETE")
            except BaseException as exc: row={"status":"FAILED","q":[],"error":repr(exc)}; result["failures"].append({"message":m,"error":repr(exc)})
            dump(dest/"observation.json",row); observations[f"message{m}_repaired"]={"source_video":str(source_mp4),"lossless_repaired_rgb":str(target.relative_to(output)),"q":row.get("q",[]),"allocation":fixed_allocation(181,"none"),"receiver":row}; dump(output/"result.json",result); print(f"C2-T1 pre-VAE repair: message {m} {row['status']}",flush=True)
        comparisons={}
        for m in (0,1):
            seq=sequences[m]; comparisons[str(m)]={}
            for label in ("normal","delete","repaired"):
                row=observations[f"message{m}_{label}"]; sel=row["allocation"][0]; kind="delete" if label=="delete" else "none"; score_by_message={str(candidate):score(row["q"],sel,sequences[candidate]) for candidate in (0,1)}; true=score_by_message[str(m)].get("score"); wrong=score_by_message[str(1-m)].get("score")
                comparisons[str(m)][label]={"score_by_message":score_by_message,"wrong_minus_true_margin":None if true is None or wrong is None else wrong-true,"qualified_observation_count_by_position":qualified_counts(180 if label=="delete" else 181,kind),"detail_true_message":detail(row["q"],sel,seq)}
        result["status"]="EXECUTED_REQUIRES_FIXED_PATH_REVIEW"; result["mapping_check"]={"received_to_repaired_length":"180_to_181","indices_137_138_139":[137,137,138],"prefix_frames":138,"suffix_unchanged":True,"g0_expected_latent_groups":46}; result["comparisons"]=comparisons; dump(output/"comparisons.json",comparisons)
    except BaseException as exc: result["status"]="FAILED"; result["failures"].append(repr(exc)); dump(output/"result.json",result); raise
    dump(output/"result.json",result); return result
def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); run(json.loads(a.config.read_text()),a.output)
if __name__=="__main__": main()
