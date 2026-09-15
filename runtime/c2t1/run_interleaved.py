"""C2-T1 interleaved-public receiver experiment; does not alter prior runners."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any, Iterator
from runtime.c2a.chain import decode_normalized_latent, ffmpeg_roundtrip
from runtime.c2a.generation import load_frozen_vae
from runtime.c2t1.run import (ceil_scaled, dump, encode_rgb, observed_count, read_mp4, receiver_observation,
                               state_sequence, write_states)
from runtime.c2t1.run_local_paths import allocate, path_time

FIT=(0,1,2,3,4); HOLDOUT=(6,8,10,12,13); PAYLOAD=(5,7,9,11)
K_VALUES=tuple(x for lo,hi in ((61,72),(85,96),(109,120),(133,144)) for x in range(lo,hi+1))

def interleaved_sequence(key: str, message: int) -> list[str]:
    old=state_sequence(key,message); p=old[10:14]
    return ["Z","A","B","C","D",p[0],"B",p[1],"Z",p[2],"D",p[3],"A","C","Z"]

def score(qs: list[list[float]], selected: list[int|None], sequence: list[str]) -> dict[str,Any]:
    import numpy as np
    if any(selected[i] is None or selected[i]>=len(qs) for i in FIT): return {"status":"PUBLIC_FIT_MISSING","score":None}
    proto={sequence[i]:np.asarray(qs[selected[i]],float) for i in FIT}
    d2=sum(float(np.sum((proto[x]-proto[y])**2)) for n,x in enumerate(("Z","A","B","C","D")) for y in ("Z","A","B","C","D")[n+1:])/10
    if not math.isfinite(d2) or d2<=0: return {"status":"PUBLIC_PROTOTYPES_DEGENERATE","score":None,"d2":d2}
    if any(selected[i] is None or selected[i]>=len(qs) for i in HOLDOUT): return {"status":"PUBLIC_HOLDOUT_MISSING","score":None}
    available=sum(selected[i] is not None and selected[i]<len(qs) for i in PAYLOAD)
    if available<3: return {"status":"PAYLOAD_COVERAGE_INSUFFICIENT","score":None,"payload_available":available}
    losses=[]; missing=0
    for i in HOLDOUT+PAYLOAD:
        j=selected[i]
        if j is None or j>=len(qs): losses.append(1.0); missing+=1
        else: losses.append(float(np.sum((np.asarray(qs[j])-proto[sequence[i]])**2)/d2))
    return {"status":"SCORED" if not missing else "SCORED_WITH_FIXED_MISSING_LOSS","score":sum(losses)/9,"d2":d2,"losses":losses,"missing":missing}

def paths(local: bool) -> Iterator[tuple[str,int|None]]:
    yield "none",None
    if local:
        for k in K_VALUES: yield "delete",k
        for k in K_VALUES: yield "repeat",k

def candidates(obs: dict[str,Any], local: bool, sequences: dict[int,list[str]]) -> Iterator[dict[str,Any]]:
    n=obs["received_frames"]
    for g in range(4):
        item=obs[str(g)]; count=item.get("complete_groups_available_from_latent",0)
        for a in (.8,1.,1.25):
            for b in range(1-ceil_scaled(a,n-1),181):
                for kind,k in paths(local):
                    selected,coverage,deviation=allocate(n,g,a,b,count,kind,k)
                    sig=json.dumps({"selected":selected,"coverage":coverage},separators=(",",":"))
                    yield {"a":a,"b":b,"g":g,"local_path":kind,"k":k,"signature":sig,"selected":selected,"coverage":coverage,"candidate_symbol_center_deviation":deviation}

def write_search(path: Path, obs: dict[str,Any], local: bool, sequences: dict[int,list[str]], reference: dict[str,Any]) -> dict[str,Any]:
    count=obs[str(reference["g"])].get("complete_groups_available_from_latent",0); ref_selected,ref_coverage,_=allocate(obs["received_frames"],reference["g"],reference["a"],reference["b"],count,reference["local_path"],reference["k"])
    ref_key=f"g={reference['g']}|"+json.dumps({"selected":ref_selected,"coverage":ref_coverage},separators=(",",":"))
    classes={}; records={}; counts={"rows":0,"status":{}}; best={0:None,1:None}; ref_class=None; exact={}
    class_path=path.with_name(path.stem+"_classes.jsonl")
    with path.open("w",encoding="utf-8") as out, class_path.open("w",encoding="utf-8") as table:
        for row in candidates(obs,local,sequences):
            key=f"g={row['g']}|{row['signature']}"; cid=classes.get(key)
            if cid is None:
                cid=len(classes); classes[key]=cid; records[cid]={str(m):score(obs[str(row["g"])].get("q",[]),row["selected"],s) for m,s in sequences.items()}
                table.write(json.dumps({"effective_alignment_class":cid,"g":row["g"],"selected":row["selected"],"coverage":row["coverage"],"scores":records[cid]},separators=(",",":"))+"\n")
            scores=records[cid]
            if key==ref_key: ref_class=cid
            out.write(json.dumps({"a":row["a"],"b":row["b"],"g":row["g"],"local_path":row["local_path"],"k":row["k"],"effective_alignment_class":cid,"candidate_symbol_center_deviation":row["candidate_symbol_center_deviation"]},separators=(",",":"))+"\n")
            counts["rows"]+=1
            if all(row[x]==reference[x] for x in ("a","b","g","local_path","k")): ref_class=cid; exact={m:scores[str(m)].get("score") for m in (0,1)}
            for m in (0,1):
                status=scores[str(m)]["status"]; counts["status"][status]=counts["status"].get(status,0)+1
                value=scores[str(m)].get("score")
                if value is not None and (best[m] is None or (value,row["a"],row["b"],row["g"],row["local_path"],row["k"])<best[m]): best[m]=(value,row["a"],row["b"],row["g"],row["local_path"],row["k"])
    wrong={0:None,1:None}
    with path.open() as src:
        for line in src:
            row=json.loads(line)
            if row["effective_alignment_class"]==ref_class: continue
            for m in (0,1):
                value=records[row["effective_alignment_class"]][str(m)].get("score")
                if value is not None and (wrong[m] is None or value<wrong[m]): wrong[m]=value
    rank={str(m):{"best_candidate":best[m],"reference_exact_path":reference,"reference_exact_path_present":m in exact,"reference_exact_path_score":exact.get(m),"reference_equivalent_class_status":"REPRESENTED" if ref_class is not None else "NOT_REPRESENTED","reference_equivalent_class_score":None if ref_class is None else records[ref_class][str(m)].get("score"),"reference_effective_alignment_class":ref_class,"best_wrong_time_score_excluding_reference_class":wrong[m],"reference_equivalent_minus_best_wrong_time":None if ref_class is None or records[ref_class][str(m)].get("score") is None or wrong[m] is None else records[ref_class][str(m)]["score"]-wrong[m]} for m in (0,1)}
    return {"candidate_rows":counts["rows"],"effective_alignment_class_count":len(classes),"status_counts":counts["status"],"ranking_and_true_wrong_time_gaps":rank}

def ideal_check() -> dict[str,Any]:
    result={"fit_indices":list(FIT),"holdout_indices":list(HOLDOUT),"payload_indices":list(PAYLOAD),"k_values":list(K_VALUES),"frame_138_logical_state":11,"frame_138_role":"payload_p3"}
    for n,label in ((181,"normal181"),(180,"delete180")):
        rows=[]
        for g in range(4):
            count,_=observed_count(n,g); base=allocate(n,g,1.,0,count,"none",None); edited=allocate(n,g,1.,0,count,"delete",138)
            changed=[i for i,(x,y) in enumerate(zip(base[0],edited[0])) if x!=y or base[1][i]!=edited[1][i]]
            rows.append({"g":g,"complete_groups":count,"coverage":base[1],"delete138_changed_indices":changed,"changed_roles":["payload" if i in PAYLOAD else "holdout" if i in HOLDOUT else "fit" if i in FIT else "tail" for i in changed]})
        result[label]=rows
    return result

def run(config: dict[str,Any], output: Path) -> dict[str,Any]:
    import torch
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); dump(output/"config.json",config)
    result={"status":"SETUP_RUNNING","science_denominator":0,"fixed_counts":{"generation_invocations":0,"transformer_calls":0,"vae_decodes":2,"vae_encodes":16,"received_videos":4},"actual_calls":{"vae_decode_attempted":0,"vae_decode_completed":0,"vae_encode_attempted":0,"vae_encode_completed":0},"failures":[]}
    ideal=ideal_check(); result["ideal_check"]=ideal; dump(output/"ideal_interleaved_check.json",ideal); dump(output/"result.json",result)
    try:
        terminal=torch.load(config["source_shared_terminal"],map_location="cuda",weights_only=True); 
        if list(terminal.shape[2:3]) != [46]: raise ValueError("shared terminal lacks special+45 groups")
        vae=load_frozen_vae(config); sequences={m:interleaved_sequence(config["payload_key_utf8"],m) for m in (0,1)}; dump(output/"state_sequences.json",{str(m):s for m,s in sequences.items()})
        videos={}
        for m in (0,1):
            marked,evidence,snap=write_states(terminal,sequences[m]); evidence_dir=output/"actual_writing_evidence"; evidence_dir.mkdir(exist_ok=True); torch.save(snap,evidence_dir/f"message_{m}_blocks.pt"); dump(evidence_dir/f"message_{m}_records.json",evidence)
            arm=output/"precodec_and_normal"/f"MESSAGE_{m}"; arm.mkdir(parents=True); result["actual_calls"]["vae_decode_attempted"]+=1; rgb=decode_normalized_latent(vae,marked); result["actual_calls"]["vae_decode_completed"]+=1; torch.save(rgb.detach().cpu(),arm/"precodec_rgb.pt"); normal=arm/"normal.mp4"; ffmpeg_roundtrip(rgb,normal,fps=config["codec"]["fps"],crf=config["codec"]["crf"]); raster=read_mp4(normal); deleted=output/"received_videos"/f"MESSAGE_{m}_DELETE138.mp4"; encode_rgb(torch.cat((raster[:138],raster[139:]),0),deleted,config["codec"]["fps"],config["codec"]["crf"]); videos[f"MESSAGE_{m}_NORMAL"]=normal; videos[f"MESSAGE_{m}_DELETE138"]=deleted
        result["status"]="FOUR_VIDEOS_COMPLETE"; dump(output/"result.json",result); obs={}
        print("C2-T1 interleaved: four videos persisted",flush=True)
        for name,path in videos.items():
            raster=read_mp4(path); expected=180 if "DELETE" in name else 181
            if int(raster.shape[0])!=expected: raise RuntimeError(f"{name} length mismatch")
            obs[name]={"received_frames":expected}
            for g in range(4):
                dest=output/"receiver_observations"/name/f"g{g}"; dest.mkdir(parents=True); result["actual_calls"]["vae_encode_attempted"]+=1
                try:
                    row=receiver_observation(vae,raster,g,dest); result["actual_calls"]["vae_encode_completed"]+=int(row["status"]=="COMPLETE")
                except BaseException as exc:
                    count,tail=observed_count(expected,g); row={"status":"FAILED","g":g,"complete_ordinary_groups_expected":count,"incomplete_tail_frames":tail,"complete_groups_available_from_latent":0,"q":[],"error":repr(exc)}; result["failures"].append({"video":name,"g":g,"error":repr(exc)})
                obs[name][str(g)]=row; dump(dest/"observation.json",row); dump(output/"result.json",result); print(f"C2-T1 interleaved: {name} g={g} {row['status']}",flush=True)
        dump(output/"receiver_observations/index.json",obs); result["status"]="SIXTEEN_RECEIVER_OBSERVATIONS_COMPLETE"; dump(output/"result.json",result); print("C2-T1 interleaved: receiver observations persisted",flush=True); summaries={}
        for name,item in obs.items():
            ref={"a":1.,"b":0,"g":0,"local_path":"delete" if "DELETE" in name else "none","k":138 if "DELETE" in name else None}; summaries[name]={}
            for label,local in (("no_local_path",False),("one_local_path",True)):
                p=output/"candidate_jsonl"/f"{name}_{label}.jsonl"; p.parent.mkdir(exist_ok=True); summaries[name][label]=write_search(p,item,local,sequences,ref); dump(output/"search_summary.json",summaries); result["status"]="SEARCH_IN_PROGRESS"; result["search_summary"]={n:{l:{k:v[k] for k in ("candidate_rows","effective_alignment_class_count")} for l,v in s.items()} for n,s in summaries.items()}; dump(output/"result.json",result); print(f"C2-T1 interleaved: {name} {label} search persisted",flush=True)
        dump(output/"search_summary.json",summaries); result["search_summary"]={n:{l:{k:v[k] for k in ("candidate_rows","effective_alignment_class_count")} for l,v in s.items()} for n,s in summaries.items()}; result["status"]="EXECUTED_REQUIRES_COMPARATIVE_RANKING_REVIEW"; result["limitations"]=["No FPR, threshold, or PASS claim.","New/old layout differences also include symbol history and re-encoding; they do not uniquely identify public spacing."]
    except BaseException as exc:
        result["status"]="FAILED"; result["failures"].append(repr(exc)); dump(output/"result.json",result); raise
    dump(output/"result.json",result); return result

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); run(json.loads(a.config.read_text()),a.output)
if __name__=="__main__": main()
