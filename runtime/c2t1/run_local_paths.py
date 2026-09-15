"""C2-T1 fixed-frame delete/repeat receiver-only local-path comparison."""
from __future__ import annotations

import argparse, json, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from runtime.c2a.generation import load_frozen_vae
from runtime.c2t1.run import (ceil_scaled, dump, encode_rgb, mapped_frame, observed_count,
                               payload_order, q, read_mp4, receiver_observation, score_candidate,
                               state_sequence)

K_VALUES = tuple(range(121, 169))

def path_time(u: int, kind: str, k: int | None) -> int:
    if kind == "none": return u
    if kind == "delete": return u + int(u >= int(k))
    if kind == "repeat": return u - int(u >= int(k) + 1)
    raise ValueError(f"unknown local path {kind}")

def allocate(n: int, g: int, a: float, b: int, count: int, kind: str, k: int | None) -> tuple[list[int|None],list[int],list[float|None]]:
    # A complete monotonically mapped group belongs to at most one half-open
    # symbol support.  Iterating groups once (not 15 times) is crucial for the
    # 97-path fixed grid and preserves the documented early-j tie rule.
    best: list[tuple[float,int,float]|None]=[None]*15
    for j in range(count):
        ts=[path_time(mapped_frame(a,g+1+4*j+h,b),kind,k) for h in range(4)]
        first,last=ts[0],ts[-1]
        if first < 1 or last >= 181: continue
        i=(first-1)//12
        if i >= 15 or (last-1)//12 != i: continue
        deviation=sum(ts)/4-(6.5+12*i); value=(abs(deviation),j,deviation)
        if best[i] is None or value[:2] < best[i][:2]: best[i]=value
    selected=[]; coverage=[]; center_deviation=[]
    for value in best:
        selected.append(None if value is None else value[1]); coverage.append(int(value is not None)); center_deviation.append(None if value is None else value[2])
    return selected,coverage,center_deviation

def paths(with_local: bool) -> Iterator[tuple[str,int|None]]:
    yield "none",None
    if with_local:
        for k in K_VALUES:
            yield "delete",k
        for k in K_VALUES:
            yield "repeat",k

def candidate_rows(observation: dict[str,Any], with_local: bool) -> Iterator[dict[str,Any]]:
    n=int(observation["received_frames"])
    for g in range(4):
        item=observation[str(g)]; count=int(item.get("complete_groups_available_from_latent",0))
        for a in (0.8,1.0,1.25):
            for b in range(1-ceil_scaled(a,n-1),181):
                for kind,k in paths(with_local):
                    selected,coverage,deviation=allocate(n,g,a,b,count,kind,k)
                    signature=json.dumps({"selected":selected,"coverage":coverage},separators=(",",":"))
                    yield {"a":a,"b":b,"g":g,"local_path":kind,"k":k,"effective_alignment_signature":signature,"selected":selected,"coverage":coverage,"candidate_symbol_center_deviation":deviation}

def write_search(path: Path, observation: dict[str,Any], with_local: bool, sequences: dict[int,list[str]], reference: dict[str,Any]) -> dict[str,Any]:
    ref_count=int(observation[str(reference["g"])].get("complete_groups_available_from_latent",0))
    ref_selected,ref_coverage,_=allocate(int(observation["received_frames"]),reference["g"],reference["a"],reference["b"],ref_count,reference["local_path"],reference["k"])
    reference_class_key=f"g={reference['g']}|"+json.dumps({"selected":ref_selected,"coverage":ref_coverage},separators=(",",":"))
    class_path=path.with_name(path.stem+"_classes.jsonl"); classes={}; class_scores={}; counts={"rows":0,"scored_message0":0,"scored_message1":0,"coverage_status":{}}; ref_rows={}; reference_class_id=None; best={0:None,1:None}; best_wrong={0:None,1:None}
    with path.open("w",encoding="utf-8") as handle, class_path.open("w",encoding="utf-8") as class_handle:
        for row in candidate_rows(observation,with_local):
            class_key=f"g={row['g']}|{row['effective_alignment_signature']}"
            class_id=classes.get(class_key)
            if class_id is None:
                class_id=len(classes); classes[class_key]=class_id; scores={str(m):score_candidate(observation[str(row["g"])].get("q",[]),row["selected"],seq) for m,seq in sequences.items()}; class_scores[class_id]=scores
                class_handle.write(json.dumps({"effective_alignment_class":class_id,"g":row["g"],"selected":row["selected"],"coverage":row["coverage"],"message":scores},ensure_ascii=False,allow_nan=False,separators=(",",":"))+"\n")
            scores=class_scores[class_id]
            handle.write(json.dumps({"a":row["a"],"b":row["b"],"g":row["g"],"local_path":row["local_path"],"k":row["k"],"effective_alignment_class":class_id,"candidate_symbol_center_deviation":row["candidate_symbol_center_deviation"]},ensure_ascii=False,separators=(",",":"))+"\n")
            counts["rows"]+=1
            if class_key == reference_class_key: reference_class_id=class_id
            for m in (0,1):
                score=scores[str(m)].get("score"); status=scores[str(m)]["status"]
                counts["coverage_status"][status]=counts["coverage_status"].get(status,0)+1
                if score is not None:
                    counts[f"scored_message{m}"]+=1
                    key=(score,row["a"],row["b"],row["g"],row["local_path"],row["k"])
                    if best[m] is None or key<best[m]: best[m]=key
                if all(row[x]==reference[x] for x in ("a","b","g","local_path","k")):
                    ref_rows[m]={**row,"message":scores}; reference_class_id=class_id
    # A second streaming pass keeps all candidate rows on disk while excluding the complete reference equivalence class.
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row=json.loads(line)
            if row["effective_alignment_class"] == reference_class_id: continue
            for m in (0,1):
                score=class_scores[row["effective_alignment_class"]][str(m)].get("score")
                if score is not None:
                    key=(score,row["a"],row["b"],row["g"],row["local_path"],row["k"])
                    if best_wrong[m] is None or key<best_wrong[m]: best_wrong[m]=key
    rank={}
    for m in (0,1):
        exact_ref=None if m not in ref_rows else ref_rows[m]["message"][str(m)].get("score")
        equivalent_ref=None if reference_class_id is None else class_scores[reference_class_id][str(m)].get("score")
        wrong=None if best_wrong[m] is None else best_wrong[m][0]
        rank[str(m)]={"best_candidate":best[m],"reference_exact_path":reference,"reference_exact_path_present":m in ref_rows,"reference_exact_path_score":exact_ref,"reference_equivalent_class_status":"REPRESENTED" if reference_class_id is not None else "NOT_REPRESENTED","reference_equivalent_class_score":equivalent_ref,"reference_effective_alignment_class":reference_class_id,"best_wrong_time_score_excluding_reference_class":wrong,"reference_equivalent_minus_best_wrong_time":None if equivalent_ref is None or wrong is None else equivalent_ref-wrong}
    return {"candidate_rows":counts["rows"],"effective_alignment_class_count":len(classes),"status_counts":counts["coverage_status"],"ranking_and_true_wrong_time_gaps":rank}

def ideal_difference_check() -> dict[str,Any]:
    """No-model check required before GPU execution, using the actual fixed edit only."""
    answer={"fixed_k":138,"by_received_length":{},"full_search_has_different_allocation":False}
    for n,label,kind in ((180,"delete180","delete"),(182,"repeat182","repeat")):
        rows=[]
        for g in range(4):
            count,_=observed_count(n,g)
            plain=allocate(n,g,1.0,0,count,"none",None)
            local=allocate(n,g,1.0,0,count,kind,138)
            different=[i for i,(x,y) in enumerate(zip(plain[0],local[0])) if x!=y or plain[1][i]!=local[1][i]]
            rows.append({"g":g,"same_selected_and_coverage":not different,"different_symbol_indices":different})
        answer["by_received_length"][label]=rows
        distinct_counts=[]
        for g in range(4):
            count,_=observed_count(n,g); signatures=set()
            for candidate_kind,k in paths(True):
                current=allocate(n,g,1.0,0,count,candidate_kind,k)
                signatures.add((tuple(current[0]),tuple(current[1])))
            distinct_counts.append(len(signatures))
        answer["by_received_length"][label+"_full_search_distinct_allocation_class_count_by_g"]=distinct_counts
        answer["full_search_has_different_allocation"] |= any(value > 1 for value in distinct_counts)
    return answer

def run(config: dict[str,Any], output: Path) -> dict[str,Any]:
    import torch
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True); dump(output/"config.json",config)
    result={"status":"SETUP_RUNNING","science_denominator":0,"fixed_counts":{"generation_invocations":0,"transformer_calls":0,"vae_decodes":0,"vae_encodes":16,"received_videos":4},"actual_calls":{"vae_encode_attempted":0,"vae_encode_completed":0},"failures":[]}
    dump(output/"calls.json",{"generation_invocations":0,"transformer_calls":0,"vae_decodes":0,"vae_encodes_planned":16,"receiver_start_offsets":[0,1,2,3],"local_paths_per_time_candidate":97})
    dump(output/"result.json",result); ideal=ideal_difference_check(); dump(output/"ideal_local_path_difference_check.json",ideal); result["ideal_local_path_difference_check"]=ideal
    source=Path(config["source_drive_run"]); source_paths={m:source/"precodec_and_normal"/f"MESSAGE_{m}"/"normal.mp4" for m in (0,1)}
    try:
        videos={}
        for m,path in source_paths.items():
            raster=read_mp4(path); k=int(config["edit"]["source_frame_zero_based"])
            if int(raster.shape[0])!=181: raise ValueError(f"source normal length for message {m} is not 181")
            deleted=output/"received_videos"/f"MESSAGE_{m}_DELETE138.mp4"; repeated=output/"received_videos"/f"MESSAGE_{m}_REPEAT138.mp4"
            encode_rgb(torch.cat((raster[:k],raster[k+1:]),dim=0),deleted,int(config["codec"]["fps"]),int(config["codec"]["crf"]))
            encode_rgb(torch.cat((raster[:k+1],raster[k:k+1],raster[k+1:]),dim=0),repeated,int(config["codec"]["fps"]),int(config["codec"]["crf"]))
            videos[f"MESSAGE_{m}_DELETE138"]=deleted; videos[f"MESSAGE_{m}_REPEAT138"]=repeated
        result["status"]="FOUR_RECEIVED_VIDEOS_COMPLETE"; result["video_paths"]={x:str(y.relative_to(output)) for x,y in videos.items()}; dump(output/"result.json",result)
        print("C2-T1 local path: four received videos persisted", flush=True)
        vae=load_frozen_vae(config); observations={}
        for name,path in videos.items():
            raster=read_mp4(path); expected=180 if "DELETE" in name else 182
            if int(raster.shape[0]) != expected: raise RuntimeError(f"{name} length {int(raster.shape[0])} != {expected}")
            observations[name]={"received_frames":int(raster.shape[0]),"path":str(path.relative_to(output))}
            for g in range(4):
                dest=output/"receiver_observations"/name/f"g{g}"; dest.mkdir(parents=True); result["actual_calls"]["vae_encode_attempted"]+=1
                try:
                    row=receiver_observation(vae,raster,g,dest)
                    if row["status"]=="COMPLETE": result["actual_calls"]["vae_encode_completed"]+=1
                except BaseException as exc:
                    count,tail=observed_count(int(raster.shape[0]),g); row={"status":"FAILED","g":g,"complete_ordinary_groups_expected":count,"incomplete_tail_frames":tail,"complete_groups_available_from_latent":0,"q":[],"error":repr(exc)}; result["failures"].append({"video":name,"g":g,"error":repr(exc)})
                observations[name][str(g)]=row; dump(dest/"observation.json",row)
                dump(output/"result.json",result); print(f"C2-T1 local path: {name} g={g} {row['status']}", flush=True)
        dump(output/"receiver_observations/index.json",observations); result["status"]="SIXTEEN_RECEIVER_OBSERVATIONS_COMPLETE"; dump(output/"result.json",result)
        print("C2-T1 local path: all receiver observations persisted", flush=True)
        sequences={0:state_sequence(config["payload_key_utf8"],0),1:state_sequence(config["payload_key_utf8"],1)}; summaries={}
        for name,obs in observations.items():
            kind="delete" if "DELETE" in name else "repeat"; reference={"a":1.0,"b":0,"g":0,"local_path":kind,"k":138}
            summaries[name]={}
            for label,local in (("no_local_path",False),("one_local_path",True)):
                target=output/"candidate_jsonl"/f"{name}_{label}.jsonl"; target.parent.mkdir(exist_ok=True)
                summaries[name][label]=write_search(target,obs,local,sequences,reference)
                print(f"C2-T1 local path: {name} {label} search persisted", flush=True)
        dump(output/"search_summary.json",summaries); result["search_summary"]={n:{l:{k:v[k] for k in ("candidate_rows","effective_alignment_class_count")} for l,v in s.items()} for n,s in summaries.items()}; result["status"]="EXECUTED_REQUIRES_COMPARATIVE_RANKING_REVIEW"; result["limitations"]=["No threshold, FPR, or PASS claim.","A lower correct score alone is not a gain because wrong messages receive the same local-path search.","The fixed-k ideal check may show no allocation difference; that combination cannot establish path gain."]
        dump(output/"calls.json",{**json.loads((output/"calls.json").read_text()),"actual_calls":result["actual_calls"]})
    except BaseException as exc:
        result["status"]="FAILED"; result["failures"].append(repr(exc)); dump(output/"result.json",result); raise
    dump(output/"result.json",result); return result

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output",type=Path,required=True); args=p.parse_args(); run(json.loads(args.config.read_text()),args.output)
if __name__ == "__main__": main()
