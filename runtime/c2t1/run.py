"""C2-T1 real-validation executor; model work occurs only through this CLI."""
from __future__ import annotations

import argparse, hashlib, hmac, json, math, subprocess
from pathlib import Path
from typing import Any

from runtime.c2a.chain import decode_normalized_latent, ffmpeg_roundtrip, reencode_rgb24_readback
from runtime.c2a.generation import generate_terminal_latent
from runtime.c2a.protocol import C2AConfig, central_block_slices, rgb_quality_metrics

STATES = {"Z": (0.0, 0.0), "A": (0.5, 0.0), "B": (0.0, 0.5), "C": (-0.5, 0.0), "D": (0.0, -0.5)}
PUBLIC_FIT, PUBLIC_HOLDOUT = ("Z", "A", "B", "C", "D"), ("B", "Z", "D", "A", "C")

def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")

def payload_order(key: str) -> list[str]:
    return sorted("ABCD", key=lambda x: (hmac.new(key.encode(), ("C2-T1/payload/"+x).encode(), hashlib.sha256).digest(), x))

def state_sequence(key: str, message: int) -> list[str]:
    p = payload_order(key); return list(PUBLIC_FIT + PUBLIC_HOLDOUT + tuple(p if message == 0 else reversed(p)) + ("Z",))

def q(block: Any) -> Any:
    import torch
    x=block.reshape(2,-1).to(torch.float64); x=x-x.mean(1,keepdim=True); cov=x@x.T/64.0
    return torch.stack(((cov[0,0]-cov[1,1])/2, cov[0,1])), cov

def write_states(terminal: Any, sequence: list[str]) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    import torch
    cfg=C2AConfig(); written=terminal.detach().clone(); _,_,groups,h,w=written.shape
    if groups != 46: raise ValueError(f"C2-T1 requires special+45 groups, got {groups}")
    rows,cols=central_block_slices(h,w,8); records=[]; snapshots={}
    for logical,state_name in enumerate(sequence):
        state=STATES[state_name]; sigma=torch.tensor(((1+.25*state[0], .25*state[1]),(.25*state[1],1-.25*state[0])),dtype=torch.float64,device=written.device)
        for repeat in range(3):
            group=1+3*logical+repeat; block=written[0,0:2,group,rows,cols]; original=block.dtype
            x=block.reshape(2,-1).to(torch.float64); mean=x.mean(1,keepdim=True); xc=x-mean; cov=xc@xc.T/64.0
            ev,vec=torch.linalg.eigh(cov); 
            if float(ev[0]) <= 1e-8*max(float(ev[-1]),1.): raise ValueError(f"SINGULAR_OR_NEAR_SINGULAR_COVARIANCE group={group}")
            root=(vec*ev.sqrt())@vec.T; iroot=(vec*ev.rsqrt())@vec.T; mev=root@sigma@root; me,mv=torch.linalg.eigh(mev)
            transform=iroot@((mv*me.sqrt())@mv.T)@iroot; block.copy_((transform@xc+mean).reshape_as(block).to(original))
            actual_q,actual_cov=q(block); records.append({"logical_state":logical,"symbol":state_name,"group":group,"actual_covariance_after_fp32_write":actual_cov.cpu().tolist(),"actual_q_after_fp32_write":actual_q.cpu().tolist()})
            snapshots[f"group_{group:02d}"]=block.detach().cpu().contiguous().clone()
    return written,records,snapshots

def read_mp4(path: Path) -> Any:
    import numpy as np, torch
    probe=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height,nb_frames","-of","json",str(path)],capture_output=True,text=True,check=True)
    stream=json.loads(probe.stdout)["streams"][0]; w,h=int(stream["width"]),int(stream["height"])
    raw=subprocess.run(["ffmpeg","-v","error","-threads","1","-noautorotate","-i",str(path),"-map","0:v:0","-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True,check=True).stdout
    one=h*w*3
    if len(raw)%one: raise RuntimeError("received MP4 is not integral RGB24 frames")
    return torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).reshape(len(raw)//one,h,w,3).copy()).float()/255

def encode_rgb(rgb: Any, path: Path, fps: int, crf: int) -> None:
    from runtime.c2a.chain import quantize_rgb8_no_codec
    path.parent.mkdir(parents=True,exist_ok=True)
    t,h,w,_=map(int,rgb.shape); pixels=quantize_rgb8_no_codec(rgb).numpy()
    subprocess.run(["ffmpeg","-v","error","-threads","1","-f","rawvideo","-pix_fmt","rgb24","-s",f"{w}x{h}","-r",str(fps),"-i","pipe:0","-an","-c:v","libx264","-crf",str(crf),"-pix_fmt","yuv420p","-n",str(path)],input=pixels.tobytes(),check=True,capture_output=True)

def observed_count(n: int, g: int) -> tuple[int,int]: return ((n-g-1)//4, (n-g-1)%4)

def receiver_observation(vae: Any, rgb: Any, g: int, destination: Path) -> dict[str, Any]:
    import torch
    n=int(rgb.shape[0]); complete,tail=observed_count(n,g); used=1+4*complete
    # This deliberately omits the incomplete tail; it is recorded, not padded into a full observation.
    recoded=reencode_rgb24_readback(vae,rgb[g:g+used]); torch.save(recoded.detach().cpu(),destination/"reencoded_normalized.pt")
    rows,cols=central_block_slices(int(recoded.shape[-2]),int(recoded.shape[-1]),8); available=max(0,int(recoded.shape[2])-1); usable=min(complete,available)
    qs=[]
    for j in range(usable):
        value,_=q(recoded[0,0:2,1+j,rows,cols]); qs.append(value.cpu().tolist())
    return {"g":g,"received_frames":n,"frames_supplied_to_vae":used,"complete_ordinary_groups_expected":complete,"incomplete_tail_frames":tail,"latent_shape":list(recoded.shape),"complete_groups_available_from_latent":usable,"q":qs,"status":"COMPLETE" if usable==complete else "FAILED_LATENT_TEMPORAL_GEOMETRY"}

def mapped_frame(a: float, tau: int, b: int) -> int:
    """Exact fixed-grid floor, matching the documented 4/5 and 5/4 semantics."""
    if a == 0.8: return 4 * tau // 5 + b
    if a == 1.0: return tau + b
    if a == 1.25: return 5 * tau // 4 + b
    raise ValueError("C2-T1 only permits the fixed speed grid")

def ceil_scaled(a: float, value: int) -> int:
    if a == 0.8: return (4 * value + 4) // 5
    if a == 1.0: return value
    if a == 1.25: return (5 * value + 3) // 4
    raise ValueError("C2-T1 only permits the fixed speed grid")

def allocation(n: int,g: int,a: float,b: int,count: int) -> tuple[list[int|None],list[float|None],list[int]]:
    selected=[]; deviations=[]; coverage=[]; used=set()
    for i in range(15):
        lo,hi=1+12*i,13+12*i; center=6.5+12*i; choices=[]
        for j in range(count):
            fs=[mapped_frame(a,g+1+4*j+h,b) for h in range(4)]
            if all(lo<=f<hi for f in fs) and j not in used: choices.append((abs(sum(fs)/4-center),j,sum(fs)/4-center))
        if choices:
            _,j,d=min(choices,key=lambda x:(x[0],x[1])); selected.append(j); deviations.append(d); used.add(j); coverage.append(1)
        else: selected.append(None); deviations.append(None); coverage.append(0)
    return selected,deviations,coverage

def score_candidate(qs: list[list[float]], selected: list[int|None], sequence: list[str]) -> dict[str,Any]:
    import numpy as np
    fit=selected[:5]
    if any(j is None or j>=len(qs) for j in fit): return {"status":"PUBLIC_FIT_MISSING","score":None,"missing":15-sum(j is not None and j<len(qs) for j in selected)}
    proto={s:np.asarray(qs[j],float) for s,j in zip(PUBLIC_FIT,fit)}; d2=sum(float(np.sum((proto[x]-proto[y])**2)) for ix,x in enumerate(PUBLIC_FIT) for y in PUBLIC_FIT[ix+1:])/10
    if not math.isfinite(d2) or d2<=0: return {"status":"PUBLIC_PROTOTYPES_DEGENERATE","score":None,"d2":d2,"missing":15-sum(j is not None and j<len(qs) for j in selected)}
    holdout=selected[5:10]
    if any(j is None or j>=len(qs) for j in holdout):
        return {"status":"PUBLIC_HOLDOUT_MISSING","score":None,"missing":15-sum(j is not None and j<len(qs) for j in selected)}
    payload=selected[10:14]
    payload_available=sum(j is not None and j<len(qs) for j in payload)
    if payload_available < 3:
        return {"status":"PAYLOAD_COVERAGE_INSUFFICIENT","score":None,"payload_available":payload_available,"missing":15-sum(j is not None and j<len(qs) for j in selected)}
    losses=[]; missing=0
    for i in range(5,14):
        j=selected[i]
        if j is None or j>=len(qs): losses.append(1.0); missing+=1
        else: losses.append(float(np.sum((np.asarray(qs[j])-proto[sequence[i]])**2)/d2))
    return {"status":"SCORED" if missing==0 else "SCORED_WITH_FIXED_MISSING_LOSS","score":sum(losses)/9,"d2":d2,"losses":losses,"missing":missing}

def evaluate(video: str, obs: dict[str,Any], config: dict[str,Any], messages: dict[int,list[str]], reference: dict[str,Any]) -> dict[str,Any]:
    n=obs["received_frames"]; rows=[]; classes={}; class_next=0
    for g in config["candidate_grid"]["g"]:
        one=obs[str(g)]; count=one["complete_groups_available_from_latent"]
        for a in config["candidate_grid"]["a"]:
            for b in range(1-ceil_scaled(a,n-1),181):
                sel,dev,cov=allocation(n,g,a,b,count); sig=json.dumps({"allocation":sel,"coverage":cov},separators=(",",":")); key=f"g={g}|{sig}"
                if key not in classes: classes[key]=class_next; class_next+=1
                item={"a":a,"b":b,"g":g,"effective_alignment_class":classes[key],"allocation":sel,"coverage":cov,"candidate_symbol_center_deviation":dev,"reporting_only_reference_parameter_delta":{"a_minus_reference":a-reference["a"],"b_minus_reference":b-reference["b"],"g_minus_reference":g-reference["g"]},"message":{}}
                for m,sequence in messages.items(): item["message"][str(m)]=score_candidate(one["q"],sel,sequence)
                rows.append(item)
    pairs=[]
    for r in rows:
        x,y=r["message"]["0"].get("score"),r["message"]["1"].get("score")
        pairs.append({"a":r["a"],"b":r["b"],"g":r["g"],"effective_alignment_class":r["effective_alignment_class"],"score_message0":x,"score_message1":y,"message0_minus_message1":None if x is None or y is None else x-y})
    ranking={}
    for m in (0,1):
        scored=[r for r in rows if r["message"][str(m)].get("score") is not None]
        ordered=sorted(scored,key=lambda r:r["message"][str(m)]["score"])
        ref=next((r for r in rows if all(r[x]==reference[x] for x in ("a","b","g"))),None)
        ref_score=None if ref is None else ref["message"][str(m)].get("score")
        reference_class=None if ref is None else ref["effective_alignment_class"]
        wrong=[r["message"][str(m)]["score"] for r in scored if r["effective_alignment_class"] != reference_class]
        best_wrong=min(wrong) if wrong else None
        equivalent_count=sum(r["effective_alignment_class"]==reference_class for r in rows) if ref is not None else 0
        ranking[str(m)]={"scored_candidate_count":len(scored),"best_candidate":None if not ordered else {x:ordered[0][x] for x in ("a","b","g","effective_alignment_class")}|{"score":ordered[0]["message"][str(m)]["score"]},"reporting_only_reference":reference,"reference_exact_path_score":ref_score,"reference_effective_alignment_class":reference_class,"equivalent_path_count_in_reference_class":equivalent_count,"best_wrong_time_score_excluding_reference_class":best_wrong,"reference_minus_best_wrong_time":None if ref_score is None or best_wrong is None else ref_score-best_wrong}
    return {"video":video,"candidate_count":len(rows),"effective_alignment_class_count":class_next,"candidates":rows,"three_condition_message_competition":pairs,"ranking_and_true_wrong_time_gaps":ranking}

def run(config: dict[str,Any], output: Path) -> dict[str,Any]:
    import torch
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True); dump(output/"config.json",config)
    result={"status":"SETUP_RUNNING","science_denominator":0,"fixed_counts":{"source_frames":181,"cut_frames":178,"speed_frames":145,"vae_decodes":3,"vae_encodes":28},"actual_calls":{"vae_decode_attempted":0,"vae_decode_completed":0,"vae_encode_attempted":0,"vae_encode_completed":0},"failures":[]}
    dump(output/"calls.json", {"generation_invocations":1,"transformer_calls":"recorded_from_actual_long_generation","vae_decodes_planned":3,"vae_encodes_planned":28,"receiver_start_offsets":[0,1,2,3],"normal_edit_lineage":"saved_normal_mp4 -> decode_rgb24 -> fixed_lossy_h264_yuv420p_encode"})
    dump(output/"result.json",result)
    try:
        generated=generate_terminal_latent(config); terminal,vae=generated.normalized_latent,generated.vae; result["generation"]=generated.metadata
        torch.save(terminal.detach().cpu(),output/"shared_terminal_normalized.pt")
        result["status"]="TERMINAL_GENERATED"; dump(output/"result.json",result)
        print("C2-T1: shared long terminal generated and persisted", flush=True)
        if int(terminal.shape[2])!=46: raise ValueError("long terminal must contain special+45 ordinary groups")
        messages={0:state_sequence(config["payload_key_utf8"],0),1:state_sequence(config["payload_key_utf8"],1)}; result["state_sequences"]={str(k):v for k,v in messages.items()}
        arms={"OFF":terminal.detach().clone()}; evidence={}
        (output/"actual_writing_evidence").mkdir(parents=True, exist_ok=True)
        for m in (0,1): arms[f"MESSAGE_{m}"],evidence[str(m)],snap=write_states(terminal,messages[m]); torch.save(snap,output/f"actual_writing_evidence/message_{m}_blocks.pt")
        dump(output/"actual_writing_evidence/write_records.json",evidence)
        result["status"]="WRITING_EVIDENCE_COMPLETE"; dump(output/"result.json",result)
        print("C2-T1: actual FP32 writing evidence persisted", flush=True)
        videos={}; received={}; decoded={}; precodec={}
        for arm,latent in arms.items():
            arm_dir=output/"precodec_and_normal"/arm; arm_dir.mkdir(parents=True,exist_ok=True); result["actual_calls"]["vae_decode_attempted"]+=1; rgb=decode_normalized_latent(vae,latent); result["actual_calls"]["vae_decode_completed"]+=1; precodec[arm]=rgb; torch.save(rgb.detach().cpu(),arm_dir/"precodec_rgb.pt")
            normal=arm_dir/"normal.mp4"; decoded[arm]=ffmpeg_roundtrip(rgb,normal,fps=int(config["generation"]["fps"]),crf=int(config["codec"]["crf"])); videos[f"{arm}_NORMAL"]=normal
        result["quality_and_temporal_change"]={arm:{"precodec_vs_off":rgb_quality_metrics(precodec["OFF"], precodec[arm]),"normal_mp4_rgb24_vs_off":rgb_quality_metrics(decoded["OFF"], decoded[arm])} for arm in arms if arm!="OFF"}
        for m in (0,1):
            base=videos[f"MESSAGE_{m}_NORMAL"]; raster=read_mp4(base); cut=output/"received_videos"/f"MESSAGE_{m}_CUT3.mp4"; speed=output/"received_videos"/f"MESSAGE_{m}_SPEED125.mp4"
            encode_rgb(raster[3:],cut,8,18); encode_rgb(raster[[math.floor(1.25*j) for j in range(145)]],speed,8,18); videos[f"MESSAGE_{m}_CUT3"],videos[f"MESSAGE_{m}_SPEED125"]=cut,speed
        result["status"]="SEVEN_RECEIVED_VIDEOS_COMPLETE"; result["video_paths"]={name:str(path.relative_to(output)) for name,path in videos.items()}; dump(output/"result.json",result)
        print("C2-T1: seven received videos persisted", flush=True)
        for name,path in videos.items():
            raster=read_mp4(path); received[name]={"path":str(path.relative_to(output)),"frames":int(raster.shape[0]),"g":{}}
            for g in range(4):
                d=output/"receiver_observations"/name/f"g{g}"; d.mkdir(parents=True); result["actual_calls"]["vae_encode_attempted"]+=1
                try:
                    row=receiver_observation(vae,raster,g,d)
                    if row["status"]=="COMPLETE": result["actual_calls"]["vae_encode_completed"]+=1
                except BaseException as exc:
                    count,tail=observed_count(int(raster.shape[0]),g); row={"g":g,"received_frames":int(raster.shape[0]),"complete_ordinary_groups_expected":count,"incomplete_tail_frames":tail,"complete_groups_available_from_latent":0,"q":[],"status":"FAILED","error":repr(exc)}; result["failures"].append({"video":name,"g":g,"error":repr(exc)})
                dump(d/"observation.json",row); received[name]["g"][str(g)]=row
                print(f"C2-T1: receiver {name} g={g} {row['status']}", flush=True)
        result["received_videos"]=received; dump(output/"receiver_observations/index.json",received)
        result["status"]="RECEIVER_OBSERVATIONS_COMPLETE"; dump(output/"result.json",result)
        print("C2-T1: all receiver observations persisted", flush=True)
        def reference_for(name: str) -> dict[str,Any]:
            kind="cut3" if name.endswith("CUT3") else "speed125" if name.endswith("SPEED125") else "normal"
            return config["reporting_only_reference"][kind]
        evaluations={name:evaluate(name,{"received_frames":row["frames"],**row["g"]},config,messages,reference_for(name)) for name,row in received.items()}
        dump(output/"candidate_evaluations.json",evaluations); result["evaluation_summary"]={k:{x:v[x] for x in ("candidate_count","effective_alignment_class_count")} for k,v in evaluations.items()}
        dump(output/"calls.json", {**json.loads((output/"calls.json").read_text()), "actual_calls":result["actual_calls"]})
        result["status"]="EXECUTED_REQUIRES_RANKING_REVIEW"; result["limitations"]=["No FPR or PASS claim; ranks and score gaps only.","Cut and speed conditions are joint edit-plus-second-lossy-reencoding conditions, not pure temporal causation.","All grid candidates and missing/failure rows are retained; reporting-only references do not filter candidates."]
    except BaseException as exc:
        result["status"]="FAILED"; result["failures"].append(repr(exc)); dump(output/"result.json",result); raise
    dump(output/"result.json",result); return result

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output",type=Path,required=True); args=p.parse_args(); run(json.loads(args.config.read_text()),args.output)
if __name__ == "__main__": main()
