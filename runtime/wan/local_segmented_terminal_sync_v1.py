"""B clean cached segmented host + terminal state only. Fixed35 slots, D1/E2."""
from pathlib import Path
import json,math,time,resource
import numpy as np
import torch
from main.tube_state import local_segmented_terminal_sync_v1 as state
from main.tube_state import b_segmented_payload_v1 as payload
from runtime.wan import local_joint_positive_projection_margin_v1 as shared
from runtime.wan.local_joint_positive_projection_margin_quality_v1 import distortion
from runtime.wan.local_segmented_terminal_sync_evaluation_v1 import evaluate
dump=shared.dump
DEFAULT_MODEL_ID=shared.DEFAULT_MODEL_ID

def execution_artifacts(output):
    output=Path(output)
    names=("attempt.json","result.json","checkpoint.json","ledger.json","protocol.json","construction.json",
           "state_completion.json","blind_completion.json","evaluation.json")
    found=[n+s for n in names for s in ("",".tmp") if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for folder in ("blind","features","media","quality","latent_difference"):
        root=output/folder
        if root.is_dir():found.extend(str(p.relative_to(output)) for p in root.rglob("*") if p.is_file() or p.is_symlink())
    return sorted(found)

def units():
    out={"source/P":"terminal_load","source/native":"native_rgb8_load","construct":"state_only_construct","vae":"vae_load",
         "S/decode":"ordinary_decode","S/quantize":"rgb8_quantize","latent/P_to_S":"latent_difference"}
    out.update({f"codebook/{k}":"public_state_codebook" for k in state.KEYS})
    for arm in state.ARMS:
        for name,kind in (("save_mp4","mp4_save"),("probe","ffprobe"),("read_mp4","mp4_decode"),("encode_mp4","posterior_encode")):
            out[f"{arm}/{name}"]=kind
        for layer in state.LAYERS:
            for key in state.KEYS:
                out[f"{arm}/{layer}/{key}/state"]="state_read"
                out[f"{arm}/{layer}/{key}/payload"]="segment_fft_read"
    out.update({f"quality/{layer}":"paired_quality" for layer in ("rgb8","mp4")})
    return out

def fixed_denominator():
    return dict(tables=8,state_positions=176,state_competitors=3696,roi_pilot_scores=15488,pooled_pilot_scores=3872,
                payload_segments=32,payload_bits=256,quality_files=2,frames_each=181,adjacent_each=180)

def unavailable_delta(status):
    return dict(status=status,times=[dict(latent_time=t,status=status) for t in range(46)])

def protocol():
    return dict(method="B segmented multi-step payload + terminal-only state",preparation_status="USER_ADOPTED_FIXED_CLEAN_CANDIDATE",
        net_bits=32,segment_times=[[1,11],[12,22],[23,33],[34,44]],guard_times=[0,45],
        generation=dict(native_steps=50,active_sampling_indices=list(range(25,50)),alpha=.5,mask_count=44160,eta=5520,CFG=5),
        actual_new_generation=0,cached_source_is_independent=False,
        state=dict(domain=state.DOMAIN,directions=22,gamma_s=state.GAMMA_S,rois=state.ROIS,channels=list(range(4,16)),
                   payload_directions=0,fp32_materializations=1,norm_cap=None),
        observation="clean public181 grid; per-window state argmax; exact ties unresolved; no edited path",
        payload="clean nominal full-cell local4x8 read; no global repeated-message surrogate",
        media=dict(D=1,E=2,VAE_load=1,codec_pairs=2,crf=18,fps=8,pixel_format="yuv420p"),
        fixed_denominator=fixed_denominator(),quality_layers=["rgb8","mp4"],
        quality_scope="cached-P uint8 versus current-new-S pipeline increment; original P VAE environment may differ; not pure state effect",
        missing_measurements=["P preclip/float baseline","edited-video state path"],
        scientific_pass=None)

class Journal(shared.Journal):
    def __init__(self,output,backend):
        self.output,self.backend=output,backend
        self.data={k:dict(kind=v,status="NOT_ENTERED") for k,v in units().items()};self.save()

class WanBackend(shared.WanBackend):
    def load_terminal(self,path):
        a=np.load(path,allow_pickle=False)
        if a.dtype!=np.float32 or a.shape!=state.LATENT_SHAPE or not np.isfinite(a).all():
            raise ValueError("source terminal must be finite FP32 normalized [1,16,46,40,64]")
        return torch.from_numpy(a.copy())
    def load_native(self,path):
        a=np.load(path,allow_pickle=False)
        if a.dtype!=np.uint8 or a.shape!=state.VIDEO_SHAPE:raise ValueError("cached native must be uint8 [181,320,512,3]")
        return torch.from_numpy(a.copy())

def run(source_terminal,source_native,output,truth_record,model_id=DEFAULT_MODEL_ID,*,backend=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    old=execution_artifacts(output)
    if old:
        result=shared.existing_attempt(output,old);result["fixed_denominator"]=fixed_denominator();return result
    try:
        with (output/"attempt.json").open("x") as f:
            json.dump(dict(started_unix=time.time(),source_terminal=str(source_terminal),source_native=str(source_native)),f)
    except FileExistsError:return shared.existing_attempt(output,["attempt.json"])
    backend=backend or WanBackend(model_id)
    ledger=Journal(output,backend)
    observations={f"{arm}/{layer}/{key}":dict(state=state.unavailable_state("NOT_ENTERED"),payload=state.unavailable_payload("NOT_ENTERED"))
                  for arm in state.ARMS for layer in state.LAYERS for key in state.KEYS}
    result=dict(status="RUNNING",protocol=protocol(),fixed_denominator=fixed_denominator(),scientific_pass=None,
                quality={layer:with_quality_peaks(shared.unavailable_quality("NOT_ENTERED")) for layer in ("rgb8","mp4")},
                construction=dict(status="NOT_ENTERED"),latent_difference=unavailable_delta("NOT_ENTERED"))
    books={};features={};media={};started=time.perf_counter()
    def checkpoint():
        result["counts"]=ledger.counts();dump(output/"checkpoint.json",result)
    def save_tensor(folder,arm,name,value):
        p=output/folder/arm/(name+".pt");p.parent.mkdir(parents=True,exist_ok=True);torch.save(value,p);return value
    def state_read(arm,layer,feature):
        for key_id,key in state.KEYS.items():
            slot=f"{arm}/{layer}/{key_id}"
            if key_id not in books:continue
            def call():
                raw=state.observe_state(feature,key,books[key_id]);dump(output/"blind"/slot/"state.json",raw);return raw
            value=ledger.call(slot+"/state",call)
            observations[slot]["state"]=value if value is not None else state.unavailable_state(ledger.data[slot+"/state"]["status"])
        checkpoint()
    dump(output/"protocol.json",result["protocol"]);checkpoint()
    if getattr(backend,"device","cpu")=="cuda":torch.cuda.reset_peak_memory_stats()
    try:
        for key_id,key in state.KEYS.items():
            b=ledger.call("codebook/"+key_id,lambda key=key:state.public_codebook(key))
            if b is not None:books[key_id]=b
        p=ledger.call("source/P",lambda:save_tensor("features","P","terminal",backend.load_terminal(source_terminal)))
        if p is not None:
            state.validate_latent(p)
            features["P","terminal"]=p;state_read("P","terminal",p)
        native=ledger.call("source/native",lambda:save_tensor("media","P","rgb8",backend.load_native(source_native)))
        if native is not None:media["P","rgb8"]=shared.validate_rgb(native,pixels=True)
        host=observations["P/terminal/CORRECT"]["state"]
        if p is not None and "CORRECT" in books and host["status"]=="READ":
            def construct():
                s,receipt=state.write_state_only(p,host,books["CORRECT"])
                save_tensor("features","S","terminal",s);dump(output/"construction.json",receipt);return s,receipt
            made=ledger.call("construct",construct)
            if made is not None:
                features["S","terminal"],result["construction"]=made;state_read("S","terminal",made[0])
                d=ledger.call("latent/P_to_S",lambda:state.delta_statistics(p,made[0]))
                result["latent_difference"]=dict(status="MEASURED",**d) if d is not None else unavailable_delta(ledger.data["latent/P_to_S"]["status"])
                dump(output/"latent_difference"/"P_to_S.json",result["latent_difference"])
            else:result["construction"]=dict(ledger.data["construct"])
        if ("S","terminal") in features or ("P","rgb8") in media:
            loaded=ledger.call("vae",backend.load);result["model"]=loaded or dict(ledger.data["vae"])
            if loaded is not None:
                if ("S","terminal") in features:
                    decoded=ledger.call("S/decode",lambda:backend.decode(features["S","terminal"]))
                    if decoded is not None:
                        def quantize():
                            rgb=shared.validate_rgb(decoded["float_rgb"])
                            return save_tensor("media","S","rgb8",backend.quantize(rgb))
                        pixels=ledger.call("S/quantize",quantize)
                        if pixels is not None:media["S","rgb8"]=pixels
                for arm in state.ARMS:
                    if (arm,"rgb8") not in media:continue
                    path=output/"media"/(arm+".mp4")
                    saved=ledger.call(arm+"/save_mp4",lambda arm=arm,path=path:backend.save_mp4(media[arm,"rgb8"],path))
                    if saved is None:continue
                    probe=ledger.call(arm+"/probe",lambda path=path:backend.probe(path))
                    if probe is None:continue
                    dump(output/"media"/(arm+"_probe.json"),probe)
                    rgb=ledger.call(arm+"/read_mp4",lambda arm=arm,path=path,probe=probe:save_tensor("media",arm,"mp4_rgb8",backend.read_mp4(path,probe)))
                    if rgb is None:continue
                    media[arm,"mp4"]=rgb
                    z=ledger.call(arm+"/encode_mp4",lambda arm=arm,rgb=rgb:save_tensor("features",arm,"mp4",backend.encode(rgb)))
                    if z is not None:features[arm,"mp4"]=z;state_read(arm,"mp4",z)
        # Freeze every state table/clean plan before any payload reader or evaluator.
        clean_plan=payload.segment_plan(list(range(181)))
        for slot,o in observations.items():
            dump(output/"blind"/slot/"state.json",o["state"]);dump(output/"blind"/slot/"clean_payload_plan.json",clean_plan)
        dump(output/"state_completion.json",dict(completed_unix=time.time(),truth_used=False,message_used=False,tables=8))
        for arm in state.ARMS:
            for layer in state.LAYERS:
                if (arm,layer) not in features:continue
                for key_id,key in state.KEYS.items():
                    slot=f"{arm}/{layer}/{key_id}";directory=output/"blind"/slot
                    def read(feature=features[arm,layer],key=key,directory=directory):
                        raw=payload.read_segments(feature,key,list(range(181)))
                        signed=raw.pop("signed_votes");zeros=raw.pop("zero_coefficient_mask")
                        np.savez(directory/"votes.npz",signed_votes=signed,zero_coefficient_mask=zeros)
                        dump(directory/"payload.json",raw);return raw
                    raw=ledger.call(slot+"/payload",read)
                    observations[slot]["payload"]=raw if raw is not None else state.unavailable_payload(ledger.data[slot+"/payload"]["status"])
        for layer in ("rgb8","mp4"):
            if all((arm,layer) in media for arm in state.ARMS):
                q=ledger.call("quality/"+layer,lambda layer=layer:with_quality_peaks(distortion(media["P",layer],media["S",layer],meaning="cached-P uint8 versus current-new-S pipeline increment; possible P/S VAE environment confounding; not pure state effect")))
                result["quality"][layer]=q if q is not None else with_quality_peaks(shared.unavailable_quality(ledger.data["quality/"+layer]["status"]))
            else:result["quality"][layer]=with_quality_peaks(shared.unavailable_quality("NOT_ENTERED"))
            dump(output/"quality"/(layer+".json"),result["quality"][layer])
    except BaseException as exc:result["fatal_error"]=f"{type(exc).__name__}: {exc}"
    # An interruption can bypass the normal post-call assignment. Retain its
    # consumed slot identity in every fixed observation/measurement row.
    for slot,o in observations.items():
        for kind,missing in (("state",state.unavailable_state),("payload",state.unavailable_payload)):
            status=ledger.data[slot+"/"+kind]["status"]
            if status in ("FAILED","INTERRUPTED"):
                o[kind]=missing(status)
    if ledger.data["latent/P_to_S"]["status"] in ("FAILED","INTERRUPTED"):
        result["latent_difference"]=unavailable_delta(ledger.data["latent/P_to_S"]["status"])
    if ledger.data["construct"]["status"] in ("FAILED","INTERRUPTED"):
        result["construction"]=dict(ledger.data["construct"])
    for layer in ("rgb8","mp4"):
        status=ledger.data["quality/"+layer]["status"]
        if status in ("FAILED","INTERRUPTED"):
            result["quality"][layer]=with_quality_peaks(shared.unavailable_quality(status))
    for slot,o in observations.items():
        for kind,raw in o.items():dump(output/"blind"/slot/(kind+".json"),raw)
    for layer,q in result["quality"].items():dump(output/"quality"/(layer+".json"),q)
    dump(output/"latent_difference"/"P_to_S.json",result["latent_difference"])
    dump(output/"blind_completion.json",dict(completed_unix=time.time(),tables=8,truth_used=False))
    # Read writer truth only after all persisted blind observations; never imported by receivers.
    try:
        truth_bits=json.loads(Path(truth_record).read_text())["fixed_method_semantics"]["bits"]
        dump(output/"evaluation_truth.json",dict(bits=truth_bits,read_after_blind_completion=True,source=str(truth_record)))
        persisted={slot:{kind:json.loads((output/"blind"/slot/(kind+".json")).read_text()) for kind in ("state","payload")} for slot in observations}
        evaluation,pairs=evaluate(persisted,truth_bits)
        result["evaluation"],result["paired_readout"]=evaluation,pairs;dump(output/"evaluation.json",evaluation)
    except Exception as exc:result["evaluation_error"]=f"{type(exc).__name__}: {exc}"
    result.update(calls=ledger.data,counts=ledger.counts(),seconds=time.perf_counter()-started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    if getattr(backend,"device","cpu")=="cuda":
        result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    result["status"]="EXECUTION_COMPLETE" if not result.get("fatal_error") and not result.get("evaluation_error") and all(x["status"]=="RETURNED" for x in ledger.data.values()) else "INCOMPLETE"
    dump(output/"result.json",result);checkpoint();return result



def with_quality_peaks(quality):
    """Reduce existing scalar rows only; retain every exact maximum location."""
    source_status=quality.get("status","UNKNOWN")
    def peak(rows,region,metric,fields,denominator):
        result=dict(status="UNAVAILABLE",source_status=source_status,value=None,locations=[],
                    row_denominator=denominator,observed_rows=0)
        if source_status != "SCORED":
            result["reason"]="QUALITY_NOT_SCORED"
            return result
        values=[]
        for row in rows:
            value=row.get(region,{}).get(metric)
            location=tuple(row.get(field) for field in fields)
            if (type(value) in (int,float) and math.isfinite(value) and value >= 0
                    and all(type(i) is int for i in location)):
                values.append((value,location))
        result["observed_rows"]=len(values)
        expected=[(i,) if len(fields)==1 else (i,i+1) for i in range(denominator or 0)]
        if not expected or len(rows) != denominator or sorted(loc for _,loc in values) != expected:
            result["reason"]="INCOMPLETE_OR_UNAVAILABLE_METRIC_ROWS"
            return result
        maximum=max(value for value,_ in values)
        locations=sorted(loc for value,loc in values if value == maximum)
        result.update(status="SCORED",value=maximum,
                      locations=[dict(zip(fields,loc)) for loc in locations])
        return result
    peaks=dict(tie_policy="exact == maximum; retain all ties in ascending frame or (before,after) order; no epsilon",
               meaning="maxima of saved per-frame/per-adjacent scalar rows, not new measurements")
    for name,regions,fields,denominator in (
        ("frames",("whole","outside_nominal_support"),("frame",),quality.get("frame_denominator")),
        ("adjacent",("whole","outside_union_nominal_support"),("frame_before","frame_after"),quality.get("adjacent_denominator"))):
        rows=quality.get(name,[])
        peaks[name]={region:{metric:peak(rows,region,metric,fields,denominator) for metric in ("rms","linf")}
                     for region in regions}
    return dict(quality,peaks=peaks)
