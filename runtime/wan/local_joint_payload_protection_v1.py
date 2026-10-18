"""Fixed user-run card51: frozen P/J0 and one new ECC terminal, D3/E3."""
import json
import math
from pathlib import Path
import platform
import resource
import time

import torch

from main.tube_state import local_joint_payload_protection_v1 as method
from main.tube_state import local_payload_secded_39_32_v1 as ecc
from main.tube_state.grow_video_reference import message_bits, read_latent_bits
from runtime.wan import local_joint_positive_projection_margin_v1 as shared
from runtime.wan.local_joint_positive_projection_margin_quality_v1 import distortion
from runtime.wan.local_joint_payload_protection_evaluation_v1 import evaluate_all

DEFAULT_MODEL_ID=shared.DEFAULT_MODEL_ID
dump=shared.dump
QUALITY_LAYERS=shared.QUALITY_LAYERS
PAIRS=(("P","J0"),("P","J_ECC"),("J0","J_ECC"))


class WanBackend(shared.WanBackend):
    def load_terminal(self,path):
        value=torch.load(path,map_location="cpu",weights_only=True)
        method.validate_latent(value)
        if value.dtype != torch.float32:
            raise ValueError("saved P/J0 terminal must already be FP32")
        return value.detach()


def execution_artifacts(output):
    output=Path(output)
    names=("attempt.json","result.json","checkpoint.json","ledger.json","cpu_ledger.json",
           "evaluation.json","protocol.json","construction.json","writer_encoding.json","blind_completion.json")
    found=[n+s for n in names for s in ("",".tmp") if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for folder in ("blind","features","media","quality","latent_difference"):
        root=output/folder
        if root.is_dir():
            found.extend(str(p.relative_to(output)) for p in root.rglob("*") if p.is_file() or p.is_symlink())
    return sorted(found)


def fixed_denominator():
    return dict(sources=1,arms=3,layers=2,keys=2,local_tables=12,state_positions=264,state_competitors=5544,
        roi_scalar_scores=64416,pooled_scalar_scores=16104,raw_symbol_observation_slots=10296,
        raw_systematic_data_bits=8448,raw_parity_observation_slots=1848,ecc_blocks=264,
        ecc_local_message_bits=8448,aggregate_bits_per_branch=384,legacy_bits=384,
        J_ECC_blocks=88,J_ECC_raw_symbol_slots=3432,J_ECC_mp4_correct_symbols=858,
        J_ECC_mp4_correct_data_bits=704,J_ECC_mp4_correct_parity_bits=154,
        frames_per_quality=181,adjacent_pairs_per_quality=180,paired_quality=12,processing_quality=9,
        meaning="observation slots include uncoded P/J0 and WRONG controls; no combined transmitted BER")


def protocol_record():
    return dict(name="local_joint_payload_protection_v1",method_status="ADOPTED_FOR_IMPLEMENTATION_CARD51",
        code=dict(n=39,k=32,d=4,data_coordinates=ecc.DATA_COORDINATES,parity_coordinates=ecc.PARITY_COORDINATES,
                  erasure_rule="any raw erasure -> whole32 unresolved",over_radius="3+ errors may miscorrect;4+ may be undetected"),
        gamma_s=method.GAMMA_S,gamma_data=method.GAMMA_P,gamma_parity=method.GAMMA_P,norm_cap=None,
        directions=61,original_prefix_directions=54,rois=method.ROIS,arms=method.ARMS,layers=method.LAYERS,keys=method.KEYS,
        path="pilot-only, unknown initial, path[k]=(initial+k)%22, fsum22, exact ties unresolved",
        aggregation="raw soft fsum/22 retained; ECC decoded sign times raw abs soft, rejected block absent, fsum/22",
        construction="from saved FP32 P, frozen54 continuous corrections plus7 parity, one FP32 materialization; never from J0",
        media=dict(VAE_load=1,D=3,E=3,fps=8,crf=18,pixel_format="yuv420p"),
        receiver_scope="complete clean public grid, terminal/MP4 endpoints only; no edit synchronization",
        fixed_denominator=fixed_denominator(),new_candidates=1)


def units():
    result={"source/P":"terminal_load","source/J0":"terminal_load","construct":"construct","vae":"vae_load"}
    result.update({f"codebook/{key}":"public_codebook" for key in method.KEYS})
    for arm in method.ARMS:
        for name,kind in (("decode","ordinary_decode"),("quantize","rgb8_quantize"),("save_mp4","mp4_save"),
                          ("probe","ffprobe"),("read_mp4","mp4_decode"),("encode_mp4","posterior_encode")):
            result[f"{arm}/{name}"]=kind
        for layer in method.LAYERS:
            for key in method.KEYS:
                result[f"{arm}/{layer}/{key}/local"]="joint_read"
                result[f"{arm}/{layer}/{key}/legacy"]="payload_fft_read"
        for transition in ("clip","quantize","codec"):
            result[f"quality/{arm}/{transition}"]="processing_distortion"
    for left,right in PAIRS:
        for layer in QUALITY_LAYERS:
            result[f"quality/{left}_to_{right}/{layer}"]="paired_quality"
    return result


def cpu_units():
    result={"encode_message":"message_encode"}
    for arm in method.ARMS:
        for layer in method.LAYERS:
            for key in method.KEYS:
                slot=f"{arm}/{layer}/{key}"
                result[slot+"/path"]="pilot_temporal"
                for k in method.STATES: result[f"{slot}/ecc/{k}"]="ecc_block"
                for branch in ("raw","ecc"): result[f"{slot}/aggregate/{branch}"]="aggregate"
    for left,right in PAIRS: result[f"latent/{left}_to_{right}"]="latent_difference"
    return result


class Journal(shared.Journal):
    def __init__(self,output,backend,plan,filename):
        self.output,self.backend,self.filename=output,backend,filename
        self.data={name:dict(kind=kind,status="NOT_ENTERED") for name,kind in plan.items()}
        self.save()
    def save(self):
        dump(self.output/self.filename,self.data)


def _unavailable_delta(status):
    return dict(status=status,global_delta=None,support=None,outside_support=None,legacy_channels_0_3=None,
                times=[dict(latent_time=t,status=status) for t in range(46)])


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


def measured_quality(left,right,*,meaning):
    return with_quality_peaks(distortion(left,right,meaning=meaning))


def run(source_p,source_j0,output,model_id=DEFAULT_MODEL_ID,*,backend=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    existing=execution_artifacts(output)
    if existing:
        result=shared.existing_attempt(output,existing)
        result["fixed_denominator"]=fixed_denominator()
        return result
    try:
        with (output/"attempt.json").open("x") as stream:
            json.dump(dict(source_p=str(source_p),source_j0=str(source_j0),started_unix=time.time()),stream)
    except FileExistsError:
        return dict(status="EXISTING_ATTEMPT_NOT_REEXECUTED",fixed_denominator=fixed_denominator())
    backend=backend or WanBackend(model_id)
    core=Journal(output,backend,units(),"ledger.json")
    cpu=Journal(output,backend,cpu_units(),"cpu_ledger.json")
    observations={f"{a}/{l}/{k}":dict(local=method.unavailable_local("NOT_ENTERED"),legacy=dict(status="NOT_ENTERED"))
                  for a in method.ARMS for l in method.LAYERS for k in method.KEYS}
    result=dict(status="RUNNING",protocol=protocol_record(),fixed_denominator=fixed_denominator(),
        source_paths=dict(P=str(source_p),J0=str(source_j0)),construction=dict(status="NOT_ENTERED"),quality={},latent_difference={},
        environment=dict(python=platform.python_version(),torch=torch.__version__),scientific_pass=None,
        evidence_ceiling="single-host paired clean terminal/MP4 coded-channel diagnostic; no native multistep or edit-sync claim",
        zero_budget=["DiT","native_sampling","gradient","VJP","retry","amplitude_change","additional_candidate"])
    for left,right in PAIRS:
        result["latent_difference"][f"{left}_to_{right}"]=_unavailable_delta("NOT_ENTERED")
        for layer in QUALITY_LAYERS: result["quality"][f"{left}_to_{right}/{layer}"]=with_quality_peaks(shared.unavailable_quality("NOT_ENTERED"))
    for arm in method.ARMS:
        for phase in ("clip","quantize","codec"): result["quality"][f"{arm}/{phase}"]=with_quality_peaks(shared.unavailable_quality("NOT_ENTERED"))
    books={};candidates={};media={};encoding=None

    def checkpoint():
        result["counts"],result["cpu_counts"]=core.counts(),cpu.counts()
        dump(output/"checkpoint.json",result)

    def save_tensor(folder,arm,name,value):
        path=output/folder/arm/f"{name}.pt";path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(value,path)
        return value

    def blind_read(arm,layer,feature):
        for key_id,key in method.KEYS.items():
            slot=f"{arm}/{layer}/{key_id}";directory=output/"blind"/arm/layer/key_id
            if key_id in books:
                local=method.unavailable_local("NOT_ENTERED")
                def observe():
                    raw=method.observe(feature,key,books[key_id]);local["raw_scores"]=raw
                    dump(directory/"raw_scores.json",raw)
                    selection=cpu.call(slot+"/path",lambda:method.select_pilot_path([r[:22] for r in raw["pooled_scores"]]))
                    if selection is None:
                        local["selection"]=dict(cpu.data[slot+"/path"],selected_path=None)
                        local["status"]="PARTIAL"
                        return local
                    local["selection"]=selection;dump(directory/"selection.json",selection)
                    # Read the persisted selection, then persist raw39 before any ECC.
                    selection=json.loads((directory/"selection.json").read_text())
                    symbols=method.despread(raw,selection);local["symbols"]=symbols
                    dump(directory/"raw_symbols.json",symbols)
                    for k,pos in enumerate(symbols["positions"]):
                        decoded=cpu.call(f"{slot}/ecc/{k}",lambda pos=pos:ecc.decode(pos["codeword_bits"]))
                        if decoded is None:
                            decoded=dict(local["ecc_blocks"][k],status=cpu.data[f"{slot}/ecc/{k}"]["status"])
                        local["ecc_blocks"][k]=dict(decoded,position=k)
                        dump(directory/"ecc.json",dict(truth_used=False,blocks=local["ecc_blocks"]))
                    raw_soft=[pos["data_soft"] for pos in symbols["positions"]]
                    post_soft=[method.ecc_soft(pos,d) for pos,d in zip(symbols["positions"],local["ecc_blocks"])]
                    for branch,soft in (("raw",raw_soft),("ecc",post_soft)):
                        value=cpu.call(f"{slot}/aggregate/{branch}",lambda soft=soft:method.aggregate(soft))
                        if value is not None:
                            local[branch+"_aggregate"]=value
                        else:
                            local[branch+"_aggregate"]=[dict(bit=b,decoded=None,status=cpu.data[f"{slot}/aggregate/{branch}"]["status"]) for b in range(32)]
                        dump(directory/(branch+"_aggregate.json"),dict(truth_used=False,rows=local[branch+"_aggregate"]))
                    local["status"]="READ" if all(row["status"] == "RETURNED" for name,row in cpu.data.items() if name.startswith(slot+"/")) else "PARTIAL"
                    dump(directory/"local.json",local)
                    return local
                observed=core.call(slot+"/local",observe)
                if observed is None:
                    local["status"]=core.data[slot+"/local"]["status"]
                    local["detail"]=core.data[slot+"/local"]
                    # Keep any raw/selection/partial ECC that was already produced.
                    observed=local
                observations[slot]["local"]=observed
                dump(directory/"local.json",observed)
            def read_legacy():
                raw=read_latent_bits(feature,key);dump(directory/"legacy.json",raw);return raw
            raw=core.call(slot+"/legacy",read_legacy)
            observations[slot]["legacy"]=raw if raw is not None else dict(core.data[slot+"/legacy"])
            checkpoint()

    def quality(name,left,right,meaning):
        value=core.call("quality/"+name,lambda:measured_quality(left,right,meaning=meaning))
        result["quality"][name]=value if value is not None else with_quality_peaks(shared.unavailable_quality(core.data["quality/"+name]["status"]))
        dump(output/"quality"/(name+".json"),result["quality"][name]);checkpoint()

    started=time.perf_counter()
    dump(output/"protocol.json",result["protocol"]);checkpoint()
    if getattr(backend,"device","cpu") == "cuda": torch.cuda.reset_peak_memory_stats()
    try:
        def encode_message():
            bits=message_bits("OKOK")
            value=dict(message=bits,codeword=ecc.encode(bits),status="ENCODED",scope="writer/evaluation only; never blind receiver input")
            dump(output/"writer_encoding.json",value)
            return value
        encoding=cpu.call("encode_message",encode_message)
        for arm,path in (("P",source_p),("J0",source_j0)):
            def load(path=path,arm=arm):
                value=backend.load_terminal(path)
                method.validate_latent(value)
                if value.dtype != torch.float32: raise ValueError("saved terminal must be FP32")
                return save_tensor("features",arm,"terminal",value)
            value=core.call("source/"+arm,load)
            if value is not None: candidates[arm]=value
        if candidates:
            for key_id,key in method.KEYS.items():
                book=core.call("codebook/"+key_id,lambda key=key:method.public_codebook(key))
                if book is not None: books[key_id]=book
            if "P" in candidates: blind_read("P","terminal",candidates["P"])
            if "J0" in candidates: blind_read("J0","terminal",candidates["J0"])
            host=observations["P/terminal/CORRECT"]["local"]["raw_scores"]
            if "P" in candidates and "CORRECT" in books and host.get("status") == "READ" and encoding is not None:
                def construct():
                    candidate,record=method.write_projection(candidates["P"],host,encoding["codeword"],books["CORRECT"])
                    save_tensor("features","J_ECC","terminal",candidate)
                    dump(output/"construction.json",record)
                    return candidate,record
                made=core.call("construct",construct)
                if made is not None:
                    candidates["J_ECC"],result["construction"]=made
                    blind_read("J_ECC","terminal",candidates["J_ECC"])
                else: result["construction"]=dict(core.data["construct"])
            for left,right in PAIRS:
                name=f"{left}_to_{right}"
                if left in candidates and right in candidates:
                    value=cpu.call("latent/"+name,lambda left=left,right=right:method.base.delta_statistics(candidates[left],candidates[right]))
                    result["latent_difference"][name]=dict(status="MEASURED",**value) if value is not None else _unavailable_delta(cpu.data["latent/"+name]["status"])
                dump(output/"latent_difference"/(name+".json"),result["latent_difference"][name])
            loaded=core.call("vae",backend.load)
            result["model"]=loaded if loaded is not None else dict(core.data["vae"])
            if loaded is not None:
                for arm in method.ARMS:  # fixed P -> frozen J0 -> J_ECC media order
                    if arm not in candidates: continue
                    decoded=core.call(arm+"/decode",lambda arm=arm:backend.decode(candidates[arm]))
                    if decoded is None: continue
                    for layer in ("preclip","float_rgb"):
                        value=shared.validate_rgb(decoded[layer],preclip=layer == "preclip")
                        media[arm,layer]=save_tensor("media",arm,layer,value)
                    quality(arm+"/clip",decoded["preclip"],decoded["float_rgb"],"within-arm clipping")
                    pixels=core.call(arm+"/quantize",lambda:save_tensor("media",arm,"rgb8",backend.quantize(decoded["float_rgb"])))
                    if pixels is None: continue
                    media[arm,"rgb8"]=pixels
                    quality(arm+"/quantize",decoded["float_rgb"],pixels,"within-arm quantization")
                    path=output/"media"/f"{arm}.mp4"
                    saved=core.call(arm+"/save_mp4",lambda:backend.save_mp4(pixels,path))
                    if saved is None: continue
                    probe=core.call(arm+"/probe",lambda:backend.probe(path))
                    if probe is None: continue
                    dump(output/"media"/(arm+"_probe.json"),probe)
                    reopened=core.call(arm+"/read_mp4",lambda:save_tensor("media",arm,"mp4_readback_rgb8",backend.read_mp4(path,probe)))
                    if reopened is None: continue
                    media[arm,"mp4"]=reopened
                    quality(arm+"/codec",pixels,reopened,"within-arm codec distortion; not additive to paired difference")
                    feature=core.call(arm+"/encode_mp4",lambda:save_tensor("features",arm,"mp4",backend.encode(reopened)))
                    if feature is not None: blind_read(arm,"mp4",feature)
            for left,right in PAIRS:
                for layer in QUALITY_LAYERS:
                    if all((arm,layer) in media for arm in (left,right)):
                        quality(f"{left}_to_{right}/{layer}",media[left,layer],media[right,layer],
                                f"{right} minus {left}; incremental paired distortion, no perceptual PASS")
    except BaseException as exc:
        result["fatal_error"]=f"{type(exc).__name__}: {exc}"
    # Finalize every fixed blind slot, including failures/not-entered, before evaluator.
    for slot,observed in observations.items():
        directory=output/"blind"/slot
        dump(directory/"local.json",observed["local"])
        dump(directory/"legacy.json",observed["legacy"])
    dump(output/"blind_completion.json",dict(completed_unix=time.time(),tables=12,truth_used=False,
        statuses={s:{k:v["status"] for k,v in o.items()} for s,o in observations.items()}))
    # Semantic truth is read from the writer record only in this separate stage.
    if encoding is not None:
        truth=json.loads((output/"writer_encoding.json").read_text())
        persisted={slot:dict(local=json.loads((output/"blind"/slot/"local.json").read_text()),
                             legacy=json.loads((output/"blind"/slot/"legacy.json").read_text())) for slot in observations}
        evaluation,pairs=evaluate_all(persisted,truth)
        result["evaluation"],result["paired_readout"]=evaluation,pairs
        dump(output/"evaluation.json",evaluation)
    else:
        result["evaluation_status"]="WRITER_TRUTH_UNAVAILABLE"
    result["counts"],result["calls"]=core.counts(),core.data
    result["cpu_counts"],result["cpu_calls"]=cpu.counts(),cpu.data
    result["cpu_arithmetic"]=dict(
        path_scores=sum(o["local"]["selection"].get("path_scores",0) for o in observations.values()),
        pilot_terms=sum(o["local"]["selection"].get("pilot_terms_reduced",0) for o in observations.values()),
        syndrome_reductions=sum(b.get("syndrome_reductions",0) for o in observations.values() for b in o["local"]["ecc_blocks"]))
    result["seconds"]=time.perf_counter()-started
    result["process_peak_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if getattr(backend,"device","cpu") == "cuda":
        result["cuda_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        result["cuda_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
    result["status"]="EXECUTION_COMPLETE" if not result.get("fatal_error") and all(row["status"] == "RETURNED" for ledger in (core.data,cpu.data) for row in ledger.values()) else "INCOMPLETE"
    dump(output/"result.json",result);checkpoint()
    return result
