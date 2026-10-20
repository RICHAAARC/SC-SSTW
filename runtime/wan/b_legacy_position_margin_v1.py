"""User-only one-shot P0/PM clean D2/codec2/full-E2/FFT2 evidence runner."""
from pathlib import Path
import json,time,resource
import torch
from main.tube_state import b_legacy_position_margin_v1 as method
from runtime.wan import c73_identity_v1 as c73
from runtime.wan.c73_identity_quality_v1 import distortion,preclip_record
from runtime.wan.b_legacy_migration_v1 import WanBackend
from runtime.wan.b_legacy_position_margin_evaluation_v1 import evaluate

dump=c73.dump
DEFAULT_MODEL_ID=c73.DEFAULT_MODEL_ID


def fixed_denominator():
    return dict(arms=2,keys=2,positions=44,ROIs=4,direction_dimension=768,sync_tables=8,
        pooled_q=15488,ROI_dots=61952,competitor_margins=15488,clean_diagonal_slots=352,post_expected_correct_key_diagonals=176,
        post_correct_key_other_candidates=7568,post_wrong_key_candidates=7744,post_other_candidates=15312,
        terminal_load=1,candidates=1,VAE_load=1,D=2,quant=2,codec=2,readback=2,E=2,
        nominal_encoder_chunks=92,nominal_decoder_chunks=92,FFT=2,payload_reads=4,
        aggregate_payload_bit_slots=128,local_payload_bit_slots=1408,signed_vote_slots=168960,
        zero_mask_slots=168960,quality_tables=2,quality_frame_rows=362,quality_adjacent_rows=360,
        preclip_tables=2,preclip_frame_rows=362,writer_units=176,ledger_units=38,
        DiT=0,gradients=0,edited_views=0,local_reset_E=0,RGB8_E=0)


def protocol(gamma=method.GAMMA):
    return dict(method="position signature + host-aware nearest halfspace",evidence_level="DIRECTION_EVIDENCE_ONLY",candidate_spec="single recommendation for principal/user review",
        gamma=gamma,support="normalized Wan channels4..15, four8x8 ROIs, times1..44; guards0/45 unchanged",
        public_reference=method.REFERENCE,direction_generator="old uniform/l2/round8; actual rounded norm squared divisor",
        normalization="existing Wan normalized posterior mode, FP32; float64 writer arithmetic then one FP32 cast",
        reference_is_public_not_source_identity=True,old22="candidate replaces old22 from payload-only P0; default/evidence preserved",
        net_user_bits=32,ROIs=[list(x) for x in method.ROIS],terminal_payload_write=False,identity_status="UNCALIBRATED",phase_status="UNCALIBRATED",
        scope="clean complete181 full-history only; canonical payload reads measure side effect, not synchronization",
        reliable_threshold=None,score_gate=None,default_receiver_changed=False,A_changed=False,
        resource_hard_caps=None,fixed_denominator=fixed_denominator(),scientific_pass=None)


def units():
    u={"source":"terminal_load","construct":"position_margin_construct","model":"vae_load",
       "native_audit":"native_support_audit","truth":"post_numeric_truth_read","evaluation":"post_numeric_evaluation"}
    u.update({f"book/{k}":"position_book" for k in method.KEY_IDS})
    for a in method.ARMS:
        for layer in ("native","mp4"):
            for k in method.KEY_IDS:u[f"sync/{a}/{layer}/{k}"]="direction_observation"
        for slot,kind in (("decode","ordinary_decode"),("preclip","preclip_record"),("quantize","rgb8_quantize"),
            ("save_mp4","mp4_save"),("probe","ffprobe"),("read_mp4","mp4_read"),("encode","posterior_encode"),("FFT","shared_payload_fft")):
            u[f"{a}/{slot}"]=kind
        for k in method.KEY_IDS:u[f"payload/{a}/{k}"]="canonical_payload_read"
    u.update({f"quality/{l}":"paired_quality" for l in ("rgb8","mp4")})
    return u


class Journal(c73.Journal):
    def __init__(self,output,backend):
        self.output,self.backend=output,backend
        self.data={k:dict(kind=v,status="NOT_ENTERED") for k,v in units().items()};self.save()


def execution_artifacts(output):
    output=Path(output)
    records=("attempt.json","result.json","checkpoint.json","ledger.json","protocol.json","construction.json","numeric_completion.json","evaluation.json","native_audit.json")
    found=[n+s for n in records for s in ("",".tmp") if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for folder in ("blind","features","media","quality","books","references"):
        root=output/folder
        if root.is_dir():found.extend(str(p.relative_to(output)) for p in root.rglob("*") if p.is_file() or p.is_symlink())
    return sorted(found)


def read_truth(path):
    raw=json.loads(Path(path).read_text());bits=raw["fixed_method_semantics"]["bits"]
    if len(bits)!=32 or any(type(b) is not int or b not in (0,1) for b in bits):raise ValueError("32 binary post-only truth bits required")
    return dict(status="AVAILABLE",bits=bits,source=str(path),role="post-only; never receiver input")


def run(source_p,output,truth_path,model_id=DEFAULT_MODEL_ID,gamma=method.GAMMA,*,backend=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    def existing(artifacts):
        value=c73.existing_attempt(output,artifacts);value["fixed_denominator"]=fixed_denominator();return value
    artifacts=execution_artifacts(output)
    if artifacts:return existing(artifacts)
    try:
        with (output/"attempt.json").open("x") as stream:
            json.dump(dict(started_unix=time.time(),source=str(source_p),failure_consumes_attempt=True),stream);c73.flush_file(stream)
    except FileExistsError:return existing(["attempt.json"])
    backend=backend or WanBackend(model_id);ledger=Journal(output,backend)
    result=dict(status="RUNNING",protocol=protocol(gamma),fixed_denominator=fixed_denominator(),scientific_pass=None)
    sync={f"{a}/{l}/{k}":method.unavailable("NOT_ENTERED") for a in method.ARMS for l in ("native","mp4") for k in method.KEY_IDS}
    payload={f"{a}/{k}":method.unavailable_payload("NOT_ENTERED") for a in method.ARMS for k in method.KEY_IDS}
    quality={l:c73.shared.unavailable_quality("NOT_ENTERED") for l in ("rgb8","mp4")}
    preclip={a:dict(status="NOT_ENTERED",frame_denominator=181,rows=[dict(frame=t,status="NOT_ENTERED") for t in range(181)]) for a in method.ARMS}
    dependencies={};books={};terminals={};pixels={};construction=None;started=time.perf_counter()
    def checkpoint():
        result["counts"]=ledger.counts();dump(output/"checkpoint.json",result)
    def save_tensor(path,value):
        p=output/path;p.parent.mkdir(parents=True,exist_ok=True);torch.save(value,p);return value
    def call(slot,fn,deps=()):
        dependencies[slot]=list(deps)
        if any(ledger.data[d]["status"]!="RETURNED" for d in deps):return None
        return ledger.call(slot,fn)
    def record(slot,fn,path,deps=()):
        def operation():
            value=fn();dump(output/path,value);return value
        return call(slot,operation,deps)
    def unavailable_reason(slot):
        seen=set()
        def visit(s):
            if s in seen:return None
            seen.add(s);row=ledger.data[s]
            if row["status"] in ("FAILED","INTERRUPTED","STARTED"):return s
            for d in dependencies.get(s,[]):
                hit=visit(d)
                if hit:return hit
            return None
        failed=visit(slot)
        if failed:return dict(status=ledger.data[failed]["status"] if failed==slot else "UPSTREAM_"+ledger.data[failed]["status"],unavailable_from=failed)
        return dict(status="NOT_ENTERED",attempt_interruption=result.get("fatal_error"))
    dump(output/"protocol.json",protocol(gamma))
    for n,v in sync.items():dump(output/"blind/sync"/(n+".json"),v)
    for n,v in payload.items():dump(output/"blind/payload"/(n+".json"),v)
    for n,v in quality.items():dump(output/"quality"/(n+".json"),v)
    for n,v in preclip.items():dump(output/"quality/preclip"/(n+".json"),v)
    checkpoint()
    # Predeclare dependencies even if a global interruption prevents reaching a slot.
    dependencies.update({"construct":["source","book/key00","sync/P0/native/key00"],"model":["source"]})
    for a in method.ARMS:
        origin="source" if a=="P0" else "construct"
        for k in method.KEY_IDS:dependencies[f"sync/{a}/native/{k}"]=[origin,"book/"+k]
        chain=[("decode",[origin,"model"]),("preclip",[f"{a}/decode"]),("quantize",[f"{a}/decode"]),
            ("save_mp4",[f"{a}/quantize"]),("probe",[f"{a}/save_mp4"]),("read_mp4",[f"{a}/probe"]),
            ("encode",[f"{a}/read_mp4"]),("FFT",[f"{a}/encode"])]
        for n,d in chain:dependencies[f"{a}/{n}"]=d
        for k in method.KEY_IDS:
            dependencies[f"sync/{a}/mp4/{k}"]=[f"{a}/encode","book/"+k]
            dependencies[f"payload/{a}/{k}"]=[f"{a}/FFT"]
    for l,op in (("rgb8","quantize"),("mp4","read_mp4")):
        dependencies[f"quality/{l}"]=[f"{a}/{op}" for a in method.ARMS]
    try:
        for k,key in zip(method.KEY_IDS,method.KEYS):
            b=call("book/"+k,lambda key=key:method.codebook(key))
            if b is not None:books[k]=b
        z=call("source",lambda:backend.load_terminal(source_p))
        if z is not None:terminals["P0"]=z
        for k in method.KEY_IDS:
            slot=f"sync/P0/native/{k}"
            v=record(slot,lambda k=k:method.observe(terminals["P0"],books[k]),Path("blind/sync/P0/native")/(k+".json"),dependencies[slot])
            if v is not None:sync[f"P0/native/{k}"]=v
        def build():
            z,receipt=method.construct(terminals["P0"],books["key00"],gamma,native_observation=sync["P0/native/key00"])
            save_tensor(Path("features/PM/terminal.pt"),z);dump(output/"construction.json",receipt);return z,receipt
        built=call("construct",build,dependencies["construct"])
        if built is not None:terminals["PM"],construction=built;result["construction"]=construction
        for k in method.KEY_IDS:
            slot=f"sync/PM/native/{k}"
            v=record(slot,lambda k=k:method.observe(terminals["PM"],books[k]),Path("blind/sync/PM/native")/(k+".json"),dependencies[slot])
            if v is not None:sync[f"PM/native/{k}"]=v
        def audit():
            before,after=terminals["P0"],terminals["PM"];delta=after.double()-before.double()
            mask=torch.zeros_like(delta,dtype=torch.bool)
            for y0,y1,x0,x1 in method.ROIS:mask[0,4:16,1:45,y0:y1,x0:x1]=True
            rows=[dict(position=t+1,roi=r,q=float(sync["PM/native/key00"]["roi_q"][t][r][t]),
                       FP32_shortage=max(0,gamma-sync["PM/native/key00"]["roi_q"][t][r][t])) for t in range(44) for r in range(4)]
            return dict(status="MEASURED",units=rows,FP32_shortage_count=sum(x["FP32_shortage"]>0 for x in rows),
                actual_squared_energy=float(delta.square().sum()),outside_support_max=float(delta[~mask].abs().max()),
                payload_channels_max=float(delta[:,:4].abs().max()),guard_max=float(delta[:,:, [0,45]].abs().max()),
                media_payload_cost="must be measured separately",no_reprojection=True)
        result["native_audit"]=record("native_audit",audit,Path("native_audit.json"),("construct","sync/PM/native/key00"))
        loaded=call("model",backend.load,dependencies["model"])
        if loaded is not None:result["model"]=loaded
        for a in method.ARMS:
            slot=f"{a}/decode"
            decoded=call(slot,lambda a=a:backend.decode(terminals[a]),dependencies[slot])
            v=record(f"{a}/preclip",lambda:preclip_record(decoded["preclip"]),Path("quality/preclip")/(a+".json"),dependencies[f"{a}/preclip"])
            if v is not None:preclip[a]=v
            slot=f"{a}/quantize"
            rgb=call(slot,lambda a=a:save_tensor(Path("media")/a/"rgb8.pt",backend.quantize(decoded["float_rgb"])),dependencies[slot])
            del decoded
            if rgb is not None:pixels[a,"rgb8"]=rgb
            path=output/"media"/a/"video.mp4"
            slot=f"{a}/save_mp4"
            record(slot,lambda:backend.save_mp4(rgb,path),Path("media")/a/"codec.json",dependencies[slot])
            slot=f"{a}/probe"
            probe=record(slot,lambda:backend.probe(path),Path("media")/a/"probe.json",dependencies[slot])
            slot=f"{a}/read_mp4"
            readback=call(slot,lambda a=a:save_tensor(Path("media")/a/"readback.pt",backend.read_mp4(path,probe)),dependencies[slot])
            if readback is not None:pixels[a,"mp4"]=readback
            slot=f"{a}/encode"
            z=call(slot,lambda a=a:save_tensor(Path("features")/a/"mp4_posterior.pt",backend.encode_full(readback)),dependencies[slot])
            for k in method.KEY_IDS:
                slot=f"sync/{a}/mp4/{k}"
                v=record(slot,lambda k=k:method.observe(z,books[k]),Path("blind/sync")/a/"mp4"/(k+".json"),dependencies[slot])
                if v is not None:sync[f"{a}/mp4/{k}"]=v
            slot=f"{a}/FFT"
            spectrum=call(slot,lambda:method.payload_spectrum(z),dependencies[slot])
            for k,key in zip(method.KEY_IDS,method.KEYS):
                slot=f"payload/{a}/{k}"
                v=record(slot,lambda key=key:method.payload_from_spectrum(spectrum,key),Path("blind/payload")/a/(k+".json"),dependencies[slot])
                if v is not None:payload[f"{a}/{k}"]=v
            del z,spectrum;checkpoint()
        for l in ("rgb8","mp4"):
            slot="quality/"+l
            v=record(slot,lambda l=l:distortion(pixels["P0",l],pixels["PM",l],includes_guard=False,
                meaning="fresh paired P0 to PM; actual media/temporal cost; no quality cutoff"),Path("quality")/(l+".json"),dependencies[slot])
            if v is not None:quality[l]=v
    except BaseException as exc:result["fatal_error"]=f"{type(exc).__name__}: {exc}"
    for n,v in list(sync.items()):
        if v["status"]=="NOT_ENTERED":
            why=unavailable_reason("sync/"+n);v=method.unavailable(why["status"]);v.update(why);sync[n]=v
        dump(output/"blind/sync"/(n+".json"),v)
    for n,v in list(payload.items()):
        if v["status"]=="NOT_ENTERED":
            why=unavailable_reason("payload/"+n);v=method.unavailable_payload(why["status"]);v.update(why);payload[n]=v
        dump(output/"blind/payload"/(n+".json"),v)
    for n,v in list(quality.items()):
        if v["status"]=="NOT_ENTERED":quality[n]=c73.shared.unavailable_quality(unavailable_reason("quality/"+n)["status"],unavailable_reason("quality/"+n))
        dump(output/"quality"/(n+".json"),quality[n])
    for a,v in preclip.items():
        if v["status"]=="NOT_ENTERED":
            why=unavailable_reason(a+"/preclip");v.update(why);v["rows"]=[dict(frame=t,**why) for t in range(181)]
        dump(output/"quality/preclip"/(a+".json"),v)
    # Every score/vote/failure row is durable before opening truth.
    dump(output/"numeric_completion.json",dict(completed_unix=time.time(),sync_tables=8,payload_tables=4,
        truth_read=False,all_fixed_rows_persisted=True))
    truth=record("truth",lambda:read_truth(truth_path),Path("references/truth.json"))
    if truth is None:truth=dict(status="UNAVAILABLE",bits=None)
    value=record("evaluation",lambda:evaluate(sync,payload,truth),Path("evaluation.json"))
    result.update(evidence_level="DIRECTION_EVIDENCE_ONLY",identity_status="UNCALIBRATED",phase_status="UNCALIBRATED",quality=quality,preclip=preclip,
        truth_status=truth["status"],counts=ledger.counts(),calls=ledger.data,seconds=time.perf_counter()-started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        optional_file_fsync_unsupported_errnos=sorted(c73.SYNC_UNAVAILABLE_ERRNOS))
    if backend.device=="cuda":result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    scientific=[v for n,v in ledger.data.items() if n not in ("truth","evaluation")]
    result["status"]="EXECUTION_COMPLETE" if all(v["status"]=="RETURNED" for v in scientific) and value is not None and not result.get("fatal_error") else "INCOMPLETE"
    dump(output/"result.json",result);checkpoint();return result
